"""Trained decision scorer — Tier 1 of the System One integration.

A linear probe on the last-token hidden state that *verifies* a candidate
answer in context:

    s(q, c) = w . h(prompt + " " + candidate) + b

The prompt is the same one SystemOneEvaluator builds (state + question +
``assistant\n<think>\n</think>\nAnswer:``); the candidate answer string is
appended and the hidden state at the final token is scored.  Per-question
softmax over candidate scores yields the answer distribution.

This is the RLCD-lite objective: the action is a probability distribution
over candidates and the loss is a strictly proper scoring rule (log-loss /
cross-entropy) against verifiable outcomes — the same family TypeSafe uses
to calibrate Jev, minus the proprietary architecture.

Follows the SIREN convention (forge/engine/safety/siren.py): small linear
probes over hidden states, trained offline, bolted on — no checkpoint or
architecture changes.  ~10 KB of weights for a 2560-d model.

Usage — train:
    from forge.engine.decision_head import fit_decision_scorer
    scorer, metrics = fit_decision_scorer(
        model, tokenizer, device, dataset)
    # dataset: [(state, question_spec, correct_candidate_idx), ...]
    scorer.save("research/checkpoints/decision_scorer.pt")

Usage — inference:
    engine.load_decision_scorer("research/checkpoints/decision_scorer.pt")
    out = engine.decide(state, questions)   # head probs replace Tier-0
"""
from __future__ import annotations

import logging
import os
import time
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

__all__ = [
    "DecisionScorer",
    "fit_decision_scorer",
    "expected_calibration_error",
]

SCORE_HEAD_VERSION = 1


def expected_calibration_error(probs: torch.Tensor,
                               labels: torch.Tensor,
                               bins: int = 10) -> float:
    """ECE for binary or argmax-conf/accuracy style probabilities.

    For binary: probs = P(yes), labels ∈ {0,1}.  For multiclass pass the
    top-class probability with a correctness indicator instead.
    """
    probs = probs.float().flatten()
    labels = labels.float().flatten()
    e = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        m = (probs >= lo) & (probs < hi if b < bins - 1 else probs <= hi)
        if m.any():
            e += m.float().mean().item() * abs(
                probs[m].mean().item() - labels[m].mean().item())
    return e


class DecisionScorer(nn.Module):
    """Linear (or small-MLP) verifier head on last-token hidden states.

    score(hidden) -> real-valued logit; softmax over a question's
    candidate scores gives the answer distribution.  ``temperature``
    scales logits before the softmax (fitted post-training, >1 softens).
    """

    def __init__(self, d_model: int, hidden_dim: int = 0,
                 temperature: float = 1.0):
        super().__init__()
        self.d_model = d_model
        if hidden_dim and hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, 1))
        else:
            self.net = nn.Linear(d_model, 1)
        self.temperature = temperature

    def score(self, hidden: torch.Tensor) -> torch.Tensor:
        """(B, d_model) -> (B,) raw scores."""
        return self.net(hidden.float()).squeeze(-1)

    def probs(self, scores: Sequence[float]) -> list[float]:
        """Candidate scores -> calibrated softmax distribution."""
        t = torch.tensor(list(scores), dtype=torch.float32)
        return F.softmax(t / max(self.temperature, 1e-6), dim=-1).tolist()

    # ── persistence ────────────────────────────────────────────────────

    def save(self, path: str, meta: dict | None = None) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            "version": SCORE_HEAD_VERSION,
            "d_model": self.d_model,
            "temperature": self.temperature,
            "state_dict": self.state_dict(),
            "meta": meta or {},
        }, path)

    @classmethod
    def load(cls, path: str, device: str | torch.device = "cpu"
             ) -> "DecisionScorer":
        blob = torch.load(path, map_location="cpu", weights_only=True)
        version = blob.get("version", 0)
        if version > SCORE_HEAD_VERSION:
            raise ValueError(
                f"decision scorer version {version} > supported "
                f"{SCORE_HEAD_VERSION}: {path}")
        hidden_dim = 0
        w0 = blob["state_dict"].get("net.0.weight")
        if w0 is not None and w0.shape[0] != 1:
            hidden_dim = w0.shape[0]
        scorer = cls(d_model=blob["d_model"], hidden_dim=hidden_dim,
                     temperature=blob.get("temperature", 1.0))
        scorer.load_state_dict(blob["state_dict"])
        return scorer.to(device).eval()


# ── training ───────────────────────────────────────────────────────────

def _harvest_hidden(evaluator, prompt_ids: list[int],
                    cand_ids: list[list[int]], pad_id: int) -> torch.Tensor:
    """Hidden at the final token of each ``prompt + candidate`` row.

    Shares SystemOneEvaluator._candidate_hidden so training sees exactly
    what inference scores.  Returns (n_candidates, d_model) CPU float32.
    """
    return evaluator._candidate_hidden(prompt_ids, cand_ids,
                                       pad_id).float().cpu()


def fit_decision_scorer(
    model, tokenizer, device,
    dataset: Sequence[tuple],
    *,
    context_limit: int | None = None,
    max_batch: int = 24,
    epochs: int = 4,
    lr: float = 3e-3,
    weight_decay: float = 1e-4,
    val_frac: float = 0.15,
    hidden_dim: int = 0,
    seed: int = 0,
    progress: bool = True,
) -> tuple[DecisionScorer, dict]:
    """Train a DecisionScorer on (state, question_spec, correct_idx) rows.

    Each example contributes one softmax group: hidden states for every
    candidate of the question, with ``correct_idx`` marking the verifiable
    answer.  Loss is group softmax cross-entropy — a strictly proper
    scoring rule, the RLCD-lite objective.

    Returns (scorer, metrics).  Metrics include val accuracy, ECE and
    Brier plus harvest/train wall-clock times.
    """
    from forge.engine.decide import (  # local import: avoid cycles
        DEFAULT_CONTEXT_LIMIT, SystemOneEvaluator, _jsonable,
        _parse_question)

    torch.manual_seed(seed)
    device = torch.device(device)
    evaluator = SystemOneEvaluator(model, tokenizer, device,
                                   max_batch=max_batch)
    limit = context_limit or DEFAULT_CONTEXT_LIMIT
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None) or 0

    # ── 1. harvest hidden states per candidate ─────────────────────────
    t0 = time.time()
    feats: list[torch.Tensor] = []   # per-example (C_i, d)
    labels: list[int] = []
    n_fwd = 0
    with torch.inference_mode():
        for state, spec, correct in dataset:
            q = _parse_question("q", spec)
            prefix, suffix = evaluator._prompt_parts(q)
            st = evaluator._fit_state(_jsonable(state), prefix, suffix, limit)
            prompt_ids = evaluator._encode(prefix + st + suffix,
                                           special=True)
            cand_ids = [evaluator._encode(c, special=False)
                        for c in q.candidates]
            cand_ids = [c if c else [0] for c in cand_ids]
            feats.append(_harvest_hidden(evaluator, prompt_ids, cand_ids,
                                         pad_id))
            labels.append(int(correct))
            n_fwd += len(cand_ids)
    harvest_s = time.time() - t0
    # inference tensors cannot backprop — clone into normal tensors
    feats = [f.clone() for f in feats]

    d_model = feats[0].shape[1]
    # ── 2. train/val split (by example, not by row) ────────────────────
    n = len(feats)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_val = max(1, int(n * val_frac)) if n > 4 else 0
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    Xtr = [feats[i].to(device) for i in tr_idx]
    ytr = [labels[i] for i in tr_idx]
    Xva = [feats[i].to(device) for i in val_idx]
    yva = [labels[i] for i in val_idx]

    scorer = DecisionScorer(d_model, hidden_dim=hidden_dim).to(device)
    opt = torch.optim.AdamW(scorer.parameters(), lr=lr,
                            weight_decay=weight_decay)

    t1 = time.time()
    for _ep in range(epochs):
        ep_perm = torch.randperm(len(Xtr), generator=g).tolist()
        for i in ep_perm:
            opt.zero_grad()
            s = scorer.score(Xtr[i])
            loss = F.cross_entropy(s.unsqueeze(0),
                                   torch.tensor([ytr[i]], device=device))
            loss.backward()
            opt.step()
    train_s = time.time() - t1

    # ── 3. temperature scaling on the validation split ─────────────────
    metrics: dict = {"harvest_s": round(harvest_s, 2),
                     "train_s": round(train_s, 2),
                     "n_examples": n, "n_val": n_val,
                     "forwards": n_fwd}
    if Xva:
        with torch.no_grad():  # no_grad (not inference_mode): sva feeds LBFGS backward
            sva = [scorer.score(x) for x in Xva]
        # single-scalar temperature fit
        t_log = torch.zeros(1, device=device, requires_grad=True)
        topt = torch.optim.LBFGS([t_log], lr=0.1, max_iter=50)
        yv = torch.tensor(yva, device=device)

        def _nll():
            topt.zero_grad()
            t = t_log.exp()
            l = sum(F.cross_entropy((s / t).unsqueeze(0), yv[i:i+1])
                    for i, s in enumerate(sva))
            l.backward()
            return l
        topt.step(_nll)
        scorer.temperature = float(t_log.exp().item())

        with torch.inference_mode():
            correct = 0
            top_probs, hit = [], []
            for i, x in enumerate(Xva):
                p = scorer.probs(scorer.score(x).tolist())
                win = max(range(len(p)), key=lambda j: p[j])
                ok = float(win == yva[i])
                correct += ok
                top_probs.append(max(p))
                hit.append(ok)
        metrics["val_acc"] = round(correct / len(Xva), 4)
        metrics["val_ece"] = round(expected_calibration_error(
            torch.tensor(top_probs), torch.tensor(hit)), 4)
        briers = []
        for i, x in enumerate(Xva):
            p = torch.tensor(scorer.probs(scorer.score(x).tolist()))
            onehot = torch.zeros(len(p)); onehot[yva[i]] = 1.0
            briers.append(((p - onehot) ** 2).sum().item())
        metrics["val_brier"] = round(sum(briers) / len(briers), 4)
    if progress:
        logger.info("[DecisionScorer] trained: %s", metrics)
    return scorer.to(device).eval(), metrics
