"""Quality / skill / compute evaluation for epoch comparison.

Each candidate epoch is scored on three axes so the epoch manager can decide
which checkpoint stays (best) and which is archived (loser):

  quality — mean cross-entropy loss on a held-out set of prompts built from
            the discovery DB (thoughts → instruction prompts). Lower is better;
            we map to a 0..1 score where higher is better.
  skill   — fraction of a fixed coding-probe battery the model solves
            correctly (executed in the sandbox). 0..1, higher better.
  compute — tokens/sec on a fixed generation task. Normalized to 0..1 against
            a reference rate so we can compare across epochs fairly.

composite = 0.5*quality + 0.3*skill + 0.2*compute  (tunable).

Reuses:
  - research.training.training_utils.compute_ce_loss, vram_gb
  - research.model_loader.generate_text
  - research.self_play.self_play_sandbox.SandboxExecutor (skill probes)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch

from research.self_play.discovery.discovery_db import DiscoveryDB


# Fixed coding-probe battery for the skill axis. Each probe is a prompt that
# should produce a function definition; we run the generated code against the
# provided asserts. Kept small + deterministic so cross-epoch comparison is fair.
_SKILL_PROBES = [
    {
        "prompt": "def is_prime(n):\n    \"\"\"Return True if n is prime.\"\"\"\n",
        "test": "print(is_prime(2), is_prime(7), is_prime(8), is_prime(1))",
        "expected": "True True False False",
    },
    {
        "prompt": "def fib(n):\n    \"\"\"Return the n-th Fibonacci number (fib(0)=0).\"\"\"\n",
        "test": "print(fib(0), fib(1), fib(10))",
        "expected": "0 1 55",
    },
    {
        "prompt": "def reverse_string(s):\n    \"\"\"Return s reversed.\"\"\"\n",
        "test": "print(reverse_string('abc'))",
        "expected": "cba",
    },
    {
        "prompt": "def max_of_list(xs):\n    \"\"\"Return the maximum of a non-empty list.\"\"\"\n",
        "test": "print(max_of_list([3,1,4,1,5,9,2,6]))",
        "expected": "9",
    },
    {
        "prompt": "def count_vowels(s):\n    \"\"\"Return the number of vowels in s.\"\"\"\n",
        "test": "print(count_vowels('hello world'))",
        "expected": "3",
    },
]

_REF_TOK_S = 80.0  # reference tokens/sec for compute normalization (lfm25-1.2b)


@dataclass
class EpochScore:
    quality: float
    skill: float
    compute: float
    composite: float
    skill_passed: int
    skill_total: int
    loss: float
    tok_s: float
    detail: dict = field(default_factory=dict)

    def as_db_tuple(self) -> dict:
        return {"quality": self.quality, "skill": self.skill,
                "compute": self.compute, "composite": self.composite}


def _build_eval_prompts(db: DiscoveryDB, n: int = 16) -> list[str]:
    """Held-out prompts from the DB's thoughts (model hasn't memorized these
    as completions — they're instructions, not answers)."""
    rows = db.query(
        "SELECT content FROM thoughts WHERE kind IN ('think','sudo_think') "
        "AND length(content) > 20 ORDER BY RANDOM() LIMIT ?", (n,))
    if len(rows) < 4:  # not enough thoughts yet — fall back to generic probes
        return ["Explain what a prime number is.",
                "Write a function that adds two numbers.",
                "What is recursion?",
                "Describe how a hash map works."]
    return [f"Discuss: {r['content'][:120]}" for r in rows]


def _quality_loss(model, tokenizer, prompts: list[str],
                  device: str, max_new: int = 48) -> float:
    """Mean token-level cross-entropy of model continuations on the prompts.

    Lower loss = the model is more confident/coherent on its own discovery
    topics. We generate a short continuation and measure the model's NLL on
    those generated tokens (on-policy quality, per the research finding that
    on-policy signals resist forgetting better than off-policy SFT).
    """
    import torch.nn.functional as F
    losses = []
    model.eval()
    for p in prompts:
        enc = tokenizer(p, return_tensors="pt")
        ids = enc.input_ids.to(device) if hasattr(enc, "to") else \
            torch.tensor(enc["input_ids"]).to(device)
        if ids.shape[1] < 2:
            continue
        with torch.no_grad():
            out = model(ids)
            logits = out[0] if isinstance(out, tuple) else out
            # NLL of predicting token t+1 from position t over the prompt itself
            lp = F.log_softmax(logits[0, :-1].float(), dim=-1)
            tgt = ids[0, 1:]
            tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            losses.append(-tok_lp.mean().item())
    return sum(losses) / max(len(losses), 1)


def _skill_score(model, tokenizer, device: str, max_new: int = 120) -> tuple[int, int, list[str]]:
    """Run the coding-probe battery. Returns (passed, total, details)."""
    from research.model_loader import ModelLoader
    from research.self_play.self_play_sandbox import SandboxExecutor
    exec = SandboxExecutor(timeout_s=5.0, use_persistent=False)
    passed = 0
    details = []
    for probe in _SKILL_PROBES:
        try:
            completion = ModelLoader.generate_text(model, tokenizer, probe["prompt"],
                                       max_new_tokens=max_new, temperature=0.0)
        except Exception as e:
            details.append(f"{probe['test']} -> gen error: {e}")
            continue
        code = probe["prompt"] + completion + "\n" + probe["test"]
        try:
            res = exec.execute(code, expected_output=probe["expected"])
            ok = (res.get("returncode") == 0 and
                  res.get("output_matches_expected"))
            if ok:
                passed += 1
                details.append(f"{probe['test']} -> PASS")
            else:
                details.append(f"{probe['test']} -> FAIL rc={res.get('returncode')}")
        except Exception as e:
            details.append(f"{probe['test']} -> exec error: {e}")
    return passed, len(_SKILL_PROBES), details


def _compute_score(model, tokenizer, device: str,
                   prompt: str = "Write a short poem about the sea.",
                   max_new: int = 64) -> float:
    """tokens/sec on a fixed generation task."""
    from research.model_loader import ModelLoader
    t0 = time.time()
    out = ModelLoader.generate_text(model, tokenizer, prompt, max_new_tokens=max_new,
                        temperature=0.0)
    dt = max(time.time() - t0, 1e-3)
    # Approx token count via tokenizer (rough but consistent across epochs).
    try:
        n = len(tokenizer(out)["input_ids"])
    except Exception:
        n = max(len(out.split()), 1)
    return n / dt


def evaluate(model, tokenizer, db: DiscoveryDB, device: str = "cuda") -> EpochScore:
    """Score a model on quality + skill + compute. Returns an EpochScore."""
    prompts = _build_eval_prompts(db)
    loss = _quality_loss(model, tokenizer, prompts, device)
    # Map loss -> quality score: loss 0 -> 1.0, loss 5+ -> ~0.0 (sigmoid-ish).
    import math
    quality = 1.0 / (1.0 + math.exp(loss / 2.0))

    passed, total, details = _skill_score(model, tokenizer, device)
    skill = passed / total if total else 0.0

    tok_s = _compute_score(model, tokenizer, device)
    compute = min(tok_s / _REF_TOK_S, 1.0)

    composite = 0.5 * quality + 0.3 * skill + 0.2 * compute
    return EpochScore(quality=quality, skill=skill, compute=compute,
                      composite=composite, skill_passed=passed,
                      skill_total=total, loss=loss, tok_s=tok_s,
                      detail={"skill_details": details, "n_prompts": len(prompts)})
