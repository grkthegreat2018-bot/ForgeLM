"""R39-6: Test-Time Scaling — FFS + Beam Search + MCTS.

Three test-time compute scaling strategies that trade extra inference
FLOPs for higher answer quality.  All three operate on a *model* object
that exposes a ``generate(prompt: str, **kwargs) -> str`` interface
(matching ``ForgeEngine.generate`` / ``ForgeEngine.generate_batch``).

Strategies
----------
FirstFinishSearch (FFS)
    Launch *N* independent stochastic samples and return the first one
    that reaches a natural stop (EOS / end-of-sentence).  Cheap and
    surprisingly effective (+15 % AIME in published reports).

BeamSearch
    Classic width-*K* beam search over cumulative log-probabilities.
    At every step all live beams are expanded, then pruned to the top-*K*
    by total log-prob.  The highest-scoring finished beam is returned.

MCTSDecoder
    Monte-Carlo Tree Search over partial sequences using the UCB1
    formula::

        UCB = Q(s, a) + c * sqrt(ln(N) / n(s, a))

    Each node is a partial sequence; expansion samples *n_children*
    continuations.  Values are back-propagated from leaf rollouts.
    The highest-value leaf after *n_iterations* is returned.

Design notes
------------
* The model interface is intentionally minimal (``generate`` returning a
  string) so the same code works with ``ForgeEngine`` and with the mock
  models used in unit tests.
* A lightweight token-probability hook (``next_token_logprobs``) is
  *optionally* supported by the model.  When present, BeamSearch uses
  real log-probs; when absent, a deterministic length-normalised
  surrogate is used so the algorithm still runs end-to-end on mock
  models.
* Everything is CPU-friendly — no CUDA tensors required.
"""
from __future__ import annotations

import math
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

__all__ = [
    "FirstFinishSearch",
    "BeamSearch",
    "MCTSDecoder",
    "GenerativeModel",
]


# ─── Model protocol ──────────────────────────────────────────────────────

class GenerativeModel(Protocol):
    """Minimal interface a model must implement for test-time scaling."""

    def generate(self, prompt: str, **kwargs: Any) -> str: ...


# ─── Helpers ─────────────────────────────────────────────────────────────

# Tokens / patterns that signal a finished generation.
_STOP_PATTERNS = [
    re.compile(r"<\|endoftext\|>", re.IGNORECASE),
    re.compile(r"<\|im_end\|>", re.IGNORECASE),
    re.compile(r"</s>", re.IGNORECASE),
    re.compile(r"<eos>", re.IGNORECASE),
]


def _looks_finished(text: str) -> bool:
    """Heuristic: does *text* end on a natural stopping point?"""
    stripped = text.rstrip()
    if not stripped:
        return True
    for pat in _STOP_PATTERNS:
        if pat.search(stripped):
            return True
    # Ends with sentence-final punctuation or a code block close.
    return stripped[-1] in ".!?)]}\n"


def _get_logprobs(model: Any, prompt: str) -> Optional[list[float]]:
    """If the model exposes ``next_token_logprobs``, return a vocab-sized
    log-prob list for the next token given *prompt*; else ``None``."""
    fn = getattr(model, "next_token_logprobs", None)
    if fn is None:
        return None
    try:
        return fn(prompt)
    except Exception:
        return None


# ─── 1. First-Finish Search ──────────────────────────────────────────────

class FirstFinishSearch:
    """First-Finish Search: race *N* samples, return the first to finish.

    Launches ``n_samples`` independent stochastic generations in parallel
    (thread pool) and returns the first one whose output reaches a
    natural stopping point.  If none finish within ``max_tokens`` the
    shortest completed sample is returned.
    """

    def __init__(self, n_samples: int = 8, max_tokens: int = 512,
                 temperature: float = 0.8, top_p: float = 0.95,
                 n_workers: Optional[int] = None):
        self.n_samples = n_samples
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.n_workers = n_workers

    # -- public API ---------------------------------------------------------

    def generate(self, model: GenerativeModel, prompt: str,
                 n_samples: Optional[int] = None,
                 max_tokens: Optional[int] = None) -> str:
        """Run FFS and return the first-finished (or best) sample."""
        n = n_samples if n_samples is not None else self.n_samples
        mt = max_tokens if max_tokens is not None else self.max_tokens
        if n <= 1:
            return model.generate(
                prompt, max_new_tokens=mt,
                temperature=self.temperature, top_p=self.top_p)

        samples: list[str] = []
        # Use threads so concurrent samples can race.
        workers = self.n_workers or min(n, 8)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {
                pool.submit(
                    self._one_sample, model, prompt, mt, seed): seed
                for seed in range(n)
            }
            # Collect results as they complete; short-circuit on finish.
            best: Optional[str] = None
            best_len = math.inf
            for fut in as_completed(futs):
                out = fut.result()
                samples.append(out)
                if _looks_finished(out):
                    return out
                # Track shortest as fallback.
                if len(out) < best_len:
                    best_len = len(out)
                    best = out
        return best if best is not None else (samples[0] if samples else "")

    # -- internals ----------------------------------------------------------

    def _one_sample(self, model: GenerativeModel, prompt: str,
                    max_tokens: int, seed: int) -> str:
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        # Allow the model to consume a seed if it supports one.
        if hasattr(model, "seed"):
            try:
                model.seed(seed)  # type: ignore[attr-defined]
            except Exception:
                pass
        elif "seed" in getattr(model, "generate_kwargs", {}):
            kwargs["seed"] = seed
        return model.generate(prompt, **kwargs)


# ─── 2. Beam Search ──────────────────────────────────────────────────────

@dataclass
class _Beam:
    """A single beam: accumulated text + cumulative log-prob + finished flag."""
    text: str = ""
    logprob: float = 0.0
    finished: bool = False
    tokens: list[str] = field(default_factory=list)


class BeamSearch:
    """Width-*K* beam search over cumulative log-probabilities.

    At each step every live beam is expanded by one token, all candidates
    are scored by cumulative log-probability, and the top-*K* survive.
    The highest-scoring *finished* beam is returned (or the best partial
    beam if none finish).
    """

    def __init__(self, beam_width: int = 4, max_tokens: int = 512,
                 length_penalty: float = 0.0):
        self.beam_width = beam_width
        self.max_tokens = max_tokens
        self.length_penalty = length_penalty

    # -- public API ---------------------------------------------------------

    def generate(self, model: GenerativeModel, prompt: str,
                 beam_width: Optional[int] = None,
                 max_tokens: Optional[int] = None) -> str:
        bw = beam_width if beam_width is not None else self.beam_width
        mt = max_tokens if max_tokens is not None else self.max_tokens
        return self._search(model, prompt, bw, mt)

    # -- internals ----------------------------------------------------------

    def _search(self, model: GenerativeModel, prompt: str,
                beam_width: int, max_tokens: int) -> str:
        beams: list[_Beam] = [_Beam(text="", logprob=0.0)]
        finished: list[_Beam] = []

        for _step in range(max_tokens):
            if not beams:
                break
            candidates: list[_Beam] = []
            for beam in beams:
                if beam.finished:
                    finished.append(beam)
                    continue
                candidates.extend(self._expand(model, prompt, beam, beam_width))
            if not candidates:
                break
            # Prune to top-K by score (length-normalised log-prob).
            candidates.sort(key=lambda b: self._score(b), reverse=True)
            beams = candidates[:beam_width]
            # Move finished beams aside.
            still_live: list[_Beam] = []
            for b in beams:
                if b.finished:
                    finished.append(b)
                else:
                    still_live.append(b)
            beams = still_live
            if not beams:
                break

        pool = finished if finished else beams
        if not pool:
            return ""
        pool.sort(key=lambda b: self._score(b), reverse=True)
        return pool[0].text

    def _expand(self, model: GenerativeModel, prompt: str,
                beam: _Beam, beam_width: int) -> list[_Beam]:
        """Expand *beam* by sampling up to *beam_width* candidate next tokens."""
        full_prompt = prompt + beam.text
        logprobs = _get_logprobs(model, full_prompt)
        if logprobs is not None:
            # Real log-probs available — pick top-K tokens.
            indexed = sorted(
                enumerate(logprobs), key=lambda iv: iv[1], reverse=True)
            top = indexed[:beam_width]
            out: list[_Beam] = []
            for tok_id, lp in top:
                tok_str = self._token_to_str(model, tok_id)
                new_text = beam.text + tok_str
                out.append(_Beam(
                    text=new_text,
                    logprob=beam.logprob + lp,
                    tokens=beam.tokens + [tok_str],
                    finished=_looks_finished(new_text),
                ))
            return out
        # No log-prob hook: fall back to a single greedy-ish expansion using
        # the model's own generate with 1 token, plus stochastic siblings.
        out: list[_Beam] = []
        base_kwargs: dict[str, Any] = {"max_new_tokens": 1}
        # Primary (greedy) continuation.
        primary = model.generate(full_prompt, **base_kwargs)
        delta = primary[len(beam.text):] if primary.startswith(beam.text) else primary
        if not delta:
            delta = " "
        new_text = beam.text + delta[:1]
        out.append(_Beam(
            text=new_text,
            logprob=beam.logprob + self._surrogate_logprob(len(beam.tokens)),
            tokens=beam.tokens + [delta[:1]],
            finished=_looks_finished(new_text),
        ))
        # Stochastic siblings (different temperatures) for beam diversity.
        for t in (0.7, 1.0, 1.3):
            if len(out) >= beam_width:
                break
            try:
                alt = model.generate(full_prompt, max_new_tokens=1,
                                     temperature=t)
            except TypeError:
                alt = model.generate(full_prompt, max_new_tokens=1)
            adelta = alt[len(beam.text):] if alt.startswith(beam.text) else alt
            if not adelta:
                adelta = " "
            atext = beam.text + adelta[:1]
            # Deduplicate by text.
            if any(o.text == atext for o in out):
                continue
            out.append(_Beam(
                text=atext,
                logprob=beam.logprob + self._surrogate_logprob(
                    len(beam.tokens)) - 0.1 * t,
                tokens=beam.tokens + [adelta[:1]],
                finished=_looks_finished(atext),
            ))
        return out

    def _score(self, beam: _Beam) -> float:
        """Length-normalised cumulative log-prob."""
        n = max(len(beam.tokens), 1)
        return beam.logprob / (n ** self.length_penalty)

    @staticmethod
    def _surrogate_logprob(step: int) -> float:
        """Deterministic surrogate log-prob when real ones are unavailable."""
        return -math.log(step + 2)

    @staticmethod
    def _token_to_str(model: Any, token_id: int) -> str:
        """Decode a token id to a string if the model supports it."""
        tok = getattr(model, "tokenizer", None) or getattr(model, "decode", None)
        if tok is not None and hasattr(tok, "decode"):
            try:
                return tok.decode([token_id])
            except Exception:
                pass
        if hasattr(model, "id_to_token"):
            try:
                return model.id_to_token(token_id)  # type: ignore[attr-defined]
            except Exception:
                pass
        # Fallback: represent token as a single char.
        return chr(32 + (token_id % 95))


# ─── 3. MCTS Decoder ─────────────────────────────────────────────────────

@dataclass
class _MCTSNode:
    """A node in the MCTS search tree (partial sequence)."""
    text: str
    parent: Optional["_MCTSNode"] = None
    children: list["_MCTSNode"] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0  # cumulative reward from rollouts
    unexpanded: list[str] = field(default_factory=list)  # pending child texts
    terminal: bool = False

    # UCB helpers -------------------------------------------------------
    @property
    def n(self) -> int:
        return self.visits

    def q(self) -> float:
        """Mean value of this node."""
        return (self.value / self.visits) if self.visits > 0 else 0.0

    def ucb(self, parent_n: int, c: float) -> float:
        """UCB1 score: exploitation + exploration."""
        if self.visits == 0:
            return math.inf
        exploit = self.q()
        explore = c * math.sqrt(math.log(max(parent_n, 1)) / self.visits)
        return exploit + explore


class MCTSDecoder:
    """Monte-Carlo Tree Search decoder using UCB1 selection.

    Each node represents a partial sequence.  Selection walks down the
    tree picking the child with the highest UCB score.  Expansion
    samples ``n_children`` continuations.  Simulation runs a rollout to
    completion (or ``max_tokens``).  Backpropagation updates visit
    counts and cumulative values along the path.

    After ``n_iterations`` the highest-value leaf is returned.
    """

    def __init__(self, n_iterations: int = 32, n_children: int = 4,
                 c: float = 1.414, max_tokens: int = 512,
                 rollout_temperature: float = 0.7):
        self.n_iterations = n_iterations
        self.n_children = n_children
        self.c = c
        self.max_tokens = max_tokens
        self.rollout_temperature = rollout_temperature

    # -- public API ---------------------------------------------------------

    def generate(self, model: GenerativeModel, prompt: str,
                 n_iterations: Optional[int] = None,
                 n_children: Optional[int] = None,
                 c: Optional[float] = None,
                 max_tokens: Optional[int] = None) -> str:
        ni = n_iterations if n_iterations is not None else self.n_iterations
        nc = n_children if n_children is not None else self.n_children
        cc = c if c is not None else self.c
        mt = max_tokens if max_tokens is not None else self.max_tokens
        return self._search(model, prompt, ni, nc, cc, mt)

    # -- internals ----------------------------------------------------------

    def _search(self, model: GenerativeModel, prompt: str,
                n_iterations: int, n_children: int, c: float,
                max_tokens: int) -> str:
        root = _MCTSNode(text="")
        self._expand_node(model, prompt, root, n_children, max_tokens)

        for _ in range(n_iterations):
            # 1. Selection
            node = self._select(root, c)
            # 2. Expansion (if not terminal)
            if not node.terminal and node.visits > 0:
                self._expand_node(
                    model, prompt, node, n_children, max_tokens)
                if node.children:
                    node = node.children[0]
            # 3. Simulation (rollout)
            reward = self._simulate(model, prompt, node, max_tokens)
            # 4. Backpropagation
            self._backprop(node, reward)

        # Return the highest-value leaf.
        best = self._best_leaf(root)
        return best.text if best is not None else ""

    # -- selection ----------------------------------------------------------

    def _select(self, root: _MCTSNode, c: float) -> _MCTSNode:
        """Walk down the tree choosing the max-UCB child until a leaf."""
        node = root
        while node.children:
            parent_n = node.visits
            best_child: Optional[_MCTSNode] = None
            best_score = -math.inf
            for child in node.children:
                score = child.ucb(parent_n, c)
                if score > best_score:
                    best_score = score
                    best_child = child
            if best_child is None:
                break
            node = best_child
        return node

    # -- expansion ----------------------------------------------------------

    def _expand_node(self, model: GenerativeModel, prompt: str,
                     node: _MCTSNode, n_children: int,
                     max_tokens: int) -> None:
        """Generate *n_children* candidate continuations for *node*."""
        if node.terminal:
            return
        full_prompt = prompt + node.text
        seen = {ch.text for ch in node.children}
        for i in range(n_children):
            try:
                cont = model.generate(
                    full_prompt, max_new_tokens=max(1, max_tokens // 4),
                    temperature=self.rollout_temperature + 0.1 * i)
            except TypeError:
                cont = model.generate(full_prompt, max_new_tokens=max(1, max_tokens // 4))
            # The continuation is the *new* part.
            delta = cont[len(node.text):] if cont.startswith(node.text) else cont
            if not delta:
                delta = " "
            child_text = node.text + delta
            if child_text in seen:
                # Perturb to keep diversity.
                child_text = node.text + delta + " "
                if child_text in seen:
                    continue
            seen.add(child_text)
            child = _MCTSNode(
                text=child_text, parent=node,
                terminal=_looks_finished(child_text))
            node.children.append(child)

    # -- simulation ---------------------------------------------------------

    def _simulate(self, model: GenerativeModel, prompt: str,
                  node: _MCTSNode, max_tokens: int) -> float:
        """Rollout from *node* to completion; return a reward in [0, 1]."""
        if node.terminal:
            return self._reward(node.text)
        full_prompt = prompt + node.text
        try:
            rollout = model.generate(
                full_prompt, max_new_tokens=max_tokens,
                temperature=self.rollout_temperature)
        except TypeError:
            rollout = model.generate(full_prompt, max_new_tokens=max_tokens)
        delta = rollout[len(node.text):] if rollout.startswith(node.text) else rollout
        full_text = node.text + delta
        return self._reward(full_text)

    @staticmethod
    def _reward(text: str) -> float:
        """Heuristic reward for a completed sequence in [0, 1].

        Favour finished, non-trivial, non-repetitive text.
        """
        if not text.strip():
            return 0.0
        score = 0.3  # base for producing something
        if _looks_finished(text):
            score += 0.4
        # Length bonus (diminishing).
        n = len(text.split())
        score += min(0.2, n / 50.0)
        # Repetition penalty.
        words = text.lower().split()
        if words:
            unique = len(set(words)) / len(words)
            score += 0.1 * unique
        return min(1.0, score)

    # -- backprop -----------------------------------------------------------

    def _backprop(self, node: _MCTSNode, reward: float) -> None:
        cur: Optional[_MCTSNode] = node
        while cur is not None:
            cur.visits += 1
            cur.value += reward
            cur = cur.parent

    # -- best leaf ----------------------------------------------------------

    def _best_leaf(self, root: _MCTSNode) -> Optional[_MCTSNode]:
        """Return the leaf with the highest mean value (min 1 visit)."""
        best: Optional[_MCTSNode] = None
        best_q = -math.inf

        def walk(n: _MCTSNode) -> None:
            nonlocal best, best_q
            if not n.children:  # leaf
                if n.visits > 0:
                    q = n.q()
                else:
                    q = self._reward(n.text)
                if q > best_q:
                    best_q = q
                    best = n
            for ch in n.children:
                walk(ch)

        walk(root)
        return best
