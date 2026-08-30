"""SharedCheckerModel — LLM-as-judge for complex domain answer evaluation.

A single shared LLM instance that evaluates answers against requirements for
domains that need semantic checking (DomainSpec.checker_type == "llm_judge").

Design:
  - Singleton: one instance shared across all domains. Boots once, stays in
    memory. Access via ``get_checker()``.
  - Ultra-compact: uses the "lfm25_tiny" config (d_model=128, 4 layers,
    vocab=256) for minimal compute. Lives on CPU by default, or on a small
    GPU partition.
  - LLM-as-judge: given (question, answer, requirements) -> score 0-100.
  - Sleep/wake: follows ForgeEngine.sleep(level=1)/wake() pattern to release
    VRAM when not actively checking. Wakes on demand.
  - Bounded LRU cache (max 10 000 entries) to avoid re-evaluating identical
    inputs.
  - Thread-safe: multiple domains can call check() concurrently.
  - Fallback: if the LLM fails to produce a parseable score (or can't be
    built), falls back to HeuristicChecker (keyword matching).

The checker model is SEPARATE from the gen model. The gen model produces
answers; the checker model evaluates them. The checker runs on MINIMAL
compute (CPU or small GPU partition) and does NOT compete with gen models
for GPU.

The model is built with random init (no checkpoint needed). When trained/
fine-tuned, it will produce meaningful scores. Until then, the random-init
output will almost always fail score parsing, triggering the heuristic
fallback — this is the designed behavior.
"""
from __future__ import annotations

import hashlib
import re
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import torch

# ── Constants ──────────────────────────────────────────────────────────────

_DEFAULT_TOKENIZER_PATH = str(
    Path(__file__).resolve().parents[1] / "checkpoints" / "lfm25_tokenizer"
)
_CACHE_MAX_ENTRIES = 10_000
_SCORE_TOKENS = 20          # enough for "Score: 85" or "85/100" style responses
_MAX_BATCH_SIZE = 32        # cap batched generation to avoid OOM on tiny model

_PROMPT_TEMPLATE = (
    "Evaluate the following answer. Score 0-100.\n"
    "\n"
    "Question: {question}\n"
    "Answer: {answer}\n"
    "Requirements: {requirements}\n"
    "\n"
    "Score (0-100):"
)

# Stop words excluded from keyword extraction in the heuristic checker.
_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "be", "to", "of", "in", "for", "and",
    "or", "not", "with", "that", "this", "it", "as", "by", "on", "at",
    "from", "must", "should", "will", "can", "may", "have", "has", "do",
    "does", "all", "any", "if", "then", "else", "when", "where", "which",
    "who", "whom", "whose", "what", "why", "how", "than", "but", "so",
    "no", "nor", "only", "own", "same", "such", "too", "very", "just",
})


# ── Bounded LRU Cache ──────────────────────────────────────────────────────

class _BoundedLRUCache:
    """Thread-safe LRU cache with a hard entry cap.

    Uses an OrderedDict internally: most-recently-used entries are kept at
    the end; eviction pops from the front (least-recently-used).
    """

    def __init__(self, max_entries: int = _CACHE_MAX_ENTRIES) -> None:
        self._data: OrderedDict[str, float] = OrderedDict()
        self._max = max_entries
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[float]:
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def put(self, key: str, value: float) -> None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = value
            while len(self._data) > self._max:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


# ── Heuristic Checker (fallback) ───────────────────────────────────────────

class HeuristicChecker:
    """Fallback checker — scores based on keyword overlap, length, and
    required-term presence.

    Scoring components (weighted):
      - Keyword overlap (60%): fraction of requirement keywords present in
        the answer.
      - Length score (25%): penalises answers that are too short (< 5 words)
        or too long (> 500 words). Sweet spot: 10-200 words.
      - Question relevance (15%): fraction of question keywords present in
        the answer.
    """

    def check(self, question: str, answer: str, requirements: str) -> float:
        answer_lower = answer.lower()
        req_lower = requirements.lower()
        question_lower = question.lower()

        # Extract keywords (alphabetic tokens, minus stop words).
        req_keywords = set(re.findall(r"[a-z]+", req_lower)) - _STOP_WORDS
        answer_words = set(re.findall(r"[a-z]+", answer_lower))
        question_keywords = set(re.findall(r"[a-z]+", question_lower)) - _STOP_WORDS

        # ── Keyword overlap score ──
        if req_keywords:
            matched = req_keywords & answer_words
            keyword_score = len(matched) / len(req_keywords) * 100.0
        else:
            # No extractable requirements — neutral score.
            keyword_score = 50.0

        # ── Length score ──
        word_count = len(answer.split())
        if word_count < 5:
            length_score = (word_count / 5.0) * 50.0
        elif word_count <= 200:
            length_score = 100.0
        elif word_count <= 500:
            # Gradual decline from 100 to 70 over 200-500 words.
            length_score = 100.0 - (word_count - 200) / 300.0 * 30.0
        else:
            # Steep decline past 500 words.
            length_score = max(20.0, 70.0 - (word_count - 500) / 50.0)

        # ── Question relevance score ──
        if question_keywords:
            q_overlap = len(answer_words & question_keywords) / len(question_keywords)
        else:
            q_overlap = 0.5
        relevance_score = q_overlap * 100.0

        # ── Weighted combination ──
        score = (
            keyword_score * 0.60
            + length_score * 0.25
            + relevance_score * 0.15
        )
        return max(0.0, min(100.0, score))


# ── Score Parsing ──────────────────────────────────────────────────────────

# Ordered by specificity — first match wins.
_SCORE_PATTERNS = [
    re.compile(r"score\s*[:=]?\s*(\d{1,3})", re.IGNORECASE),              # "Score: 85"
    re.compile(r"(\d{1,3})\s*/\s*100", re.IGNORECASE),                    # "85/100"
    re.compile(r"^\s*(\d{1,3})\s*$", re.MULTILINE),                       # just "85" on a line
    re.compile(r"\b(\d{1,3})\s*(?:points?|pts?)\s*$", re.IGNORECASE | re.MULTILINE),  # "85 points"
]


def _parse_score(text: str) -> Optional[float]:
    """Extract a numeric score (0-100) from the LLM's response text.

    Handles formats like "Score: 85", "85/100", "85", "85 points", etc.
    Returns None if no valid score is found.
    """
    if not text:
        return None
    for pattern in _SCORE_PATTERNS:
        match = pattern.search(text)
        if match:
            val = int(match.group(1))
            if 0 <= val <= 100:
                return float(val)
    # Last resort: first standalone 1-3 digit number in the text.
    fallback = re.search(r"\b(\d{1,3})\b", text)
    if fallback:
        val = int(fallback.group(1))
        if 0 <= val <= 100:
            return float(val)
    return None


# ── Shared Checker Model ───────────────────────────────────────────────────

class SharedCheckerModel:
    """Singleton LLM-as-judge that evaluates answers for complex domains.

    Boots once with a tiny random-init model and stays in memory. Uses
    sleep/wake (following ForgeEngine's pattern) to release VRAM when idle.
    Falls back to HeuristicChecker when the LLM can't produce a parseable
    score or when the model can't be built.

    Thread-safe: multiple domains can call check() concurrently.
    """

    def __init__(self, config_name: str = "lfm25_tiny",
                 device: str = "cpu",
                 tokenizer_path: str | None = None) -> None:
        self._lock = threading.RLock()
        self._cache = _BoundedLRUCache(max_entries=_CACHE_MAX_ENTRIES)
        self._heuristic = HeuristicChecker()
        self._is_sleeping = False
        self._model_failed = False
        self._device = torch.device(device)
        self._tokenizer_path = tokenizer_path or _DEFAULT_TOKENIZER_PATH

        # Attempt to build the LLM. If anything fails, mark as failed and
        # rely on the heuristic fallback for all checks.
        try:
            from research.config import get_config
            from research.model_loader import ModelLoader, unpack_output_with_kv
            from research.tokenizer_cache import get_tokenizer

            self._unpack_output = unpack_output_with_kv

            cfg = get_config(config_name, device=device)
            self._config = cfg
            self._vocab_size = cfg.vocab_size
            self._max_seq_len = cfg.max_seq_len

            # Build model with random init (no checkpoint needed).
            self._model = ModelLoader.build_model_fast(
                cfg, checkpoint_path=None)
            self._model.eval()

            # Load tokenizer (LFM2.5 tokenizer — vocab 65536).
            # Token IDs are modded by the model's vocab_size (256) before
            # feeding to the model, since the tiny config uses a reduced
            # vocab for minimal compute.
            self._tokenizer = get_tokenizer(self._tokenizer_path)

            n_params = sum(p.numel() for p in self._model.parameters())
            print(f"  [SharedCheckerModel] Built {config_name} on {device} "
                  f"({n_params / 1e6:.2f}M params, vocab={self._vocab_size})")
        except Exception as exc:
            print(f"  [SharedCheckerModel] Failed to build LLM checker: {exc}")
            print("  [SharedCheckerModel] Falling back to HeuristicChecker")
            self._model_failed = True
            self._model = None
            self._tokenizer = None
            self._config = None
            self._vocab_size = 256
            self._max_seq_len = 128
            self._unpack_output = None

    # ── Public API ──────────────────────────────────────────────────────

    def check(self, question: str, answer: str, requirements: str) -> float:
        """Evaluate an answer and return a score in [0, 100].

        Checks the cache first. On a miss, wakes the model if sleeping,
        generates an LLM response, parses the score, and falls back to the
        heuristic checker if parsing fails.
        """
        cache_key = self._cache_key(question, answer, requirements)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        with self._lock:
            score = self._check_uncached(question, answer, requirements)
            self._cache.put(cache_key, score)
            return score

    def check_batch(self, items: list[dict]) -> list[float]:
        """Evaluate multiple answers in one call.

        Checks the cache for all items first. For uncached items, runs a
        batched forward pass when possible (up to _MAX_BATCH_SIZE per batch).
        Falls back to the heuristic for any item where the LLM score can't
        be parsed.

        Each item dict must have keys: "question", "answer", "requirements".
        """
        if not items:
            return []

        results: list[Optional[float]] = [None] * len(items)
        uncached: list[tuple[int, dict]] = []

        # Phase 1: cache lookup.
        for i, item in enumerate(items):
            key = self._cache_key(
                item["question"], item["answer"], item["requirements"])
            cached = self._cache.get(key)
            if cached is not None:
                results[i] = cached
            else:
                uncached.append((i, item))

        # Phase 2: evaluate uncached items.
        if uncached:
            with self._lock:
                # Process in batches.
                for start in range(0, len(uncached), _MAX_BATCH_SIZE):
                    batch = uncached[start:start + _MAX_BATCH_SIZE]
                    scores = self._check_batch_uncached(batch)
                    for (idx, item), score in zip(batch, scores):
                        key = self._cache_key(
                            item["question"], item["answer"],
                            item["requirements"])
                        self._cache.put(key, score)
                        results[idx] = score

        return [s if s is not None else 0.0 for s in results]

    def wake(self) -> None:
        """Wake the model from sleep (restore to device).

        Follows ForgeEngine.wake() pattern: CPU -> device copy.
        No-op if already awake or if the model failed to build.
        """
        with self._lock:
            self._wake_unlocked()

    def sleep(self) -> None:
        """Release VRAM by offloading model weights to CPU.

        Follows ForgeEngine.sleep(level=1) pattern: move weights to CPU,
        clear CUDA cache. No-op if already sleeping, on CPU, or if the
        model failed to build.
        """
        with self._lock:
            if self._model_failed or self._model is None:
                return
            if self._is_sleeping:
                return
            if self._device.type == "cuda":
                self._model.to("cpu", non_blocking=True)
                torch.cuda.synchronize(self._device)
                torch.cuda.empty_cache()
                torch.cuda.synchronize(self._device)
            self._is_sleeping = True
            print("  [SharedCheckerModel] Sleep: weights offloaded to CPU")

    def clear_cache(self) -> None:
        """Clear the score cache."""
        self._cache.clear()

    # ── Internal: scoring ───────────────────────────────────────────────

    def _check_uncached(self, question: str, answer: str,
                        requirements: str) -> float:
        """Score a single uncached item (assumes lock is held)."""
        if self._model_failed or self._model is None:
            return self._heuristic.check(question, answer, requirements)

        # Wake if sleeping.
        if self._is_sleeping:
            self._wake_unlocked()

        prompt = _PROMPT_TEMPLATE.format(
            question=question, answer=answer, requirements=requirements)

        try:
            response = self._generate(prompt, max_new_tokens=_SCORE_TOKENS)
            score = _parse_score(response)
        except Exception:
            score = None

        if score is None:
            score = self._heuristic.check(question, answer, requirements)

        return score

    def _check_batch_uncached(
        self, batch: list[tuple[int, dict]]
    ) -> list[float]:
        """Score a batch of uncached items (assumes lock is held).

        Tries batched LLM generation first. Falls back to per-item
        heuristic for any item where the score can't be parsed.
        """
        items = [item for _, item in batch]

        if self._model_failed or self._model is None:
            return [
                self._heuristic.check(
                    item["question"], item["answer"], item["requirements"])
                for item in items
            ]

        # Wake if sleeping.
        if self._is_sleeping:
            self._wake_unlocked()

        prompts = [
            _PROMPT_TEMPLATE.format(
                question=item["question"],
                answer=item["answer"],
                requirements=item["requirements"])
            for item in items
        ]

        try:
            responses = self._generate_batch(prompts,
                                             max_new_tokens=_SCORE_TOKENS)
        except Exception:
            # Batch generation failed entirely — heuristic for all.
            return [
                self._heuristic.check(
                    item["question"], item["answer"], item["requirements"])
                for item in items
            ]

        scores: list[float] = []
        for response, item in zip(responses, items):
            score = _parse_score(response)
            if score is None:
                score = self._heuristic.check(
                    item["question"], item["answer"], item["requirements"])
            scores.append(score)
        return scores

    # ── Internal: generation ────────────────────────────────────────────

    def _generate(self, prompt: str, max_new_tokens: int = _SCORE_TOKENS) -> str:
        """Greedy-decode a response from a single prompt.

        Token IDs from the LFM2.5 tokenizer (vocab 65536) are modded by the
        model's vocab_size (256) to fit the tiny embedding table.
        """
        enc = self._tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=self._max_seq_len - max_new_tokens)
        ids = enc.input_ids if hasattr(enc, "input_ids") else enc["input_ids"]
        ids = (ids % self._vocab_size).to(self._device)

        generated: list[int] = []
        with torch.no_grad():
            for _ in range(max_new_tokens):
                if ids.shape[1] >= self._max_seq_len:
                    break
                out = self._model(ids, use_cache=False)
                logits, _ = self._unpack_output(out)
                next_tok = logits[0, -1, :].argmax(dim=-1)
                generated.append(int(next_tok.item()))
                ids = torch.cat(
                    [ids, next_tok.reshape(1, 1).to(self._device)], dim=1)

        return self._decode_tokens(generated)

    def _generate_batch(
        self, prompts: list[str], max_new_tokens: int = _SCORE_TOKENS
    ) -> list[str]:
        """Batched greedy generation — one forward pass per decode step for
        all prompts simultaneously.

        Sequences are left-padded to the same length so the last position
        always corresponds to the real last token for every sequence.
        """
        # Tokenize all prompts.
        all_ids: list[torch.Tensor] = []
        for prompt in prompts:
            enc = self._tokenizer(
                prompt, return_tensors="pt", truncation=True,
                max_length=self._max_seq_len - max_new_tokens)
            ids = enc.input_ids if hasattr(enc, "input_ids") else enc["input_ids"]
            ids = (ids[0] % self._vocab_size).to(self._device)
            all_ids.append(ids)

        if not all_ids:
            return []

        batch_size = len(all_ids)
        max_len = max(t.shape[0] for t in all_ids)

        # Left-pad to max_len.
        padded = torch.zeros(
            batch_size, max_len, dtype=torch.long, device=self._device)
        for i, ids in enumerate(all_ids):
            offset = max_len - ids.shape[0]
            padded[i, offset:] = ids

        all_generated: list[list[int]] = [[] for _ in range(batch_size)]
        with torch.no_grad():
            for _ in range(max_new_tokens):
                if padded.shape[1] >= self._max_seq_len:
                    break
                out = self._model(padded, use_cache=False)
                logits, _ = self._unpack_output(out)
                # Last position is the real last token for all (left-padded).
                next_tokens = logits[:, -1, :].argmax(dim=-1)  # (B,)
                for i in range(batch_size):
                    all_generated[i].append(int(next_tokens[i].item()))
                padded = torch.cat(
                    [padded, next_tokens.unsqueeze(1)], dim=1)

        return [self._decode_tokens(tokens) for tokens in all_generated]

    def _decode_tokens(self, token_ids: list[int]) -> str:
        """Decode a list of token IDs back to text via the tokenizer."""
        if not token_ids:
            return ""
        ids_tensor = torch.tensor(token_ids, dtype=torch.long)
        try:
            return self._tokenizer.decode(
                ids_tensor, skip_special_tokens=True)
        except Exception:
            # Some tokenizer wrappers expect a list or a 2D tensor.
            try:
                return self._tokenizer.decode(
                    token_ids, skip_special_tokens=True)
            except Exception:
                return ""

    # ── Internal: sleep/wake ────────────────────────────────────────────

    def _wake_unlocked(self) -> None:
        """Restore model to device (assumes lock is held).

        Follows ForgeEngine.wake() level-1 pattern: CPU -> device copy.
        """
        if self._model_failed or self._model is None:
            return
        if not self._is_sleeping:
            return
        self._model.to(self._device, non_blocking=True)
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        self._model.eval()
        self._is_sleeping = False
        print("  [SharedCheckerModel] Woke from sleep")

    # ── Internal: cache key ─────────────────────────────────────────────

    @staticmethod
    def _cache_key(question: str, answer: str, requirements: str) -> str:
        """Compute a stable SHA-256 hash key for the cache."""
        h = hashlib.sha256()
        h.update(question.encode("utf-8"))
        h.update(b"\x00")
        h.update(answer.encode("utf-8"))
        h.update(b"\x00")
        h.update(requirements.encode("utf-8"))
        return h.hexdigest()

    # ── Introspection ───────────────────────────────────────────────────

    @property
    def is_awake(self) -> bool:
        return not self._is_sleeping

    @property
    def is_llm_available(self) -> bool:
        return not self._model_failed and self._model is not None

    def cache_size(self) -> int:
        return len(self._cache)


# ── Singleton Access ───────────────────────────────────────────────────────

_checker_instance: Optional[SharedCheckerModel] = None
_checker_lock = threading.Lock()


def get_checker(config_name: str = "lfm25_tiny",
                device: str = "cpu",
                tokenizer_path: str | None = None) -> SharedCheckerModel:
    """Get the shared singleton SharedCheckerModel instance.

    The first call boots the model; subsequent calls return the same
    instance regardless of the arguments passed.
    """
    global _checker_instance
    if _checker_instance is None:
        with _checker_lock:
            if _checker_instance is None:
                _checker_instance = SharedCheckerModel(
                    config_name=config_name,
                    device=device,
                    tokenizer_path=tokenizer_path)
    return _checker_instance


def reset_checker() -> None:
    """Tear down the singleton (sleeps the model and clears the reference).

    The next get_checker() call will boot a fresh instance.
    """
    global _checker_instance
    with _checker_lock:
        if _checker_instance is not None:
            try:
                _checker_instance.sleep()
            except Exception:
                pass
        _checker_instance = None
