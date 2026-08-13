"""Adaptive speculative decoding router.

Auto-selects the optimal decoding strategy based on prompt characteristics:
- Code generation: n-gram + EAGLE-3 (high repetition, high acceptance)
- Short prompts (<32 tokens): MTP (lower overhead, no feature extraction)
- Medium prompts (32-512 tokens): EAGLE-3 (best all-around at batch=1)
- Long prompts (>512 tokens): Standard (draft overhead > benefit)

Reference benchmarks (SpecDecode-Bench, MLSys 2026):
- EAGLE-3: 1.96x speedup at batch=1 on Llama-3-70B
- MTP (k=3): ~1.3x on Qwen3-8B
- n-gram: 4.9x on code-editing workloads (with EAGLE combo)
- Standard: baseline, no overhead

Usage:
    from research.decoding.adaptive_router import AdaptiveRouter

    router = AdaptiveRouter(model, eagle3_head, mtp_head)
    strategy = router.select(prompt="def fibonacci(n):")
    output = router.generate(prompt, max_new_tokens=100)
"""
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch


class DecodeMode(Enum):
    STANDARD = "standard"
    MTP = "mtp"
    EAGLE3 = "eagle3"
    NGRAM = "ngram"
    HYBRID = "hybrid"  # n-gram + EAGLE-3 for code


@dataclass
class PromptProfile:
    """Analyzed prompt characteristics for strategy selection."""
    token_count: int
    is_code: bool
    has_repetition: bool          # high n-gram overlap potential
    complexity_score: float       # 0-1, higher = more complex
    estimated_tokens: int         # expected output length


class AdaptiveRouter:
    """Auto-selects optimal speculative decoding strategy per prompt.

    Strategy selection heuristics (from SpecDecode-Bench findings):
    - Code prompts with repetition → n-gram + EAGLE-3 (highest acceptance)
    - Short prompts (<32 tokens) → MTP (no feature extraction overhead)
    - General prompts (32-512 tokens) → EAGLE-3 (best all-around)
    - Long prompts (>512 tokens) → Standard (draft overhead dominates)
    - If EAGLE-3 head unavailable → fallback to MTP
    - If neither available → Standard

    Args:
        model: the target model
        eagle3_head: optional EAGLE-3 head (from eagle.py)
        mtp_head: optional MTP head (from mtp.py or model.mtp_head)
        tokenizer: optional tokenizer for prompt analysis
    """

    # Code indicators in prompts
    CODE_PATTERNS = [
        r'\bdef\s+\w+\s*\(',     # def function(
        r'\bclass\s+\w+',        # class Name
        r'\bimport\s+\w+',       # import module
        r'\bfrom\s+\w+\s+import', # from module import
        r'```',                   # markdown code block
        r'\bfunction\b',          # JS function
        r'\bconst\b.*\b=\b',      # JS const =
        r'#include\b',            # C/C++
        r'package\s+\w+',         # Go/Java
        r'SELECT\s+.*\bFROM\b',  # SQL
    ]
    CODE_PATTERNS_RE = re.compile('|'.join(CODE_PATTERNS), re.IGNORECASE)

    def __init__(self, model, eagle3_head=None, mtp_head=None,
                 tokenizer=None):
        self.model = model
        self.eagle3_head = eagle3_head
        self.mtp_head = mtp_head or getattr(model, 'mtp_head', None)
        self.tokenizer = tokenizer
        self._stats = {"standard": 0, "mtp": 0, "eagle3": 0, "ngram": 0, "hybrid": 0}

    def profile(self, prompt: str) -> PromptProfile:
        """Analyze a prompt to determine characteristics."""
        # Token count
        if self.tokenizer:
            token_count = len(self.tokenizer.encode(prompt))
        else:
            token_count = len(prompt.split()) * 1.3  # rough estimate

        # Code detection
        is_code = bool(self.CODE_PATTERNS_RE.search(prompt))

        # Repetition (high n-gram overlap potential)
        words = prompt.lower().split()
        has_repetition = len(set(words)) / max(len(words), 1) < 0.6

        # Complexity: ratio of unique tokens to total
        complexity_score = min(1.0, token_count / 512.0)

        # Estimated output length (code tends to be longer)
        estimated_tokens = 200 if is_code else 100

        return PromptProfile(
            token_count=int(token_count),
            is_code=is_code,
            has_repetition=has_repetition,
            complexity_score=complexity_score,
            estimated_tokens=estimated_tokens,
        )

    def select(self, prompt: str) -> DecodeMode:
        """Select optimal decoding strategy for a prompt.

        Returns the recommended DecodeMode. The caller should then use
        the appropriate decoding path.
        """
        profile = self.profile(prompt)

        # Code with repetition: n-gram + EAGLE-3 hybrid
        if profile.is_code and profile.has_repetition and self.eagle3_head is not None:
            self._stats["hybrid"] += 1
            return DecodeMode.HYBRID

        # Code: EAGLE-3 (high acceptance on code)
        if profile.is_code and self.eagle3_head is not None:
            self._stats["eagle3"] += 1
            return DecodeMode.EAGLE3

        # Short prompts: MTP (lower overhead)
        if profile.token_count < 32 and self.mtp_head is not None:
            self._stats["mtp"] += 1
            return DecodeMode.MTP

        # Long prompts: Standard (draft overhead > benefit)
        if profile.token_count > 512:
            self._stats["standard"] += 1
            return DecodeMode.STANDARD

        # General: EAGLE-3 if available
        if self.eagle3_head is not None:
            self._stats["eagle3"] += 1
            return DecodeMode.EAGLE3

        # Fallback: MTP if available
        if self.mtp_head is not None:
            self._stats["mtp"] += 1
            return DecodeMode.MTP

        self._stats["standard"] += 1
        return DecodeMode.STANDARD

    def generate(self, prompt: str, max_new_tokens: int = 100,
                 temperature: float = 0.0, top_p: float = 1.0) -> str:
        """Generate text using the auto-selected optimal strategy.

        This is the main entry point. Call this instead of manually
        choosing a decoding strategy.
        """
        mode = self.select(prompt)
        return self._generate_with(prompt, mode, max_new_tokens, temperature, top_p)

    def _generate_with(self, prompt: str, mode: DecodeMode,
                       max_new_tokens: int, temperature: float,
                       top_p: float) -> str:
        """Execute generation with the selected strategy."""
        from research.inference.decoding import (
            StandardDecoding, MTPSelfSpecDecoding,
        )

        if self.tokenizer:
            ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
                next(self.model.parameters()).device)
        else:
            # Fallback: assume tokenizer is accessible via model
            raise RuntimeError("AdaptiveRouter requires tokenizer for generation")

        if mode == DecodeMode.STANDARD:
            dec = StandardDecoding()
            output_ids = dec.generate(self.model, ids, max_new_tokens, temperature, top_p)
            return self.tokenizer.decode(output_ids[0, ids.shape[1]:], skip_special_tokens=True)

        elif mode == DecodeMode.MTP:
            dec = MTPSelfSpecDecoding(k=4, mtp_module=self.mtp_head)
            output_ids = dec.generate(self.model, ids, max_new_tokens, temperature, top_p)
            return self.tokenizer.decode(output_ids[0, ids.shape[1]:], skip_special_tokens=True)

        elif mode == DecodeMode.EAGLE3:
            from research.decoding.eagle import eagle3_speculative_generate
            return eagle3_speculative_generate(
                self.model, self.eagle3_head, self.tokenizer,
                prompt, max_new_tokens=max_new_tokens, k=4,
                temperature=temperature,
                device=str(next(self.model.parameters()).device),
            )

        elif mode == DecodeMode.HYBRID:
            # n-gram + EAGLE-3: use n-gram cache for repeated patterns,
            # fall back to EAGLE-3 for novel tokens
            return self._generate_hybrid(prompt, max_new_tokens, temperature)

        else:
            # Fallback
            dec = StandardDecoding()
            output_ids = dec.generate(self.model, ids, max_new_tokens, temperature, top_p)
            return self.tokenizer.decode(output_ids[0, ids.shape[1]:], skip_special_tokens=True)

    def _generate_hybrid(self, prompt: str, max_new_tokens: int,
                         temperature: float) -> str:
        """Hybrid n-gram + EAGLE-3 for code generation.

        n-gram cache catches repeated patterns (variable names, keywords)
        with near-zero overhead. EAGLE-3 handles novel tokens.
        """
        # Simple n-gram cache (3-gram) for code repetition
        ngram_cache: dict[tuple, int] = {}
        n = 3

        # Tokenize
        ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
            next(self.model.parameters()).device)
        device = ids.device
        generated = ids
        eos_id = self.tokenizer.eos_token_id

        for _ in range(max_new_tokens):
            # Check n-gram cache
            if generated.shape[1] >= n:
                key = tuple(generated[0, -n:].tolist())
                if key in ngram_cache:
                    next_token = torch.tensor([[ngram_cache[key]]], device=device)
                    generated = torch.cat([generated, next_token], dim=1)
                    if eos_id and next_token.item() == eos_id:
                        break
                    continue

            # Fallback to EAGLE-3 for one speculative round
            from research.decoding.eagle import eagle3_speculative_generate
            # Generate a few tokens with EAGLE-3
            current_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            new_text = eagle3_speculative_generate(
                self.model, self.eagle3_head, self.tokenizer,
                current_text, max_new_tokens=4, k=4,
                temperature=temperature, device=str(device),
            )

            # Tokenize the new tokens and append
            new_ids = self.tokenizer(new_text, return_tensors="pt").input_ids.to(device)
            # Only take the newly generated part
            new_part = new_ids[0, len(self.tokenizer.encode(current_text)):]
            if len(new_part) == 0:
                break

            # Update n-gram cache
            for i in range(len(new_part) - n + 1):
                ngram_key = tuple(new_part[i:i+n].tolist())
                if i + n < len(new_part):
                    ngram_cache[ngram_key] = new_part[i + n].item()

            generated = torch.cat([generated, new_part.unsqueeze(0)], dim=1)
            if eos_id and (new_part == eos_id).any():
                break

        return self.tokenizer.decode(
            generated[0, ids.shape[1]:], skip_special_tokens=True)

    def stats(self) -> dict:
        """Get strategy usage statistics."""
        total = sum(self._stats.values()) or 1
        return {
            **self._stats,
            "total": sum(self._stats.values()),
            "eagle3_pct": self._stats["eagle3"] / total * 100,
            "mtp_pct": self._stats["mtp"] / total * 100,
            "standard_pct": self._stats["standard"] / total * 100,
            "hybrid_pct": self._stats["hybrid"] / total * 100,
        }


# ── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    router = AdaptiveRouter(None)  # model=None for testing profile only

    test_prompts = [
        "def fibonacci(n):\n    if n <= 1:\n        return n",
        "Explain the theory of relativity in simple terms",
        "What is 2+2?",
        "Write a SQL query to find all users who joined in the last 30 days",
    ]

    for prompt in test_prompts:
        profile = router.profile(prompt)
        mode = router.select(prompt)
        print(f"\nPrompt: {prompt[:60]}...")
        print(f"  tokens={profile.token_count} code={profile.is_code} "
              f"rep={profile.has_repetition} complexity={profile.complexity_score:.2f}")
        print(f"  → selected: {mode.value}")
