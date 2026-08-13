"""Compound speculative decoding — L1 SpecAttn + EAGLE-3 + MTP.

Combines orthogonal optimization layers for multiplicative speedup:
  L1: Speculative Attention — 57% attention compute cut (lossless)
  L2: EAGLE-3 — ~50% fewer forward passes via draft+verify
  L3: MTP — baked-in multi-token prediction heads

These are additive in savings: each layer reduces a different component
of the inference cost. Together they can achieve 2.5-3.5x speedup at
batch_size=1 on consumer GPUs.

Architecture:
    ┌──────────────────────────────────────────┐
    │              CompoundSpecDecoder          │
    │                                           │
    │  ┌─────────────────────────────────────┐  │
    │  │ L1: SpeculativeAttention (per-layer) │  │
    │  │  - Low-rank draft attn               │  │
    │  │  - Entropy-based verify              │  │
    │  │  - 57% attn compute cut              │  │
    │  └─────────────────────────────────────┘  │
    │                    ↓                      │
    │  ┌─────────────────────────────────────┐  │
    │  │ L2: EAGLE-3 Draft Head              │  │
    │  │  - Multi-layer feature fusion        │  │
    │  │  - k=4 draft tokens per round        │  │
    │  │  - ~50% fewer forward passes         │  │
    │  └─────────────────────────────────────┘  │
    │                    ↓                      │
    │  ┌─────────────────────────────────────┐  │
    │  │ L3: MTP Heads (model-baked)         │  │
    │  │  - Parallel prediction heads         │  │
    │  │  - Zero-draft-cost speculation       │  │
    │  └─────────────────────────────────────┘  │
    └──────────────────────────────────────────┘

Usage:
    from research.decoding.compound_spec import CompoundSpecDecoder

    decoder = CompoundSpecDecoder(model, eagle3_head)
    decoder.activate()  # patches model, activates all layers
    output = decoder.generate("def fibonacci(n):", max_new_tokens=100)
"""
import time
from typing import Optional

import torch


class CompoundSpecDecoder:
    """Orchestrates all speculative decoding layers for maximum speedup.

    Applies L1 Speculative Attention (patches model layers), then uses
    EAGLE-3 for draft generation and MTP for free bonus predictions.

    Args:
        model: the target model (ForgeAI ConfigurableResearchLLM)
        eagle3_head: trained EAGLE-3 head (from eagle.py)
        mtp_head: optional MTP head (from model.mtp_head)
        draft_rank: L1 SpecAttn low-rank dimension (default 32)
        k: EAGLE-3 draft tokens per round (default 4)
    """

    def __init__(self, model, eagle3_head=None, mtp_head=None,
                 draft_rank: int = 32, k: int = 4, tokenizer=None):
        self.model = model
        self.eagle3_head = eagle3_head
        self.mtp_head = mtp_head or getattr(model, 'mtp_head', None)
        self.draft_rank = draft_rank
        self.k = k
        self.tokenizer = tokenizer
        self._l1_active = False
        self._l2_active = False
        self._l3_active = False

    def activate(self, l1: bool = True, l2: bool = True, l3: bool = True):
        """Activate speculative decoding layers.

        Args:
            l1: Enable L1 Speculative Attention (patches model)
            l2: Enable L2 EAGLE-3 draft head
            l3: Enable L3 MTP heads
        """
        active = []

        if l1:
            from research.keys.speculative.speculative_keys import (
                SpeculativeAttentionKey,
            )
            self._spec_attn = SpeculativeAttentionKey(draft_rank=self.draft_rank)
            n_patched = self._spec_attn.apply(self.model)
            self._l1_active = n_patched > 0
            if self._l1_active:
                active.append(f"L1-SpecAttn({n_patched} layers)")

        if l2 and self.eagle3_head is not None:
            self._l2_active = True
            active.append(f"L2-EAGLE3(k={self.k})")

        if l3 and self.mtp_head is not None:
            self._l3_active = True
            active.append("L3-MTP")

        # Estimate compound speedup
        speedup = self._estimate_speedup()
        print(f"  [CompoundSpec] Activated: {' + '.join(active)}")
        print(f"  [CompoundSpec] Estimated speedup: {speedup:.1f}x")

    def _estimate_speedup(self) -> float:
        """Estimate compound speedup from active layers.

        L1: 57% attention cut → attention is ~40% of forward pass
            → 1 / (1 - 0.57 * 0.4) = 1.30x
        L2: k=4, acceptance ~0.75 → 1 / (1 - 0.75 * 4/5) = 2.5x on draft round
            But verification costs 1 forward pass, so effective ~1.8x
        L3: MTP adds ~10% free tokens → 1.1x
        Compound: 1.30 * 1.8 * 1.1 ≈ 2.6x
        """
        factor = 1.0
        if self._l1_active:
            factor *= 1.30
        if self._l2_active:
            factor *= 1.80
        if self._l3_active:
            factor *= 1.10
        return factor

    def generate(self, prompt: str, max_new_tokens: int = 100,
                 temperature: float = 0.0, top_p: float = 1.0) -> str:
        """Generate text with all active speculative decoding layers.

        Falls back gracefully: EAGLE-3 → MTP → Standard.
        """
        if not self.tokenizer:
            raise RuntimeError("CompoundSpecDecoder requires tokenizer")

        if self._l2_active and self.eagle3_head is not None:
            return self._generate_eagle3(prompt, max_new_tokens, temperature)
        elif self._l3_active and self.mtp_head is not None:
            return self._generate_mtp(prompt, max_new_tokens, temperature, top_p)
        else:
            return self._generate_standard(prompt, max_new_tokens, temperature, top_p)

    def _generate_eagle3(self, prompt: str, max_new_tokens: int,
                         temperature: float) -> str:
        """EAGLE-3 generation with L1 SpecAttn already patched into model."""
        from research.decoding.eagle import eagle3_speculative_generate

        device = str(next(self.model.parameters()).device)
        return eagle3_speculative_generate(
            self.model, self.eagle3_head, self.tokenizer,
            prompt, max_new_tokens=max_new_tokens, k=self.k,
            temperature=temperature, device=device,
        )

    def _generate_mtp(self, prompt: str, max_new_tokens: int,
                      temperature: float, top_p: float) -> str:
        """MTP generation with L1 SpecAttn patched."""
        from research.inference.decoding import MTPSelfSpecDecoding

        ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
            next(self.model.parameters()).device)
        dec = MTPSelfSpecDecoding(k=4, mtp_module=self.mtp_head)
        output_ids = dec.generate(self.model, ids, max_new_tokens, temperature, top_p)
        return self.tokenizer.decode(
            output_ids[0, ids.shape[1]:], skip_special_tokens=True)

    def _generate_standard(self, prompt: str, max_new_tokens: int,
                           temperature: float, top_p: float) -> str:
        """Standard generation with L1 SpecAttn patched."""
        from research.inference.decoding import StandardDecoding

        ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
            next(self.model.parameters()).device)
        dec = StandardDecoding()
        output_ids = dec.generate(self.model, ids, max_new_tokens, temperature, top_p)
        return self.tokenizer.decode(
            output_ids[0, ids.shape[1]:], skip_special_tokens=True)

    def benchmark(self, prompt: str, max_new_tokens: int = 50,
                  n_runs: int = 3) -> dict:
        """Benchmark compound speculative decoding vs standard."""
        # Warmup
        self.generate(prompt, max_new_tokens=10)

        # Compound
        times_compound = []
        for _ in range(n_runs):
            torch.cuda.synchronize()
            t0 = time.time()
            self.generate(prompt, max_new_tokens=max_new_tokens)
            torch.cuda.synchronize()
            times_compound.append(time.time() - t0)
        avg_compound = sum(times_compound) / len(times_compound)

        # Standard (deactivate L2/L3, keep L1 for fair comparison of overhead)
        was_l2, was_l3 = self._l2_active, self._l3_active
        self._l2_active = False
        self._l3_active = False
        times_std = []
        for _ in range(n_runs):
            torch.cuda.synchronize()
            t0 = time.time()
            self._generate_standard(prompt, max_new_tokens, 0.0, 1.0)
            torch.cuda.synchronize()
            times_std.append(time.time() - t0)
        avg_std = sum(times_std) / len(times_std)
        self._l2_active, self._l3_active = was_l2, was_l3

        speedup = avg_std / avg_compound
        print(f"  [CompoundSpec] {speedup:.1f}x speedup "
              f"({avg_compound*1000:.0f}ms vs {avg_std*1000:.0f}ms)")
        return {
            "speedup": speedup,
            "compound_ms": avg_compound * 1000,
            "standard_ms": avg_std * 1000,
            "tokens": max_new_tokens,
            "estimated_speedup": self._estimate_speedup(),
        }
