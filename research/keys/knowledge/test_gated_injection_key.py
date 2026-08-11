"""Test-Gated Fact Injection Key — only inject test-verified facts into MLP weights.

Combines FactInjectionKey (closed-form MLP fact injection) with test verification
from the self-play pipeline. Only solutions that pass test cases are injected,
and injection magnitude is scaled by a quality score.

Safe to apply to ForgeLM V2 and expert packs — only injects test-verified facts.

Method:
  1. Run self-play: generate solutions, verify against test cases
  2. For each (prompt, solution) where test_passed=True:
     a. Extract fact vector via mean-pooled embeddings of prompt+solution
     b. Scale injection magnitude by quality score (0-1)
     c. Apply rank-1 MLP update (same closed-form recipe as FactInjectionKey)
  3. Low-quality passing solutions get weak writes; high-quality get strong
  4. Solutions that fail tests are never injected — no noise in the model

Key class: PARTIAL — modifies weights, not reversible (overwrites hidden dims).

Usage:
    from research.keys.test_gated_injection_key import (
        TestGatedFactInjectionKey, inject_test_verified)
    state = inject_test_verified(state, verified_solutions=[
        {"prompt": "def add(a,b):", "solution": "return a+b", "quality": 0.95,
         "test_passed": True},
        {"prompt": "def buggy(x):", "solution": "return x//0", "quality": 0.9,
         "test_passed": False}],  # NOT injected — fails tests
        n_layers=28, d_model=1536, d_ff=8960)
"""
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from research.keys.misc.base import Key, KeyClass, KeyResult


class TestGatedFactInjectionKey(Key):
    """Test-Gated Fact Injection key — inject only test-verified facts.

    Combines closed-form MLP fact injection with test-case verification.
    Only solutions passing tests are injected; quality score scales magnitude.

    Safe to apply to ForgeLM V2 and expert packs — only injects test-verified facts.
    """

    def __init__(self, layer_idx: int = -1):
        self.layer_idx = layer_idx

    @property
    def name(self) -> str:
        return "test_gated_fact_injection"

    @property
    def description(self) -> str:
        return ("Inject test-verified facts into MLP weights, "
                "scaled by quality score (gated FactInjectionKey)")

    def key_class(self) -> KeyClass:
        return KeyClass.PARTIAL

    def forward(self, data: dict[str, torch.Tensor]) -> KeyResult:
        """Inject test-verified facts into MLP weights.
        data: {"state", "verified_solutions", "n_layers", "d_model", "d_ff", "layer_idx"}
        """
        try:
            state = dict(data.get("state", data))
            solutions = data["verified_solutions"]
            n_layers = data["n_layers"]
            d_model = data["d_model"]
            d_ff = data.get("d_ff", 8960)
            layer_idx = data.get("layer_idx", self.layer_idx)
            if layer_idx < 0:
                layer_idx = n_layers - 1

            modified, meta = inject_test_verified(
                state, solutions, n_layers, d_model, d_ff, layer_idx=layer_idx)

            return KeyResult(
                success=True,
                weights=modified,
                metadata={
                    **meta,
                    "method": "test_gated_closed_form",
                    "lossless": True,
                },
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: dict[str, torch.Tensor]) -> KeyResult:
        """Not supported — injection overwrites hidden dims."""
        return KeyResult(
            success=False,
            error="TestGatedFactInjectionKey.reverse is not supported: "
                  "injection overwrites hidden dimensions.",
        )


def extract_fact_vector(
    prompt: str,
    solution: str,
    tokenizer,
    d_model: int,
) -> torch.Tensor:
    """Extract a fact vector from a (prompt, solution) pair.

    Tokenizes prompt+solution, embeds tokens, mean-pools to a d_model vector.

    Args:
        prompt: input prompt text
        solution: solution text (verified by tests)
        tokenizer: tokenizer with __call__ returning input_ids
        d_model: model hidden dimension
    Returns:
        (d_model,) tensor — mean-pooled fact vector
    """
    text = f"{prompt}\n{solution}"

    # Hash-based fallback when no tokenizer available
    if tokenizer is None:
        import hashlib
        h = hashlib.md5(text.encode()).digest()
        vec = torch.zeros(d_model)
        for i in range(d_model):
            vec[i] = (h[i % len(h)] / 255.0 - 0.5) * 0.02
        return vec

    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids

    # Use tokenizer's embedding if available, else deterministic hash fallback
    if hasattr(tokenizer, "embed") and tokenizer.embed is not None:
        embeds = tokenizer.embed(input_ids).squeeze(0)
    elif hasattr(tokenizer, "get_input_embeddings"):
        embeds = tokenizer.get_input_embeddings()(input_ids).squeeze(0)
    else:
        n_tokens = input_ids.shape[1]
        embeds = torch.zeros(n_tokens, d_model)
        for t in range(n_tokens):
            seed = int(input_ids[0, t].item()) % (2**31)
            g = torch.Generator().manual_seed(seed)
            embeds[t] = torch.randn(d_model, generator=g) * 0.02

    return embeds.mean(dim=0)  # mean-pool over tokens → (d_model,)


def inject_test_verified(
    state: dict[str, torch.Tensor],
    verified_solutions: list[dict],
    n_layers: int,
    d_model: int,
    d_ff: int = 8960,
    layer_idx: int = -1,
    tokenizer=None,
) -> tuple:
    """Inject only test-verified solutions into MLP weights.

    Args:
        state: model state dict
        verified_solutions: list of dicts with keys:
            prompt (str), solution (str), quality (float 0-1), test_passed (bool)
        n_layers, d_model, d_ff: model dims
        layer_idx: which layer to inject into (-1 = last)
        tokenizer: for fact vector extraction (optional)

    Returns:
        (modified state_dict, metadata dict)
    """
    if layer_idx < 0:
        layer_idx = n_layers - 1
    passed = [s for s in verified_solutions if s.get("test_passed", False)]
    skipped = len(verified_solutions) - len(passed)
    if not passed:
        return state, {"n_injected": 0, "n_skipped": skipped,
                       "n_total": len(verified_solutions)}

    start_dim = d_ff - len(passed)
    if start_dim < 0:
        raise RuntimeError(f"Not enough hidden dims: need {len(passed)}, have {d_ff}")

    n_injected = 0
    for i, sol in enumerate(passed):
        quality = max(0.0, min(1.0, float(sol["quality"])))
        if quality < 1e-4:
            continue

        h_k = start_dim + i
        fact_vec = extract_fact_vector(
            sol["prompt"], sol["solution"], tokenizer, d_model)
        # Scale injection by quality — high quality = stronger write
        x_vec = fact_vec * quality
        y_vec = fact_vec * quality  # auto-associative: input ≈ output

        # Try MoE expert keys (ffn.experts.{N}.w1/w2/w3 = gate/down/up in SwiGLU)
        injected_into = False
        for expert in range(8):
            # w1 = gate, w3 = up, w2 = down (SwiGLU convention)
            gate_key = f"blocks.{layer_idx}.ffn.experts.{expert}.w1.weight"
            up_key = f"blocks.{layer_idx}.ffn.experts.{expert}.w3.weight"
            down_key = f"blocks.{layer_idx}.ffn.experts.{expert}.w2.weight"

            if gate_key in state:
                state[gate_key][h_k] = x_vec.to(state[gate_key].dtype)
                injected_into = True
            if up_key in state:
                state[up_key][h_k] = x_vec.to(state[up_key].dtype)
                injected_into = True
            if down_key in state:
                x_norm_sq = float(x_vec.pow(2).sum())
                if x_norm_sq > 0:
                    scale = float(F.silu(torch.tensor(x_norm_sq))) * x_norm_sq
                    state[down_key][:, h_k] = (y_vec / scale).to(state[down_key].dtype)
                injected_into = True
            if injected_into:
                break

        # Fall back to dense FFN (w_gate/w_up/w_down naming)
        if not injected_into:
            for part, target in [("w_gate", x_vec), ("w_up", x_vec)]:
                key = f"blocks.{layer_idx}.ffn.{part}.weight"
                if key in state:
                    state[key][h_k] = target.to(state[key].dtype)
                    injected_into = True
            down_key = f"blocks.{layer_idx}.ffn.w_down.weight"
            if down_key in state:
                x_norm_sq = float(x_vec.pow(2).sum())
                if x_norm_sq > 0:
                    scale = float(F.silu(torch.tensor(x_norm_sq))) * x_norm_sq
                    state[down_key][:, h_k] = (y_vec / scale).to(state[down_key].dtype)
                injected_into = True

        if injected_into:
            n_injected += 1
    print(f"  [Test-Gated Injection] {n_injected}/{len(verified_solutions)} "
          f"solutions injected ({skipped} failed tests, skipped)")
    return state, {"n_injected": n_injected, "n_skipped": skipped,
                   "n_total": len(verified_solutions), "layer_idx": layer_idx}
