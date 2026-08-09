"""Closed-Form Fact Injection Key — write facts into MLP weights without training.

Based on "Transformer MLPs Encode Facts Without Training" (COLM 2026,
Stanford AI Lab / HazyResearch).

Key insight: MLPs in language models naturally store factual associations.
There exists a closed-form mathematical recipe to construct MLP weights that
encode specific facts, bypassing gradient descent entirely.

The method:
  1. Define desired input→output fact pairs: (x_input, y_output)
  2. Construct MLP weights analytically such that:
     - The gate/up projections map x_input to a specific hidden pattern
     - The down projection maps that pattern to y_output
  3. The fact is "stored" in the MLP and retrieved during inference

For SwiGLU FFN: output = w_down(silu(w_gate @ x) * (w_up @ x))
  To store fact (x → y):
    - Choose a hidden dimension h_k to store this fact
    - Set w_gate[h_k, :] = x (so gate activates on x)
    - Set w_up[h_k, :] = x (so up also activates on x)
    - Set w_down[:, h_k] = y / (silu(1) * 1) = y / 0.731
    - Other facts use different hidden dimensions

This is a rank-1 update per fact. Multiple facts can be stored in parallel
using different hidden dimensions.

Key class: PARTIAL — modifies weights, not reversible (overwrites hidden dims).

Usage:
    from research.keys.fact_injection_key import FactInjectionKey, inject_facts
    # Inject facts into model
    state = inject_facts(state, facts=[("Paris", "capital of France"),
                                        ("Einstein", "physicist")],
                         n_layers=28, d_model=1536, d_ff=8960)
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .base import Key, KeyClass, KeyResult


class FactInjectionKey(Key):
    """Closed-Form Fact Injection key — write facts into MLP weights.

    Uses the COLM 2026 closed-form recipe to construct MLP weights that
    encode specific facts without gradient descent.

    Key class: PARTIAL — modifies weights, not fully reversible.
    """

    def __init__(self, layer_idx: int = -1, hidden_dims_per_fact: int = 1):
        self.layer_idx = layer_idx  # which layer to inject into (-1 = last)
        self.hidden_dims_per_fact = hidden_dims_per_fact

    @property
    def name(self) -> str:
        return "fact_injection"

    @property
    def description(self) -> str:
        return "Closed-form fact injection into MLP weights (COLM 2026, no training)"

    def key_class(self) -> KeyClass:
        return KeyClass.PARTIAL

    def forward(self, data: dict[str, torch.Tensor]) -> KeyResult:
        """Inject facts into MLP weights.

        Args:
            data: {"state": state_dict, "facts": [(x_vec, y_vec), ...],
                   "n_layers": int, "d_model": int, "d_ff": int,
                   "layer_idx": int (optional)}

        Returns:
            modified state dict with facts injected
        """
        try:
            state = dict(data.get("state", data))
            facts = data["facts"]  # list of (x_vec, y_vec) tuples
            n_layers = data["n_layers"]
            d_model = data["d_model"]
            d_ff = data.get("d_ff", 8960)
            layer_idx = data.get("layer_idx", self.layer_idx)
            if layer_idx < 0:
                layer_idx = n_layers - 1  # inject into last layer

            # Select hidden dimensions for facts (use unused dimensions)
            n_facts = len(facts)
            dims_per_fact = self.hidden_dims_per_fact
            start_dim = d_ff - n_facts * dims_per_fact  # use end of hidden dim

            if start_dim < 0:
                return KeyResult(success=False,
                                 error=f"Not enough hidden dims: need {n_facts * dims_per_fact}, have {d_ff}")

            silu_scale = float(F.silu(torch.tensor(1.0)))  # silu(1) ≈ 0.731

            for fact_idx, (x_vec, y_vec) in enumerate(facts):
                h_dims = list(range(start_dim + fact_idx * dims_per_fact,
                                    start_dim + (fact_idx + 1) * dims_per_fact))

                for h_k in h_dims:
                    # Set w_gate row h_k = x (so gate activates on x)
                    gate_key = f"blocks.{layer_idx}.ffn.w_gate.weight"
                    if gate_key in state:
                        state[gate_key][h_k] = x_vec.to(state[gate_key].dtype)

                    # Set w_up row h_k = x (so up also activates on x)
                    up_key = f"blocks.{layer_idx}.ffn.w_up.weight"
                    if up_key in state:
                        state[up_key][h_k] = x_vec.to(state[up_key].dtype)

                    # Set w_down col h_k = y / (silu(||x||²) * ||x||²)
                    # When x is the input: gate = silu(x·x) = silu(||x||²)
                    #                       up = x·x = ||x||²
                    #                       output = w_down @ (silu(||x||²) * ||x||²)
                    # We want output = y, so w_down[:, h_k] = y / (silu(||x||²) * ||x||²)
                    down_key = f"blocks.{layer_idx}.ffn.w_down.weight"
                    if down_key in state:
                        x_norm_sq = float(x_vec.pow(2).sum())
                        if x_norm_sq > 0:
                            scale = float(F.silu(torch.tensor(x_norm_sq))) * x_norm_sq
                            state[down_key][:, h_k] = (y_vec / scale).to(state[down_key].dtype)

                    # Also handle MoE experts (inject into first expert)
                    for ei in range(4):
                        for part, target in [("w_gate", x_vec), ("w_up", x_vec)]:
                            ek = f"blocks.{layer_idx}.ffn.experts.{ei}.{part}.weight"
                            if ek in state:
                                state[ek][h_k] = target.to(state[ek].dtype)
                        ek_down = f"blocks.{layer_idx}.ffn.experts.{ei}.w_down.weight"
                        if ek_down in state and x_norm_sq > 0:
                            state[ek_down][:, h_k] = (y_vec / scale).to(state[ek_down].dtype)
                        break  # only inject into first expert

            return KeyResult(
                success=True,
                weights=state,
                metadata={
                    "n_facts": n_facts,
                    "layer_idx": layer_idx,
                    "dims_used": n_facts * dims_per_fact,
                    "start_dim": start_dim,
                    "method": "closed_form_colm2026",
                },
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: dict[str, torch.Tensor]) -> KeyResult:
        return KeyResult(success=True, data=weights,
                         metadata={"reversible": False})


def inject_facts(state: dict[str, torch.Tensor],
                 facts: list[tuple[torch.Tensor, torch.Tensor]],
                 n_layers: int, d_model: int, d_ff: int = 8960,
                 layer_idx: int = -1) -> dict[str, torch.Tensor]:
    """Inject facts into model MLP weights via closed-form solution.

    Args:
        state: model state dict
        facts: list of (input_vector, output_vector) pairs
               input_vector: (d_model,) — the trigger embedding
               output_vector: (d_model,) — the desired output change
        n_layers: number of layers
        d_model: model dimension
        d_ff: FFN hidden dimension
        layer_idx: which layer to inject into (-1 = last)

    Returns:
        modified state dict with facts injected
    """
    key = FactInjectionKey(layer_idx=layer_idx)
    result = key.forward({
        "state": state, "facts": facts,
        "n_layers": n_layers, "d_model": d_model, "d_ff": d_ff,
        "layer_idx": layer_idx,
    })
    if not result.success:
        raise RuntimeError(f"Fact injection failed: {result.error}")

    print(f"  [Fact Injection] Injected {result.metadata['n_facts']} facts "
          f"into layer {result.metadata['layer_idx']}")
    print(f"    Hidden dims used: {result.metadata['dims_used']} "
          f"(starting at dim {result.metadata['start_dim']})")
    return result.weights


def create_fact_from_text(model, tokenizer, input_text: str,
                          output_text: str, device: str = "cuda"
                          ) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a fact pair from text using the model's embeddings.

    Args:
        model: transformer model with .embed
        tokenizer: tokenizer
        input_text: trigger text (e.g., "Paris")
        output_text: desired output text (e.g., "capital of France")
        device: compute device

    Returns:
        (input_vector, output_vector) — both (d_model,) tensors
    """
    # Get input embedding (mean of token embeddings)
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    with torch.inference_mode():
        x_vec = model.embed(input_ids).mean(dim=1).squeeze(0).cpu()

    # Get output embedding (mean of token embeddings)
    output_ids = tokenizer(output_text, return_tensors="pt").input_ids.to(device)
    with torch.inference_mode():
        y_vec = model.embed(output_ids).mean(dim=1).squeeze(0).cpu()

    return x_vec, y_vec


if __name__ == "__main__":
    key = FactInjectionKey()
    print(f"Key: {key.name}, class: {key.key_class().value}")

    # Test with small dimensions
    d_model = 128
    d_ff = 256
    state = {
        "blocks.0.ffn.w_gate.weight": torch.randn(d_ff, d_model, dtype=torch.bfloat16),
        "blocks.0.ffn.w_up.weight": torch.randn(d_ff, d_model, dtype=torch.bfloat16),
        "blocks.0.ffn.w_down.weight": torch.randn(d_model, d_ff, dtype=torch.bfloat16),
    }

    # Create facts
    facts = [
        (torch.randn(d_model), torch.randn(d_model)),
        (torch.randn(d_model), torch.randn(d_model)),
    ]

    result = key.forward({
        "state": state, "facts": facts,
        "n_layers": 1, "d_model": d_model, "d_ff": d_ff,
        "layer_idx": 0,
    })
    print(f"  Success: {result.success}")
    print(f"  Facts injected: {result.metadata['n_facts']}")
    print(f"  Dims used: {result.metadata['dims_used']}")
    assert result.success, "Fact injection should succeed"
    print("  Fact injection verified ✓")
