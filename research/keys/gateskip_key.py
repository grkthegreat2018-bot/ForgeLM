"""GateSkip weight-stealing key — derive gate weights from activation deltas.

GateSkip adds a sigmoid gate g = sigmoid(W_g @ x) per layer. If g ≈ 0 the
block is skipped. Instead of training W_g, we derive it from calibration
data: tokens whose residual stream changes a lot under the block need the
layer (gate → 1), tokens that barely change can skip (gate → 0).

Key class: PARTIAL — forward only (gate weights are not invertible).
"""
from typing import Dict

import torch

from .base import Key, KeyClass, KeyResult


class GateSkipKey(Key):
    """Derive GateSkip gate weights from activation delta statistics.

    Computes W_g such that sigmoid(W_g @ x) ≈ 1 when the block significantly
    changes x (token needs this layer) and ≈ 0 when it doesn't (token can skip).

    Key class: PARTIAL — needs calibration data (a few hundred tokens).
    """

    @property
    def name(self) -> str:
        return "gateskip"

    def key_class(self) -> KeyClass:
        return KeyClass.PARTIAL

    def forward(self, data: dict[str, torch.Tensor]) -> KeyResult:
        """data -> gate weights.

        Args:
            data: {"deltas": (n_tokens, d_model) or (n_tokens,),
                   "inputs": (n_tokens, d_model)}
            delta_i = ||block(x_i) - x_i|| (block's contribution to token i)
            inputs_i = x_i (residual stream entering the block)

        Returns KeyResult with weights:
            {"gate_weight": (1, d_model), "gate_bias": (1,)}
        """
        deltas = data["deltas"]
        inputs = data["inputs"]  # (n_tokens, d_model)

        # Reduce deltas to per-token norm if multi-dimensional.
        delta_norm = deltas.norm(dim=-1) if deltas.dim() > 1 else deltas  # (n_tokens,)

        # Least-squares fit: W_g = (sum delta_i * x_i) / (sum x_i^2)
        # Makes W_g @ x correlate with delta, so sigmoid(W_g @ x) ≈ gate.
        weighted_sum = (delta_norm.unsqueeze(-1) * inputs).sum(dim=0)  # (d_model,)
        norm_sum = (inputs * inputs).sum(dim=0)  # (d_model,)
        gate_weight = (weighted_sum / (norm_sum + 1e-8)).unsqueeze(0)  # (1, d_model)

        # Bias so ~50% of tokens are gated on initially.
        scores = inputs @ gate_weight.squeeze(0)  # (n_tokens,)
        gate_bias = -scores.median().detach().reshape(1)  # (1,)

        return KeyResult(
            success=True,
            weights={"gate_weight": gate_weight, "gate_bias": gate_bias},
            metadata={"n_tokens": inputs.shape[0], "d_model": inputs.shape[1]},
        )

    def reverse(self, weights: dict[str, torch.Tensor]) -> KeyResult:
        """Gate weights are not invertible — passthrough."""
        return KeyResult(
            success=True, data=weights,
            metadata={"note": "GateSkip reverse is passthrough (not invertible)."},
        )


def _find_blocks(model):
    """Find the transformer block list in model."""
    for attr in ("blocks", "h", "layers", "transformer_blocks"):
        if hasattr(model, attr):
            return getattr(model, attr)
    for _, module in model.named_children():
        for attr in ("blocks", "h", "layers"):
            if hasattr(module, attr):
                return getattr(module, attr)
    raise ValueError("Could not find block list in model.")


def compute_gateskip_from_model(model, input_ids, n_layers=None):
    """Run model with hooks, compute gate weights per layer.

    Hooks into each block to capture input (x) and output (block(x)),
    computes delta = ||block(x) - x||, then derives W_g via GateSkipKey.
    Returns dict of {layer_idx: {"gate_weight": tensor, "gate_bias": tensor}}.
    """
    key = GateSkipKey()
    blocks = _find_blocks(model)
    if n_layers is not None:
        blocks = blocks[:n_layers]

    captures = {i: {"inputs": [], "deltas": []} for i in range(len(blocks))}
    hooks = []

    def make_hook(idx):
        def hook_fn(module, inp, out):
            x = inp[0] if isinstance(inp, tuple) else inp
            o = out[0] if isinstance(out, tuple) else out
            x_flat = x.reshape(-1, x.shape[-1]).detach()
            o_flat = o.reshape(-1, o.shape[-1]).detach()
            captures[idx]["inputs"].append(x_flat)
            captures[idx]["deltas"].append((o_flat - x_flat).norm(dim=-1))
        return hook_fn

    for i, block in enumerate(blocks):
        hooks.append(block.register_forward_hook(make_hook(i)))

    model.eval()
    with torch.no_grad():
        model(input_ids)
    for h in hooks:
        h.remove()

    results = {}
    for idx in range(len(blocks)):
        cap = captures[idx]
        if not cap["inputs"]:
            continue
        inputs = torch.cat(cap["inputs"], dim=0)
        deltas = torch.cat(cap["deltas"], dim=0)
        result = key.forward({"deltas": deltas, "inputs": inputs})
        if result.success:
            results[idx] = result.weights
    return results
