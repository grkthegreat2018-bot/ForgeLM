"""SSA key — derive sparse attention top-k threshold from calibration data.

Sparse attention keeps only the top-k attention scores per query. The optimal k
is derived by running the model on calibration data, computing attention scores
per layer, and finding the k that retains 99% of the attention mass (cumulative
softmax). The threshold is stored as a per-layer scalar — no weight training.

Key class: PARTIAL — needs calibration data (a few hundred tokens).
Reference: arXiv:2511.20102 (SSA: Sparse Sparse Attention)
"""
from typing import Dict, List

import torch
import torch.nn as nn

from research.keys.base import Key, KeyClass, KeyResult


class SSAKey(Key):
    """Sparse attention sparsity key — derive top-k from calibration data.

    Computes the optimal number of attention entries to keep per layer
    by analyzing attention score distributions on calibration data.
    Key class: PARTIAL — needs calibration data (a few hundred tokens).
    """

    @property
    def name(self) -> str:
        return "ssa"

    @property
    def description(self) -> str:
        return "SSA: derive sparse attention top-k threshold from calibration data"

    def key_class(self) -> KeyClass:
        return KeyClass.PARTIAL

    def forward(self, data: dict) -> KeyResult:
        """Compute per-layer sparsity k and threshold from attention scores.

        data: {"attention_scores": (n_layers, n_heads, seq, seq) tensor or
               list of per-layer (n_heads, seq, seq), "retention": float=0.99}
        Returns: {"sparsity_k": list[int], "thresholds": list[float]}
        """
        try:
            scores = data["attention_scores"]
            retention = data.get("retention", 0.99)
            layer_scores = scores if isinstance(scores, list) else [
                scores[i] for i in range(scores.shape[0])]
            sparsity_k: list[int] = []
            thresholds: list[float] = []
            for ls in layer_scores:
                if ls.dim() == 4:
                    ls = ls.mean(0)  # average over batch -> (n_heads, seq, seq)
                probs = torch.softmax(ls, dim=-1)
                sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                # First index where cumulative mass >= retention, per (head, query)
                reached = (cumsum >= retention).float()
                k = max(1, int((reached.argmax(dim=-1) + 1).float().mean().item()))
                thresh = sorted_probs[..., min(k, ls.shape[-1]) - 1].mean().item()
                sparsity_k.append(k)
                thresholds.append(thresh)
            return KeyResult(
                success=True,
                weights={"sparsity_k": sparsity_k, "thresholds": thresholds},
                metadata={"retention": retention, "n_layers": len(sparsity_k)})
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: dict) -> KeyResult:
        """Passthrough — sparsity threshold is not invertible."""
        return KeyResult(
            success=True, data=weights,
            metadata={"note": "SSA threshold not invertible — passthrough"})


def compute_sparsity_from_model(model, input_ids, retention=0.99):
    """Run model with attention hooks, compute optimal sparsity per layer.

    Hooks into each attention layer to capture attention weights,
    then finds the k that retains `retention` fraction of attention mass.
    Returns dict of {layer_idx: {"k": int, "threshold": float}}.
    """
    captured: dict[int, torch.Tensor] = {}
    hooks = []
    def _make_hook(idx):
        def hook(module, inputs, outputs):
            attn = None
            if isinstance(outputs, tuple) and len(outputs) > 1:
                attn = outputs[1]
            elif isinstance(outputs, torch.Tensor) and outputs.dim() >= 3 \
                    and outputs.shape[-1] == outputs.shape[-2]:
                attn = outputs
            if attn is None:
                attn = getattr(module, "_last_attn_weights", None)
            if attn is not None:
                captured[idx] = attn.detach().cpu()
        return hook
    idx = [0]
    def _register(module):
        cls_name = type(module).__name__.lower()
        if "attention" in cls_name or "attn" in cls_name:
            hooks.append(module.register_forward_hook(_make_hook(idx[0])))
            idx[0] += 1
        for child in module.children():
            _register(child)
    _register(model)
    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        for h in hooks:
            h.remove()

    result = SSAKey().forward({
        "attention_scores": [captured[i] for i in sorted(captured)],
        "retention": retention})
    if not result.success:
        raise RuntimeError(f"SSA key failed: {result.error}")
    ks, ths = result.weights["sparsity_k"], result.weights["thresholds"]
    return {i: {"k": ks[i], "threshold": ths[i]} for i in range(len(ks))}
