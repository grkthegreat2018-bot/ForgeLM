"""GQA → MQA key — mean-pool KV heads to reduce cache by n_kv_heads×."""
import torch
from typing import Dict
from .base import Key, KeyClass, KeyResult


class GQAToMQAKey(Key):
    """Convert GQA to MQA by mean-pooling KV heads.

    Averages n_kv_heads KV heads into a single head, reducing KV cache
    by n_kv_heads×. No training needed — pure weight averaging.

    Key class: FULL — exact weight transform, no training.
    """

    @property
    def name(self) -> str:
        return "gqa_to_mqa"

    @property
    def description(self) -> str:
        return "Mean-pool GQA KV heads into single MQA head (n_kv_heads× cache reduction)."

    def key_class(self) -> KeyClass:
        return KeyClass.PARTIAL

    def forward(self, data: Dict[str, torch.Tensor]) -> KeyResult:
        """data -> weights. Mean-pool n_kv_heads KV heads into one.

        Expected data:
            k_weight: (n_kv_heads * head_dim, d_model)
            v_weight: (n_kv_heads * head_dim, d_model)
            n_kv_heads: int
            head_dim: int
            Optional: k_bias, v_bias with same head grouping.
        """
        try:
            n_kv_heads = int(data["n_kv_heads"])
            head_dim = int(data["head_dim"])
            weights: Dict[str, torch.Tensor] = {}

            for prefix in ("k", "v"):
                w = data[f"{prefix}_weight"]
                d_model = w.shape[-1]
                w = w.view(n_kv_heads, head_dim, d_model).mean(dim=0)
                weights[f"{prefix}_weight"] = w

                bias_key = f"{prefix}_bias"
                if bias_key in data and data[bias_key] is not None:
                    b = data[bias_key]
                    b = b.view(n_kv_heads, head_dim).mean(dim=0)
                    weights[bias_key] = b

            return KeyResult(
                success=True,
                weights=weights,
                metadata={"n_kv_heads": n_kv_heads, "head_dim": head_dim,
                          "new_n_kv_heads": 1},
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: Dict[str, torch.Tensor]) -> KeyResult:
        """Cannot reverse — mean-pooling is lossy (individual heads lost)."""
        return KeyResult(
            success=False,
            error="GQA→MQA is lossy: mean-pooling destroys per-head information.",
        )


def convert_gqa_to_mqa_in_model(model) -> int:
    """Convert all attention layers from GQA to MQA (in-place).

    Mean-pools KV heads and sets n_kv_heads=1.
    Returns number of layers converted.
    """
    count = 0
    for module in model.modules():
        n_kv = getattr(module, "n_kv_heads", None) or getattr(module, "num_key_value_heads", None)
        if n_kv is None or n_kv <= 1:
            continue
        head_dim = getattr(module, "head_dim", None)
        if head_dim is None:
            continue

        for prefix in ("k_proj", "v_proj", "k_weight", "v_weight"):
            w = getattr(module, prefix, None)
            if w is None or not hasattr(w, "weight"):
                continue
            tensor = w.weight
            d_model = tensor.shape[-1]
            pooled = tensor.data.view(n_kv, head_dim, d_model).mean(dim=0)
            w.weight = torch.nn.Parameter(pooled, requires_grad=w.weight.requires_grad)
            if w.bias is not None:
                b = w.bias.data.view(n_kv, head_dim).mean(dim=0)
                w.bias = torch.nn.Parameter(b, requires_grad=w.bias.requires_grad)

        if hasattr(module, "n_kv_heads"):
            module.n_kv_heads = 1
        if hasattr(module, "num_key_value_heads"):
            module.num_key_value_heads = 1
        count += 1

    return count
