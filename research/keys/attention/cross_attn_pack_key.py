"""Cross-Attention Pack Key — portable cross-attention adapter + knowledge KV.

Novel insight: A domain fine-tuned model that uses cross-attention adapters
(e.g. TokenMem-style retrieval) contains two portable artifacts: (1) the
adapter weight matrices (W_q, W_k, W_v, W_o) that project between the base
model's hidden space and a knowledge-augmented space, and (2) the pre-computed
KV cache from running knowledge passages through the domain model. By
extracting both as a "Cross-Attention Pack", we can inject the domain
knowledge into *any* base model of the same dimensionality — no fine-tuning
required on the target.

Pipeline:
  1. extract: copy adapter weights from domain model, run knowledge passages
     through domain model to capture KV
  2. inject: add cross-attention layers to base model using pack weights,
     compute h_new = h + cross_attn(h, knowledge_kv)

Key class: PARTIAL — extract adapter weights and KV, no training.
"""
from collections.abc import Callable
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from research.keys.misc.base import Key, KeyClass, KeyResult


class CrossAttnPack:
    """Portable cross-attention adapter weights + pre-computed knowledge KV.

    Attributes:
        adapter_weights: {layer: {"W_q", "W_k", "W_v", "W_o"}}
        knowledge_kv: {layer: {"K", "V}}  pre-computed from knowledge passages
    """

    def __init__(self):
        self.adapter_weights: dict[int, dict[str, torch.Tensor]] = {}
        self.knowledge_kv: dict[int, dict[str, torch.Tensor]] = {}

    def extract(
        self,
        domain_model: Callable,
        knowledge_tokens: torch.Tensor,
        layers: list | None = None,
    ) -> None:
        """Extract adapter weights and knowledge KV from a domain model.

        Args:
            domain_model: callable returning (hidden_states, kv_dict, adapter_dict)
            knowledge_tokens: (1, T_know, d_model) knowledge passage embeddings
            layers: which layers to extract (None = all)
        """
        with torch.inference_mode():
            hidden, kv_dict, adapter_dict = domain_model(knowledge_tokens)
        for layer in (layers if layers is not None else sorted(adapter_dict)):
            if layer in adapter_dict:
                self.adapter_weights[layer] = {
                    k: v.clone().cpu() for k, v in adapter_dict[layer].items()
                }
            if layer in kv_dict:
                self.knowledge_kv[layer] = {
                    k: v.clone().cpu() for k, v in kv_dict[layer].items()
                }

    def inject(self, base_hidden: dict[int, torch.Tensor],
               scale: float = 1.0) -> dict[int, torch.Tensor]:
        """Apply cross-attention from knowledge KV to base model hidden states.

        h_new = h + scale * W_o(softmax(h W_q @ K^T / sqrt(d)) @ V)

        Args:
            base_hidden: {layer: (B, T, d_model)}
            scale: injection strength

        Returns:
            Modified hidden states
        """
        out = dict(base_hidden)
        for layer, h in base_hidden.items():
            if layer not in self.adapter_weights or layer not in self.knowledge_kv:
                continue
            aw = self.adapter_weights[layer]
            kv = self.knowledge_kv[layer]
            q = h @ aw["W_q"].to(h.dtype)
            k = kv["K"].to(h.dtype)
            v = kv["V"].to(h.dtype)
            d_k = q.shape[-1]
            attn = F.softmax(q @ k.transpose(-1, -2) / (d_k ** 0.5), dim=-1)
            ctx = attn @ v
            out[layer] = h + scale * (ctx @ aw["W_o"].to(h.dtype))
        return out


class CrossAttnPackKey(Key):
    """Cross-Attention Pack key — extract/inject cross-attention adapter packs.

    Key class: PARTIAL — extract adapter weights, no training.
    """

    @property
    def name(self) -> str:
        return "cross_attn_pack"

    @property
    def description(self) -> str:
        return ("Portable cross-attention adapter + knowledge KV pack: extract "
                "from domain model, inject into any base of same dim")

    def key_class(self) -> KeyClass:
        return KeyClass.PARTIAL

    def forward(self, data: dict[str, torch.Tensor]) -> KeyResult:
        """Store adapter weights and knowledge KV as a portable pack.

        Args:
            data: {"adapter_weights": {layer: {W_q,W_k,W_v,W_o}},
                   "knowledge_kv": {layer: {K,V}}}
        """
        try:
            weights = {}
            aw = data.get("adapter_weights", {})
            kv = data.get("knowledge_kv", {})
            for layer, params in aw.items():
                for name, tensor in params.items():
                    weights[f"adapter_L{layer}_{name}"] = tensor
            for layer, params in kv.items():
                for name, tensor in params.items():
                    weights[f"knowledge_L{layer}_{name}"] = tensor
            return KeyResult(
                success=True, weights=weights,
                metadata={"n_adapter_layers": len(aw), "n_kv_layers": len(kv)},
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: dict[str, torch.Tensor]) -> KeyResult:
        """Reconstruct adapter weights and knowledge KV from a stored pack."""
        try:
            adapter_weights, knowledge_kv = {}, {}
            for key, tensor in weights.items():
                if key.startswith("adapter_L"):
                    # Format: adapter_L{layer}_{name} where name can contain underscores (W_q, W_k, etc)
                    rest = key[len("adapter_L"):]
                    layer_str, name = rest.split("_", 1)
                    layer = int(layer_str)
                    adapter_weights.setdefault(layer, {})[name] = tensor
                elif key.startswith("knowledge_L"):
                    rest = key[len("knowledge_L"):]
                    layer_str, name = rest.split("_", 1)
                    layer = int(layer_str)
                    knowledge_kv.setdefault(layer, {})[name] = tensor
            return KeyResult(
                success=True,
                data={"adapter_weights": adapter_weights,
                      "knowledge_kv": knowledge_kv},
                metadata={"n_adapter_layers": len(adapter_weights),
                          "n_kv_layers": len(knowledge_kv)},
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))


if __name__ == "__main__":
    torch.manual_seed(42)
    d_model, d_attn, n_layers = 64, 32, 2

    # Synthetic adapter weights
    adapter = {i: {n: torch.randn(d_model, d_attn) * 0.1
                   for n in ("W_q", "W_k", "W_v", "W_o")}
               for i in range(n_layers)}
    # Synthetic knowledge KV
    knowledge_kv = {i: {"K": torch.randn(1, 10, d_attn),
                        "V": torch.randn(1, 10, d_attn)}
                    for i in range(n_layers)}

    key = CrossAttnPackKey()
    result = key.forward({"adapter_weights": adapter, "knowledge_kv": knowledge_kv})
    assert result.success, f"Forward failed: {result.error}"
    assert result.metadata["n_adapter_layers"] == n_layers
    print(f"[CrossAttnPackKey] forward: {len(result.weights)} tensors, "
          f"{result.metadata['n_adapter_layers']} adapter layers")

    rev = key.reverse(result.weights)
    assert rev.success, f"Reverse failed: {rev.error}"
    assert len(rev.data["adapter_weights"]) == n_layers
    assert len(rev.data["knowledge_kv"]) == n_layers
    print(f"[CrossAttnPackKey] reverse: {len(rev.data['adapter_weights'])} "
          f"adapter, {len(rev.data['knowledge_kv'])} kv layers")

    # Verify injection changes hidden states
    pack = CrossAttnPack()
    pack.adapter_weights = adapter
    pack.knowledge_kv = knowledge_kv
    base_h = {i: torch.randn(2, 5, d_model) for i in range(n_layers)}
    injected = pack.inject(base_h, scale=1.0)
    diff = (injected[0] - base_h[0]).abs().mean().item()
    assert diff > 1e-6, "Injection had no effect"
    print(f"[CrossAttnPackKey] injection delta: {diff:.6f} (non-zero, OK)")
    print("[CrossAttnPackKey] all tests passed")
