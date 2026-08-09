"""SAE Pack Key — portable SAE feature-delta steering packs.

Novel insight: A domain fine-tuned model shifts its sparse-autoencoder (SAE)
feature activations relative to the base model. By encoding both models'
hidden states through a shared SAE, we can compute the *delta* in feature
space (what the fine-tune learned), keep only the top-k non-zero deltas
(sparse), and decode them back into steering vectors. The resulting
"SAE Pack" is portable: it can be injected into *any* base model that
shares the same SAE, without retraining.

Pipeline:
  1. extract: encode base & domain activations -> feature deltas -> top-k
     sparse -> decode to steering vectors -> store as pack
  2. apply_steering: at inference, add steering vectors to hidden states

Key class: PARTIAL — closed-form SAE encode/decode, no training.
"""
from collections.abc import Callable
from typing import Dict, Optional, Tuple

import torch

from .base import Key, KeyClass, KeyResult


class SAEPack:
    """Portable SAE feature-delta steering pack.

    Stores per-(layer, feature) coefficients that represent the difference
    between a domain fine-tuned model and the base model in SAE feature space.
    """

    def __init__(self, top_k: int = 128):
        self.packs: dict[tuple[int, int], float] = {}
        self.steering_vectors: dict[int, torch.Tensor] = {}
        self.top_k = top_k

    def extract(
        self,
        base_activations: dict[int, torch.Tensor],
        domain_activations: dict[int, torch.Tensor],
        sae_encoder: Callable[[torch.Tensor], torch.Tensor],
        sae_decoder: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        """Extract sparse feature deltas and build steering vectors.

        Args:
            base_activations: {layer: hidden_states (B, T, d_model)}
            domain_activations: same shape, from domain fine-tuned model
            sae_encoder: maps hidden -> sparse features (..., d_sae)
            sae_decoder: maps sparse features -> hidden (..., d_model)
        """
        for layer in base_activations:
            if layer not in domain_activations:
                continue
            # Support both callable and tensor SAE encoder/decoder
            def _encode(x, enc):
                if callable(enc):
                    return enc(x)
                return x @ enc.T  # (..., d_model) @ (d_sae, d_model).T
            def _decode(x, dec):
                if callable(dec):
                    return dec(x)
                return x @ dec.T  # (..., d_sae) @ (d_model, d_sae).T
            f_base = _encode(base_activations[layer], sae_encoder)
            f_dom = _encode(domain_activations[layer], sae_encoder)
            delta = f_dom - f_base  # (..., d_sae)
            # Average over batch/time dims to get per-feature delta
            delta_flat = delta.reshape(-1, delta.shape[-1]).mean(dim=0)
            # Keep top-k by absolute magnitude
            topk_vals, topk_idx = delta_flat.abs().topk(min(self.top_k, delta_flat.numel()))
            sparse_delta = torch.zeros_like(delta_flat)
            sparse_delta[topk_idx] = delta_flat[topk_idx]
            for idx, coeff in zip(topk_idx.tolist(), sparse_delta[topk_idx].tolist()):
                if abs(coeff) > 1e-8:
                    self.packs[(layer, idx)] = coeff
            # Decode sparse delta into a steering vector in hidden space
            steering = _decode(sparse_delta.unsqueeze(0), sae_decoder).squeeze(0)
            self.steering_vectors[layer] = steering

    def apply_steering(self, hidden_states: dict[int, torch.Tensor],
                       scale: float = 1.0) -> dict[int, torch.Tensor]:
        """Add steering vectors to hidden states at each layer.

        Args:
            hidden_states: {layer: (B, T, d_model)}
            scale: steering strength multiplier

        Returns:
            Modified hidden states dict
        """
        out = dict(hidden_states)
        for layer, vec in self.steering_vectors.items():
            if layer in out:
                out[layer] = out[layer] + scale * vec.to(out[layer].dtype)
        return out


class SAEPackKey(Key):
    """SAE Pack key — extract/inject SAE feature-delta steering packs.

    Key class: PARTIAL — closed-form SAE encode/decode, no training.
    """

    def __init__(self, top_k: int = 128):
        self.top_k = top_k

    @property
    def name(self) -> str:
        return "sae_pack"

    @property
    def description(self) -> str:
        return ("Portable SAE feature-delta steering pack: extract from "
                "domain model, inject into any base with the same SAE")

    def key_class(self) -> KeyClass:
        return KeyClass.PARTIAL

    def forward(self, data: dict[str, torch.Tensor]) -> KeyResult:
        """Extract SAE feature deltas and return as a steering pack.

        Args:
            data: {"base_activations": {layer: tensor},
                   "domain_activations": {layer: tensor},
                   "sae_encoder": callable, "sae_decoder": callable}
        """
        try:
            pack = SAEPack(top_k=self.top_k)
            pack.extract(
                data["base_activations"],
                data["domain_activations"],
                data["sae_encoder"],
                data["sae_decoder"],
            )
            weights = {f"sae_pack_L{L}_F{F}": torch.tensor(c)
                       for (L, F), c in pack.packs.items()}
            weights["sae_pack_steering"] = torch.stack(
                [pack.steering_vectors[k] for k in sorted(pack.steering_vectors)]
            ) if pack.steering_vectors else torch.zeros(0)
            return KeyResult(
                success=True, weights=weights,
                metadata={"n_features": len(pack.packs),
                          "n_layers": len(pack.steering_vectors)},
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: dict[str, torch.Tensor]) -> KeyResult:
        """Reconstruct steering vectors from a stored pack."""
        try:
            steering = weights.get("sae_pack_steering")
            if steering is None:
                return KeyResult(success=False, error="No steering tensor in weights")
            return KeyResult(
                success=True,
                data={"steering_vectors": steering},
                metadata={"n_layers": steering.shape[0] if steering.dim() > 1 else 0},
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))


if __name__ == "__main__":
    torch.manual_seed(42)
    d_model, d_sae, n_layers = 64, 128, 3

    # Synthetic SAE: random encoder/decoder (linear)
    W_enc = torch.randn(d_sae, d_model)
    W_dec = torch.randn(d_model, d_sae)
    enc = lambda h: torch.relu(h @ W_enc.T)
    dec = lambda f: f @ W_dec.T

    # Synthetic activations
    base_acts = {i: torch.randn(2, 5, d_model) for i in range(n_layers)}
    domain_acts = {i: base_acts[i] + 0.3 * torch.randn_like(base_acts[i])
                   for i in range(n_layers)}

    key = SAEPackKey(top_k=32)
    result = key.forward({
        "base_activations": base_acts,
        "domain_activations": domain_acts,
        "sae_encoder": enc,
        "sae_decoder": dec,
    })
    assert result.success, f"Forward failed: {result.error}"
    assert result.metadata["n_features"] > 0, "No features extracted"
    print(f"[SAEPackKey] forward: {result.metadata['n_features']} features, "
          f"{result.metadata['n_layers']} layers")

    rev = key.reverse(result.weights)
    assert rev.success, f"Reverse failed: {rev.error}"
    assert rev.data["steering_vectors"].shape[0] == n_layers
    print(f"[SAEPackKey] reverse: steering shape {rev.data['steering_vectors'].shape}")

    # Verify steering changes hidden states
    pack = SAEPack(top_k=32)
    pack.extract(base_acts, domain_acts, enc, dec)
    steered = pack.apply_steering(base_acts, scale=1.0)
    diff = (steered[0] - base_acts[0]).abs().mean().item()
    assert diff > 1e-6, "Steering had no effect"
    print(f"[SAEPackKey] steering delta: {diff:.6f} (non-zero, OK)")
    print("[SAEPackKey] all tests passed")
