"""Norm Folding Key (V2-opt version) — fold ln1/ln2/ln_f γ into adjacent weights.

Unlike the v3 norm_folding_key.py which deletes norm weights, this version
SETS them to 1.0 (identity) after folding. This way:
1. The model code doesn't need changes (RMSNorm with weight=1.0 = just RMS scalar)
2. All norm weights become identical (all-ones) → can be deduped to 1 tensor
3. The folded γ is absorbed into adjacent Linear weights (lossless)

LOSSLESS: RMSNorm(x) = (x / rms(x)) * γ
  W @ RMSNorm(x) = W @ (x/rms * γ) = (W * γ) @ (x/rms)
  So scale W's columns by γ, set norm weight to 1.0, keep dynamic rms division.

Key class: FULL — reversible (un-fold to recover γ), composable.
"""
from typing import Dict

import torch

from research.keys.misc.base import Key, KeyClass, KeyResult


class NormFoldingV2Key(Key):
    """Norm Folding V2 — fold γ into adjacent weights, set norm to identity.

    Unlike NormFoldingKey (which deletes norms), this keeps them as 1.0
    so the model code doesn't need changes and norms can be deduped.

    Key class: FULL — reversible, composable.
    """

    @property
    def name(self) -> str:
        return "norm_folding_v2"

    @property
    def description(self) -> str:
        return "Fold RMSNorm γ into adjacent weights, set norm to identity (lossless)"

    def key_class(self) -> KeyClass:
        return KeyClass.FULL

    def forward(self, data: dict[str, torch.Tensor]) -> KeyResult:
        """Fold norm weights into adjacent linear weights, set norms to 1.0."""
        try:
            state = dict(data)
            n_layers = 0
            for k in state:
                if "ln1.weight" in k and "blocks." in k:
                    layer = int(k.split(".")[1])
                    n_layers = max(n_layers, layer + 1)

            folded = 0
            for i in range(n_layers):
                # ln1 (pre-attention): fold into q_proj, kv_down_proj (column scale)
                ln1_key = f"blocks.{i}.ln1.weight"
                if ln1_key in state:
                    gamma = state[ln1_key].float()
                    # Only fold if not already identity
                    if not (gamma == 1.0).all().item():
                        for reader in ["q_proj", "kv_down_proj"]:
                            rk = f"blocks.{i}.attn.{reader}.weight"
                            if rk in state:
                                w = state[rk].float()
                                # Linear: y = x @ W.T + b, W shape [out, in]
                                # RMSNorm: x_normed = x / rms * γ
                                # W @ (x/rms * γ) = (W * γ) @ (x/rms)
                                # Scale W columns (dim=1, input dim) by γ
                                state[rk] = (w * gamma.unsqueeze(0)).to(state[rk].dtype)
                            # Bias is added AFTER, not affected by input norm
                            # Do NOT scale bias
                        folded += 1
                    # Set to identity (1.0)
                    state[ln1_key] = torch.ones_like(state[ln1_key])

                # ln2 (pre-FFN): fold into w_gate, w_up, experts (column scale)
                ln2_key = f"blocks.{i}.ln2.weight"
                if ln2_key in state:
                    gamma = state[ln2_key].float()
                    if not (gamma == 1.0).all().item():
                        for reader in ["w_gate", "w_up", "ffn.w_gate", "ffn.w_up"]:
                            rk = f"blocks.{i}.{reader}.weight"
                            if rk in state:
                                w = state[rk].float()
                                state[rk] = (w * gamma.unsqueeze(0)).to(state[rk].dtype)
                        # MoE experts also read from normed x
                        for ei in range(4):
                            for part in ["w_gate", "w_up", "w1", "w3"]:
                                rk = f"blocks.{i}.ffn.experts.{ei}.{part}.weight"
                                if rk in state:
                                    w = state[rk].float()
                                    state[rk] = (w * gamma.unsqueeze(0)).to(state[rk].dtype)
                        # Shared expert
                        for part in ["w1", "w3"]:
                            rk = f"blocks.{i}.ffn.shared.{part}.weight"
                            if rk in state:
                                w = state[rk].float()
                                state[rk] = (w * gamma.unsqueeze(0)).to(state[rk].dtype)
                        folded += 1
                    # Set to identity (1.0)
                    state[ln2_key] = torch.ones_like(state[ln2_key])

            # ln_f (final norm): SKIP if head.weight was deduped to embed.weight
            # (can't fold into embed — it's used for token lookup, not output projection)
            # If head is deduped, keep ln_f with its real γ (don't set to identity)
            lnf_key = "ln_f.weight"
            if lnf_key in state:
                gamma = state[lnf_key].float()
                if not (gamma == 1.0).all().item():
                    head_key = "head.weight"
                    if head_key in state:
                        # head is in state (not deduped) — safe to fold
                        w = state[head_key].float()
                        state[head_key] = (w * gamma.unsqueeze(0)).to(state[head_key].dtype).clone()
                        folded += 1
                        state[lnf_key] = torch.ones_like(state[lnf_key])
                    # else: head was deduped (tied to embed) — SKIP ln_f folding entirely
                    # Keep ln_f with real γ so model applies it at runtime

            # Count norms that are now identity
            norm_keys = [k for k in state if "norm" in k.lower() and "weight" in k and "post_" not in k]
            identity_count = sum(1 for k in norm_keys if (state[k] == 1.0).all().item())

            return KeyResult(
                success=True,
                weights=state,
                metadata={
                    "n_layers": n_layers,
                    "n_folded": folded,
                    "n_identity_norms": identity_count,
                    "n_total_norms": len(norm_keys),
                    "saves_compute": True,
                    "note": "Norm weights set to 1.0 (identity), γ absorbed into adjacent weights",
                },
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: dict[str, torch.Tensor]) -> KeyResult:
        """Cannot un-fold without original γ values (they were overwritten)."""
        return KeyResult(
            success=True,
            data=weights,
            metadata={"reversible": False, "note": "Original γ values overwritten with 1.0"},
        )


def apply_norm_folding_v2(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Apply norm folding to state dict (lossless)."""
    key = NormFoldingV2Key()
    result = key.forward(state)
    if not result.success:
        raise RuntimeError(f"Norm folding V2 failed: {result.error}")

    meta = result.metadata
    print(f"  [NormFoldingV2] Folded {meta['n_folded']} norms into adjacent weights")
    print(f"  [NormFoldingV2] {meta['n_identity_norms']}/{meta['n_total_norms']} norms now identity (dedupable)")
    return result.weights


if __name__ == "__main__":
    key = NormFoldingV2Key()
    print(f"Key: {key.name}, class: {key.key_class().value}")
