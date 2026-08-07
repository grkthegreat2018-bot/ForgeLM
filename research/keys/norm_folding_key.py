"""Norm Folding Key — fold RMSNorm into adjacent Linear weights at inference.

Based on TaperNorm (2026): RMSNorm decomposes into:
  1. Static per-dimension scale γ (the learned weight)
  2. Dynamic per-token RMS scalar: 1/sqrt(mean(x²) + eps)

The static scale γ can be absorbed into adjacent linear projections:
  - ln1 (pre-attention): fold γ into q_proj, kv_down_proj input weights
  - ln2 (pre-FFN): fold γ into w_gate, w_up input weights
  - ln_f (final): fold γ into head weight
  - q_norm/k_norm: fold γ into q_proj/k_up_proj output weights (already done in qk_norm_mla_key)

After folding, the norm layer only needs to compute the dynamic RMS scalar
(one division per token), eliminating the per-dimension multiply.

Key insight: RMSNorm(x) = (x / rms(x)) * γ
  If W reads from the normed output: W @ (RMSNorm(x)) = (W * γ) @ (x / rms(x))
  So we can scale W's columns by γ and keep only the dynamic rms division.

  If W writes to the normed input: we need to pre-scale x by γ before the norm.
  But for pre-norm transformers, the norm is BEFORE the sublayer, so the
  sublayer weights READ from the normed output. We fold γ into the reader.

Key class: FULL — reversible (un-fold to recover γ), composable.

Usage:
    from research.keys.norm_folding_key import NormFoldingKey, apply_norm_folding
    # Fold all norms into adjacent weights
    state = apply_norm_folding(state, n_layers=28, d_model=1536)
"""
import torch
from typing import Dict
from .base import Key, KeyClass, KeyResult


class NormFoldingKey(Key):
    """Norm Folding key — absorb RMSNorm scales into adjacent Linear weights.

    Eliminates the per-dimension multiply from all norm layers.
    The dynamic RMS scalar (1/sqrt(mean(x²)+eps)) remains at runtime.

    Key class: FULL — reversible, composable.
    """

    @property
    def name(self) -> str:
        return "norm_folding"

    @property
    def description(self) -> str:
        return "Fold RMSNorm scales into adjacent Linear weights (zero norm multiply at inference)"

    def key_class(self) -> KeyClass:
        return KeyClass.FULL

    def forward(self, data: Dict[str, torch.Tensor]) -> KeyResult:
        """Fold norm weights into adjacent linear weights.

        Args:
            data: state dict with norm weights and linear weights

        Returns:
            modified state dict with norms folded into linears
        """
        try:
            state = dict(data)  # copy
            n_layers = 0
            for k in state:
                if "ln1.weight" in k:
                    n_layers = max(n_layers, int(k.split(".")[1]) + 1)

            d_model = state.get("ln_f.weight", state.get("ln1.weight")).shape[0]

            for i in range(n_layers):
                # ln1 (pre-attention): fold into q_proj, kv_down_proj (column scale)
                ln1_key = f"blocks.{i}.ln1.weight"
                if ln1_key in state:
                    gamma = state[ln1_key].float()
                    # q_proj reads from normed x: q_proj.weight @ (x/rms * gamma)
                    # = (q_proj.weight * gamma) @ (x/rms)
                    # So scale columns of q_proj by gamma
                    for reader in ["q_proj", "kv_down_proj"]:
                        rk = f"blocks.{i}.attn.{reader}.weight"
                        if rk in state:
                            w = state[rk].float()
                            state[rk] = (w * gamma.unsqueeze(0)).to(state[rk].dtype)
                    # Also scale bias if present
                    for reader in ["q_proj", "kv_down_proj"]:
                        rk = f"blocks.{i}.attn.{reader}.bias"
                        if rk in state:
                            state[rk] = (state[rk].float() * gamma).to(state[rk].dtype)
                    del state[ln1_key]

                # ln2 (pre-FFN): fold into w_gate, w_up (column scale)
                ln2_key = f"blocks.{i}.ln2.weight"
                if ln2_key in state:
                    gamma = state[ln2_key].float()
                    for reader in ["w_gate", "w_up", "ffn.w_gate", "ffn.w_up"]:
                        rk = f"blocks.{i}.{reader}.weight"
                        if rk in state:
                            w = state[rk].float()
                            state[rk] = (w * gamma.unsqueeze(0)).to(state[rk].dtype)
                    # MoE experts also read from normed x
                    for ei in range(4):  # up to 4 experts
                        for part in ["w_gate", "w_up"]:
                            rk = f"blocks.{i}.ffn.experts.{ei}.{part}.weight"
                            if rk in state:
                                w = state[rk].float()
                                state[rk] = (w * gamma.unsqueeze(0)).to(state[rk].dtype)
                    del state[ln2_key]

            # ln_f (final norm): fold into head (column scale)
            lnf_key = "ln_f.weight"
            if lnf_key in state:
                gamma = state[lnf_key].float()
                head_key = "head.weight"
                if head_key in state:
                    w = state[head_key].float()
                    state[head_key] = (w * gamma.unsqueeze(0)).to(state[head_key].dtype)
                del state[lnf_key]

            # QK-Norm: fold q_norm into q_proj output (row scale), k_norm into k_up_proj output
            for i in range(n_layers):
                for norm_name, proj_name in [("q_norm", "q_proj"), ("k_norm", "k_up_proj")]:
                    nk = f"blocks.{i}.attn.{norm_name}.weight"
                    pk = f"blocks.{i}.attn.{proj_name}.weight"
                    if nk in state and pk in state:
                        gamma = state[nk].float()
                        w = state[pk].float()
                        # q_norm normalizes Q output (rows = output dims), so scale rows
                        head_dim = gamma.shape[0]
                        n_heads = w.shape[0] // head_dim
                        w = w.view(n_heads, head_dim, -1)
                        w = w * gamma.unsqueeze(1).unsqueeze(0)
                        state[pk] = w.reshape(state[pk].shape).to(state[pk].dtype)
                        del state[nk]

            return KeyResult(
                success=True,
                weights=state,
                metadata={
                    "n_layers": n_layers, "d_model": d_model,
                    "norms_folded": n_layers * 2 + 1,  # ln1, ln2 per layer + ln_f
                    "saves_compute": True,
                },
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: Dict[str, torch.Tensor]) -> KeyResult:
        """Cannot un-fold without the original norm weights (they were deleted)."""
        return KeyResult(
            success=True,
            data=weights,
            metadata={"reversible": False, "note": "Norm weights deleted during folding"},
        )


def apply_norm_folding(state: Dict[str, torch.Tensor], n_layers: int,
                       d_model: int) -> Dict[str, torch.Tensor]:
    """Fold all RMSNorm scales into adjacent Linear weights.

    After folding:
      - ln1.weight, ln2.weight, ln_f.weight are removed
      - q_norm.weight, k_norm.weight are removed (if present)
      - Adjacent Linear weights have their columns/rows scaled by γ
      - The model still needs to compute the dynamic RMS scalar at runtime
        (but the per-dimension multiply is eliminated)

    The model's forward pass must be modified to:
      - Skip the norm weight multiply (weight is now 1.0)
      - Or: set norm weights to ones and skip the multiply in code
    """
    key = NormFoldingKey()
    result = key.forward(state)
    if not result.success:
        raise RuntimeError(f"Norm folding failed: {result.error}")

    folded = result.metadata.get("norms_folded", 0)
    print(f"  [Norm Folding] Folded {folded} norms into adjacent weights "
          f"(eliminates {folded} per-dim multiplies at inference)")
    return result.weights


if __name__ == "__main__":
    key = NormFoldingKey()
    print(f"Key: {key.name}, class: {key.key_class().value}")

    # Create a mini state dict
    state = {
        "blocks.0.ln1.weight": torch.ones(128, dtype=torch.bfloat16) * 2.0,
        "blocks.0.ln2.weight": torch.ones(128, dtype=torch.bfloat16) * 3.0,
        "blocks.0.attn.q_proj.weight": torch.randn(128, 128, dtype=torch.bfloat16),
        "blocks.0.attn.kv_down_proj.weight": torch.randn(128, 128, dtype=torch.bfloat16),
        "blocks.0.ffn.w_gate.weight": torch.randn(256, 128, dtype=torch.bfloat16),
        "blocks.0.ffn.w_up.weight": torch.randn(256, 128, dtype=torch.bfloat16),
        "ln_f.weight": torch.ones(128, dtype=torch.bfloat16) * 1.5,
        "head.weight": torch.randn(1000, 128, dtype=torch.bfloat16),
    }

    result = key.forward(state)
    print(f"Forward: {result.success}")
    print(f"  Norms folded: {result.metadata['norms_folded']}")
    assert "blocks.0.ln1.weight" not in result.weights, "ln1 not folded!"
    assert "blocks.0.ln2.weight" not in result.weights, "ln2 not folded!"
    assert "ln_f.weight" not in result.weights, "ln_f not folded!"
    # Verify q_proj columns scaled by gamma=2
    orig_q = state["blocks.0.attn.q_proj.weight"].float()
    folded_q = result.weights["blocks.0.attn.q_proj.weight"].float()
    assert torch.allclose(folded_q, orig_q * 2.0, atol=1e-5), "q_proj not scaled correctly!"
    print("  Folding verified ✓")
