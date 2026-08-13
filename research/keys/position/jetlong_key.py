"""Jet-Long key — training-free zero-shot long-context extension via bifocal RoPE.

Jet-Long (arXiv:2607.07740) is a tuning-free context extension that pairs a
local RoPE-faithful window with a long-range window whose rescaling factor
adapts DYNAMICALLY to the current sequence length via a parameter-free
analytic schedule. It recovers the base model exactly at short inputs and
extrapolates cleanly at long ones (+4.79pp over the strongest baseline on
RULER at 128K for Qwen3-1.7B). Strictly better than YaRN for zero-shot
extension.

This is a TRIVIAL key — pure runtime formula, no weight changes, no training.
The model's weights are completely unchanged; only the position encoding
applied to Q/K is modified at inference time.

Core mechanism (single-window "Jet-Long lite" — drop-in via position_ids):
  - For seq_len <= L_train: identity position map (standard RoPE, zero
    overhead, byte-identical output to the base model).
  - For seq_len > L_train: positions within the local window (the last
    L_train positions) keep their absolute positions (RoPE-faithful);
    older positions are logarithmically compressed so the effective
    position grows sub-linearly:
        p_eff = L_train * (1 + ln(p / L_train))   for p > L_train
    This keeps the well-trained RoPE range intact for recent context while
    smoothly extrapolating to arbitrary lengths. The compression adapts to
    seq_len automatically (longer sequences compress distant positions more).

The full Jet-Long paper uses a bifocal inclusion-exclusion attention merge
(two position encodings, one local-faithful and one long-range-rescaled,
merged via inclusion-exclusion). That requires an attention-side change and
is deferred to the FlexAttention integration (Tier 1E). This key provides
the position-remapping primitive that both the single-window and the full
bifocal paths build on.

Integration: pass the remapped positions as `position_ids` to the model's
RoPE forward (already supported by RotaryEmbedding.forward). No architecture
change needed.
"""
import math

import torch

from research.keys.misc.base import Key, KeyClass, KeyResult


def jetlong_position_map(
    seq_len: int,
    L_train: int,
    mode: str = "log",
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Compute Jet-Long effective positions for each absolute position.

    The local window is the RECENT context (the last L_train positions) —
    these keep their absolute positions (RoPE-faithful, byte-identical to
    the base model for the recent span). Older positions are compressed so
    their effective distance from the end grows sub-linearly.

    Args:
        seq_len: current sequence length (number of positions).
        L_train: the model's training context window. The last L_train
            positions are encoded with base RoPE (identity map).
        mode: "log" (default, sub-linear compression of distant positions) or
            "linear" (uniform compression, YaRN-like, for ablation).
        device: torch device for the returned tensor.

    Returns:
        tensor (seq_len,) of effective positions to feed to RoPE as
        position_ids. For seq_len <= L_train this is arange(seq_len) (identity).
    """
    positions = torch.arange(seq_len, dtype=torch.float32, device=device)
    if seq_len <= L_train:
        return positions
    if mode == "linear":
        # Uniform compression to fit [0, L_train). Ablation baseline.
        return positions * (L_train / seq_len)
    # Default: logarithmic compression of the long-range (early) positions.
    # Work in distance-from-end: d = seq_len - p.
    #   g(d) = d                              for d <= L_train  (local, faithful)
    #   g(d) = L_train * (1 + ln(d / L_train)) for d >  L_train  (compressed)
    # p_eff = seq_len - g(d).
    # C1-continuous at d = L_train: g(L_train)=L_train, g'(L_train)=L_train/L_train=1.
    d = (seq_len - positions).clamp(min=1e-6)
    g = d.clone()
    mask = d > L_train
    g[mask] = L_train * (1.0 + torch.log(d[mask] / L_train))
    return seq_len - g


class JetLongKey(Key):
    """Jet-Long bifocal RoPE key — training-free long-context extension.

    Produces a position remapping that extends context zero-shot with no
    weight changes. Identity for seq_len <= L_train (exact base model).

    Key class: TRIVIAL — pure runtime formula, no data or training.
    """

    @property
    def name(self) -> str:
        return "jetlong"

    @property
    def description(self) -> str:
        return ("Jet-Long training-free long-context extension via bifocal "
                "RoPE (identity for seq<=L_train, log-compressed tail beyond)")

    def key_class(self) -> KeyClass:
        return KeyClass.TRIVIAL

    def forward(self, data: dict) -> KeyResult:
        """Compute Jet-Long effective positions.

        Args:
            data: {"seq_len": int (current sequence length),
                   "L_train": int (training context window),
                   "mode": str ("log" | "linear", default "log"),
                   "device": str | torch.device (default "cpu")}

        Returns:
            {"position_ids": tensor (seq_len,) — effective positions to feed
             to RoPE as position_ids,
             "extended": bool — whether Jet-Long rescaling was applied,
             "compression_ratio": float — seq_len / max(p_eff), >= 1.0}
        """
        try:
            seq_len = int(data.get("seq_len", 2048))
            L_train = int(data.get("L_train", data.get("orig_context", 1024)))
            mode = data.get("mode", "log")
            device = data.get("device", "cpu")

            pos = jetlong_position_map(seq_len, L_train, mode=mode, device=device)
            extended = seq_len > L_train
            # Compression ratio of the early (long-range) window: how much the
            # distant positions are squeezed. 1.0 = no compression, >1 = compressed.
            if extended and seq_len > L_train:
                early_abs_span = seq_len - L_train
                early_eff_span = float((pos[L_train] - pos[0]).item())
                ratio = early_abs_span / max(early_eff_span, 1.0)
            else:
                ratio = 1.0

            return KeyResult(
                success=True,
                weights={"position_ids": pos, "extended": extended,
                         "compression_ratio": ratio},
                metadata={"seq_len": seq_len, "L_train": L_train, "mode": mode},
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: dict) -> KeyResult:
        """Jet-Long is a runtime formula, not a weight transform — nothing to reverse."""
        return KeyResult(
            success=True,
            data={"note": "Jet-Long is a runtime position remap (no weights to reverse)"},
            metadata={"lossy": False, "reversible": True, "trivial": True},
        )


def apply_jetlong_to_rope(rope_module, seq_len: int, L_train: int,
                          mode: str = "log"):
    """Precompute Jet-Long cos/sin caches on a RotaryEmbedding.

    Rebuilds the module's cos_cached / sin_cached (and bf16 variants) so that
    the standard offset-based forward path uses Jet-Long effective positions.
    This avoids passing position_ids on every call (the offset path is faster).

    For seq_len <= L_train this is a no-op (the existing caches already cover
    the training window with base RoPE).

    Args:
        rope_module: a RotaryEmbedding instance.
        seq_len: current sequence length to support.
        L_train: training context window.
        mode: "log" (default) or "linear".
    """
    if seq_len <= L_train:
        return  # base caches already correct — zero overhead.

    pos = jetlong_position_map(seq_len, L_train, mode=mode,
                               device=rope_module.inv_freq.device)
    # Rebuild cos/sin from effective positions: freqs = outer(pos, inv_freq)
    freqs = torch.outer(pos, rope_module.inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    rope_module.cos_cached = emb.cos()
    rope_module.sin_cached = emb.sin()
    rope_module.cos_cached_bf16 = emb.cos().to(torch.bfloat16)
    rope_module.sin_cached_bf16 = emb.sin().to(torch.bfloat16)


def apply_jetlong_to_model(model, L_train: int | None = None,
                           mode: str = "log", seq_len: int | None = None,
                           safe: bool = True):
    """Apply Jet-Long position remapping to all RoPE modules in a model.

    Uses safety validation. For seq_len <= L_train, Jet-Long is identity
    (zero overhead, exact base model). For seq_len > L_train, the position
    remapping is applied — this changes the forward output by design
    (long-context extension), so identity_init=False for extended sequences.

    Args:
        model: ConfigurableResearchLLM (or any module with .blocks[*].attn.rope).
        L_train: training context window. If None, inferred from model.config.
        mode: "log" (default) or "linear".
        seq_len: current sequence length. If None, uses model.config.max_seq_len.
        safe: if True, validate that parameters are finite after application.

    Returns:
        {"L_train": int, "seq_len": int, "extended": bool, "n_rope_modules": int}
    """
    cfg = getattr(model, "config", None)
    if L_train is None:
        L_train = getattr(cfg, "max_seq_len", 1024) if cfg else 1024
    if seq_len is None:
        seq_len = getattr(cfg, "max_seq_len", 2048) if cfg else 2048

    # For short sequences (identity), use safe_apply with identity_init=True.
    # For long sequences (extended), just validate finiteness.
    is_identity = seq_len <= L_train

    def _apply(m):
        n = 0
        for block in getattr(m, "blocks", []):
            rope = getattr(getattr(block, "attn", None), "rope", None)
            if rope is not None:
                apply_jetlong_to_rope(rope, seq_len, L_train, mode=mode)
                n += 1
        return m

    if safe:
        from research.keys.safety import safe_apply
        # Only check identity if seq_len <= L_train (otherwise output changes by design).
        safe_apply(model, _apply, identity_init=is_identity,
                   test_input=None, atol=1e-5, rtol=1e-4)
    else:
        _apply(model)

    n = 0
    for block in getattr(model, "blocks", []):
        if getattr(getattr(block, "attn", None), "rope", None) is not None:
            n += 1
    return {"L_train": L_train, "seq_len": seq_len,
            "extended": seq_len > L_train, "n_rope_modules": n}


if __name__ == "__main__":
    key = JetLongKey()
    print(f"Key: {key.name}, class: {key.key_class().value}")

    # Short context — identity (zero overhead, exact base model).
    r = key.forward({"seq_len": 1024, "L_train": 1024})
    pos = r.weights["position_ids"]
    print(f"Short (seq=L_train): extended={r.weights['extended']}, "
          f"max_pos={pos.max().item():.1f}, identity={torch.equal(pos, torch.arange(1024))}")

    # Long context — log-compressed distant (early) positions.
    r = key.forward({"seq_len": 8192, "L_train": 1024})
    pos = r.weights["position_ids"]
    print(f"Long (8x): extended={r.weights['extended']}, "
          f"compression_ratio={r.weights['compression_ratio']:.2f}, "
          f"max_eff_pos={pos.max().item():.1f} (vs seq_len=8192)")
    # Local window (last L_train positions) keeps absolute positions.
    local = pos[-1024:]
    expected_local = torch.arange(7168, 8192, dtype=torch.float32)
    print(f"Recent window faithful: {torch.allclose(local, expected_local)}")
    # Monotonic increasing; early positions compressed (step < 1).
    diffs = pos[1:] - pos[:-1]
    print(f"Monotonic increasing: {(diffs >= -1e-6).all().item()}, "
          f"min step in early tail: {diffs[:1024].min().item():.3f} (<1.0 = compressed), "
          f"recent step: {diffs[-1].item():.3f} (=1.0 = faithful)")
