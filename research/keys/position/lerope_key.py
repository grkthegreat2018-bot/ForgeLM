"""LeRoPE / AdaRoPE — learnable per-head RoPE frequencies.

LeRoPE (arXiv:2607.10134): learn one scalar per frequency band (~32 params total).
  RoPE needs 3.4% more compute to match LeRoPE at 2.5B scale. Emerges a
  high-norm positional band that the model can use for long-range retrieval.

AdaRoPE (arXiv:2607.19363): per-head learnable frequencies AND attention
  scaling factors. Heads with different functional roles (local syntax vs
  long-range retrieval) need distinct frequency ranges. Outperforms partial
  RoPE and NoPE; better context extension than YaRN.

Both variants initialize from the standard geometric RoPE schedule, making
them LOSSLESS at the start of training (byte-identical to vanilla RoPE).
The learnable parameters are then fine-tuned during training to discover
optimal per-head/per-band frequency schedules.

Usage:
    from research.keys.position.lerope_key import LeRoPEEmbedding, AdaRoPEEmbedding
    rope = LeRoPEEmbedding(dim=64, n_heads=12, max_seq_len=2048)
    # or
    rope = AdaRoPEEmbedding(dim=64, n_heads=12, max_seq_len=2048)
    rotated = rope(x, position_ids=position_ids)

The forward signature matches RotaryEmbedding so this is a drop-in replacement.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LeRoPEEmbedding(nn.Module):
    """LeRoPE: learnable per-frequency-band scaling of RoPE frequencies.

    Instead of the fixed geometric schedule inv_freq_i = base^(-2i/dim),
    LeRoPE learns a scalar multiplier s_i per frequency band:
        inv_freq_i = s_i * base^(-2i/dim)

    Only dim//2 parameters (typically 32 for head_dim=64), initialized to 1.0
    (identity = standard RoPE). The model learns to up-weight important
    frequency bands and down-weight less useful ones.

    Research: arXiv:2607.10134 — RoPE needs 3.4% more compute to match LeRoPE
    at 2.5B scale. A high-norm positional band emerges naturally.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048,
                 base: float = 10000.0, rope_scaling=None):
        super().__init__()
        self.dim = dim
        n_freqs = dim // 2

        # Standard geometric RoPE frequencies (frozen base).
        base_inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        if rope_scaling and rope_scaling.get("type") == "yarn":
            base_inv_freq = self._yarn_inv_freq(base_inv_freq, rope_scaling, max_seq_len)

        self.register_buffer("base_inv_freq", base_inv_freq, persistent=False)

        # Learnable per-band scalar (init=1.0 = identity = standard RoPE).
        self.freq_scale = nn.Parameter(torch.ones(n_freqs))

        # Pre-compute cos/sin cache (updated when freq_scale changes during
        # training — recomputed in forward if in training mode).
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Build cos/sin caches from current freq_scale."""
        inv_freq = self.base_inv_freq * self.freq_scale.detach()
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self.register_buffer("cos_cached_bf16", emb.cos().to(torch.bfloat16), persistent=False)
        self.register_buffer("sin_cached_bf16", emb.sin().to(torch.bfloat16), persistent=False)

    @staticmethod
    def _yarn_inv_freq(inv_freq, cfg, max_seq_len):
        """Apply YaRN wavelength-aware scaling to inv_freq."""
        factor = cfg.get("factor", 4.0)
        orig_len = cfg.get("original_max_position_embeddings", max_seq_len // factor)
        beta_fast = cfg.get("beta_fast", 32.0)
        beta_slow = cfg.get("beta_slow", 1.0)

        def _gamma_fn(x):
            return 1.0 - torch.tanh(x * math.pi / 2.0)

        low_freq_wavelen = orig_len / beta_fast
        high_freq_wavelen = orig_len * 2.0 / beta_slow
        wavelens = 2.0 * math.pi / inv_freq

        scale = torch.ones_like(inv_freq)
        x = (inv_freq * low_freq_wavelen - 1.0) / (high_freq_wavelen / low_freq_wavelen - 1.0)
        x = x.clamp(0.0, 1.0)
        ramp = _gamma_fn(x)
        scale = ramp * factor + (1.0 - ramp) * 1.0
        scale = torch.where(wavelens > high_freq_wavelen, torch.full_like(scale, factor), scale)
        scale = torch.where(wavelens < low_freq_wavelen, torch.ones_like(scale), scale)
        return inv_freq / scale

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _get_cos_sin(self, seq_len: int, offset: int,
                     position_ids: torch.Tensor | None,
                     dtype: torch.dtype):
        """Get cos/sin for the given positions, rebuilding cache if needed."""
        # In training mode, rebuild cache to reflect updated freq_scale.
        if self.training and seq_len > self.max_seq_len:
            self.max_seq_len = seq_len
            self._build_cache(seq_len)

        if position_ids is not None:
            if dtype == torch.bfloat16:
                cos = self.cos_cached_bf16[position_ids]
                sin = self.sin_cached_bf16[position_ids]
            else:
                cos = self.cos_cached[position_ids].to(dtype)
                sin = self.sin_cached[position_ids].to(dtype)
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        else:
            if dtype == torch.bfloat16:
                cos = self.cos_cached_bf16[offset : offset + seq_len, :].unsqueeze(0).unsqueeze(0)
                sin = self.sin_cached_bf16[offset : offset + seq_len, :].unsqueeze(0).unsqueeze(0)
            else:
                cos = self.cos_cached[offset : offset + seq_len, :].unsqueeze(0).unsqueeze(0).to(dtype)
                sin = self.sin_cached[offset : offset + seq_len, :].unsqueeze(0).unsqueeze(0).to(dtype)
        return cos, sin

    def forward(self, x: torch.Tensor, offset: int = 0,
                position_ids: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = x.shape[-2]
        cos, sin = self._get_cos_sin(seq_len, offset, position_ids, x.dtype)
        return (x * cos) + (self._rotate_half(x) * sin)

    @torch.no_grad()
    def rebuild_cache(self):
        """Rebuild cos/sin cache after freq_scale has been updated.
        Call this after optimizer.step() in training loops."""
        self._build_cache(self.max_seq_len)


class AdaRoPEEmbedding(nn.Module):
    """AdaRoPE: per-head learnable frequencies AND attention scaling.

    Extends LeRoPE with per-head frequency scales:
        inv_freq[h, i] = s[h, i] * base^(-2i/dim)

    Each head gets its own frequency schedule. Heads specializing in local
    syntax can learn high-frequency-dominant patterns, while long-range
    retrieval heads can learn low-frequency-dominant patterns.

    Parameters: n_heads × (dim//2) frequency scales + n_heads attention
    scaling factors. Init: freq_scale=1.0 (identity), attn_scale=1.0.

    Research: arXiv:2607.19363 — outperforms partial RoPE and NoPE; better
    context extension than YaRN.
    """

    def __init__(self, dim: int, n_heads: int, max_seq_len: int = 2048,
                 base: float = 10000.0, rope_scaling=None):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        n_freqs = dim // 2

        base_inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        if rope_scaling and rope_scaling.get("type") == "yarn":
            base_inv_freq = self._yarn_inv_freq(base_inv_freq, rope_scaling, max_seq_len)

        self.register_buffer("base_inv_freq", base_inv_freq, persistent=False)

        # Per-head learnable frequency scales: [n_heads, n_freqs]
        # Init=1.0 = identity = standard RoPE for all heads.
        self.freq_scale = nn.Parameter(torch.ones(n_heads, n_freqs))

        # Per-head attention scaling factor (applied to the attention logits
        # before softmax, like 1/sqrt(d_k) but per-head learnable).
        # Init=1.0 = standard scaling.
        self.attn_scale = nn.Parameter(torch.ones(n_heads))

        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Build per-head cos/sin caches from current freq_scale."""
        # inv_freq: [n_heads, n_freqs]
        inv_freq = self.base_inv_freq.unsqueeze(0) * self.freq_scale.detach()
        t = torch.arange(seq_len, dtype=torch.float32)  # [seq_len]
        # freqs: [seq_len, n_heads, n_freqs]
        freqs = torch.einsum("s,hf->shf", t, inv_freq)
        # emb: [seq_len, n_heads, dim] (cat freqs with itself)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self.register_buffer("cos_cached_bf16", emb.cos().to(torch.bfloat16), persistent=False)
        self.register_buffer("sin_cached_bf16", emb.sin().to(torch.bfloat16), persistent=False)

    @staticmethod
    def _yarn_inv_freq(inv_freq, cfg, max_seq_len):
        """Apply YaRN scaling to base inv_freq."""
        factor = cfg.get("factor", 4.0)
        orig_len = cfg.get("original_max_position_embeddings", max_seq_len // factor)
        beta_fast = cfg.get("beta_fast", 32.0)
        beta_slow = cfg.get("beta_slow", 1.0)

        def _gamma_fn(x):
            return 1.0 - torch.tanh(x * math.pi / 2.0)

        low_freq_wavelen = orig_len / beta_fast
        high_freq_wavelen = orig_len * 2.0 / beta_slow
        wavelens = 2.0 * math.pi / inv_freq

        scale = torch.ones_like(inv_freq)
        x = (inv_freq * low_freq_wavelen - 1.0) / (high_freq_wavelen / low_freq_wavelen - 1.0)
        x = x.clamp(0.0, 1.0)
        ramp = _gamma_fn(x)
        scale = ramp * factor + (1.0 - ramp) * 1.0
        scale = torch.where(wavelens > high_freq_wavelen, torch.full_like(scale, factor), scale)
        scale = torch.where(wavelens < low_freq_wavelen, torch.ones_like(scale), scale)
        return inv_freq / scale

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor, offset: int = 0,
                position_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Apply per-head RoPE.

        x shape: (B, n_heads, seq_len, head_dim)
        cos/sin cached shape: (seq_len, n_heads, head_dim)
        """
        seq_len = x.shape[-2]

        if self.training and seq_len > self.max_seq_len:
            self.max_seq_len = seq_len
            self._build_cache(seq_len)

        if position_ids is not None:
            # position_ids: (B, seq_len)
            if x.dtype == torch.bfloat16:
                cos = self.cos_cached_bf16[position_ids]  # (B, seq_len, n_heads, dim)
                sin = self.sin_cached_bf16[position_ids]
            else:
                cos = self.cos_cached[position_ids].to(x.dtype)
                sin = self.sin_cached[position_ids].to(x.dtype)
            # (B, seq_len, n_heads, dim) -> (B, n_heads, seq_len, dim)
            cos = cos.permute(0, 2, 1, 3)
            sin = sin.permute(0, 2, 1, 3)
        else:
            if x.dtype == torch.bfloat16:
                cos = self.cos_cached_bf16[offset : offset + seq_len, :, :]  # (seq_len, n_heads, dim)
                sin = self.sin_cached_bf16[offset : offset + seq_len, :, :]
            else:
                cos = self.cos_cached[offset : offset + seq_len, :, :].to(x.dtype)
                sin = self.sin_cached[offset : offset + seq_len, :, :].to(x.dtype)
            # (seq_len, n_heads, dim) -> (1, n_heads, seq_len, dim)
            cos = cos.unsqueeze(0).permute(0, 2, 1, 3)
            sin = sin.unsqueeze(0).permute(0, 2, 1, 3)

        return (x * cos) + (self._rotate_half(x) * sin)

    def get_attn_scale(self) -> torch.Tensor:
        """Return per-head attention scaling factors.

        Use this in the attention computation:
            scale = rope.get_attn_scale()  # [n_heads]
            attn = softmax(q @ k.T * scale.unsqueeze(-1).unsqueeze(-1), dim=-1)
        """
        return self.attn_scale

    @torch.no_grad()
    def rebuild_cache(self):
        """Rebuild cos/sin cache after freq_scale has been updated."""
        self._build_cache(self.max_seq_len)


def apply_lerope_to_model(model, rope_variant: str = "lerope",
                          test_input=None, safe: bool = True):
    """Replace all RotaryEmbedding modules in a model with LeRoPE/AdaRoPE.

    Uses safety validation to ensure the model is not corrupted. LeRoPE and
    AdaRoPE are identity-init (freq_scale=1.0), so the forward output should
    be numerically identical after application.

    Args:
        model: the model with RotaryEmbedding modules.
        rope_variant: "lerope" or "adarope".
        test_input: optional input tensor for forward pass validation.
            If safe=True and this is None, a dummy input will be used.
        safe: if True, use safe_apply with rollback on corruption.

    Returns:
        The model with replaced RoPE modules (in-place).
    """
    if rope_variant not in ("lerope", "adarope"):
        raise ValueError(f"Unknown rope_variant: {rope_variant}")

    def _apply(m):
        for name, module in m.named_modules():
            if isinstance(module, RotaryEmbedding):
                dim = module.dim
                max_seq_len = module.max_seq_len if hasattr(module, "max_seq_len") else 2048

                if rope_variant == "lerope":
                    new_rope = LeRoPEEmbedding(dim=dim, max_seq_len=max_seq_len)
                else:
                    n_heads = getattr(module, "n_heads", 12)
                    new_rope = AdaRoPEEmbedding(
                        dim=dim, n_heads=n_heads, max_seq_len=max_seq_len)

                parent_name = name.rsplit(".", 1)[0] if "." in name else ""
                attr_name = name.rsplit(".", 1)[1] if "." in name else name
                if parent_name:
                    parent = m.get_submodule(parent_name)
                    setattr(parent, attr_name, new_rope)
                else:
                    setattr(m, attr_name, new_rope)
        return m

    if safe:
        from research.keys.safety import safe_apply
        return safe_apply(model, _apply, identity_init=True,
                          test_input=test_input, atol=1e-4, rtol=1e-3)
    return _apply(model)
