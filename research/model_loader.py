"""Modular model factory and inference engine for ForgeAI research."""
import math
import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from research.config import ModelConfig


KVCache = Tuple[torch.Tensor, torch.Tensor]


def flash_attention(q, k, v, is_causal=True):
    """Use FlashAttention-2 via PyTorch's SDPA when available.

    PyTorch 2.x automatically dispatches to FlashAttention-2 (FA2) on CUDA
    when using F.scaled_dot_product_attention with is_causal=True.
    This is ~2x faster than manual attention and uses O(1) memory.

    Falls back to manual computation on CPU.
    """
    if q.is_cuda:
        # FA2 is automatically used by SDPA on CUDA in PyTorch 2.x
        return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    else:
        # Manual attention for CPU
        scale = 1.0 / math.sqrt(q.size(-1))
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        if is_causal:
            T = scores.size(-2)
            mask = torch.tril(torch.ones(T, T, device=scores.device, dtype=torch.bool))
            scores = scores.masked_fill(~mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, v)


def _causal_mask(seq_len: int, total_len: int, past_len: int, device: torch.device,
                 dtype: torch.dtype = None) -> torch.Tensor:
    """Create a causal mask for a query of length `seq_len` attending to `total_len` keys."""
    # For standard prefill (past_len == 0, seq_len == total_len) this is the usual upper-triangular mask.
    if dtype is None:
        dtype = torch.float32
    return torch.triu(torch.full((seq_len, total_len), float("-inf"), device=device, dtype=dtype),
                      diagonal=past_len + 1)


class RMSNorm(nn.Module):
    """RMSNorm — faster than LayerNorm, no mean subtraction or bias."""

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        variance = x.float().pow(2).mean(-1, keepdim=True)
        norm = torch.rsqrt(variance + self.eps)
        return (x.float() * norm).to(x.dtype) * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Positional Embeddings (RoPE) for query/key tensors.

    Supports optional YaRN scaling (Peng et al. 2023) for context extension.
    YaRN non-uniformly interpolates RoPE frequencies: high-freq bands extrapolate
    unchanged, low-freq bands are linearly interpolated, and a smooth ramp
    (controlled by beta_fast/beta_slow) blends the two zones.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0, rope_scaling=None):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

        if rope_scaling and rope_scaling.get("type") == "yarn":
            inv_freq = self._yarn_inv_freq(inv_freq, rope_scaling, max_seq_len)

        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    @staticmethod
    def _yarn_inv_freq(inv_freq, cfg, max_seq_len):
        """Apply YaRN wavelength-aware scaling to inv_freq."""
        import math
        factor = cfg.get("factor", 4.0)
        orig_len = cfg.get("original_max_position_embeddings", max_seq_len // factor)
        beta_fast = cfg.get("beta_fast", 32.0)
        beta_slow = cfg.get("beta_slow", 1.0)

        def _gamma_fn(x):
            # Smooth ramp from 0 to 1 using tanh.
            return 1.0 - torch.tanh(x * math.pi / 2.0)

        # Wavelengths for each freq band: lambda_i = 2*pi / inv_freq_i
        # YaRN defines low/high freq w.r.t. the original context length.
        low_freq_wavelen = orig_len / beta_fast
        high_freq_wavelen = orig_len * 2.0 / beta_slow
        wavelens = 2.0 * math.pi / inv_freq

        # Bands: extrapolate (high freq), interpolate (low freq), ramp (middle).
        # inv_freq_scaled = inv_freq / scale, where scale=1 for high freq, =factor for low.
        scale = torch.ones_like(inv_freq)
        # Smooth ramp factor across the middle zone.
        x = (inv_freq * low_freq_wavelen - 1.0) / (high_freq_wavelen / low_freq_wavelen - 1.0)
        x = x.clamp(0.0, 1.0)
        ramp = _gamma_fn(x)
        scale = ramp * factor + (1.0 - ramp) * 1.0
        # Bands fully above high_freq_wavelen get full interpolation.
        scale = torch.where(wavelens > high_freq_wavelen, torch.full_like(scale, factor), scale)
        # Bands fully below low_freq_wavelen extrapolate (scale=1).
        scale = torch.where(wavelens < low_freq_wavelen, torch.ones_like(scale), scale)
        return inv_freq / scale

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        # x shape: (..., seq_len, head_dim)
        seq_len = x.shape[-2]
        cos = self.cos_cached[offset : offset + seq_len, :].unsqueeze(0).unsqueeze(0).to(x.dtype)
        sin = self.sin_cached[offset : offset + seq_len, :].unsqueeze(0).unsqueeze(0).to(x.dtype)
        return (x * cos) + (self._rotate_half(x) * sin)


class DifferentialAttention(nn.Module):
    """
    Causal Differential Attention.
    Splits Q/K into two groups and subtracts a scaled softmax attention map.
    """

    def __init__(self, d_model: int = 768, n_heads: int = 12, max_seq_len: int = 2048, base: float = 10000.0, rope_scaling=None):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // (2 * n_heads)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len, base=base, rope_scaling=rope_scaling)

        self.lambda_init = 0.8
        self.lambda_q1 = nn.Parameter(torch.randn(n_heads, self.head_dim) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(n_heads, self.head_dim) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(n_heads, self.head_dim) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(n_heads, self.head_dim) * 0.1)

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        B, T, C = x.shape

        # Q/K use 2*n_heads heads, V uses n_heads heads with doubled head dim.
        q = self.q_proj(x).view(B, T, 2 * self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, 2 * self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, 2 * self.head_dim).transpose(1, 2)

        past_len = past_key_value[0].shape[-2] if past_key_value is not None else 0
        q = self.rope(q, offset=past_len)
        k = self.rope(k, offset=past_len)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=-2)
            v = torch.cat([past_key_value[1], v], dim=-2)

        q1, q2 = q[:, : self.n_heads], q[:, self.n_heads :]
        k1, k2 = k[:, : self.n_heads], k[:, self.n_heads :]

        total_len = k1.shape[-2]

        lambda_val = (
            torch.exp((self.lambda_q1 * self.lambda_k1).sum(dim=-1, keepdim=True))
            - torch.exp((self.lambda_q2 * self.lambda_k2).sum(dim=-1, keepdim=True))
            + self.lambda_init
        )
        lambda_val = lambda_val.unsqueeze(0).unsqueeze(-1)  # (1, n_heads, 1, 1)

        # Differential attention: out = (attn1 - lambda * attn2) @ v
        #   = attn1 @ v - lambda * (attn2 @ v) = out1 - lambda * out2.
        # We compute out1/out2 via flash_attention (FA2 on CUDA) for each
        # sub-attention, then combine — FA2 cannot be applied to the diff
        # directly because it does not expose attention weights.
        if T == 1 and total_len > 1:
            # Single-token decode: all cached keys are valid; no causal mask needed.
            out1 = flash_attention(q1, k1, v, is_causal=False)
            out2 = flash_attention(q2, k2, v, is_causal=False)
        elif past_len == 0 and T == total_len:
            # Standard prefill: use FlashAttention-2 via SDPA (is_causal=True).
            out1 = flash_attention(q1, k1, v, is_causal=True)
            out2 = flash_attention(q2, k2, v, is_causal=True)
        else:
            # Chunked prefill with a custom causal mask — manual attention.
            scale = 1.0 / math.sqrt(self.head_dim)
            mask = _causal_mask(T, total_len, past_len, x.device, q.dtype)
            scores1 = torch.matmul(q1, k1.transpose(-2, -1)) * scale + mask
            scores2 = torch.matmul(q2, k2.transpose(-2, -1)) * scale + mask
            attn1 = F.softmax(scores1, dim=-1)
            attn2 = F.softmax(scores2, dim=-1)
            out1 = torch.matmul(attn1, v)
            out2 = torch.matmul(attn2, v)

        out = out1 - lambda_val * out2
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        if use_cache:
            return out, (k, v)
        return out, None


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA) with low-rank KV compression.
    For generation we cache the full up-projected K/V (simplest and fastest).
    """

    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 12,
        kv_compression_dim: int = 128,
        max_seq_len: int = 2048,
        base: float = 10000.0,
        rope_scaling=None,
        use_qk_norm: bool = False,
        attn_scale: float = None,
        attn_bias: bool = False,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.kv_compression_dim = kv_compression_dim

        self.kv_down_proj = nn.Linear(d_model, kv_compression_dim, bias=False)
        self.k_up_proj = nn.Linear(kv_compression_dim, d_model, bias=attn_bias)
        self.v_up_proj = nn.Linear(kv_compression_dim, d_model, bias=attn_bias)
        self.q_proj = nn.Linear(d_model, d_model, bias=attn_bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len, base=base, rope_scaling=rope_scaling)

        # QK-norm: RMSNorm on Q and K before RoPE (NanoGPT speedrun / IMU-1).
        self.use_qk_norm = use_qk_norm
        self._qk_norm_identity = True  # assume identity until weights loaded
        if use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        # Fixed attention scale (0.12 from NanoGPT speedrun) instead of head_dim**-0.5.
        self.attn_scale = attn_scale

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        c_kv = self.kv_down_proj(x)

        k = self.k_up_proj(c_kv).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_up_proj(c_kv).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # QK-norm: normalize Q and K before RoPE for training stability.
        # When weights are identity (all 1.0), skip — it's a no-op that would
        # otherwise divide by RMS and break the lossless property.
        if self.use_qk_norm:
            if not getattr(self, '_qk_norm_identity', True):
                q = self.q_norm(q)
                k = self.k_norm(k)
            # else: identity init, skip normalization (lossless)

        past_len = past_key_value[0].shape[-2] if past_key_value is not None else 0
        q = self.rope(q, offset=past_len)
        k = self.rope(k, offset=past_len)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=-2)
            v = torch.cat([past_key_value[1], v], dim=-2)

        total_len = k.shape[-2]
        if T == 1 and total_len > 1:
            # Single-token decode: all cached keys are valid; no causal mask needed.
            out = flash_attention(q, k, v, is_causal=False)
        elif past_len == 0 and T == total_len:
            out = flash_attention(q, k, v, is_causal=True)
        else:
            mask = _causal_mask(T, total_len, past_len, x.device, q.dtype)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        if use_cache:
            return out, (k, v)
        return out, None


class StandardSDPA(nn.Module):
    """Standard causal scaled dot-product attention with RoPE."""

    def __init__(self, d_model: int = 768, n_heads: int = 12, max_seq_len: int = 2048, base: float = 10000.0, rope_scaling=None):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len, base=base, rope_scaling=rope_scaling)

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        B, T, C = x.shape
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        past_len = past_key_value[0].shape[-2] if past_key_value is not None else 0
        q = self.rope(q, offset=past_len)
        k = self.rope(k, offset=past_len)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=-2)
            v = torch.cat([past_key_value[1], v], dim=-2)

        total_len = k.shape[-2]
        if T == 1 and total_len > 1:
            # Single-token decode: all cached keys are valid; no causal mask needed.
            out = flash_attention(q, k, v, is_causal=False)
        elif past_len == 0 and T == total_len:
            out = flash_attention(q, k, v, is_causal=True)
        else:
            mask = _causal_mask(T, total_len, past_len, x.device, q.dtype)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        if use_cache:
            return out, (k, v)
        return out, None


class GroupedQueryAttention(nn.Module):
    """GQA: n_heads query heads share n_kv_heads KV heads (saves KV cache memory)."""

    def __init__(self, d_model=768, n_heads=12, n_kv_heads=None, max_seq_len=2048, base=10000.0, rope_scaling=None,
                 use_qk_norm=False, attn_scale=None, attn_bias=False):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads  # default to MHA
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // self.n_kv_heads
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=attn_bias)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=attn_bias)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=attn_bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)  # o_proj never has bias (Qwen2 convention)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len, base=base, rope_scaling=rope_scaling)

    def _repeat_kv(self, x):
        """Repeat KV heads to match query heads."""
        if self.n_rep == 1:
            return x
        B, n_kv, T, hd = x.shape
        return x[:, :, None, :, :].expand(B, n_kv, self.n_rep, T, hd).reshape(B, n_kv * self.n_rep, T, hd)

    def forward(self, x, past_key_value=None, use_cache=False):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        past_len = past_key_value[0].shape[-2] if past_key_value is not None else 0
        q = self.rope(q, offset=past_len)
        k = self.rope(k, offset=past_len)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=-2)
            v = torch.cat([past_key_value[1], v], dim=-2)

        new_kv = (k, v) if use_cache else None

        # Repeat KV heads to match Q heads.
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        # Single-token decode with cached KV: no causal mask needed (all keys are valid)
        total_len = k.shape[-2]
        if T == 1 and total_len > 1:
            out = flash_attention(q, k, v, is_causal=False)
        else:
            out = flash_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out), new_kv


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, d_model: int = 768, hidden_dim: Optional[int] = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(8 * d_model / 3)
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_up = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class ModularBlock(nn.Module):
    """One transformer block with swappable attention and FFN types."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        norm = RMSNorm if getattr(config, 'norm_type', 'layernorm') == 'rmsnorm' else nn.LayerNorm
        self.ln1 = norm(config.d_model)
        self.attn = build_attention(config)
        self.ln2 = norm(config.d_model)
        self.ffn = build_ffn(config)

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        # Activation checkpointing: recompute forward during backward to save VRAM.
        # Only applies during training (use_cache=False); inference materializes normally.
        if self.training and not use_cache and hasattr(self, '_gradient_checkpointing') and self._gradient_checkpointing:
            def custom_forward(x_inner):
                attn_out, present = self.attn(self.ln1(x_inner), past_key_value=past_key_value, use_cache=False)
                x_inner = x_inner + attn_out
                ffn_out = self.ffn(self.ln2(x_inner))
                # MoE returns (output, aux_loss) tuple; dense returns just output.
                if isinstance(ffn_out, tuple):
                    ffn_out = ffn_out[0]
                x_inner = x_inner + ffn_out
                return x_inner, present
            x, present = torch.utils.checkpoint.checkpoint(custom_forward, x, use_reentrant=False)
        else:
            attn_out, present = self.attn(self.ln1(x), past_key_value=past_key_value, use_cache=use_cache)
            x = x + attn_out
            ffn_out = self.ffn(self.ln2(x))
            if isinstance(ffn_out, tuple):
                ffn_out = ffn_out[0]
            x = x + ffn_out
        return x, present


def build_attention(config: ModelConfig) -> nn.Module:
    kwargs = dict(d_model=config.d_model, n_heads=config.n_heads, max_seq_len=config.max_seq_len, base=config.rope_base, rope_scaling=config.rope_scaling,
                  use_qk_norm=getattr(config, 'use_qk_norm', False), attn_scale=getattr(config, 'attn_scale', None))
    if config.attn_type == "diff":
        return DifferentialAttention(**kwargs)
    if config.attn_type == "mla":
        return MultiHeadLatentAttention(kv_compression_dim=config.kv_compression_dim,
                                        attn_bias=getattr(config, 'attn_bias', False), **kwargs)
    if config.attn_type == "standard":
        return StandardSDPA(**kwargs)
    if config.attn_type == "gqa":
        return GroupedQueryAttention(**kwargs, n_kv_heads=getattr(config, 'n_kv_heads', None),
                                     attn_bias=getattr(config, 'attn_bias', False))
    raise ValueError(f"Unknown attention type: {config.attn_type}")


def build_ffn(config: ModelConfig) -> nn.Module:
    if config.ffn_type == "swiglu":
        return SwiGLUFFN(config.d_model, hidden_dim=getattr(config, 'intermediate_size', None))
    if config.ffn_type == "standard":
        hidden = 4 * config.d_model
        return nn.Sequential(
            nn.Linear(config.d_model, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, config.d_model, bias=False),
        )
    raise ValueError(f"Unknown FFN type: {config.ffn_type}")


class EAGLESpeculativeDraftHead(nn.Module):
    """Auxiliary draft head predicting t+2 from hidden states."""

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.draft_fc = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.SiLU(),
            nn.Linear(d_model, d_model, bias=False),
        )
        self.draft_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, draft_targets: Optional[torch.Tensor] = None):
        draft_features = self.draft_fc(hidden_states)
        draft_logits = self.draft_head(draft_features)

        draft_loss = None
        if draft_targets is not None:
            draft_loss = F.cross_entropy(draft_logits.view(-1, draft_logits.size(-1)), draft_targets.view(-1))
        return draft_logits, draft_loss


class ConfigurableResearchLLM(nn.Module):
    """Full config-driven research language model."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([ModularBlock(config) for _ in range(config.n_layers)])
        norm = RMSNorm if getattr(config, 'norm_type', 'layernorm') == 'rmsnorm' else nn.LayerNorm
        self.ln_f = norm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.embed.weight = self.head.weight  # Weight tying

        # Zero-init residual: output projections start at zero so the residual
        # stream is unchanged at init — cleaner gradient flow early in training
        # (NanoGPT speedrun technique by @Grad62304977).
        if getattr(config, 'zero_init_residual', False):
            for block in self.blocks:
                if hasattr(block.attn, 'out_proj'):
                    block.attn.out_proj.weight.detach().zero_()
                if hasattr(block.ffn, 'w_down'):
                    block.ffn.w_down.weight.detach().zero_()

        self.draft_head: Optional[nn.Module] = None
        if config.use_gradient_checkpointing:
            self.enable_gradient_checkpointing()
        if config.enable_draft_head:
            self.draft_head = EAGLESpeculativeDraftHead(config.d_model, config.vocab_size)

    def enable_gradient_checkpointing(self):
        """Enable activation checkpointing on all transformer blocks."""
        for block in self.blocks:
            block._gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        """Disable activation checkpointing (e.g., for inference)."""
        for block in self.blocks:
            block._gradient_checkpointing = False

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Optional[KVCache]]] = None,
        use_cache: bool = False,
        return_hidden: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]] | Tuple[torch.Tensor, Optional[torch.Tensor], List[Optional[KVCache]]]:
        x = self.embed(idx)
        presents: List[Optional[KVCache]] = []
        for i, block in enumerate(self.blocks):
            past = past_key_values[i] if past_key_values is not None else None
            x, present = block(x, past_key_value=past, use_cache=use_cache)
            if use_cache:
                presents.append(present)
        hidden = self.ln_f(x)

        # Chunked CE path: skip materializing full [B*T, V] logits for loss.
        # The head Linear + CE are fused into chunked passes over the token dim,
        # saving ~2.8 GB at batch 2 / seq 1024 / vocab 151665.
        if self.config.use_chunked_ce and targets is not None and not use_cache:
            from research.training.chunked_ce import chunked_linear_cross_entropy
            loss = chunked_linear_cross_entropy(
                hidden.view(-1, hidden.size(-1)),
                self.head.weight,
                targets.view(-1),
                chunk_size=self.config.ce_chunk_size,
            )
            # Logits are not computed in this path; return None for the logits
            # slot since the training loop only uses the loss.
            logits = None
        elif getattr(self.config, 'use_liger_ce', False) and targets is not None and not use_cache:
            # Liger-Kernel fused linear cross-entropy: one Triton kernel for
            # head matmul + CE, avoids [B*T, V] logits entirely. Saves ~620MB
            # VRAM (bf16 logits at batch 2 / seq 1024 / vocab 151665).
            # NOT compatible with --compile (graph break kills backward compilation,
            # 5x slower). Use with --no-compile for memory-constrained scenarios.
            from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
            if not hasattr(self, '_liger_fce'):
                self._liger_fce = LigerFusedLinearCrossEntropyLoss()
            loss = self._liger_fce(
                self.head.weight,
                hidden.view(-1, hidden.size(-1)),
                targets.view(-1),
            )
            logits = None
        else:
            logits = self.head(hidden)
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        if self.draft_head is not None and targets is not None and targets.size(1) > 2:
            draft_logits, draft_loss = self.draft_head(hidden[:, :-2], targets[:, 2:])
            if loss is not None and draft_loss is not None:
                loss = loss + 0.1 * draft_loss

        if return_hidden:
            if use_cache:
                return logits, loss, presents, hidden
            return logits, loss, hidden
        if use_cache:
            return logits, loss, presents
        return logits, loss


class ModelLoader:
    """Convenience helpers for building, loading, and generating from models."""

    # Cache of blank state dicts keyed by config signature — avoids rebuilding
    # the same architecture every time we just need tensor names/shapes.
    _blank_cache: dict = {}

    @staticmethod
    def _config_signature(config: ModelConfig) -> str:
        """A stable hashable signature for caching blank models."""
        return f"{config.d_model}_{config.n_layers}_{config.attn_type}_{config.ffn_type}_{config.norm_type}_{getattr(config, 'kv_compression_dim', 0)}_{getattr(config, 'n_kv_heads', 0)}_{getattr(config, 'attn_bias', False)}"

    @staticmethod
    def blank_state_dict(config: ModelConfig) -> dict:
        """Return a blank state dict (tensor names + shapes) for a config.

        Uses a cache so the same architecture is only built once per session.
        Returns zero-filled tensors — callers only need names/shapes.
        """
        sig = ModelLoader._config_signature(config)
        if sig not in ModelLoader._blank_cache:
            cfg_cpu = ModelConfig(**{**config.__dict__, "device": "cpu"})
            model = ConfigurableResearchLLM(cfg_cpu)
            # Store only names and shapes (tiny dict), not actual weight values
            ModelLoader._blank_cache[sig] = {k: v.shape for k, v in model.state_dict().items()}
            del model
        # Return zero-filled tensors with correct shapes (cheap — no big allocs)
        return {k: torch.zeros(s, dtype=torch.bfloat16) for k, s in ModelLoader._blank_cache[sig].items()}

    # Cache of built model architectures (on CPU) for fast cloning
    _model_cache: dict = {}

    @staticmethod
    def build_model_fast(config: ModelConfig, checkpoint_path: Optional[str] = None,
                         compile: bool = False, moe_top_k: Optional[int] = None,
                         dtype: Optional[torch.dtype] = None):
        """Fast model build — caches architecture, only loads weights.

        First call builds the architecture (~3s). Subsequent calls with the
        same config clone the cached model (~0.5s) and just load weights.

        moe_top_k: override MoE top-k routing (default: all experts).
                   Set to 2 for 4-expert model to halve FFN activations/VRAM.
        dtype: convert model to this dtype before loading weights (e.g. torch.bfloat16).
               Prevents upcasting bf16 checkpoint weights to fp32, saving ~50% VRAM.
        """
        import time
        t0 = time.time()
        device = torch.device(config.device)
        sig = ModelLoader._config_signature(config)

        # Build or clone architecture
        if sig not in ModelLoader._model_cache:
            t_arch = time.time()
            model = ConfigurableResearchLLM(config).to(device)
            ModelLoader._model_cache[sig] = model
            t_arch = time.time() - t_arch
            print(f"  [FastBuild] Architecture built in {t_arch:.1f}s (cached)")
        else:
            t_clone = time.time()
            cached = ModelLoader._model_cache[sig]
            # Deep copy the cached model (much faster than rebuilding)
            import copy
            model = copy.deepcopy(cached)
            t_clone = time.time() - t_clone
            print(f"  [FastBuild] Architecture cloned in {t_clone:.1f}s (from cache)")

        # Convert dtype before loading weights (prevents bf16→fp32 upcast)
        if dtype is not None:
            model = model.to(dtype)

        if compile:
            model = torch.compile(model, mode="reduce-overhead", dynamic=True)

        if checkpoint_path and os.path.exists(checkpoint_path):
            t_load = time.time()
            from research.checkpoint_io import load_checkpoint
            state = load_checkpoint(checkpoint_path, map_location=device)
            if isinstance(state, dict) and "model_state" in state and not any(k.startswith("blocks.") for k in state):
                state = state["model_state"]
            # Auto-detect MoE
            if any("ffn.experts." in k for k in state):
                from research.moe import replace_ffn_with_moe
                n_experts = len(set(k.split("ffn.experts.")[1].split(".")[0] for k in state if "ffn.experts." in k))
                has_shared = any("ffn.shared." in k for k in state)
                w1_key = next((k for k in state if "ffn.experts.0.w1.weight" in k), None)
                d_ff = state[w1_key].shape[0] if w1_key else None
                effective_top_k = moe_top_k if moe_top_k else n_experts
                # Use dense_bypass when moe_top_k=0 (skip router, run all experts).
                # This reproduces dense FFN output when router is untrained.
                use_dense_bypass = (moe_top_k is not None and moe_top_k == 0)
                replace_ffn_with_moe(model, n_experts=n_experts, top_k=effective_top_k,
                                     d_model=config.d_model,
                                     shared_expert=has_shared, d_ff=d_ff,
                                     dense_bypass=use_dense_bypass)
                model = model.to(device)
                # Re-apply dtype after MoE replacement (new layers are fp32 by default)
                if dtype is not None:
                    model = model.to(dtype)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                print("Missing keys:", missing[:5], "..." if len(missing) > 5 else "")
            if unexpected:
                print("Unexpected keys:", unexpected[:5], "..." if len(unexpected) > 5 else "")

            # Post-load: detect non-identity QK-Norm weights.
            # If all q_norm/k_norm weights are 1.0, skip normalization (lossless).
            # If any differ, enable normalization (trained QK-Norm).
            for block in model.blocks:
                attn = block.attn
                if hasattr(attn, 'q_norm') and hasattr(attn, '_qk_norm_identity'):
                    q_id = (attn.q_norm.weight == 1.0).all()
                    k_id = (attn.k_norm.weight == 1.0).all()
                    attn._qk_norm_identity = bool(q_id and k_id)
            # Log
            n_identity = sum(1 for b in model.blocks if getattr(b.attn, '_qk_norm_identity', True))
            if getattr(config, 'use_qk_norm', False):
                print(f"  [FastBuild] QK-Norm: {n_identity}/{len(model.blocks)} layers identity (skipped)")

            t_load = time.time() - t_load
            print(f"  [FastBuild] Weights loaded in {t_load:.1f}s")
        elif checkpoint_path:
            print(f"Warning: checkpoint {checkpoint_path} not found, using random weights.")

        t_total = time.time() - t0
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  [FastBuild] Total: {t_total:.1f}s ({param_count:.1f}M params)")
        return model

    @staticmethod
    def build_model(config: ModelConfig, checkpoint_path: Optional[str] = None, compile: bool = False):
        print(
            f"Building {config.d_model}d x {config.n_layers}L {config.attn_type.upper()} model "
            f"(FFN: {config.ffn_type.upper()}, draft: {config.enable_draft_head})..."
        )
        device = torch.device(config.device)
        model = ConfigurableResearchLLM(config).to(device)
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"Total parameters: {param_count:.2f}M")

        if compile:
            model = torch.compile(model, mode="reduce-overhead", dynamic=True)

        if checkpoint_path and os.path.exists(checkpoint_path):
            from research.checkpoint_io import load_checkpoint
            state = load_checkpoint(checkpoint_path, map_location=device)
            # If the checkpoint was saved with metadata wrapper (e.g. dpo_align),
            # unwrap the model_state key.
            if isinstance(state, dict) and "model_state" in state and not any(k.startswith("blocks.") for k in state):
                state = state["model_state"]
            # Auto-detect MoE checkpoint and replace FFN layers before loading
            if any("ffn.experts." in k for k in state):
                from research.moe import replace_ffn_with_moe
                # Infer MoE config from checkpoint keys
                expert_keys = [k for k in state if "ffn.experts.0." in k]
                n_experts = len(set(k.split("ffn.experts.")[1].split(".")[0] for k in state if "ffn.experts." in k))
                has_shared = any("ffn.shared." in k for k in state)
                # Infer d_ff from expert weight shape
                w1_key = next((k for k in state if "ffn.experts.0.w1.weight" in k), None)
                d_ff = state[w1_key].shape[0] if w1_key else None
                effective_top_k = moe_top_k if moe_top_k else n_experts
                replace_ffn_with_moe(model, n_experts=n_experts, top_k=effective_top_k,
                                     d_model=config.d_model,
                                     shared_expert=has_shared, d_ff=d_ff)
                model = model.to(device)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                print("Missing keys:", missing)
            if unexpected:
                print("Unexpected keys:", unexpected)
            print(f"Loaded checkpoint from {checkpoint_path}")
        elif checkpoint_path:
            print(f"Warning: checkpoint {checkpoint_path} not found, using random weights.")

        return model

    @staticmethod
    def generate_text(
        model: ConfigurableResearchLLM,
        tokenizer,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_k: Optional[int] = None,
    ):
        model.eval()
        device = next(model.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        idx = inputs.input_ids
        past_key_values = None

        with torch.no_grad():
            for _ in range(max_new_tokens):
                if past_key_values is None:
                    idx_cond = idx[:, -model.config.max_seq_len :]
                    logits, _, past_key_values = model(idx_cond, use_cache=True)
                else:
                    logits, _, past_key_values = model(idx[:, -1:], past_key_values=past_key_values, use_cache=True)

                logits = logits[:, -1, :] / max(temperature, 1e-5)

                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, next_token), dim=1)
                if next_token.item() == tokenizer.eos_token_id:
                    break

        return tokenizer.decode(idx[0], skip_special_tokens=True)


if __name__ == "__main__":
    from research.config import get_config

    for name in ["360m_mla", "250m_diff", "tiny_test"]:
        print("\n" + "=" * 50)
        cfg = get_config(name)
        cfg.device = "cpu"
        ModelLoader.build_model(cfg)
