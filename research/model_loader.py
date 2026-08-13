"""Modular model factory and inference engine for ForgeAI research."""
import math
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Enable TensorFloat32 tensor cores for float32 matmuls (RTX 5070 supports this).
# Free ~8x speedup on fp32 matmuls with negligible precision loss (~1e-5).
torch.set_float32_matmul_precision("high")

from research.config import ModelConfig

KVCache = tuple[torch.Tensor, torch.Tensor]


class PreAllocatedKVCache:
    """Pre-allocated KV cache — O(1) append instead of O(n) torch.cat per token.

    Allocates max_seq_len slots upfront. Each attention layer writes new k/v
    into the buffer by index, then reads a view of the filled portion.
    This eliminates the O(n²) tensor growth from torch.cat in generation loops.

    With quantize="int8": stores K/V as int8 with per-token scale (q8_0 style),
    halving cache memory. Dequantization happens on read in get_layer().
    This matches llama.cpp's q8_0 KV cache approach.

    Usage:
        cache = PreAllocatedKVCache(n_layers, batch, n_kv_heads, max_seq_len, head_dim, dtype, device)
        # In generation loop:
        cache.advance()  # increment position
        # Pass cache.get_layer(i) as past_key_value to attention
        # Attention writes new k/v via cache.append(i, k_new, v_new)
    """

    def __init__(self, n_layers: int, batch: int, n_kv_heads: int,
                 max_seq_len: int, head_dim: int, dtype: torch.dtype,
                 device: torch.device, n_kv_heads_per_layer: list = None,
                 quantize: str = "none"):
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.position = 0  # current fill position (shared across layers)
        self.quantize = quantize  # "none", "int8"
        self._dtype = dtype

        # Support per-layer head counts (e.g. GQA with different n_kv_heads)
        if n_kv_heads_per_layer is None:
            n_kv_heads_per_layer = [n_kv_heads] * n_layers

        # Skip cache allocation for layers with 0 KV heads (conv layers)
        self.n_kv_heads_per_layer = n_kv_heads_per_layer

        if quantize == "int8":
            # INT8 quantized cache: store int8 values + fp16 per-token scale
            # Memory: 1 byte/element + 2 bytes/scale per (B, n_kv, T) = ~50% of fp16
            cache_dtype = torch.int8
            scale_dtype = torch.float16
        else:
            cache_dtype = dtype
            scale_dtype = dtype

        self.k_caches = []
        self.v_caches = []
        self.k_scales = []
        self.v_scales = []
        for i in range(n_layers):
            nkvh = n_kv_heads_per_layer[i]
            if nkvh == 0:
                # Conv layer — no KV cache needed
                self.k_caches.append(None)
                self.v_caches.append(None)
                self.k_scales.append(None)
                self.v_scales.append(None)
                continue
            k = torch.zeros(batch, nkvh, max_seq_len, head_dim, dtype=cache_dtype, device=device)
            v = torch.zeros(batch, nkvh, max_seq_len, head_dim, dtype=cache_dtype, device=device)
            self.k_caches.append(k)
            self.v_caches.append(v)
            if quantize == "int8":
                # Per-token scale: (B, n_kv, max_seq_len, 1)
                ks = torch.zeros(batch, nkvh, max_seq_len, 1, dtype=scale_dtype, device=device)
                vs = torch.zeros(batch, nkvh, max_seq_len, 1, dtype=scale_dtype, device=device)
                self.k_scales.append(ks)
                self.v_scales.append(vs)
            else:
                self.k_scales.append(None)
                self.v_scales.append(None)

    def get_layer(self, layer_idx: int) -> KVCache | None:
        """Get the (k, v) view for a layer, or None if at position 0."""
        if self.position == 0 or self.k_caches[layer_idx] is None:
            return None
        pos = self.position
        if self.quantize == "int8":
            # Dequantize: int8 * scale → original dtype
            k = self.k_caches[layer_idx][:, :, :pos].to(self._dtype) * \
                self.k_scales[layer_idx][:, :, :pos].to(self._dtype)
            v = self.v_caches[layer_idx][:, :, :pos].to(self._dtype) * \
                self.v_scales[layer_idx][:, :, :pos].to(self._dtype)
            return (k, v)
        k = self.k_caches[layer_idx][:, :, :pos]
        v = self.v_caches[layer_idx][:, :, :pos]
        return (k, v)

    def append(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor):
        """Write new k/v at the current position and advance (per-layer)."""
        if self.k_caches[layer_idx] is None:
            return  # conv layer, no cache
        T = k_new.shape[-2]
        pos = self.position
        if self.quantize == "int8":
            # Quantize: scale = max(abs(x)) / 127, q = round(x / scale)
            k_scale = k_new.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
            v_scale = v_new.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
            k_q = torch.clamp(torch.round(k_new / k_scale), -128, 127).to(torch.int8)
            v_q = torch.clamp(torch.round(v_new / v_scale), -128, 127).to(torch.int8)
            self.k_caches[layer_idx][:, :, pos:pos + T] = k_q
            self.v_caches[layer_idx][:, :, pos:pos + T] = v_q
            self.k_scales[layer_idx][:, :, pos:pos + T] = k_scale.to(torch.float16)
            self.v_scales[layer_idx][:, :, pos:pos + T] = v_scale.to(torch.float16)
        else:
            self.k_caches[layer_idx][:, :, pos:pos + T] = k_new
            self.v_caches[layer_idx][:, :, pos:pos + T] = v_new

    def advance(self, n: int = 1):
        """Advance the fill position by n tokens."""
        self.position += n

    def reset(self):
        """Reset to empty (start of new sequence)."""
        self.position = 0

    @property
    def filled(self) -> int:
        return self.position

    def cache_memory_mb(self) -> float:
        """Estimate current KV cache memory usage in MB."""
        total = 0
        for i in range(self.n_layers):
            if self.k_caches[i] is None:
                continue
            k_bytes = self.k_caches[i].element_size() * self.k_caches[i].numel()
            v_bytes = self.v_caches[i].element_size() * self.v_caches[i].numel()
            total += k_bytes + v_bytes
            if self.quantize == "int8":
                total += self.k_scales[i].element_size() * self.k_scales[i].numel()
                total += self.v_scales[i].element_size() * self.v_scales[i].numel()
        return total / 1e6


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
    """RMSNorm — faster than LayerNorm, no mean subtraction or bias.

    Uses torch.nn.functional.rms_norm (available in PyTorch 2.4+) for a single
    fused kernel with fp32 internal accumulation. This replaces the manual
    x.float().pow(2).mean() → rsqrt → x.float() * norm → .to(dtype) * weight
    chain that launched 5+ kernels and allocated fp32 temporaries per call.
    With 113 norm calls per forward (57 ln + 56 qk_norm), this saves ~200
    kernel launches and ~50MB of temporary allocations per forward pass.
    """

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps
        self.normalized_shape = [d_model]

    def forward(self, x):
        return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)


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
        # Pre-compute bf16 versions to avoid .to(x.dtype) per forward call.
        # 56 calls/forward (28 layers × Q+K), each saving a dtype conversion kernel.
        self.register_buffer("cos_cached_bf16", emb.cos().to(torch.bfloat16), persistent=False)
        self.register_buffer("sin_cached_bf16", emb.sin().to(torch.bfloat16), persistent=False)

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

    def forward(self, x: torch.Tensor, offset: int = 0,
                position_ids: torch.Tensor | None = None) -> torch.Tensor:
        # x shape: (B, n_heads, seq_len, head_dim)
        seq_len = x.shape[-2]
        if position_ids is not None:
            # Per-sequence positions (for batched left-padded generation).
            # position_ids: (B, seq_len) — each sequence has its own position indices.
            if x.dtype == torch.bfloat16:
                cos = self.cos_cached_bf16[position_ids]  # (B, seq_len, dim)
                sin = self.sin_cached_bf16[position_ids]
            else:
                cos = self.cos_cached[position_ids].to(x.dtype)
                sin = self.sin_cached[position_ids].to(x.dtype)
            # Broadcast: (B, seq_len, dim) -> (B, 1, seq_len, dim) for x (B, n_heads, seq_len, dim)
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
            return (x * cos) + (self._rotate_half(x) * sin)
        # Scalar offset (original path — single sequence or uniform batch).
        if x.dtype == torch.bfloat16:
            cos = self.cos_cached_bf16[offset : offset + seq_len, :].unsqueeze(0).unsqueeze(0)
            sin = self.sin_cached_bf16[offset : offset + seq_len, :].unsqueeze(0).unsqueeze(0)
        else:
            cos = self.cos_cached[offset : offset + seq_len, :].unsqueeze(0).unsqueeze(0).to(x.dtype)
            sin = self.sin_cached[offset : offset + seq_len, :].unsqueeze(0).unsqueeze(0).to(x.dtype)
        return (x * cos) + (self._rotate_half(x) * sin)


class GroupedQueryAttention(nn.Module):
    """GQA: n_heads query heads share n_kv_heads KV heads (saves KV cache memory).

    With use_qk_norm=True, applies RMSNorm on Q and K per-head (head_dim)
    after projection, before RoPE — matching LFM2's q_layernorm/k_layernorm.
    Identity init (all 1.0) = no-op, lossless at start.
    """

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

        # QK-norm: RMSNorm on Q and K before RoPE (LFM2 / Gemma3 / Qwen3 style).
        # When weights are identity (all 1.0), skip — it's a no-op.
        self.use_qk_norm = use_qk_norm
        self._qk_norm_identity = True  # assume identity until weights loaded
        if use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

    def _repeat_kv(self, x):
        """Repeat KV heads to match query heads."""
        if self.n_rep == 1:
            return x
        B, n_kv, T, hd = x.shape
        return x[:, :, None, :, :].expand(B, n_kv, self.n_rep, T, hd).reshape(B, n_kv * self.n_rep, T, hd)

    def forward(self, x, past_key_value=None, use_cache=False,
                preallocated_cache: Optional["PreAllocatedKVCache"] = None, layer_idx: int = 0,
                attention_bias: torch.Tensor | None = None,
                position_ids: torch.Tensor | None = None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # QK-norm: normalize Q and K per-head before RoPE (LFM2 style).
        # When weights are identity (all 1.0), skip — no-op.
        if self.use_qk_norm and not self._qk_norm_identity:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Pre-allocated cache path: O(1) append, no torch.cat.
        if preallocated_cache is not None:
            past_len = preallocated_cache.position
            q = self.rope(q, offset=past_len, position_ids=position_ids)
            k = self.rope(k, offset=past_len, position_ids=position_ids)
            preallocated_cache.append(layer_idx, k, v)
            k = preallocated_cache.k_caches[layer_idx][:, :, :past_len + T]
            v = preallocated_cache.v_caches[layer_idx][:, :, :past_len + T]
        else:
            past_len = past_key_value[0].shape[-2] if past_key_value is not None else 0
            q = self.rope(q, offset=past_len, position_ids=position_ids)
            k = self.rope(k, offset=past_len, position_ids=position_ids)

            if past_key_value is not None:
                k = torch.cat([past_key_value[0], k], dim=-2)
                v = torch.cat([past_key_value[1], v], dim=-2)

        new_kv = (k, v) if use_cache else None

        # Repeat KV heads to match Q heads.
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        # Single-token decode with cached KV: no causal mask needed (all keys are valid)
        total_len = k.shape[-2]
        if attention_bias is not None:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_bias)
        elif T == 1 and total_len > 1:
            out = flash_attention(q, k, v, is_causal=False)
        else:
            out = flash_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out), new_kv


class DoubleGatedConvLayer(nn.Module):
    """LFM2-style double-gated short convolution layer.

    Matches LiquidAI LFM2 architecture exactly:
      BCx = in_proj(x)       # Linear(d → 3d), split into B, C, x
      Bx = B * x             # input gate (raw multiply, NO sigmoid)
      conv_out = conv(Bx)    # short depthwise causal conv
      out = C * conv_out     # output gate (raw multiply, NO sigmoid)
      out = out_proj(out)    # Linear(d → d)

    Key difference from our earlier (broken) version:
      - NO sigmoid gates (sigmoid squashes to [0,1], killing residual stream norm)
      - Conv applied to GATED input (B*x), not raw input
      - in_proj projects to 3*d_model (more capacity, matches LFM2)

    The conv state is a fixed-size buffer (kernel_size-1 past tokens) enabling
    O(1) per-token generation — no growing KV cache for conv layers.
    """

    def __init__(self, d_model: int, kernel_size: int = 3, bias: bool = False):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        # Input projection: d_model → 3*d_model (splits into B, C, x)
        self.in_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        # Depthwise causal conv: (d_model, 1, kernel_size), groups=d_model
        self.conv = nn.Conv1d(
            d_model, d_model, kernel_size,
            groups=d_model, bias=bias, padding=0,
        )
        # Output projection (like attention out_proj)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        # Conv state buffer for incremental generation: (B, d_model, kernel_size-1)
        self._conv_state = None

    def _init_conv_state(self, batch: int, device: torch.device, dtype: torch.dtype):
        """Initialize conv state buffer for incremental decoding."""
        state = torch.zeros(
            batch, self.d_model, self.kernel_size - 1,
            device=device, dtype=dtype,
        )
        self._conv_state = state

    def _causal_conv(self, x: torch.Tensor) -> torch.Tensor:
        """Causal depthwise conv via left-padding + Conv1d.

        Args:
            x: (B, T, d_model)
        Returns:
            (B, T, d_model)
        """
        # Transpose to (B, d_model, T) for Conv1d
        x_t = x.transpose(1, 2)  # (B, d_model, T)
        # Left-pad with (kernel_size - 1) zeros for causal conv
        pad = (self.kernel_size - 1, 0)
        x_padded = F.pad(x_t, pad)  # (B, d_model, T + k - 1)
        out = self.conv(x_padded)   # (B, d_model, T)
        return out.transpose(1, 2)  # (B, T, d_model)

    def _incremental_conv(self, x: torch.Tensor) -> torch.Tensor:
        """Single-token conv using state buffer (O(1) per token).

        Args:
            x: (B, 1, d_model) — one token at a time
        Returns:
            (B, 1, d_model)
        """
        B, T, D = x.shape
        assert T == 1, f"incremental_conv expects T=1, got T={T}"
        x_t = x.transpose(1, 2)  # (B, d_model, 1)

        if self._conv_state is None:
            self._init_conv_state(B, x.device, x.dtype)

        # Concatenate state + new token: (B, d_model, kernel_size)
        window = torch.cat([self._conv_state, x_t], dim=-1)
        # Apply conv (no padding needed — window is exactly kernel_size)
        out = self.conv(window)  # (B, d_model, 1)
        # Update state: shift window, drop oldest
        self._conv_state = window[:, :, 1:].clone()

        return out.transpose(1, 2)  # (B, 1, d_model)

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, KVCache | None]:
        """LFM2-style forward: in_proj → gate → conv → gate → out_proj.

        For prefill (T > 1): full causal conv.
        For decode (T == 1): incremental conv with state buffer.
        """
        B, T, D = x.shape

        # Project to 3*d_model and split into B (input gate), C (output gate), x_proj
        BCx = self.in_proj(x)  # (B, T, 3*D)
        B_gate, C_gate, x_proj = BCx.chunk(3, dim=-1)  # each (B, T, D)

        # Input gate: raw multiply (NO sigmoid — LFM2 uses multiplicative gates)
        Bx = B_gate * x_proj  # (B, T, D)

        if T == 1 and self._conv_state is not None:
            conv_out = self._incremental_conv(Bx)
        else:
            # Prefill or no state: full causal conv
            conv_out = self._causal_conv(Bx)
            # Initialize state from last (kernel_size - 1) tokens of GATED input
            if use_cache:
                self._init_conv_state(B, x.device, x.dtype)
                if T >= self.kernel_size - 1:
                    self._conv_state = Bx[:, -(self.kernel_size - 1):, :].transpose(1, 2).clone()
                else:
                    pad_len = self.kernel_size - 1 - T
                    pad = torch.zeros(B, pad_len, D, device=x.device, dtype=x.dtype)
                    self._conv_state = torch.cat([pad, Bx], dim=1).transpose(1, 2).clone()

        # Output gate: raw multiply (NO sigmoid)
        gated = C_gate * conv_out  # (B, T, D)

        out = self.out_proj(gated)
        # Conv layers don't produce KV cache entries
        return out, None


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, d_model: int = 768, hidden_dim: int | None = None):
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

    def __init__(self, config: ModelConfig, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        norm = RMSNorm if getattr(config, 'norm_type', 'layernorm') == 'rmsnorm' else nn.LayerNorm
        self.ln1 = norm(config.d_model)
        # Determine layer type: conv or attention
        layer_types = getattr(config, 'layer_types', None)
        if layer_types is not None and layer_idx < len(layer_types):
            ltype = layer_types[layer_idx].lower()
        else:
            ltype = "attention"
        self.layer_type = ltype
        if ltype in ("conv", "liquid"):
            ksize = getattr(config, 'conv_kernel_size', 3)
            self.attn = DoubleGatedConvLayer(config.d_model, kernel_size=ksize)
        else:
            self.attn = build_attention(config)
        self.ln2 = norm(config.d_model)
        self.ffn = build_ffn(config)
        # Cache whether attn supports pre-allocated KV cache (avoids inspect.signature per forward).
        self._supports_prealloc_cache = isinstance(self.attn, GroupedQueryAttention)
        self._is_conv = isinstance(self.attn, DoubleGatedConvLayer)
        self._gradient_checkpointing = False

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
        preallocated_cache: Optional["PreAllocatedKVCache"] = None,
        layer_idx: int = 0,
        attention_bias: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, KVCache | None]:
        # Activation checkpointing: recompute forward during backward to save VRAM.
        # Only applies during training (use_cache=False); inference materializes normally.
        if self.training and not use_cache and self._gradient_checkpointing:
            def custom_forward(x_inner):
                attn_out, present = self.attn(self.ln1(x_inner), past_key_value=past_key_value, use_cache=False)
                x_inner = x_inner + attn_out
                ffn_out = self.ffn(self.ln2(x_inner))
                aux = None
                if isinstance(ffn_out, tuple):
                    aux = ffn_out[1]
                    ffn_out = ffn_out[0]
                x_inner = x_inner + ffn_out
                return x_inner, present, aux
            x, present, aux = torch.utils.checkpoint.checkpoint(custom_forward, x, use_reentrant=False)
            self._last_aux_loss = aux
        else:
            if preallocated_cache is not None and self._supports_prealloc_cache:
                attn_out, present = self.attn(
                    self.ln1(x), past_key_value=past_key_value, use_cache=use_cache,
                    preallocated_cache=preallocated_cache, layer_idx=layer_idx,
                    attention_bias=attention_bias, position_ids=position_ids,
                )
            else:
                attn_out, present = self.attn(
                    self.ln1(x), past_key_value=past_key_value, use_cache=use_cache,
                    attention_bias=attention_bias, position_ids=position_ids,
                )
            x = x + attn_out
            ffn_out = self.ffn(self.ln2(x))
            self._last_aux_loss = None
            if isinstance(ffn_out, tuple):
                self._last_aux_loss = ffn_out[1]
                ffn_out = ffn_out[0]
            x = x + ffn_out
        return x, present


def build_attention(config: ModelConfig) -> nn.Module:
    kwargs = dict(d_model=config.d_model, n_heads=config.n_heads, max_seq_len=config.max_seq_len, base=config.rope_base, rope_scaling=config.rope_scaling,
                  use_qk_norm=getattr(config, 'use_qk_norm', False), attn_scale=getattr(config, 'attn_scale', None))
    if config.attn_type == "gqa":
        return GroupedQueryAttention(**kwargs, n_kv_heads=getattr(config, 'n_kv_heads', None),
                                     attn_bias=getattr(config, 'attn_bias', False))
    raise ValueError(
        f"Unknown attention type: '{config.attn_type}'. "
        f"Valid options: 'gqa'")


def build_ffn(config: ModelConfig) -> nn.Module:
    if config.ffn_type == "swiglu":
        return SwiGLUFFN(config.d_model, hidden_dim=getattr(config, 'intermediate_size', None))
    raise ValueError(
        f"Unknown FFN type: '{config.ffn_type}'. "
        f"Valid options: 'swiglu'")


class ConfigurableResearchLLM(nn.Module):
    """Full config-driven research language model."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([ModularBlock(config, layer_idx=i) for i in range(config.n_layers)])
        norm = RMSNorm if getattr(config, 'norm_type', 'layernorm') == 'rmsnorm' else nn.LayerNorm
        self.ln_f = norm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Weight tying: skip if PIT is enabled (PIT replaces tying),
        # or if config explicitly disables it (e.g., Qwen2.5).
        if not getattr(config, 'use_pit', False) and getattr(config, 'tie_word_embeddings', True):
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

        self.draft_head: nn.Module | None = None
        if config.use_gradient_checkpointing:
            self.enable_gradient_checkpointing()

        # MTP heads (Nemotron Lightning): shared-weight multi-token prediction
        self.mtp_module: nn.Module | None = None
        if getattr(config, 'use_mtp', False):
            from research.architecture.mtp import MTPModule
            self.mtp_module = MTPModule(
                d_model=config.d_model,
                vocab_size=config.vocab_size,
                n_heads=getattr(config, 'mtp_n_heads', 2),
                loss_weight=getattr(config, 'mtp_loss_weight', 0.3),
                identity_init=getattr(config, 'zero_init_residual', True),
            )
            # Tie MTP head to model head (shared weight design)
            self.mtp_module.tie_head_to_model(self.head.weight)

        # Cache config flags to avoid getattr/hasattr per forward.
        self._use_liger_ce = getattr(config, 'use_liger_ce', False)
        self._liger_fce = None  # lazy-init on first use

    def enable_gradient_checkpointing(self):
        """Enable activation checkpointing on all transformer blocks."""
        for block in self.blocks:
            block._gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        """Disable activation checkpointing (e.g., for inference)."""
        for block in self.blocks:
            block._gradient_checkpointing = False

    def compile_for_inference(self, mode: str = "default"):
        """torch.compile the model for inference speedup.

        Uses mode="default" (kernel fusion, no CUDA graphs) by default because
        the pre-allocated KV cache has dynamic fill lengths that are incompatible
        with CUDA graph capture (which requires static memory addresses).

        For training or non-cached inference, use mode="reduce-overhead" for
        CUDA graph acceleration (1.3-2x additional speedup).

        Args:
            mode: "default" (kernel fusion only), "reduce-overhead" (+CUDA graphs),
                  "max-autotune" (kernel autotuning, slower compile).

        Returns:
            The compiled model (replaces self in-place via torch.compile wrapper).
        """
        self.eval()
        return torch.compile(self, mode=mode, dynamic=True)

    def compile_decode_step(self, batch_size: int = 1):
        """Compile a dedicated decode-step forward for CUDA graph acceleration.

        The decode step (single token, B×1) has fixed shapes, enabling
        mode="reduce-overhead" with dynamic=False for proper CUDA graph capture.
        This eliminates per-kernel CPU launch overhead — the primary source of
        CPU spikes during autoregressive generation.

        Must be called once per unique batch_size. The returned callable is
        a compiled forward that accepts (idx, past_key_values, use_cache, attention_mask).

        Args:
            batch_size: fixed batch size for this compiled decode step.

        Returns:
            Compiled forward callable for decode steps of shape (batch_size, 1).
        """
        self.eval()
        # Wrap just the forward with reduce-overhead + static shapes for CUDA graphs.
        return torch.compile(
            self.forward,
            mode="reduce-overhead",
            dynamic=False,
            fullgraph=False,  # allow graph breaks for attention_bias conditional
        )

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        past_key_values: list[KVCache | None] | None = None,
        use_cache: bool = False,
        return_hidden: bool = False,
        preallocated_cache: Optional["PreAllocatedKVCache"] = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | tuple[torch.Tensor, torch.Tensor | None, list[KVCache | None]]:
        # Compute position_ids and attention_bias ONCE from attention_mask.
        # These are shared across all 28 layers to avoid per-layer allocation (CPU spike fix).
        position_ids = None
        attention_bias = None  # additive mask for SDPA: (B, 1, T, total_len)
        if attention_mask is not None:
            B, T = idx.shape[:2]
            total_len = attention_mask.shape[1]  # cached + new tokens
            # position_ids: cumsum of mask - 1, clamped to 0 for pad positions.
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids = position_ids.clamp(min=0)
            position_ids = position_ids[:, -T:]  # only for tokens being processed

            # Build additive attention bias ONCE: (B, 1, T, total_len)
            # 0 for real tokens, -inf for pad. Combined with causal for prefill.
            dtype = next(self.parameters()).dtype
            pad_mask = (attention_mask == 0)  # (B, total_len)
            if T == 1 and total_len > 1:
                # Decode: only padding mask, no causal needed.
                # Shape: (B, 1, 1, total_len)
                attention_bias = torch.zeros(B, 1, 1, total_len, device=idx.device, dtype=dtype)
                attention_bias = attention_bias.masked_fill(
                    pad_mask.unsqueeze(1).unsqueeze(1), float('-inf'))
            elif total_len == T:
                # Prefill: combine causal + padding.
                causal = _causal_mask(T, total_len, 0, idx.device, dtype)
                pad_add = torch.zeros(B, 1, T, total_len, device=idx.device, dtype=dtype)
                pad_add = pad_add.masked_fill(
                    pad_mask.unsqueeze(1).unsqueeze(1), float('-inf'))
                attention_bias = causal + pad_add
            else:
                # Chunked prefill with cache.
                past_len = total_len - T
                causal = _causal_mask(T, total_len, past_len, idx.device, dtype)
                pad_add = torch.zeros(B, 1, T, total_len, device=idx.device, dtype=dtype)
                pad_add = pad_add.masked_fill(
                    pad_mask.unsqueeze(1).unsqueeze(1), float('-inf'))
                attention_bias = causal + pad_add

        # Move input to embed's device (handles hybrid offload)
        embed_device = next(self.embed.parameters()).device
        if idx.device != embed_device:
            idx = idx.to(embed_device)
        x = self.embed(idx)
        presents: list[KVCache | None] = []
        # Track device for hybrid offload: move x to each block's device
        cur_device = x.device
        for i, block in enumerate(self.blocks):
            block_device = next(block.parameters()).device
            if block_device != cur_device:
                x = x.to(block_device)
                cur_device = block_device
            past = past_key_values[i] if past_key_values is not None else None
            x, present = block(x, past_key_value=past, use_cache=use_cache,
                               preallocated_cache=preallocated_cache, layer_idx=i,
                               attention_bias=attention_bias, position_ids=position_ids)
            if use_cache:
                presents.append(present)
        # Advance the pre-allocated cache position after all layers processed.
        if preallocated_cache is not None:
            preallocated_cache.advance(idx.shape[1])
        # Move x to ln_f's device (GPU for hybrid offload)
        ln_f_device = next(self.ln_f.parameters()).device
        if x.device != ln_f_device:
            x = x.to(ln_f_device)
        hidden = self.ln_f(x)

        # Collect MoE aux_loss from all blocks that have it
        moe_aux_loss = torch.tensor(0.0, device=idx.device, dtype=hidden.dtype)
        for block in self.blocks:
            if hasattr(block, '_last_aux_loss') and block._last_aux_loss is not None:
                moe_aux_loss = moe_aux_loss + block._last_aux_loss

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
        elif self._use_liger_ce and targets is not None and not use_cache:
            # Liger-Kernel fused linear cross-entropy: one Triton kernel for
            # head matmul + CE, avoids [B*T, V] logits entirely. Saves ~620MB
            # VRAM (bf16 logits at batch 2 / seq 1024 / vocab 151665).
            # NOT compatible with --compile (graph break kills backward compilation,
            # 5x slower). Use with --no-compile for memory-constrained scenarios.
            from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
            if self._liger_fce is None:
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

        # Add MoE aux_loss (load balancing) to total loss
        if loss is not None and moe_aux_loss.requires_grad:
            loss = loss + moe_aux_loss

        # MTP loss (Nemotron Lightning): multi-token prediction with shared weights
        if self.mtp_module is not None and targets is not None and not use_cache and targets.size(1) > 3:
            token_embeds = self.embed(idx)  # ground truth token embeddings
            mtp_loss, _ = self.mtp_module(hidden, token_embeds, targets)
            if mtp_loss is not None and loss is not None:
                loss = loss + mtp_loss

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
        layer_types_sig = tuple(getattr(config, 'layer_types', None) or [])
        mtp_sig = f"mtp{getattr(config, 'use_mtp', False)}_{getattr(config, 'mtp_n_heads', 0)}"
        return f"{config.d_model}_{config.n_layers}_{config.attn_type}_{config.ffn_type}_{config.norm_type}_{getattr(config, 'kv_compression_dim', 0)}_{getattr(config, 'n_kv_heads', 0)}_{getattr(config, 'attn_bias', False)}_{layer_types_sig}_{mtp_sig}"

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

    # Cache of built model architectures (on CPU) for fast cloning.
    # Bounded LRU (max 4 entries) — each entry holds a full model in RAM,
    # so an unbounded cache would grow without limit across many configs.
    _model_cache: OrderedDict = OrderedDict()
    _MODEL_CACHE_MAXSIZE = 4

    @staticmethod
    def clear_cache():
        """Clear the architecture cache and blank state-dict cache."""
        ModelLoader._model_cache.clear()
        ModelLoader._blank_cache.clear()

    @staticmethod
    def _load_safetensors_mmap(path: str, model: nn.Module,
                               device: torch.device = None) -> dict:
        """Load safetensors weights via memory-mapped access.

        Uses safetensors' safe_open for lazy per-tensor loading. Each tensor
        is read from the mmap'd file and copied directly to the target device,
        avoiding the 2x RAM overhead of loading the full state dict to CPU
        first. This is ~2x faster for GPU loading and uses ~0 peak CPU RAM.

        For CPU-only models, tensors stay mmap'd (zero-copy) until accessed.
        """
        from safetensors import safe_open
        if device is None:
            device = next(model.parameters()).device

        state = {}
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)  # mmap'd read
                state[key] = tensor.to(device) if device.type != "cpu" else tensor
        return state

    @staticmethod
    def _load_sharded_safetensors(ckpt_dir: Path, model: nn.Module,
                                  device: torch.device = None) -> dict:
        """Load weights from sharded safetensors (model-00001-of-00002.safetensors)."""
        from safetensors import safe_open
        if device is None:
            device = next(model.parameters()).device
        state = {}
        for sf_path in sorted(ckpt_dir.glob("model-*.safetensors")):
            with safe_open(str(sf_path), framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    state[key] = tensor.to(device) if device.type != "cpu" else tensor
        return state

    @staticmethod
    def _remap_hf_keys(state: dict, config) -> dict:
        """Remap HuggingFace Qwen/Llama-style keys to ForgeAI internal names."""
        import re
        new_state = {}
        for key, tensor in state.items():
            # model.embed_tokens.weight → embed.weight
            if key == "model.embed_tokens.weight":
                new_state["embed.weight"] = tensor
            # model.norm.weight → ln_f.weight
            elif key == "model.norm.weight":
                new_state["ln_f.weight"] = tensor
            # lm_head.weight → head.weight
            elif key == "lm_head.weight":
                new_state["head.weight"] = tensor
            # model.layers.{i}.self_attn.{proj}.{weight|bias} → blocks.{i}.attn.{proj}.{weight|bias}
            elif m := re.match(r"model\.layers\.(\d+)\.self_attn\.(.+)", key):
                layer = m.group(1)
                rest = m.group(2)
                # q_proj/k_proj/v_proj → same name
                # o_proj → out_proj
                rest = rest.replace("o_proj", "out_proj")
                new_state[f"blocks.{layer}.attn.{rest}"] = tensor
            # model.layers.{i}.input_layernorm.weight → blocks.{i}.ln1.weight
            elif m := re.match(r"model\.layers\.(\d+)\.input_layernorm\.(.+)", key):
                new_state[f"blocks.{m.group(1)}.ln1.{m.group(2)}"] = tensor
            # model.layers.{i}.post_attention_layernorm.weight → blocks.{i}.ln2.weight
            elif m := re.match(r"model\.layers\.(\d+)\.post_attention_layernorm\.(.+)", key):
                new_state[f"blocks.{m.group(1)}.ln2.{m.group(2)}"] = tensor
            # model.layers.{i}.mlp.gate_proj.weight → blocks.{i}.ffn.w_gate.weight
            elif m := re.match(r"model\.layers\.(\d+)\.mlp\.gate_proj\.(.+)", key):
                new_state[f"blocks.{m.group(1)}.ffn.w_gate.{m.group(2)}"] = tensor
            # model.layers.{i}.mlp.up_proj.weight → blocks.{i}.ffn.w_up.weight
            elif m := re.match(r"model\.layers\.(\d+)\.mlp\.up_proj\.(.+)", key):
                new_state[f"blocks.{m.group(1)}.ffn.w_up.{m.group(2)}"] = tensor
            # model.layers.{i}.mlp.down_proj.weight → blocks.{i}.ffn.w_down.weight
            elif m := re.match(r"model\.layers\.(\d+)\.mlp\.down_proj\.(.+)", key):
                new_state[f"blocks.{m.group(1)}.ffn.w_down.{m.group(2)}"] = tensor
            else:
                new_state[key] = tensor  # pass through unknown keys
        return new_state

    @staticmethod
    def build_model_fast(config: ModelConfig, checkpoint_path: str | None = None,
                         compile: bool = False, moe_top_k: int | None = None,
                         dtype: torch.dtype | None = None):
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
            # Cache on CPU to avoid VRAM duplication on deepcopy
            ModelLoader._model_cache[sig] = model.cpu()
            ModelLoader._model_cache.move_to_end(sig)
            # LRU eviction: keep at most _MODEL_CACHE_MAXSIZE architectures.
            while len(ModelLoader._model_cache) > ModelLoader._MODEL_CACHE_MAXSIZE:
                ModelLoader._model_cache.popitem(last=False)
            model = model.to(device)
            t_arch = time.time() - t_arch
            print(f"  [FastBuild] Architecture built in {t_arch:.1f}s (cached)")
        else:
            t_clone = time.time()
            cached = ModelLoader._model_cache[sig]
            ModelLoader._model_cache.move_to_end(sig)  # mark as recently used
            # Deep copy the cached model (CPU — no VRAM overhead)
            import copy
            model = copy.deepcopy(cached).to(device)
            t_clone = time.time() - t_clone
            print(f"  [FastBuild] Architecture cloned in {t_clone:.1f}s (from cache)")

        # Convert dtype before loading weights (prevents bf16→fp32 upcast)
        if dtype is not None:
            model = model.to(dtype)

        if compile:
            model = torch.compile(model, mode="reduce-overhead", dynamic=True)

        if checkpoint_path and os.path.exists(checkpoint_path):
            t_load = time.time()
            ckpt_path = Path(checkpoint_path)

            # Handle sharded models (directory with model.safetensors.index.json)
            if ckpt_path.is_dir():
                index_file = ckpt_path / "model.safetensors.index.json"
                if index_file.exists():
                    state = ModelLoader._load_sharded_safetensors(
                        ckpt_path, model, device=device)
                else:
                    # Single safetensors in directory
                    sf_files = list(ckpt_path.glob("*.safetensors"))
                    if len(sf_files) == 1:
                        state = ModelLoader._load_safetensors_mmap(
                            str(sf_files[0]), model, device=device)
                    else:
                        state = ModelLoader._load_sharded_safetensors(
                            ckpt_path, model, device=device)
            elif str(checkpoint_path).endswith(".safetensors"):
                state = ModelLoader._load_safetensors_mmap(
                    checkpoint_path, model, device=device)
            else:
                from research.checkpoint_io import load_checkpoint
                state = load_checkpoint(checkpoint_path, map_location="cpu")
                if isinstance(state, dict) and "model_state" in state and not any(k.startswith("blocks.") for k in state):
                    state = state["model_state"]

            # Auto-detect and remap HuggingFace keys to ForgeAI internal names.
            if state and any(k.startswith("model.") for k in state):
                state = ModelLoader._remap_hf_keys(state, config)
                print(f"  [FastBuild] Remapped {len(state)} HF keys to ForgeAI format")

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
    def hybrid_offload(model: nn.Module, gpu_layers: int = -1,
                       device: str = "cuda") -> nn.Module:
        """Offload specific layers to GPU, keep rest on CPU.

        For LFM2.5 hybrid models, conv layers are cheap (O(T*k*d)) and can
        run on CPU, while attention layers benefit from GPU (O(T^2*d) matmuls).
        This enables running larger models on limited VRAM.

        Args:
            model: the model to offload (must have .blocks ModuleList)
            gpu_layers: number of layers to put on GPU (from the end).
                       -1 = put attention layers on GPU, conv on CPU.
                       N = last N layers on GPU, rest on CPU.
            device: GPU device string ("cuda", "cuda:0", etc.)

        Returns:
            The model with per-layer device placement applied.
        """
        gpu_dev = torch.device(device)
        cpu_dev = torch.device("cpu")

        if not hasattr(model, 'blocks'):
            return model.to(gpu_dev)

        layer_types = getattr(model.config, 'layer_types', None)

        # Always keep embed, head, ln_f on GPU
        model.embed = model.embed.to(gpu_dev)
        model.head = model.head.to(gpu_dev)
        if hasattr(model, 'ln_f'):
            model.ln_f = model.ln_f.to(gpu_dev)

        gpu_count = 0
        cpu_count = 0
        for i, block in enumerate(model.blocks):
            if gpu_layers == -1 and layer_types:
                # Auto: attention on GPU, conv on CPU
                lt = layer_types[i] if i < len(layer_types) else "attention"
                target = gpu_dev if lt == "attention" else cpu_dev
            elif gpu_layers == -1:
                target = gpu_dev
            else:
                # Last N layers on GPU
                target = gpu_dev if i >= len(model.blocks) - gpu_layers else cpu_dev

            block.to(target)
            if target == gpu_dev:
                gpu_count += 1
            else:
                cpu_count += 1

        print(f"  [HybridOffload] {gpu_count} layers on {device}, {cpu_count} on CPU")
        return model

    @staticmethod
    def build_model(config: ModelConfig, checkpoint_path: str | None = None, compile: bool = False):
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
        tokenizer: "Any",
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_k: int | None = None,
    ) -> str:
        """Generate text with pre-allocated KV cache and batched EOS check.

        Optimizations vs the original loop:
        - PreAllocatedKVCache: O(1) append per token instead of O(n) torch.cat
        - GPU-side EOS check: (next_token == eos_id).any() — one sync instead of .item() per token
        - Pre-allocated output buffer: write by index instead of torch.cat per token
        """
        model.eval()
        device = next(model.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt")
        # Handle both HF BatchEncoding (.to()) and plain dict (gigatoken)
        if hasattr(inputs, 'to'):
            inputs = inputs.to(device)
        else:
            inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        prompt_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
        B, prompt_len = prompt_ids.shape
        eos_id = tokenizer.eos_token_id

        # Pre-allocate output buffer (prompt + max_new_tokens).
        max_total = prompt_len + max_new_tokens
        out_ids = torch.zeros(B, max_total, dtype=prompt_ids.dtype, device=device)
        out_ids[:, :prompt_len] = prompt_ids

        # Determine KV cache shape from the attention type.
        cfg = model.config
        n_kv_heads = cfg.n_kv_heads if cfg.attn_type == "gqa" else cfg.n_heads
        head_dim = cfg.d_model // cfg.n_heads
        dtype = next(model.parameters()).dtype

        # For hybrid conv/attention models (v4+): conv layers don't use KV cache.
        # Allocate 0 heads for conv layers to save VRAM (22 layers × ~3MB each).
        layer_types = getattr(cfg, 'layer_types', None)
        if layer_types is not None:
            n_kv_heads_per_layer = [
                n_kv_heads if (i < len(layer_types) and layer_types[i] in ("attention", "attn"))
                else 0
                for i in range(cfg.n_layers)
            ]
        else:
            n_kv_heads_per_layer = None

        cache = PreAllocatedKVCache(
            n_layers=cfg.n_layers, batch=B, n_kv_heads=n_kv_heads,
            max_seq_len=min(cfg.max_seq_len, max_total),
            head_dim=head_dim, dtype=dtype, device=device,
            n_kv_heads_per_layer=n_kv_heads_per_layer,
        )

        with torch.no_grad():
            for step in range(max_new_tokens):
                pos = prompt_len + step
                if step == 0:
                    # Prefill: feed the full prompt.
                    # use_cache=True so conv layers initialize their state buffer.
                    idx_cond = out_ids[:, :prompt_len]
                    out = model(idx_cond, preallocated_cache=cache, use_cache=True)
                    logits = out[0]
                else:
                    # Decode: feed only the last generated token.
                    # use_cache=True so conv layers use incremental conv with state.
                    idx_cond = out_ids[:, pos - 1:pos]
                    out = model(idx_cond, preallocated_cache=cache, use_cache=True)
                    logits = out[0]

                logits = logits[:, -1, :] / max(temperature, 1e-5)

                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
                out_ids[:, pos:pos + 1] = next_token

                # Batch EOS check on GPU — one .any() sync instead of .item() per token.
                if eos_id is not None and (next_token == eos_id).any().item():
                    break

        # Decode only the generated portion (trim trailing zeros).
        actual_len = prompt_len + step + 1
        return tokenizer.decode(out_ids[0, :actual_len], skip_special_tokens=True)


def load_default_model(
    config_name: str = "lfm25_1.2b",
    checkpoint_path: str | None = None,
    device: str = "cuda",
    dtype: torch.dtype | None = None,
    moe_top_k: int = 0,
    compile_mode: str | None = None,
):
    """Load a model + tokenizer in one call.

    Centralizes the common pattern used across 10+ files:
        cfg = get_config(name, device=device)
        model = ModelLoader.build_model_fast(cfg, checkpoint_path=..., moe_top_k=..., dtype=...)
        tokenizer = get_tokenizer(...)

    Args:
        config_name: model config name (default "lfm25_1.2b")
        checkpoint_path: path to .safetensors checkpoint (default: config default)
        device: "cuda" or "cpu"
        dtype: torch.bfloat16 or torch.float32 (default: bf16 for cuda, fp32 for cpu)
        moe_top_k: MoE top-k routing (0 = dense_bypass)
        compile_mode: torch.compile mode if set (e.g. "default", "reduce-overhead")

    Returns:
        (model, tokenizer) tuple
    """
    from research.config import get_config
    from research.tokenizer_cache import get_tokenizer

    cfg = get_config(config_name, device=device)
    if dtype is None:
        dtype = torch.bfloat16 if "cuda" in device else torch.float32
    model = ModelLoader.build_model_fast(
        cfg, checkpoint_path=checkpoint_path,
        moe_top_k=moe_top_k, dtype=dtype)
    model.to(device).eval()

    if compile_mode is not None:
        try:
            model = model.compile_for_inference(mode=compile_mode)
        except Exception:
            pass

    tokenizer = get_tokenizer("research/checkpoints/lfm25_tokenizer")
    return model, tokenizer


def quantize_int4(model: torch.nn.Module, group_size: int = 32) -> torch.nn.Module:
    """Apply int4 weight-only quantization using torchao.

    Reduces model VRAM by ~58% (bf16 → int4) with minimal accuracy loss.
    Works with torch.compile and FSDP2.

    Requires MSLK (mslk-cuda>=1.0.0) for the int4 packing kernels.
    If MSLK is not available, falls back to int8 weight-only quantization
    (50% VRAM reduction instead of 58%).

    Args:
        model: model to quantize (must be on CUDA)
        group_size: quantization group size (32 = good balance, 64 = faster)

    Returns:
        The quantized model (modified in-place).

    Example:
        model = load_default_model("lfm25_1.2b")
        model = quantize_int4(model)  # 2.3GB → ~0.7GB VRAM (int4)
    """
    try:
        from torchao.quantization import quantize_
    except ImportError:
        print("torchao not installed — skipping quantization")
        return model

    if not torch.cuda.is_available():
        print("quantization requires CUDA — skipping")
        return model

    # Try int4 first (requires MSLK), fall back to int8
    try:
        from torchao.quantization import Int4WeightOnlyConfig
        quantize_(model, Int4WeightOnlyConfig(group_size=group_size))
        print("  [torchao] Applied int4 weight-only quantization")
    except (ImportError, RuntimeError) as e:
        print(f"  [torchao] int4 unavailable ({e}), falling back to int8")
        from torchao.quantization import Int8WeightOnlyConfig
        quantize_(model, Int8WeightOnlyConfig())
        print("  [torchao] Applied int8 weight-only quantization")
    return model


if __name__ == "__main__":
    from research.config import get_config

    for name in ["lfm25_tiny", "lfm25_1.2b"]:
        print("\n" + "=" * 50)
        cfg = get_config(name)
        cfg.device = "cpu"
        ModelLoader.build_model(cfg)
