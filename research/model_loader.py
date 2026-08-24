"""Modular model factory and inference engine for ForgeAI research."""
import copy
import math
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Enable safetensors fast CUDA loading (pinned-memory + async DMA to GPU).
# This dramatically speeds up weight loading on CUDA by using pinned host
# memory and asynchronous copies instead of synchronous per-tensor CPU→GPU.
os.environ.setdefault("SAFETENSORS_FAST_CUDA", "1")

# Short Triton/inductor cache dirs. The default (%TEMP%\torchinductor_<user>
# with a "triton\<device>\<key>" suffix) plus Triton's long fused kernel names
# (~130 chars) exceeds Windows MAX_PATH (260 chars), so open() fails with
# FileNotFoundError during torch.compile. Project-local short dirs keep paths
# well under the limit and persist compiled kernels across runs.
from research.paths import TORCH_CACHE_DIR  # noqa: E402
os.environ.setdefault("TRITON_CACHE_DIR", str(TORCH_CACHE_DIR / "triton"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(TORCH_CACHE_DIR))
os.environ.setdefault("TORCHINDUCTOR_PERSISTENT_AUTOTUNE_DIR", str(TORCH_CACHE_DIR))

# Enable TensorFloat32 tensor cores for float32 matmuls (RTX 5070 supports this).
# Free ~8x speedup on fp32 matmuls with negligible precision loss (~1e-5).
torch.set_float32_matmul_precision("high")
# Also enable TF32 for cuDNN convolutions (affects conv layers, attention padding ops)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

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
        if pos + T > self.max_seq_len:
            raise ValueError(
                f"KV cache overflow: pos={pos} + T={T} > max_seq_len={self.max_seq_len}. "
                f"Increase max_seq_len or use a paged/evicting KV cache strategy."
            )
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


def create_kv_cache(model: nn.Module, max_total: int, batch: int = 1,
                    device: Optional[torch.device] = None) -> PreAllocatedKVCache:
    """Build a PreAllocatedKVCache sized for *model* with per-layer head counts.

    Handles hybrid conv/attention architectures (e.g. LFM2.5): conv layers get
    0 KV heads to avoid wasting VRAM.  This replaces the open-coded cache
    construction that was duplicated across self_play and inference modules.
    """
    cfg = model.config
    n_kv_heads = cfg.n_kv_heads if cfg.n_kv_heads is not None else cfg.n_heads
    head_dim = cfg.d_model // cfg.n_heads
    if getattr(cfg, 'use_diff_attn', False) or cfg.attn_type == "diff":
        # Differential attention stores two q/k groups per head slot.
        head_dim *= 2
    dtype = next(model.parameters()).dtype
    if device is None:
        device = next(model.parameters()).device

    layer_types = getattr(cfg, "layer_types", None)
    if layer_types is not None:
        n_kv_heads_per_layer = [
            n_kv_heads if (i < len(layer_types) and layer_types[i] in ("attention", "attn"))
            else 0
            for i in range(cfg.n_layers)
        ]
    else:
        n_kv_heads_per_layer = None

    return PreAllocatedKVCache(
        n_layers=cfg.n_layers, batch=batch, n_kv_heads=n_kv_heads,
        max_seq_len=min(cfg.max_seq_len, max_total),
        head_dim=head_dim, dtype=dtype, device=device,
        n_kv_heads_per_layer=n_kv_heads_per_layer,
    )


def unpack_output_with_kv(out) -> Tuple[torch.Tensor, Optional[KVCache]]:
    """Unpack a model forward output into (logits, past_kv).

    Handles the (logits, loss, presents) and (logits, presents) tuple shapes
    emitted by ConfigurableResearchLLM.  Replaces the 3-4 line
    ``if isinstance(out, tuple): ...`` block repeated across inference paths.
    """
    if isinstance(out, tuple):
        logits = out[0]
        past_kv = out[2] if len(out) > 2 else out[1]
        return logits, past_kv
    return out, None


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
        self.base = base
        self.max_seq_len = max_seq_len
        self.rope_scaling = rope_scaling
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

        # ValueResidual: V_0 from layer 0 is injected by the parent model.
        # When set, v = v + gate * v0 (applied before attention).
        self._v0_residual: torch.Tensor | None = None
        self._v0_gate: torch.Tensor | None = None
        self._v0_capture: torch.Tensor | None = None  # set by parent for layer 0

        # LearnedSink (GPT-OSS): per-head attention sink bias.
        # init=0 → lossless (no bias added). Training learns sink values.
        self.sinks: nn.Parameter | None = None

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

        # ValueResidual (ResFormer): add V_0 from layer 0 to this layer's V.
        # gate=0 at init → v unchanged (lossless). Training opens the gate.
        if self._v0_residual is not None and self._v0_gate is not None:
            gate_val = self._v0_gate
            if gate_val.item() != 0.0:
                # V_0 is (B, n_kv_heads, T, head_dim) — same shape as v.
                v = v + gate_val * self._v0_residual
        # Capture V_0 from layer 0 (set by parent model for the first layer).
        if self._v0_capture is not None:
            self._v0_capture = v.detach()

        # Fused QK-Norm + RoPE path (opt-in via FORGE_FUSED_ROPE_QKNORM=1).
        # Fuses RMSNorm and RoPE into a single Triton kernel, halving HBM
        # traffic for Q/K preprocessing. Only used when QK-norm is active
        # (non-identity weights) and position_ids is None (offset path).
        _use_fused = (
            os.environ.get("FORGE_FUSED_ROPE_QKNORM", "0") == "1"
            and self.use_qk_norm and not self._qk_norm_identity
            and position_ids is None and q.is_cuda
        )
        if _use_fused:
            from research.decoding.fused_rope_qknorm import fused_qk_norm_rope
            if preallocated_cache is not None:
                past_len = preallocated_cache.position
            else:
                past_len = past_key_value[0].shape[-2] if past_key_value is not None else 0
            # Slice cos/sin tables to the current sequence positions
            cos_slice = self.rope.cos_cached[past_len:past_len + T, :].to(q.dtype)
            sin_slice = self.rope.sin_cached[past_len:past_len + T, :].to(q.dtype)
            q, k = fused_qk_norm_rope(
                q, k, self.q_norm.weight, self.k_norm.weight,
                cos_slice, sin_slice, eps=self.q_norm.eps)
        else:
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
            else:
                past_len = past_key_value[0].shape[-2] if past_key_value is not None else 0
                q = self.rope(q, offset=past_len, position_ids=position_ids)
                k = self.rope(k, offset=past_len, position_ids=position_ids)

        # Cache append + KV retrieval (same for both paths)
        if preallocated_cache is not None:
            preallocated_cache.append(layer_idx, k, v)
            k = preallocated_cache.k_caches[layer_idx][:, :, :past_len + T]
            v = preallocated_cache.v_caches[layer_idx][:, :, :past_len + T]
        elif past_key_value is not None and not _use_fused:
            k = torch.cat([past_key_value[0], k], dim=-2)
            v = torch.cat([past_key_value[1], v], dim=-2)

        new_kv = (k, v) if use_cache else None

        # Repeat KV heads to match Q heads.
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        # Single-token decode with cached KV: no causal mask needed (all keys are valid)
        total_len = k.shape[-2]
        # CSA (Compressed Sparse Attention): for long sequences, use top-k
        # position selection to reduce O(S^2) → O(S*k). Short sequences
        # use full attention (lossless when S <= top_k).
        csa_top_k = getattr(self, '_csa_top_k', 0)
        if getattr(self, '_csa_enabled', False) and csa_top_k > 0 and total_len > csa_top_k and T > 1:
            from research.keys.attention.csa_key import CSAAttention
            csa = CSAAttention(
                self.d_model if hasattr(self, 'd_model') else C,
                self.n_heads, top_k=csa_top_k,
                head_dim=self.head_dim)
            out = csa(q, k, v, is_causal=True)
            out = out.transpose(1, 2).reshape(B, T, C)
            return self.out_proj(out), new_kv
        # LearnedSink: add per-head sink bias to attention scores.
        # sink is (n_heads,) → broadcast to (1, n_heads, 1, total_len).
        # init=0 → no-op (lossless). Training learns positive sink values.
        sink_bias = None
        if self.sinks is not None and self.sinks.abs().max().item() != 0.0:
            sink_bias = self.sinks.view(1, self.n_heads, 1, 1).expand(1, self.n_heads, 1, total_len)
        if attention_bias is not None:
            if sink_bias is not None:
                attention_bias = attention_bias + sink_bias.to(attention_bias.dtype)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_bias)
        elif sink_bias is not None:
            # Build a combined bias: causal + sink
            if T == 1 and total_len > 1:
                bias = sink_bias.to(q.dtype).expand(B, self.n_heads, 1, total_len)
            else:
                causal = _causal_mask(T, total_len, 0, q.device, q.dtype)
                bias = causal + sink_bias.to(q.dtype).expand(B, self.n_heads, T, total_len)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
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

        # New sequence (no KV cache carried over): stale conv state from a
        # previous sequence must not be reused. Prefill (T>1) re-initializes
        # state below, but a T==1-first call would silently reuse old state.
        if past_key_value is None and T == 1:
            self._conv_state = None

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
    """SwiGLU feed-forward network.

    With use_clamp=True, uses GPT-OSS clamped SwiGLU: gate clamped to
    [None, limit], up clamped to [-limit, limit], scaled sigmoid (α=1.702),
    and +1 residual on the linear path. Prevents outlier activations.
    """

    def __init__(self, d_model: int = 768, hidden_dim: int | None = None,
                 use_clamp: bool = False, clamp_alpha: float = 1.702,
                 clamp_limit: float = 7.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(8 * d_model / 3)
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_up = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=False)
        self.use_clamp = use_clamp
        self.clamp_alpha = clamp_alpha
        self.clamp_limit = clamp_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w_gate(x)
        up = self.w_up(x)
        if self.use_clamp:
            # GPT-OSS clamped SwiGLU: scaled sigmoid + clamp + (up+1) residual
            gate = gate.clamp(min=None, max=self.clamp_limit)
            up = up.clamp(min=-self.clamp_limit, max=self.clamp_limit)
            glu = gate * torch.sigmoid(self.clamp_alpha * gate)
            return self.w_down((up + 1) * glu)
        return self.w_down(F.silu(gate) * up)


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
        self._supports_prealloc_cache = isinstance(self.attn, GroupedQueryAttention) or \
            type(self.attn).__name__ in (
                "DifferentialAttention",
                "GroupedTiedAttention",
                "GroupedLatentAttention")
        self._is_conv = isinstance(self.attn, DoubleGatedConvLayer)
        self._gradient_checkpointing = False
        # Selective checkpoint strategy: "all" (full block), "ffn" (recompute
        # only FFN — biggest activation consumer, ~2-4x VRAM savings on
        # intermediates with minimal compute penalty), "attn" (recompute only
        # attention), "none" (no recomputation).
        self._gradient_checkpointing_strategy = getattr(
            config, 'selective_gradient_checkpointing', 'all')

        # FFN-SkipLLM: skip FFN on saturated layers during eval.
        # Disabled by default — calibration shows no saturation in V3 (16 layers).
        # See docs/FFN_RESEARCH.md. Infrastructure kept for future 32+ layer models.
        self._ffn_skip_threshold = getattr(config, 'ffn_skip_threshold', 0.0)
        self._ffn_skip_count = 0
        self._static_skip_layers: set[int] = set()  # populated when threshold > 0

        # SandwichNorm: post-sublayer RMSNorm (identity init = lossless).
        # Applied after attention output and after FFN output, before the
        # residual add. Stabilizes MoE training by bounding activations.
        self._use_sandwich = getattr(config, 'use_sandwich_norm', False)
        if self._use_sandwich:
            self.post_attn_norm = norm(config.d_model)
            self.post_ffn_norm = norm(config.d_model)

        # TITAN neural memory + MoD token router (zero-init => lossless at
        # start; the ported checkpoint loads and behaves identically).
        self._memory = None
        self._mod = None
        self._mhc = None
        self._mem_gate_zero: bool | None = None  # cached eval gate state
        if getattr(config, 'use_titan_memory', False):
            from research.keys.architecture.titan_memory_key import TitanMemory
            self._memory = TitanMemory(
                config.d_model,
                rank=getattr(config, 'titan_memory_rank', 0))
        if getattr(config, 'use_mod', False):
            from research.keys.architecture.mod_router_key import ModRouter
            self._mod = ModRouter(
                config.d_model,
                keep_fraction=getattr(config, 'mod_keep_fraction', 1.0))
        # MHC: Manifold Hyper-Connections (gate=0 → lossless at start).
        if getattr(config, 'use_mhc', False):
            from research.keys.architecture.mhc_key import MHCModule
            mhc_rank = getattr(config, 'mhc_rank', 0)
            # rank=0 means "auto" (d_model // 4); MHCModule needs None for auto.
            self._mhc = MHCModule(
                config.d_model,
                rank=mhc_rank if mhc_rank > 0 else None)
            self._mhc_gate_zero: bool | None = None

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
        preallocated_cache: Optional["PreAllocatedKVCache"] = None,
        layer_idx: int = 0,
        attention_bias: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        # DiffusionBlocks: AdaLN modulation (shift_msa, scale_msa, gate_msa,
        # shift_mlp, scale_mlp, gate_mlp) — 6 * d_model values
        modulation: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, KVCache | None]:
        x0 = x  # pre-update residual (for TITAN read + MoD gating)

        # TRUE Mixture-of-Depths: in training (no KV cache, no attention
        # mask) router-skipped tokens genuinely BYPASS attention + FFN —
        # the block only computes the top-k kept tokens, so FLOPs scale
        # with keep_fraction instead of T. (Inference keeps all tokens to
        # preserve KV cache position alignment.)
        if (self.training and not use_cache and self._mod is not None
                and self._mod.keep_fraction < 1.0
                and attention_bias is None and position_ids is None):
            return self._forward_mod_skip(
                x, layer_idx, attention_bias, position_ids), None

        # Activation checkpointing: recompute forward during backward to save VRAM.
        # Only applies during training (use_cache=False); inference materializes normally.
        if self.training and not use_cache and self._gradient_checkpointing:
            strategy = self._gradient_checkpointing_strategy
            if strategy == "all":
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
            elif strategy == "ffn":
                # Keep attention activations (cheap: ~2*d_model per token),
                # recompute only the FFN (hidden_dim >> d_model) during backward.
                if preallocated_cache is not None and self._supports_prealloc_cache:
                    attn_out, present = self.attn(
                        self.ln1(x), past_key_value=past_key_value, use_cache=False,
                        preallocated_cache=preallocated_cache, layer_idx=layer_idx,
                        attention_bias=attention_bias, position_ids=position_ids,
                    )
                else:
                    attn_out, present = self.attn(
                        self.ln1(x), past_key_value=past_key_value, use_cache=False,
                        attention_bias=attention_bias, position_ids=position_ids,
                    )
                x = x + attn_out
                ffn_out = torch.utils.checkpoint.checkpoint(
                    self.ffn, self.ln2(x), use_reentrant=False)
                self._last_aux_loss = None
                if isinstance(ffn_out, tuple):
                    self._last_aux_loss = ffn_out[1]
                    ffn_out = ffn_out[0]
                x = x + ffn_out
            elif strategy == "attn":
                # Recompute only attention; FFN activations stay materialized.
                def _attn_forward(x_inner):
                    return self.attn(
                        x_inner, past_key_value=past_key_value, use_cache=False,
                        preallocated_cache=preallocated_cache if self._supports_prealloc_cache else None,
                        layer_idx=layer_idx, attention_bias=attention_bias,
                        position_ids=position_ids)
                attn_out, present = torch.utils.checkpoint.checkpoint(
                    _attn_forward, self.ln1(x), use_reentrant=False)
                x = x + attn_out
                ffn_out = self.ffn(self.ln2(x))
                self._last_aux_loss = None
                if isinstance(ffn_out, tuple):
                    self._last_aux_loss = ffn_out[1]
                    ffn_out = ffn_out[0]
                x = x + ffn_out
            else:  # "none" — no recomputation this block
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
        else:
            # DiffusionBlocks: extract AdaLN modulation (4 * d_model)
            # No gates — shift/scale only (zero-init = identity, gradients flow)
            shift_msa = scale_msa = shift_mlp = scale_mlp = None
            if modulation is not None:
                # Cast modulation to x's dtype (AdaLN may be float32, x may be bf16)
                mod = modulation.to(x.dtype)
                chunks = mod.chunk(4, dim=-1)
                shift_msa, scale_msa, shift_mlp, scale_mlp = chunks

            # Attention path
            attn_in = self.ln1(x)
            if shift_msa is not None:
                attn_in = attn_in * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
            if preallocated_cache is not None and self._supports_prealloc_cache:
                attn_out, present = self.attn(
                    attn_in, past_key_value=past_key_value, use_cache=use_cache,
                    preallocated_cache=preallocated_cache, layer_idx=layer_idx,
                    attention_bias=attention_bias, position_ids=position_ids,
                )
            else:
                attn_out, present = self.attn(
                    attn_in, past_key_value=past_key_value, use_cache=use_cache,
                    attention_bias=attention_bias, position_ids=position_ids,
                )
            x = x + attn_out
            # SandwichNorm: post-attention norm (identity init = no-op).
            if self._use_sandwich:
                x = self.post_attn_norm(x)
            # FFN-SkipLLM: skip FFN on saturated layers during eval.
            # Uses a static skip set (calibrated via cosine similarity).
            # Safe with KV cache: FFN doesn't touch KV state, only attention.
            # Disabled by default — V3 has no saturation region (see docs/FFN_RESEARCH.md).
            if (self._ffn_skip_threshold > 0.0 and not self.training
                    and self.layer_idx in self._static_skip_layers):
                self._ffn_skip_count += 1
                ffn_out = torch.zeros_like(x)
                self._last_aux_loss = None
            else:
                ffn_in = self.ln2(x)
                if shift_mlp is not None:
                    ffn_in = ffn_in * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
                ffn_out = self.ffn(ffn_in)
                self._last_aux_loss = None
                if isinstance(ffn_out, tuple):
                    self._last_aux_loss = ffn_out[1]
                    ffn_out = ffn_out[0]
            x = x + ffn_out
            # SandwichNorm: post-FFN norm (identity init = no-op).
            if self._use_sandwich:
                x = self.post_ffn_norm(x)

        # TITAN memory read + MoD token gating (zero-init => lossless).
        # Applied on the combined block update for every branch above.
        if self._memory is not None or self._mod is not None:
            # Fast lossless path: TITAN gate closed + MoD keep-all => no-op,
            # return x untouched (bit-exact vs. a plain block).
            # The gate .item() check is a GPU->CPU sync; the gate value is
            # static (zero-init, only changed by optimizer steps), so it is
            # cached after the first check (16 syncs/token otherwise — the
            # biggest decode overhead on V3). Caching applies to both training
            # and eval: the gate parameter only changes via optimizer weight
            # updates, not during forward passes.
            if self._memory is None:
                mem_noop = True
            elif self._mem_gate_zero is None:
                self._mem_gate_zero = (self._memory.gate.item() == 0.0)
                mem_noop = self._mem_gate_zero
            else:
                mem_noop = self._mem_gate_zero
            mod_noop = (self._mod is None
                        or self._mod.token_mask(x0) is None)
            if not (mem_noop and mod_noop):
                update = x - x0
                if self._memory is not None and not mem_noop:
                    update = update + self._memory(x0)
                if self._mod is not None and not mod_noop:
                    update = self._mod.apply(x0, update)
                x = x0 + update

        # MHC: Manifold Hyper-Connection (gate=0 → lossless at start).
        # Wraps the full block update: x = x0 + update + gate * proj(update).
        # At gate=0, this is x = x0 + update (standard residual, bit-exact).
        if self._mhc is not None:
            if self._mhc_gate_zero is None:
                self._mhc_gate_zero = (self._mhc.gate.item() == 0.0)
            if not self._mhc_gate_zero:
                update = x - x0
                x = self._mhc(x0, update)

        return x, present

    def _forward_mod_skip(self, x: torch.Tensor, layer_idx: int,
                          attention_bias: torch.Tensor | None,
                          position_ids: torch.Tensor | None
                          ) -> torch.Tensor:
        """Run the block only on router-kept tokens (per batch row).

        Matches the MoD paper: at this depth, skipped tokens are absent from
        attention entirely (causal within the kept subsequence). Kept tokens
        still get the full attention + FFN + TITAN update. The hard top-k
        selection is non-differentiable, so the router is trained by the
        aux loss attached to `_last_aux_loss` (see ModRouter.aux_loss).
        """
        mask = self._mod.token_mask(x)
        out = x.clone()
        for b in range(x.shape[0]):
            idx = mask[b].nonzero(as_tuple=False).squeeze(-1)
            x_k = x[b][idx].unsqueeze(0)  # (1, T_k, D)
            attn_out, _ = self.attn(
                self.ln1(x_k), past_key_value=None, use_cache=False,
                preallocated_cache=None, layer_idx=layer_idx,
                attention_bias=attention_bias, position_ids=position_ids)
            h = x_k + attn_out
            ffn_out = self.ffn(self.ln2(h))
            self._last_aux_loss = None
            if isinstance(ffn_out, tuple):
                self._last_aux_loss = ffn_out[1]
                ffn_out = ffn_out[0]
            h = h + ffn_out
            if self._memory is not None:
                h = h + self._memory(x_k)
            out[b][idx] = h[0]
        if self._last_aux_loss is None:
            self._last_aux_loss = self._mod.aux_loss(x, mask)
        return out


def build_attention(config: ModelConfig) -> nn.Module:
    kwargs = dict(d_model=config.d_model, n_heads=config.n_heads, max_seq_len=config.max_seq_len, base=config.rope_base, rope_scaling=config.rope_scaling,
                  use_qk_norm=getattr(config, 'use_qk_norm', False), attn_scale=getattr(config, 'attn_scale', None))
    # LeRoPE/AdaRoPE: learnable RoPE frequencies (identity init = lossless).
    # Applied post-construction by replacing the RotaryEmbedding module.
    rope_variant = getattr(config, 'rope_variant', 'standard')
    if config.attn_type == "gqa":
        attn = GroupedQueryAttention(**kwargs, n_kv_heads=getattr(config, 'n_kv_heads', None),
                                     attn_bias=getattr(config, 'attn_bias', False))
        attn = _maybe_bitnet_attention(config, attn)
        attn = _maybe_apply_lerope(config, attn)
        attn = _maybe_apply_learned_sink(config, attn)
        attn = _maybe_apply_csa(config, attn)
        return _maybe_fuse_qkv(config, attn)
    if config.attn_type == "diff":
        # Differential Attention (Diff-Transformer): dual-softmax subtraction.
        from research.keys.attention.differential_attn_key import DifferentialAttention
        attn = DifferentialAttention(
            d_model=config.d_model, n_heads=config.n_heads,
            n_kv_heads=getattr(config, 'n_kv_heads', None),
            max_seq_len=config.max_seq_len, base=config.rope_base,
            rope_scaling=config.rope_scaling,
            use_qk_norm=getattr(config, 'use_qk_norm', False),
            attn_bias=getattr(config, 'attn_bias', False),
            n_layers=config.n_layers, layer_idx=0,
            lambda_init=getattr(config, 'diff_attn_lambda_init', None))
        return _maybe_bitnet_attention(config, attn)
    if config.attn_type == "gta":
        # Grouped-Tied Attention (arXiv 2505.21487): V=K at init (lossless),
        # halves KV cache bandwidth. Training unties V from K.
        from research.keys.attention.gta_key import GroupedTiedAttention
        attn = GroupedTiedAttention(
            d_model=config.d_model, n_heads=config.n_heads,
            n_kv_heads=getattr(config, 'n_kv_heads', None),
            max_seq_len=config.max_seq_len, base=config.rope_base,
            rope_scaling=config.rope_scaling,
            use_qk_norm=getattr(config, 'use_qk_norm', False),
            attn_bias=getattr(config, 'attn_bias', False),
            n_layers=config.n_layers, layer_idx=0)
        attn = _maybe_bitnet_attention(config, attn)
        attn = _maybe_apply_lerope(config, attn)
        attn = _maybe_apply_learned_sink(config, attn)
        attn = _maybe_apply_csa(config, attn)
        return _maybe_fuse_qkv(config, attn)
    if config.attn_type == "gla":
        # Grouped Latent Attention (arXiv 2505.21487): latent-compressed KV,
        # identity warm start (lossless). Shifts decode to compute-bound.
        from research.keys.attention.gla_key import GroupedLatentAttention
        latent = getattr(config, 'gla_latent_dim', 0)
        attn = GroupedLatentAttention(
            d_model=config.d_model, n_heads=config.n_heads,
            n_kv_heads=getattr(config, 'n_kv_heads', None),
            max_seq_len=config.max_seq_len, base=config.rope_base,
            rope_scaling=config.rope_scaling,
            use_qk_norm=getattr(config, 'use_qk_norm', False),
            attn_bias=getattr(config, 'attn_bias', False),
            latent_dim=latent if latent > 0 else None,
            n_layers=config.n_layers, layer_idx=0)
        return _maybe_bitnet_attention(config, attn)
    raise ValueError(
        f"Unknown attention type: '{config.attn_type}'. "
        f"Valid options: 'gqa', 'diff', 'gta', 'gla'")


def _maybe_apply_lerope(config: ModelConfig, attn: nn.Module) -> nn.Module:
    """Replace RotaryEmbedding with LeRoPE/AdaRoPE (identity init = lossless).

    LeRoPE: learnable per-frequency-band scaling (dim//2 params, init=1.0).
    AdaRoPE: per-head learnable frequencies + attention scaling.
    Both init to standard RoPE → byte-identical at start.
    """
    rope_variant = getattr(config, 'rope_variant', 'standard')
    if rope_variant in ("lerope", "adarope"):
        from research.keys.position.lerope_key import LeRoPEEmbedding, AdaRoPEEmbedding
        if hasattr(attn, 'rope') and isinstance(attn.rope, RotaryEmbedding):
            head_dim = attn.rope.dim
            max_seq = attn.rope.max_seq_len if hasattr(attn.rope, 'max_seq_len') else config.max_seq_len
            if rope_variant == "lerope":
                attn.rope = LeRoPEEmbedding(
                    dim=head_dim, max_seq_len=max_seq, base=config.rope_base,
                    rope_scaling=config.rope_scaling)
            else:  # adarope
                attn.rope = AdaRoPEEmbedding(
                    dim=head_dim, n_heads=config.n_heads,
                    max_seq_len=max_seq, base=config.rope_base,
                    rope_scaling=config.rope_scaling)
    return attn


def _maybe_apply_learned_sink(config: ModelConfig, attn: nn.Module) -> nn.Module:
    """Add learned attention sink bias to attention layers (GPT-OSS style).

    init=0 → lossless (no bias added). Training learns sink values.
    """
    if not getattr(config, 'use_learned_sink', False):
        return attn
    if hasattr(attn, 'n_heads'):
        n_heads = attn.n_heads
        init_val = getattr(config, 'learned_sink_init', 0.0)
        init_method = getattr(config, 'learned_sink_init_method', 'zero')
        if init_method == "zero":
            sinks = torch.zeros(n_heads)
        elif init_method == "constant":
            sinks = torch.full((n_heads,), init_val, dtype=torch.float32)
        elif init_method == "random":
            sinks = torch.randn(n_heads) * 0.1 + init_val
        else:
            sinks = torch.zeros(n_heads)
        attn.sinks = nn.Parameter(sinks)
    return attn


def _maybe_apply_csa(config: ModelConfig, attn: nn.Module) -> nn.Module:
    """Enable Compressed Sparse Attention (CSA) for long-context efficiency.

    CSA selects the top-k most relevant key positions per query, reducing
    attention complexity from O(S^2) to O(S*k). When the sequence is shorter
    than top_k, full attention is used (lossless for short sequences).

    Applied as a flag on the attention module — the forward checks the flag
    and uses CSA's top-k selection for long sequences.
    """
    pattern = getattr(config, 'attention_pattern', 'standard')
    if pattern not in ('csa', 'csa_hca_hybrid'):
        return attn
    top_k = getattr(config, 'csa_top_k', 256)
    attn._csa_top_k = top_k
    attn._csa_enabled = True
    return attn


def _maybe_bitnet_attention(config: ModelConfig, attn: nn.Module) -> nn.Module:
    """Swap attention projections for BitNet b1.58 QAT linears (when enabled).

    Eval stays full-precision (ternary only in training) so the lossless
    checkpoint load is preserved; QAT then quantizes q/k/v/o matmuls too.
    """
    if not getattr(config, 'use_bitnet', False):
        return attn
    from research.keys.quantization.bitnet_b158_key import build_bitnet_linear
    for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
        lin = getattr(attn, name, None)
        if lin is None:
            continue
        setattr(attn, name, build_bitnet_linear(
            config, config.d_model, lin.out_features,
            bias=lin.bias is not None))
    return attn


def _maybe_fuse_qkv(config: ModelConfig, attn: nn.Module) -> nn.Module:
    """Fuse separate Q/K/V projections into a single GEMM (when enabled).

    Replaces q_proj, k_proj, v_proj with a single FusedQKVLinear that does
    all three projections in one matmul. Halves kernel launches per attention
    layer. The fused weights are initialized from the separate projections
    (lossless — same math, just one GEMM instead of three).

    Skipped for GLA (uses kv_down_proj instead of separate k/v) and
    DifferentialAttention (doubled q/k dims make fusion less beneficial).
    """
    if not getattr(config, 'use_fused_gemm', False):
        return attn
    # Only fuse for GQA and GTA (standard q/k/v projections)
    if type(attn).__name__ not in ("GroupedQueryAttention", "GroupedTiedAttention"):
        return attn
    from research.keys.quantization.fused_gemm_key import FusedQKVLinear, fuse_qkv_weights
    q_proj = getattr(attn, 'q_proj', None)
    k_proj = getattr(attn, 'k_proj', None)
    v_proj = getattr(attn, 'v_proj', None)
    if q_proj is None or k_proj is None or v_proj is None:
        return attn
    # Don't fuse if already BitNet (BitNet handles its own kernel)
    if type(q_proj).__name__ == "BitNetLinear":
        return attn
    fused_w, fused_b = fuse_qkv_weights(
        q_proj.weight, k_proj.weight, v_proj.weight,
        q_proj.bias, k_proj.bias, v_proj.bias)
    fused = FusedQKVLinear(
        config.d_model, q_proj.out_features, k_proj.out_features, v_proj.out_features,
        bias=fused_b is not None)
    with torch.no_grad():
        fused.weight.copy_(fused_w)
        if fused_b is not None:
            fused.bias.copy_(fused_b)
    attn.qkv_proj = fused
    # Keep original projections for weight loading compat (set to identity/no-op)
    # They won't be called in forward — the fused path is used instead.
    attn._fused_qkv = True
    return attn


def _build_moe_ffn(config: ModelConfig) -> nn.Module:
    """Build a MoE FFN layer with BitNet experts and shared expert.

    Uses the existing MoELayer from research.moe, then swaps expert linears
    for BitNetLinear when use_bitnet=True. The shared expert is also BitNet.

    With dense_bypass=True (default for V5), the router is skipped at init
    and all experts run with equal weight → exact dense FFN output.
    Training gradually enables routing (disable dense_bypass after warmup).
    """
    from research.moe.moe import MoELayer

    d_model = config.d_model
    d_ff = getattr(config, 'moe_d_ff', None) or getattr(config, 'intermediate_size', None) or d_model * 2
    n_experts = getattr(config, 'moe_n_experts', 8)
    top_k = getattr(config, 'moe_top_k', 2)
    shared = getattr(config, 'moe_shared_expert', True)
    dense_bypass = getattr(config, 'moe_dense_bypass', True)
    noisy = getattr(config, 'moe_noisy_gating', True)
    lb_weight = getattr(config, 'moe_load_balance_weight', 0.01)
    router_mode = getattr(config, 'moe_router_mode', 'switch')

    moe = MoELayer(
        d_model, n_experts=n_experts, top_k=top_k, d_ff=d_ff,
        shared_expert=shared, capacity_factor=None,
        noisy_gating=noisy, dense_bypass=dense_bypass,
        use_clamp=getattr(config, 'use_swiglu_clamp', False),
        clamp_alpha=getattr(config, 'swiglu_clamp_alpha', 1.702),
        clamp_limit=getattr(config, 'swiglu_clamp_limit', 7.0),
        router_mode=router_mode)

    # Override load balance weight
    moe.router.load_balance_loss_weight = lb_weight

    # Apply BitNet to all expert linears + shared expert
    if getattr(config, 'use_bitnet', False):
        from research.keys.quantization.bitnet_b158_key import build_bitnet_linear
        for expert in moe.experts:
            expert.w1 = build_bitnet_linear(config, d_model, d_ff)
            expert.w3 = build_bitnet_linear(config, d_model, d_ff)
            expert.w2 = build_bitnet_linear(config, d_ff, d_model)
        if hasattr(moe, 'shared'):
            moe.shared.w1 = build_bitnet_linear(config, d_model, d_ff)
            moe.shared.w3 = build_bitnet_linear(config, d_model, d_ff)
            moe.shared.w2 = build_bitnet_linear(config, d_ff, d_model)

    return moe


def build_ffn(config: ModelConfig) -> nn.Module:
    # V5: MoE FFN — replace dense FFN with shared expert + routed experts.
    if getattr(config, 'use_moe', False):
        return _build_moe_ffn(config)
    if config.ffn_type == "swiglu":
        # V5.2: FFN compression (Monarch/Kronecker/TT) — replaces dense linear
        # layers with factored versions. Conversion from dense checkpoint
        # happens in build_model_fast() via _convert_ffn_compression().
        ffn_compression = getattr(config, 'ffn_compression', 'none')
        if ffn_compression == 'monarch':
            from research.keys.compression.monarch_ffn_key import MonarchSwiGLUFFN
            ffn = MonarchSwiGLUFFN(
                config.d_model,
                hidden_dim=getattr(config, 'intermediate_size', None),
                block_size=getattr(config, 'monarch_block_size', 32),
                use_clamp=getattr(config, 'use_swiglu_clamp', False),
                clamp_alpha=getattr(config, 'swiglu_clamp_alpha', 1.702),
                clamp_limit=getattr(config, 'swiglu_clamp_limit', 7.0))
            return ffn
        elif ffn_compression == 'kron':
            from research.keys.compression.kron_ffn_key import KroneckerSwiGLUFFN
            # kron_a*kron_b should = intermediate, kron_c*kron_d should = d_model
            # Use config values as the (a, b) split for gate/up output factorization
            gate_kron = (getattr(config, 'kron_a', 64),
                         getattr(config, 'kron_b', 32))
            down_kron = (getattr(config, 'kron_c', 32),
                         getattr(config, 'kron_d', 256))
            ffn = KroneckerSwiGLUFFN(
                config.d_model,
                hidden_dim=getattr(config, 'intermediate_size', None),
                gate_kron=gate_kron,
                down_kron=down_kron,
                use_clamp=getattr(config, 'use_swiglu_clamp', False),
                clamp_alpha=getattr(config, 'swiglu_clamp_alpha', 1.702),
                clamp_limit=getattr(config, 'swiglu_clamp_limit', 7.0))
            return ffn
        elif ffn_compression == 'tt':
            from research.keys.compression.tt_ffn_key import TTSwiGLUFFN
            ffn = TTSwiGLUFFN(
                config.d_model,
                hidden_dim=getattr(config, 'intermediate_size', None),
                tt_rank=getattr(config, 'tt_rank', 4))
            return ffn
        elif ffn_compression == 'nlrq':
            from research.keys.compression.nlrq_ffn_key import NLRQSwiGLUFFN
            ffn = NLRQSwiGLUFFN(
                config.d_model,
                hidden_dim=getattr(config, 'intermediate_size', None),
                rank=getattr(config, 'nlrq_rank', 256),
                factor_bits=getattr(config, 'nlrq_factor_bits', 8),
                use_residual=getattr(config, 'nlrq_use_residual', False),
                residual_group_size=getattr(config, 'nlrq_residual_group_size', 128),
                use_clamp=getattr(config, 'use_swiglu_clamp', False),
                clamp_alpha=getattr(config, 'swiglu_clamp_alpha', 1.702),
                clamp_limit=getattr(config, 'swiglu_clamp_limit', 7.0))
            return ffn
        # Smooth-SwiGLU: per-channel RMSNorm on gate output for FP8 stability.
        # When use_smooth_swiglu=True, uses SmoothSwiGLUFFN (bounds SiLU outliers).
        # Otherwise standard SwiGLUFFN.
        if getattr(config, 'use_smooth_swiglu', False):
            from research.training.optim.fp8_training import SmoothSwiGLUFFN
            ffn = SmoothSwiGLUFFN(
                config.d_model,
                hidden_dim=getattr(config, 'intermediate_size', None))
        else:
            ffn = SwiGLUFFN(
                config.d_model,
                hidden_dim=getattr(config, 'intermediate_size', None),
                use_clamp=getattr(config, 'use_swiglu_clamp', False),
                clamp_alpha=getattr(config, 'swiglu_clamp_alpha', 1.702),
                clamp_limit=getattr(config, 'swiglu_clamp_limit', 7.0))
        if getattr(config, 'use_bitnet', False):
            # BitNet b1.58 QAT: swap linear layers for ternary-STE versions
            # (learned per-layer scales; ternary only in training by default).
            from research.keys.quantization.bitnet_b158_key import build_bitnet_linear
            hidden = ffn.w_gate.out_features
            ffn.w_gate = build_bitnet_linear(config, config.d_model, hidden)
            ffn.w_up = build_bitnet_linear(config, config.d_model, hidden)
            ffn.w_down = build_bitnet_linear(config, hidden, config.d_model)
        elif getattr(config, 'use_fused_gemm', False):
            # Fused Gate-Up GEMM: single matmul for w_gate + w_up.
            from research.keys.quantization.fused_gemm_key import (
                FusedGateUpLinear, fuse_gateup_weights)
            hidden = ffn.w_gate.out_features
            fused_w = fuse_gateup_weights(ffn.w_gate.weight, ffn.w_up.weight)
            fused = FusedGateUpLinear(config.d_model, hidden, bias=False)
            with torch.no_grad():
                fused.weight.copy_(fused_w)
            ffn.gate_up_proj = fused
            ffn._fused_gate_up = True
        return ffn
    raise ValueError(
        f"Unknown FFN type: '{config.ffn_type}'. "
        f"Valid options: 'swiglu'")


class ConfigurableResearchLLM(nn.Module):
    """Full config-driven research language model."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        # V5: Factorized embedding (ALBERT pattern, 7.8x param reduction).
        # Init via SVD of original embedding when loading from checkpoint.
        # Combined with BitNet embedding for ~60x total reduction.
        if getattr(config, 'use_factorized_embeddings', False):
            from research.keys.architecture.factorized_embed_key import (
                FactorizedEmbedding, FactorizedLMHead)
            rank = getattr(config, 'embed_factorized_rank', 256)
            self.embed = FactorizedEmbedding(config.vocab_size, config.d_model, rank=rank)
            self.head = FactorizedLMHead(self.embed)
        # PIT: Pseudo-Inverse Tying (L=I → standard weight tying, lossless).
        elif getattr(config, 'use_pit', False):
            from research.keys.misc.pit_key import PITEmbedding, PITLMHead
            self.embed = PITEmbedding(config.vocab_size, config.d_model)
            self.head = PITLMHead.from_embedding(self.embed)
        # V5: BitNet embedding (ternary QAT on embedding weight).
        elif getattr(config, 'use_bitnet_embedding', False):
            from research.keys.quantization.bitnet_b158_key import build_bitnet_embedding
            self.embed = build_bitnet_embedding(config, config.vocab_size, config.d_model)
            self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        else:
            self.embed = nn.Embedding(config.vocab_size, config.d_model)
            self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.blocks = nn.ModuleList([ModularBlock(config, layer_idx=i) for i in range(config.n_layers)])
        norm = RMSNorm if getattr(config, 'norm_type', 'layernorm') == 'rmsnorm' else nn.LayerNorm
        self.ln_f = norm(config.d_model)
        # Weight tying: skip if PIT or factorized is enabled (they handle tying),
        # or if config explicitly disables it (e.g., Qwen2.5).
        if (not getattr(config, 'use_pit', False)
                and not getattr(config, 'use_factorized_embeddings', False)
                and getattr(config, 'tie_word_embeddings', True)):
            self.embed.weight = self.head.weight  # Weight tying

        # V5: Expert tying — share expert weights across consecutive layer groups.
        # Applied after blocks are built but before weight loading.
        # Pointer aliasing: odd layers in each group point to even layers.
        if getattr(config, 'use_moe', False) and getattr(config, 'moe_expert_tying', False):
            from research.keys.moe.expert_tying_key import ExpertTyingKey
            tie_key = ExpertTyingKey()
            tie_key.apply(self)
            self._expert_tying_applied = True

        # AttnRes: cross-layer retrieval (shared module, gates=0 → lossless).
        # Applied after each block; maintains a buffer of past layer outputs.
        self._attn_res: nn.Module | None = None
        if getattr(config, 'use_attn_residual', False):
            from research.keys.architecture.attn_residual_key import AttnResModule
            self._attn_res = AttnResModule(
                config.d_model, config.n_layers,
                k=getattr(config, 'attn_res_k', 4),
                n_heads=min(4, config.n_heads))
            self._attn_res_gate_zero: bool | None = None

        # ValueResidual (ResFormer): add V_0 residual to all layers' V.
        # gate=0 at init → lossless. Training opens the gate.
        # V_0 is captured from layer 0's first attention forward and stored
        # as a buffer (detached, no grad) for use by subsequent layers.
        self._use_value_residual = getattr(config, 'use_value_residual', False)
        self._v0_mode = getattr(config, 'value_residual_mode', 'resformer')
        self._v0: torch.Tensor | None = None  # captured during first forward
        self._v0_gates = None
        if self._use_value_residual:
            # Per-layer gate (scalar), init=0 → lossless at start.
            import torch.nn.init as init
            self._v0_gates = nn.ParameterList([
                nn.Parameter(torch.zeros(1)) for _ in range(config.n_layers)
            ])

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
            self.enable_gradient_checkpointing(
                strategy=getattr(config, 'selective_gradient_checkpointing', 'all'))

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

        # V5.2: LiSA — cross-layer Q/K sharing with alignment FFN.
        # Lossless at init: shared Q/K + alignment are zero-init (gate=0).
        # Per-layer Q/K loaded from checkpoint unchanged. Training opens gate.
        self.lisa: nn.Module | None = None
        if getattr(config, 'use_lisa', False):
            from research.keys.attention.lisa_key import LisaKey
            LisaKey.apply(self, config)

        # V5.2: Hyperloop — looped middle blocks for layer param reduction.
        # Lossless at init: all layers unique, loop gate=0. Training opens gate.
        # We only register the loop block + gates as submodules (not the wrapper,
        # which would create a circular ref: model → wrapper → model).
        if getattr(config, 'use_hyperloop', False):
            from research.keys.architecture.hyperloop_key import HyperloopKey
            _hl = HyperloopKey.apply(
                self,
                begin=getattr(config, 'hyperloop_begin', 2),
                end=getattr(config, 'hyperloop_end', 2),
                loop_iters=getattr(config, 'hyperloop_loop_iters', 3))
            # Register loop block and gates as direct submodules (no wrapper)
            self.loop_block = _hl.loop_block
            self.loop_gates = _hl.loop_gates
            self.middle_gates = _hl.middle_gates
            # Store config for forward-time hyperloop logic
            self._hyperloop_begin = _hl.begin
            self._hyperloop_end = _hl.end
            self._hyperloop_loop_iters = _hl.loop_iters
            self._hyperloop_n_middle = _hl.n_middle

        # Cache config flags to avoid getattr/hasattr per forward.
        self._use_liger_ce = getattr(config, 'use_liger_ce', False)
        self._liger_fce = None  # lazy-init on first use

        # Device placement cache — scanning next(param).device per layer per
        # forward (17+ Python calls) is pure CPU overhead. Placement is fixed
        # after load/hybrid-offload, so cache it once on first forward.
        self._embed_device: torch.device | None = None
        self._ln_f_device: torch.device | None = None
        self._block_devices: list[torch.device] | None = None

    def cache_devices(self):
        """Scan and cache module device placement (lazy, on first forward)."""
        self._embed_device = next(self.embed.parameters()).device
        self._ln_f_device = next(self.ln_f.parameters()).device
        self._block_devices = [next(b.parameters()).device for b in self.blocks]

    def invalidate_device_cache(self):
        """Clear cached device placement (call after moving modules)."""
        self._embed_device = None
        self._ln_f_device = None
        self._block_devices = None

    def enable_gradient_checkpointing(self, strategy: str = "all"):
        """Enable activation checkpointing on all transformer blocks.

        Args:
            strategy: "all" (full block recompute), "ffn" (recompute only the
                FFN — largest activation consumer), "attn" (recompute only
                attention), "none" (no recomputation).
        """
        valid = {"all", "ffn", "attn", "none"}
        if strategy not in valid:
            import warnings
            warnings.warn(
                f"Unknown gradient checkpointing strategy '{strategy}' — "
                f"falling back to 'none' (no checkpointing). "
                f"Valid strategies: {sorted(valid)}",
                stacklevel=2,
            )
        for block in self.blocks:
            block._gradient_checkpointing = True
            block._gradient_checkpointing_strategy = strategy

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
        # DiffusionBlocks support: run only specific layers, with noise conditioning
        layer_indices: list[int] | None = None,
        noisy_embeds: torch.Tensor | None = None,
        modulation: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | tuple[torch.Tensor, torch.Tensor | None, list[KVCache | None]]:
        # FP8 training autocast: wrap forward in FP8 for 2x throughput on
        # Hopper/Blackwell. Falls back to BF16 on older GPUs.
        if getattr(self.config, 'use_fp8_training', False):
            from research.training.optim.fp8_training import enable_fp8_training
            with enable_fp8_training():
                return self._forward_impl(
                    idx, targets, past_key_values, use_cache, return_hidden,
                    preallocated_cache, attention_mask, layer_indices,
                    noisy_embeds, modulation)
        return self._forward_impl(
            idx, targets, past_key_values, use_cache, return_hidden,
            preallocated_cache, attention_mask, layer_indices,
            noisy_embeds, modulation)

    def _forward_impl(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        past_key_values: list[KVCache | None] | None = None,
        use_cache: bool = False,
        return_hidden: bool = False,
        preallocated_cache: Optional["PreAllocatedKVCache"] = None,
        attention_mask: torch.Tensor | None = None,
        layer_indices: list[int] | None = None,
        noisy_embeds: torch.Tensor | None = None,
        modulation: torch.Tensor | None = None,
    ):
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

        # Move input to embed's device (handles hybrid offload).
        # Device placement is cached after the first forward (it only changes
        # via explicit .to() / hybrid_offload, which call invalidate_device_cache).
        if self._embed_device is None:
            self.cache_devices()
        embed_device = self._embed_device
        if idx.device != embed_device:
            idx = idx.to(embed_device)
        x = self.embed(idx)
        # DiffusionBlocks: add noisy target embeddings to the input
        if noisy_embeds is not None:
            # noisy_embeds: (B, T, d_model) — added to input embeddings
            if noisy_embeds.shape[:2] == x.shape[:2]:
                x = x + noisy_embeds.to(x.dtype).to(x.device)
            else:
                # Broadcast (B, d_model) → (B, T, d_model)
                x = x + noisy_embeds.unsqueeze(1).expand(-1, x.shape[1], -1).to(x.dtype).to(x.device)
        # DiffusionBlocks: convert layer_indices to a set for O(1) lookup
        active_layers = set(layer_indices) if layer_indices is not None else None
        presents: list[KVCache | None] = []
        # Track device for hybrid offload: move x to each block's device
        cur_device = x.device
        block_devices = self._block_devices
        # AttnRes: only build past_outputs buffer when gates are non-zero.
        # At init (gates=0), skip entirely — zero overhead, bit-exact.
        use_attn_res = self._attn_res is not None
        if use_attn_res and self._attn_res_gate_zero is None:
            self._attn_res_gate_zero = (
                self._attn_res.gates.abs().max().item() == 0.0)
        attn_res_active = use_attn_res and not self._attn_res_gate_zero
        past_outputs: list[torch.Tensor] = [] if attn_res_active else None
        for i, block in enumerate(self.blocks):
            # DiffusionBlocks: skip layers not in the active set
            if active_layers is not None and i not in active_layers:
                if use_cache:
                    presents.append(past_key_values[i] if past_key_values is not None else None)
                continue
            block_device = block_devices[i]
            if block_device != cur_device:
                x = x.to(block_device)
                cur_device = block_device
            # ValueResidual: inject V_0 into layers 1+ (gate=0 → lossless).
            # V_0 is captured from layer 0's V projection on the first forward.
            if self._use_value_residual and i > 0 and self._v0 is not None:
                attn = block.attn
                if hasattr(attn, '_v0_residual'):
                    attn._v0_residual = self._v0
                    attn._v0_gate = self._v0_gates[i] if self._v0_gates is not None else None
            # Enable V_0 capture on layer 0 (first forward only).
            if self._use_value_residual and i == 0 and self._v0 is None and not use_cache:
                attn = block.attn
                if hasattr(attn, '_v0_capture'):
                    attn._v0_capture = True  # signal to capture
            past = past_key_values[i] if past_key_values is not None else None
            x, present = block(x, past_key_value=past, use_cache=use_cache,
                               preallocated_cache=preallocated_cache, layer_idx=i,
                               attention_bias=attention_bias, position_ids=position_ids,
                               modulation=modulation)
            # Capture V_0 from layer 0 after its first forward (for training).
            if self._use_value_residual and i == 0 and self._v0 is None and not use_cache:
                attn = block.attn
                if hasattr(attn, '_v0_capture') and attn._v0_capture is not None:
                    self._v0 = attn._v0_capture.detach()
            # AttnRes: cross-layer retrieval (gates=0 → lossless at start).
            if attn_res_active:
                x = x + self._attn_res(x, i, past_outputs)
                past_outputs.append(x)
            if use_cache:
                presents.append(present)
        # Advance the pre-allocated cache position after all layers processed.
        if preallocated_cache is not None:
            preallocated_cache.advance(idx.shape[1])
        # Move x to ln_f's device (GPU for hybrid offload)
        ln_f_device = self._ln_f_device
        if x.device != ln_f_device:
            x = x.to(ln_f_device)
        hidden = self.ln_f(x)

        # Collect MoE aux_loss from all blocks that have it.
        # Stored as self._last_moe_aux_loss so callers that don't pass targets
        # (e.g. GRPO/RLVR policy-gradient forward) can still add it to their
        # loss — otherwise the MoE router gets no load-balancing signal during
        # RL and can collapse to a single expert.
        moe_aux_loss = torch.tensor(0.0, device=idx.device, dtype=hidden.dtype)
        for block in self.blocks:
            if hasattr(block, '_last_aux_loss') and block._last_aux_loss is not None:
                moe_aux_loss = moe_aux_loss + block._last_aux_loss
        self._last_moe_aux_loss = moe_aux_loss

        # Chunked CE path: skip materializing full [B*T, V] logits for loss.
        # The head Linear + CE are fused into chunked passes over the token dim,
        # saving ~2.8 GB at batch 2 / seq 1024 / vocab 151665.
        if self.config.use_chunked_ce and targets is not None and not use_cache:
            ent_alpha = getattr(self.config, 'entropy_alpha', 0.0)
            if ent_alpha > 0.0:
                from research.training.losses.chunked_ce import chunked_entropy_weighted_ce
                loss = chunked_entropy_weighted_ce(
                    hidden.view(-1, hidden.size(-1)),
                    self.head.weight,
                    targets.view(-1),
                    chunk_size=self.config.ce_chunk_size,
                    entropy_alpha=ent_alpha,
                )
            else:
                from research.training.losses.chunked_ce import chunked_linear_cross_entropy
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
        """A stable hashable signature for caching blank models.

        Must capture ALL architecture-affecting fields so that configs
        differing in any key produce different signatures (no cache collisions).
        """
        layer_types_sig = tuple(getattr(config, 'layer_types', None) or [])
        mtp_sig = f"mtp{getattr(config, 'use_mtp', False)}_{getattr(config, 'mtp_n_heads', 0)}"
        arch_sig = (f"{getattr(config, 'use_bitnet', False)}_"
                    f"{getattr(config, 'use_titan_memory', False)}_"
                    f"{getattr(config, 'titan_memory_rank', 0)}_"
                    f"{getattr(config, 'use_mod', False)}_"
                    f"{getattr(config, 'mod_keep_fraction', 1.0)}_"
                    f"{getattr(config, 'use_qk_norm', False)}_"
                    f"{getattr(config, 'use_mhc', False)}_"
                    f"{getattr(config, 'mhc_rank', 0)}_"
                    f"{getattr(config, 'use_attn_residual', False)}_"
                    f"{getattr(config, 'attn_res_k', 4)}")
        # V5.1 keys — all affect architecture structure (new params/modules)
        v51_sig = (f"vr{getattr(config, 'use_value_residual', False)}_"
                   f"sn{getattr(config, 'use_sandwich_norm', False)}_"
                   f"ls{getattr(config, 'use_learned_sink', False)}_"
                   f"sc{getattr(config, 'use_swiglu_clamp', False)}_"
                   f"rv{getattr(config, 'rope_variant', 'standard')}_"
                   f"ap{getattr(config, 'attention_pattern', 'standard')}_"
                   f"ck{getattr(config, 'csa_top_k', 0)}_"
                   f"be{getattr(config, 'use_bitnet_embedding', False)}_"
                   f"fg{getattr(config, 'use_fused_gemm', False)}_"
                   f"moe{getattr(config, 'use_moe', False)}_"
                   f"ne{getattr(config, 'moe_n_experts', 0)}_"
                   f"tk{getattr(config, 'moe_top_k', 0)}_"
                   f"se{getattr(config, 'moe_shared_expert', False)}_"
                   f"et{getattr(config, 'moe_expert_tying', 0)}_"
                   f"rm{getattr(config, 'moe_router_mode', 'switch')}_"
                   f"fe{getattr(config, 'use_factorized_embedding', False)}_"
                   f"fc{getattr(config, 'ffn_compression', 'none')}_"
                   f"mb{getattr(config, 'monarch_block_size', 32)}_"
                   f"hl{getattr(config, 'use_hyperloop', False)}_"
                   f"hb{getattr(config, 'hyperloop_begin', 2)}_"
                   f"he{getattr(config, 'hyperloop_end', 2)}_"
                   f"hi{getattr(config, 'hyperloop_loop_iters', 3)}_"
                   f"li{getattr(config, 'use_lisa', False)}_"
                   f"lc{getattr(config, 'lisa_compress', 6)}")
        return f"{config.d_model}_{config.n_layers}_{config.attn_type}_{config.ffn_type}_{config.norm_type}_{getattr(config, 'kv_compression_dim', 0)}_{getattr(config, 'n_kv_heads', 0)}_{getattr(config, 'attn_bias', False)}_{layer_types_sig}_{mtp_sig}_{arch_sig}_{v51_sig}"

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

        For CUDA targets, uses fastsafetensors (pinned-memory + async DMA) when
        available, falling back to safetensors direct device loading. This loads
        weights directly to GPU, skipping the CPU→GPU copy entirely.

        For CPU-only models, tensors stay mmap'd (zero-copy) until accessed.
        """
        if device is None:
            device = next(model.parameters()).device

        # Fast path: fastsafetensors async GPU loading (pinned mem + async DMA)
        if device.type == "cuda":
            try:
                from fastsafetensors import fastsafe_open
                state = {}
                with fastsafe_open(path, framework="pt", device=str(device),
                                   nogds=True) as f:
                    for key in f.keys():
                        # Clone — tensors are backed by a device buffer
                        # that is freed when the context exits.
                        state[key] = f.get_tensor(key).clone()
                return state
            except Exception as e:
                # fastsafetensors may fail on Windows (missing DirectStorage
                # DLLs or CUDA runtime version mismatch). The standard
                # safetensors path still loads directly to GPU
                # (SAFETENSORS_FAST_CUDA=1). Suppress expected errors.
                err = str(e).lower()
                if any(kw in err for kw in ("directstorage", "dstorage",
                        "gpu runtime", "cudart", "libcudart")):
                    pass  # expected — fallback handles it
                else:
                    print(f"  [FastBuild] fastsafetensors unavailable ({e}), "
                          f"using safetensors direct device loading")

        # Fallback: safetensors safe_open with direct device loading.
        # SAFETENSORS_FAST_CUDA=1 (set at module import) enables pinned async.
        from safetensors import safe_open
        load_device = str(device) if device.type != "cpu" else "cpu"
        state = {}
        with safe_open(path, framework="pt", device=load_device) as f:
            for key in f.keys():
                state[key] = f.get_tensor(key)
        return state

    @staticmethod
    def _load_sharded_safetensors(ckpt_dir: Path, model: nn.Module,
                                  device: torch.device = None) -> dict:
        """Load weights from sharded safetensors (model-00001-of-00002.safetensors).

        For CUDA targets, uses fastsafetensors async GPU loading when available,
        falling back to safetensors direct device loading.
        """
        if device is None:
            device = next(model.parameters()).device

        sf_paths = sorted(ckpt_dir.glob("model-*.safetensors"))
        if not sf_paths:
            sf_paths = sorted(ckpt_dir.glob("*.safetensors"))

        # Fast path: fastsafetensors async GPU loading (all shards at once)
        if device.type == "cuda":
            try:
                from fastsafetensors import fastsafe_open
                state = {}
                with fastsafe_open([str(p) for p in sf_paths], framework="pt",
                                   device=str(device), nogds=True) as f:
                    for key in f.keys():
                        state[key] = f.get_tensor(key).clone()
                return state
            except Exception as e:
                err = str(e).lower()
                if any(kw in err for kw in ("directstorage", "dstorage",
                        "gpu runtime", "cudart", "libcudart")):
                    pass
                else:
                    print(f"  [FastBuild] fastsafetensors unavailable ({e}), "
                          f"using safetensors direct device loading")

        # Fallback: safetensors safe_open with direct device loading
        from safetensors import safe_open
        load_device = str(device) if device.type != "cpu" else "cpu"
        state = {}
        for sf_path in sf_paths:
            with safe_open(str(sf_path), framework="pt", device=load_device) as f:
                for key in f.keys():
                    state[key] = f.get_tensor(key)
        return state

    @staticmethod
    def _convert_ffn_compression(state: dict, config, compression: str) -> dict:
        """Convert dense FFN weights to factored format (Monarch/Kron/TT).

        Transforms blocks.{i}.ffn.w_{gate,up,down}.weight (dense) into the
        factored parameter names expected by the compressed FFN modules.
        This is a one-time conversion; subsequent saves store factored weights.
        """
        import re
        new_state = {}
        # Pattern: blocks.{i}.ffn.w_{gate,up,down}.weight
        ffn_pattern = re.compile(r'blocks\.(\d+)\.ffn\.w_(gate|up|down)\.weight')

        for key, tensor in state.items():
            m = ffn_pattern.match(key)
            if m:
                layer_idx = int(m.group(1))
                proj = m.group(2)  # gate, up, or down
                weight = tensor.float()  # (out, in) — nn.Linear format

                if compression == 'monarch':
                    from research.keys.compression.monarch_ffn_key import MonarchLinear
                    block_size = getattr(config, 'monarch_block_size', 32)
                    ml = MonarchLinear.from_dense(weight, block_size=block_size)
                    new_state[f'blocks.{layer_idx}.ffn.w_{proj}.L'] = ml.L.data.to(tensor.dtype)
                    new_state[f'blocks.{layer_idx}.ffn.w_{proj}.R'] = ml.R.data.to(tensor.dtype)
                    # perm_idx is a buffer, not a parameter — skip (re-init'd in module)

                elif compression == 'kron':
                    from research.keys.compression.kron_ffn_key import KroneckerLinear
                    a = getattr(config, 'kron_a', 64)
                    b = getattr(config, 'kron_b', 32)
                    c = getattr(config, 'kron_c', 32)
                    d = getattr(config, 'kron_d', 256)
                    out_features, in_features = weight.shape
                    # Adjust factorization to match actual dimensions
                    # For w_gate/w_up: out=intermediate, in=d_model
                    # For w_down: out=d_model, in=intermediate
                    # Find factors that work for the actual dimensions
                    kl = KroneckerLinear.from_dense(weight, a, b, c, d)
                    new_state[f'blocks.{layer_idx}.ffn.w_{proj}.A'] = kl.A.data.to(tensor.dtype)
                    new_state[f'blocks.{layer_idx}.ffn.w_{proj}.B'] = kl.B.data.to(tensor.dtype)

                elif compression == 'tt':
                    from research.keys.compression.tt_ffn_key import TTLinear
                    tt_rank = getattr(config, 'tt_rank', 4)
                    tl = TTLinear.from_dense(weight, tt_rank=tt_rank)
                    for ci, core in enumerate(tl.cores):
                        new_state[f'blocks.{layer_idx}.ffn.w_{proj}.cores.{ci}'] = core.data.to(tensor.dtype)

                elif compression == 'nlrq':
                    from research.keys.compression.nlrq_ffn_key import NLRQLinear
                    rank = getattr(config, 'nlrq_rank', 256)
                    factor_bits = getattr(config, 'nlrq_factor_bits', 8)
                    use_residual = getattr(config, 'nlrq_use_residual', False)
                    residual_gs = getattr(config, 'nlrq_residual_group_size', 128)
                    nl = NLRQLinear.from_dense(weight, rank=rank,
                                               factor_bits=factor_bits,
                                               use_residual=use_residual,
                                               residual_group_size=residual_gs)
                    # INT8 buffers (real quantized storage)
                    new_state[f'blocks.{layer_idx}.ffn.w_{proj}.U_q'] = nl.U_q.to(torch.int8)
                    new_state[f'blocks.{layer_idx}.ffn.w_{proj}.V_q'] = nl.V_q.to(torch.int8)
                    new_state[f'blocks.{layer_idx}.ffn.w_{proj}.S'] = nl.S.data.to(tensor.dtype)
                    new_state[f'blocks.{layer_idx}.ffn.w_{proj}.U_scale'] = nl.U_scale.to(torch.float16)
                    new_state[f'blocks.{layer_idx}.ffn.w_{proj}.V_scale'] = nl.V_scale.to(torch.float16)
                    if use_residual and nl.residual_q is not None:
                        new_state[f'blocks.{layer_idx}.ffn.w_{proj}.residual_q'] = nl.residual_q.to(torch.int8)
                        new_state[f'blocks.{layer_idx}.ffn.w_{proj}.residual_scales'] = nl.residual_scales.to(torch.float16)
            else:
                new_state[key] = tensor
        return new_state

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
    def _reset_non_persistent_buffers(model: nn.Module,
                                      target_device: torch.device):
        """Re-initialize non-persistent buffers left on meta after meta-init.

        After meta device init + load_state_dict(assign=True), buffers
        registered with persistent=False (e.g. RoPE cos/sin tables) remain
        on meta. This re-computes them on the target device.
        """
        for module in model.modules():
            if isinstance(module, RotaryEmbedding):
                base = getattr(module, 'base', 10000.0)
                max_seq_len = getattr(module, 'max_seq_len',
                                      module.cos_cached.shape[0])
                rope_scaling = getattr(module, 'rope_scaling', None)
                inv_freq = 1.0 / (base ** (
                    torch.arange(0, module.dim, 2, device=target_device,
                                 dtype=torch.float32) / module.dim))
                if rope_scaling and rope_scaling.get("type") == "yarn":
                    inv_freq = RotaryEmbedding._yarn_inv_freq(
                        inv_freq, rope_scaling, max_seq_len)
                t = torch.arange(max_seq_len, device=target_device,
                                 dtype=torch.float32)
                freqs = torch.outer(t, inv_freq)
                emb = torch.cat((freqs, freqs), dim=-1)
                module.inv_freq = inv_freq
                module.cos_cached = emb.cos()
                module.sin_cached = emb.sin()
                module.cos_cached_bf16 = emb.cos().to(torch.bfloat16)
                module.sin_cached_bf16 = emb.sin().to(torch.bfloat16)

    @staticmethod
    def _prefetch_file(path: str, block: int = 16 * 1024 * 1024):
        """Background thread: read file in blocks to warm OS page cache.

        Mirrors vLLM PR #36012 prefetch strategy. Overlaps I/O with arch build.
        """
        import threading
        def _read():
            try:
                size = os.path.getsize(path)
                with open(path, "rb") as f:
                    read = 0
                    while read < size:
                        chunk = f.read(min(block, size - read))
                        if not chunk:
                            break
                        read += len(chunk)
            except Exception:
                pass
        t = threading.Thread(target=_read, daemon=True)
        t.start()
        return t

    @staticmethod
    def build_model_fast(config: ModelConfig, checkpoint_path: str | None = None,
                         compile: bool = False, moe_top_k: int | None = None,
                         dtype: torch.dtype | None = None,
                         fast_load: bool = True):
        """Fast model build — caches architecture, only loads weights.

        First call builds the architecture (~3s). Subsequent calls with the
        same config clone the cached model (~0.5s) and just load weights.

        moe_top_k: override MoE top-k routing (default: all experts).
                   Set to 2 for 4-expert model to halve FFN activations/VRAM.
        dtype: convert model to this dtype before loading weights (e.g. torch.bfloat16).
               Prevents upcasting bf16 checkpoint weights to fp32, saving ~50% VRAM.
        fast_load: when True (default), uses meta-device init + assign=True for
               3-6x faster cold boot. Skips parameter init kernels entirely by
               building the model on torch.device("meta"), then directly
               replaces meta params with state_dict tensors via
               load_state_dict(assign=True). Also starts OS page cache
               prefetch in a background thread. Set to False for the
               traditional build path (needed if checkpoint is missing or
               for debugging weight loading issues).
        """
        import time
        t0 = time.time()
        device = torch.device(config.device)
        sig = ModelLoader._config_signature(config)

        # Fast load path: meta init + assign=True + parallel weight load.
        # 5x faster than the traditional path (11.3s → 2.0s on V3).
        # Weight loading runs in a background thread, overlapping with meta init.
        if fast_load and checkpoint_path and os.path.exists(checkpoint_path):
            # Start state_dict load in background thread (overlaps with meta init).
            # The weight load is I/O bound (fastsafetensors reads + H2D copy),
            # so it can run concurrently with the CPU-bound meta init.
            import threading
            state_result = {}

            def _bg_load_state():
                try:
                    ckpt_path = Path(checkpoint_path)
                    if ckpt_path.is_dir():
                        sf_files = list(ckpt_path.glob("*.safetensors"))
                        if len(sf_files) == 1:
                            s = ModelLoader._load_safetensors_mmap(
                                str(sf_files[0]), None, device=device)
                        else:
                            s = ModelLoader._load_sharded_safetensors(
                                ckpt_path, None, device=device)
                    elif str(checkpoint_path).endswith(".safetensors"):
                        s = ModelLoader._load_safetensors_mmap(
                            checkpoint_path, None, device=device)
                    else:
                        from research.checkpoint_io import load_checkpoint
                        s = load_checkpoint(checkpoint_path, map_location="cpu")
                        if isinstance(s, dict) and "model_state" in s \
                                and not any(k.startswith("blocks.") for k in s):
                            s = s["model_state"]
                    # Auto-remap HF keys
                    if s and any(k.startswith("model.") for k in s):
                        s = ModelLoader._remap_hf_keys(s, config)
                    state_result["state"] = s
                except Exception as e:
                    state_result["error"] = e

            state_thread = threading.Thread(target=_bg_load_state, daemon=True)
            state_thread.start()

            # Meta init in main thread (overlaps with background weight load).
            # Caches the meta-init architecture so subsequent boots with the
            # same config skip the ~3-7s build and just deepcopy (~0.1s).
            t_arch = time.time()
            meta_sig = f"meta_{sig}"
            if meta_sig in ModelLoader._model_cache:
                model = copy.deepcopy(ModelLoader._model_cache[meta_sig])
                ModelLoader._model_cache.move_to_end(meta_sig)
                t_arch = time.time() - t_arch
                print(f"  [FastBuild] Meta architecture cloned in {t_arch:.1f}s (cached)")
            else:
                cfg_meta = ModelConfig(**{**config.__dict__, "device": "meta"})
                with torch.device("meta"):
                    model = ConfigurableResearchLLM(cfg_meta)
                # Cache the meta model (cheap — no real tensors, just shapes)
                ModelLoader._model_cache[meta_sig] = copy.deepcopy(model)
                ModelLoader._model_cache.move_to_end(meta_sig)
                while len(ModelLoader._model_cache) > ModelLoader._MODEL_CACHE_MAXSIZE:
                    ModelLoader._model_cache.popitem(last=False)
                t_arch = time.time() - t_arch
                print(f"  [FastBuild] Meta-init architecture in {t_arch:.1f}s")

            # Wait for background weight load to complete
            t_weights = time.time()
            state_thread.join()
            if "error" in state_result:
                raise state_result["error"]
            state = state_result.get("state", {})

            # Pre-quantized BitNet: int8 weights need cast to bf16 for
            # load_state_dict(assign=True). The weights are ternary {-1,0,+1}
            # so this cast is lossless. convert_model_to_int8() re-quantizes
            # post-load for VRAM savings.
            _has_int8 = any(t.dtype == torch.int8 for t in state.values())
            if _has_int8:
                for k in list(state.keys()):
                    if state[k].dtype == torch.int8:
                        state[k] = state[k].to(torch.bfloat16)

            # GQA -> diff warm start (CPU-side transform on loaded state_dict)
            if config.attn_type == "diff":
                qk = next((k for k in state
                           if "attn.q_proj.weight" in k), None)
                if qk is not None:
                    exp_rows = config.n_heads * (config.d_model // config.n_heads)
                    if state[qk].shape[0] == exp_rows:
                        from research.keys.attention.differential_attn_key import (
                            DifferentialAttentionKey)
                        res = DifferentialAttentionKey(
                            n_layers=config.n_layers,
                            n_heads=config.n_heads, identity=True).forward(state)
                        if res.success:
                            state = res.weights
                            print("  [FastBuild] GQA -> diff warm start "
                                  "(lossless, lambda=0)")

            # GQA -> GTA warm start (V=K, v_mix_gate=0, lossless)
            # Also handles V3 (diff) -> V4 (GTA) auto-conversion.
            if config.attn_type == "gta":
                qk = next((k for k in state
                           if "attn.q_proj.weight" in k), None)
                if qk is not None:
                    # Check if checkpoint is V3 (diff) — q_proj rows are doubled
                    exp_gqa_rows = config.n_heads * (config.d_model // config.n_heads)
                    if state[qk].shape[0] == 2 * exp_gqa_rows:
                        # V3 diff checkpoint → reverse diff + forward GTA
                        from research.architecture.v3_to_v4 import convert_v3_to_v4_state
                        state = convert_v3_to_v4_state(
                            state,
                            n_heads=config.n_heads,
                            n_kv_heads=config.n_kv_heads,
                            head_dim=config.d_model // config.n_heads,
                        )
                        print("  [FastBuild] V3 (diff) -> V4 (GTA) auto-convert "
                              "(reverse diff + GTA warm start)")
                    else:
                        # Plain GQA checkpoint → forward GTA only
                        from research.keys.attention.gta_key import GTAKey
                        res = GTAKey(
                            n_layers=config.n_layers,
                            n_heads=config.n_heads).forward(state)
                        if res.success:
                            state = res.weights
                            print("  [FastBuild] GQA -> GTA warm start "
                                  "(lossless, V=K, gate=0)")

            # GQA -> GLA warm start (kv_down_proj=k_proj, identity up-projs, lossless)
            if config.attn_type == "gla":
                qk = next((k for k in state
                           if "attn.q_proj.weight" in k), None)
                if qk is not None:
                    from research.keys.attention.gla_key import GLAKey
                    latent = getattr(config, 'gla_latent_dim', 0)
                    res = GLAKey(
                        n_layers=config.n_layers, n_heads=config.n_heads,
                        n_kv_heads=getattr(config, 'n_kv_heads', 8),
                        latent_dim=latent if latent > 0 else None).forward(state)
                    if res.success:
                        state = res.weights
                        print("  [FastBuild] GQA -> GLA warm start "
                              "(lossless, identity up-projs, gate=0)")

            # V5.2: FFN compression (Monarch/Kronecker/TT) — convert dense
            # FFN weights to factored format on first load. Subsequent saves
            # store the factored weights directly (no re-conversion needed).
            ffn_compression = getattr(config, 'ffn_compression', 'none')
            if ffn_compression != 'none':
                ffn_gate_key = next((k for k in state
                                     if 'ffn.w_gate.weight' in k), None)
                if ffn_gate_key is not None:
                    state = ModelLoader._convert_ffn_compression(
                        state, config, ffn_compression)
                    print(f"  [FastBuild] Dense FFN -> {ffn_compression} "
                          f"compression (one-time conversion)")

            t_weights = time.time() - t_weights

            # assign=True: directly replace meta params with state_dict tensors.
            # Skips the copy-into-existing-storage path of normal load_state_dict.
            t_gpu = time.time()
            missing, unexpected = model.load_state_dict(
                state, strict=False, assign=True)
            # Re-tie weights (assign breaks parameter sharing; head.weight
            # is not in the checkpoint because of weight tying).
            if getattr(config, 'tie_word_embeddings', True) \
                    and not getattr(config, 'use_pit', False):
                model.head.weight = model.embed.weight
            # Re-initialize non-persistent buffers (RoPE cos/sin) left on meta.
            ModelLoader._reset_non_persistent_buffers(model, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_gpu = time.time() - t_gpu

            # Convert dtype (state may already be bf16 from fastsafetensors)
            if dtype is not None:
                model = model.to(dtype)

            if missing:
                # head.weight is expected to be missing (weight tying)
                real_missing = [k for k in missing if k != "head.weight"]
                if real_missing:
                    print("Missing keys:", real_missing[:5],
                          "..." if len(real_missing) > 5 else "")
            if unexpected:
                print("Unexpected keys:", unexpected[:5],
                      "..." if len(unexpected) > 5 else "")

            # Post-load QK-norm + diff-attn identity scan
            for block in model.blocks:
                attn = block.attn
                if hasattr(attn, 'q_norm') and hasattr(attn, '_qk_norm_identity'):
                    q_id = (attn.q_norm.weight == 1.0).all()
                    k_id = (attn.k_norm.weight == 1.0).all()
                    attn._qk_norm_identity = bool(q_id and k_id)
                if hasattr(attn, 'lambda_param') and hasattr(attn, 'set_identity'):
                    attn.set_identity((attn.lambda_param == 0.0).all().item())

            t_total = time.time() - t0
            param_count = sum(p.numel() for p in model.parameters()) / 1e6
            print(f"  [FastBuild] Weights: {t_weights:.1f}s | assign: {t_gpu:.1f}s | "
                  f"Total: {t_total:.1f}s ({param_count:.1f}M params)")
            return model

        # Traditional build path (fast_load=False or no checkpoint)
        # Build or clone architecture
        if sig not in ModelLoader._model_cache:
            t_arch = time.time()
            model = ConfigurableResearchLLM(config).to(device)
            # μScaling: unit-variance init for FP8 training stability.
            # Only applies to fresh models (not checkpoint loading).
            if getattr(config, 'use_mu_scaling', False):
                from research.training.optim.fp8_training import mu_scale_init
                mu_scale_init(model, verbose=True)
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
            t_arch = time.time()
            cached = ModelLoader._model_cache[sig]
            ModelLoader._model_cache.move_to_end(sig)  # mark as recently used
            # Deep copy the cached model (CPU — no VRAM overhead)
            model = copy.deepcopy(cached).to(device)
            t_arch = time.time() - t_arch
            print(f"  [FastBuild] Architecture cloned in {t_arch:.1f}s (from cache)")

        # Convert dtype before loading weights (prevents bf16→fp32 upcast)
        if dtype is not None:
            model = model.to(dtype)

        if compile:
            model = torch.compile(model, mode="reduce-overhead", dynamic=True)

        if checkpoint_path and os.path.exists(checkpoint_path):
            t_weights = time.time()
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

            # Lossless GQA -> DifferentialAttention conversion: when the
            # config asks for diff attention but the checkpoint has plain
            # GQA q/k projections, duplicate rows and set lambda=0 (identity
            # mode) so the loaded model is bit-exact until trained.
            if config.attn_type == "diff":
                qk = next((k for k in state
                           if "attn.q_proj.weight" in k), None)
                if qk is not None:
                    exp_rows = config.n_heads * (config.d_model // config.n_heads)
                    if state[qk].shape[0] == exp_rows:
                        from research.keys.attention.differential_attn_key import (
                            DifferentialAttentionKey)
                        res = DifferentialAttentionKey(
                            n_layers=config.n_layers,
                            n_heads=config.n_heads, identity=True).forward(state)
                        if res.success:
                            state = res.weights
                            print("  [FastBuild] GQA -> diff warm start "
                                  "(lossless, lambda=0)")

            # GQA -> GTA warm start (V=K, v_mix_gate=0, lossless)
            # Also handles V3 (diff) -> V4 (GTA) auto-conversion.
            if config.attn_type == "gta":
                qk = next((k for k in state
                           if "attn.q_proj.weight" in k), None)
                if qk is not None:
                    # Check if checkpoint is V3 (diff) — q_proj rows are doubled
                    exp_gqa_rows = config.n_heads * (config.d_model // config.n_heads)
                    if state[qk].shape[0] == 2 * exp_gqa_rows:
                        # V3 diff checkpoint → reverse diff + forward GTA
                        from research.architecture.v3_to_v4 import convert_v3_to_v4_state
                        state = convert_v3_to_v4_state(
                            state,
                            n_heads=config.n_heads,
                            n_kv_heads=config.n_kv_heads,
                            head_dim=config.d_model // config.n_heads,
                        )
                        print("  [FastBuild] V3 (diff) -> V4 (GTA) auto-convert "
                              "(reverse diff + GTA warm start)")
                    else:
                        # Plain GQA checkpoint → forward GTA only
                        from research.keys.attention.gta_key import GTAKey
                        res = GTAKey(
                            n_layers=config.n_layers,
                            n_heads=config.n_heads).forward(state)
                        if res.success:
                            state = res.weights
                            print("  [FastBuild] GQA -> GTA warm start "
                                  "(lossless, V=K, gate=0)")

            # GQA -> GLA warm start (identity up-projs, lossless)
            if config.attn_type == "gla":
                qk = next((k for k in state
                           if "attn.q_proj.weight" in k), None)
                if qk is not None:
                    from research.keys.attention.gla_key import GLAKey
                    latent = getattr(config, 'gla_latent_dim', 0)
                    res = GLAKey(
                        n_layers=config.n_layers, n_heads=config.n_heads,
                        n_kv_heads=getattr(config, 'n_kv_heads', 8),
                        latent_dim=latent if latent > 0 else None).forward(state)
                    if res.success:
                        state = res.weights
                        print("  [FastBuild] GQA -> GLA warm start "
                              "(lossless, identity up-projs, gate=0)")

            t_weights = time.time() - t_weights

            # load_state_dict copies weights into model parameters (GPU→GPU
            # if weights were loaded directly to device, CPU→GPU otherwise).
            t_gpu = time.time()
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                print("Missing keys:", missing[:5], "..." if len(missing) > 5 else "")
            if unexpected:
                print("Unexpected keys:", unexpected[:5], "..." if len(unexpected) > 5 else "")
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_gpu = time.time() - t_gpu

            # Post-load: detect non-identity QK-Norm weights.
            # If all q_norm/k_norm weights are 1.0, skip normalization (lossless).
            # If any differ, enable normalization (trained QK-Norm).
            for block in model.blocks:
                attn = block.attn
                if hasattr(attn, 'q_norm') and hasattr(attn, '_qk_norm_identity'):
                    q_id = (attn.q_norm.weight == 1.0).all()
                    k_id = (attn.k_norm.weight == 1.0).all()
                    attn._qk_norm_identity = bool(q_id and k_id)
                # Sync diff-attn identity mode with the loaded lambda weights.
                if hasattr(attn, 'lambda_param') and hasattr(attn, 'set_identity'):
                    attn.set_identity((attn.lambda_param == 0.0).all().item())
            # Log
            n_identity = sum(1 for b in model.blocks if getattr(b.attn, '_qk_norm_identity', True))
            if getattr(config, 'use_qk_norm', False):
                print(f"  [FastBuild] QK-Norm: {n_identity}/{len(model.blocks)} layers identity (skipped)")

            print(f"  [FastBuild] Weights: {t_weights:.1f}s | GPU transfer: {t_gpu:.1f}s")
        elif checkpoint_path:
            print(f"Warning: checkpoint {checkpoint_path} not found, using random weights.")

        t_total = time.time() - t0
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"  [FastBuild] Architecture: {t_arch:.1f}s | Weights: {t_weights:.1f}s | "
                  f"GPU transfer: {t_gpu:.1f}s | Total: {t_total:.1f}s ({param_count:.1f}M params)")
        else:
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

        # Auxiliary modules that participate in forward pass — keep on GPU
        # so they don't cause device mismatch with GPU-resident blocks.
        for attr in ('loop_block', 'lisa', 'mtp_module', '_attn_res',
                     '_hyperloop', 'loop_gates', 'middle_gates', '_v0_gates'):
            mod = getattr(model, attr, None)
            if mod is not None and hasattr(mod, 'to'):
                mod.to(gpu_dev)

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

        # Expert tying fix: tied expert pairs share the same Parameter objects.
        # When hybrid offload puts paired layers on different devices (e.g.,
        # layer 2=attention→GPU, layer 3=conv→CPU), the last .to() wins and
        # experts end up on the wrong device. Fix: for each tied pair, move
        # experts to the GPU layer's device (attention layers need GPU).
        if getattr(model.config, 'moe_expert_tying', False):
            tie_g = getattr(model.config, 'moe_tie_group_size', 2)
            for even_idx in range(0, len(model.blocks), tie_g):
                odd_idx = even_idx + 1
                if odd_idx >= len(model.blocks):
                    break
                even_moe = getattr(model.blocks[even_idx].ffn, 'experts', None)
                odd_moe = getattr(model.blocks[odd_idx].ffn, 'experts', None)
                if even_moe is None or odd_moe is None:
                    continue
                # Check if they're actually tied (same object)
                if len(even_moe) > 0 and len(odd_moe) > 0 \
                        and even_moe[0] is odd_moe[0]:
                    # Tied: move experts to whichever block is on GPU
                    even_dev = next(model.blocks[even_idx].parameters()).device
                    odd_dev = next(model.blocks[odd_idx].parameters()).device
                    if even_dev.type == 'cuda':
                        for exp in even_moe:
                            exp.to(gpu_dev)
                        if hasattr(model.blocks[odd_idx].ffn, 'shared'):
                            model.blocks[odd_idx].ffn.shared.to(gpu_dev)
                    elif odd_dev.type == 'cuda':
                        for exp in odd_moe:
                            exp.to(gpu_dev)
                        if hasattr(model.blocks[odd_idx].ffn, 'shared'):
                            model.blocks[odd_idx].ffn.shared.to(gpu_dev)

        # Placement changed — drop any cached device scan from a prior forward.
        if hasattr(model, 'invalidate_device_cache'):
            model.invalidate_device_cache()

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

        cache = create_kv_cache(model, max_total, batch=B, device=device)

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
    config_name: str = "forgelm_v7",
    checkpoint_path: str | None = None,
    device: str = "cuda",
    dtype: torch.dtype | None = None,
    moe_top_k: int = 0,
    compile_mode: str | None = None,
    fast_load: bool = True,
):
    """Load a model + tokenizer in one call.

    Centralizes the common pattern used across 10+ files:
        cfg = get_config(name, device=device)
        model = ModelLoader.build_model_fast(cfg, checkpoint_path=..., moe_top_k=..., dtype=...)
        tokenizer = get_tokenizer(...)

    Args:
        config_name: model config name (default "forgelm_v7")
        checkpoint_path: path to .safetensors checkpoint (default: config default)
        device: "cuda" or "cpu"
        dtype: torch.bfloat16 or torch.float32 (default: bf16 for cuda, fp32 for cpu)
        moe_top_k: MoE top-k routing (0 = dense_bypass)
        compile_mode: torch.compile mode if set (e.g. "default", "reduce-overhead")
        fast_load: when True (default), uses meta-init + assign=True + parallel
            tokenizer + OS prefetch for 3-6x faster cold boot. Set to False
            for the traditional build path.

    Returns:
        (model, tokenizer) tuple
    """
    from research.config import get_config
    from research.tokenizer_cache import get_tokenizer

    cfg = get_config(config_name, device=device)
    if dtype is None:
        dtype = torch.bfloat16 if "cuda" in device else torch.float32

    # Fast load: start tokenizer in parallel with model build (hides ~2.7s)
    tok_fut = None
    _tok_ex = None
    if fast_load:
        from concurrent.futures import ThreadPoolExecutor
        _tok_ex = ThreadPoolExecutor(max_workers=1)
        tok_fut = _tok_ex.submit(
            get_tokenizer, "research/checkpoints/lfm25_tokenizer")

    try:
        model = ModelLoader.build_model_fast(
            cfg, checkpoint_path=checkpoint_path,
            moe_top_k=moe_top_k, dtype=dtype, fast_load=fast_load)
        model.to(device).eval()

        if compile_mode is not None:
            try:
                model = model.compile_for_inference(mode=compile_mode)
            except Exception:
                pass

        if tok_fut is not None:
            tokenizer = tok_fut.result()
        else:
            tokenizer = get_tokenizer("research/checkpoints/lfm25_tokenizer")
    finally:
        if _tok_ex is not None:
            _tok_ex.shutdown(wait=False)
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


# Pre-import key modules at the bottom of the file (after all classes are
# defined) to avoid ~660ms of lazy import overhead during
# ConfigurableResearchLLM.__init__. This moves the import cost from the
# first model build to module load time. Wrapped in try/except so missing
# optional dependencies don't break the import.
try:
    from research.keys.architecture.titan_memory_key import TitanMemory  # noqa: F401,E402
    from research.keys.architecture.mod_router_key import ModRouter  # noqa: F401,E402
    from research.keys.architecture.mhc_key import MHCModule  # noqa: F401,E402
    from research.keys.architecture.attn_residual_key import AttnResModule  # noqa: F401,E402
    from research.keys.attention.differential_attn_key import DifferentialAttention  # noqa: F401,E402
    from research.keys.quantization.bitnet_b158_key import build_bitnet_linear  # noqa: F401,E402
    from research.keys.misc.pit_key import PITEmbedding, PITLMHead  # noqa: F401,E402
    from research.training.bitnet_lora import convert_to_bitnet_everywhere  # noqa: F401,E402
    # Pre-import tokenizer dependencies (avoid GIL contention when tokenizer
    # loads in a background thread during fast_load)
    import tokenizers  # noqa: F401,E402
    import gigatoken  # noqa: F401,E402
except ImportError:
    pass

# Pre-warm the class hierarchy by building a tiny model on meta device.
# The first ConfigurableResearchLLM() call takes ~640ms due to Python's
# first-time class instantiation overhead (nn.Module.__init__, meta tensor
# creation, etc.). Subsequent calls are ~50ms. By pre-building at import
# time, we move this cost to module load time (before the user's critical
# path), making the actual model build fast.
# Also pre-initialize the CUDA context so background threads can use CUDA
# immediately without ~500ms context creation overhead.
try:
    from research.config import get_config as _get_config_warmup
    _warmup_cfg = _get_config_warmup("forgelm_v7", device="meta")
    with torch.device("meta"):
        _warmup_model = ConfigurableResearchLLM(_warmup_cfg)
    del _warmup_model, _warmup_cfg
    # Pre-initialize CUDA context (needed for background weight load threads)
    if torch.cuda.is_available():
        torch.cuda.init()
except Exception:
    pass


if __name__ == "__main__":
    from research.config import get_config

    for name in ["lfm25_tiny", "forgelm_v7"]:
        print("\n" + "=" * 50)
        cfg = get_config(name)
        cfg.device = "cpu"
        ModelLoader.build_model(cfg)
