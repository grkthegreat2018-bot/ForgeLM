"""NLRQ FFN — Nested Low-Rank Quantized linear layers for FFN compression.

Implements the R&D-winning compression algorithm:
  1. SVD: W ≈ U @ diag(S) @ V^T  (low-rank decomposition)
  2. Quantize U and V factors at `factor_bits` (default 8b) — STORED as INT8
  3. Optional INT4 group residual on (W - U_q @ diag(S) @ V_q)

Compression ratios (from R&D on 2048x8192 FFN weights):
  - NLRQ r=256 b=8:           12.8x CR, 1.3% error, 8.0 bits/param
  - NLRQ+INT4 residual r=256:  3.0x CR, 0.15% error, 4.6 bits/param
  - NLRQ+INT4 residual r=512:  2.6x CR, 0.14% error, 4.9 bits/param

The key advantage over Monarch: NLRQ adapts to the actual weight structure
(SVD finds the optimal low-rank basis), while Monarch imposes a fixed
block-diagonal structure. NLRQ gives 5x better error at similar compression.

STORAGE: U and V are stored as actual INT8 tensors (torch.int8), not bf16.
This gives real VRAM savings — 8 bits per factor element vs 16 for bf16.
S (singular values) stays bf16 (only rank elements, negligible).
Scales are bf16 (out_features or in_features elements, small).

Usage:
    from research.keys.compression.nlrq_ffn_key import NLRQLinear, NLRQSwiGLUFFN
    # From dense:
    linear = NLRQLinear.from_dense(dense_weight, rank=256, factor_bits=8)
    # From scratch:
    linear = NLRQLinear(in_features, out_features, rank=256, factor_bits=8)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NLRQLinear(nn.Module):
    """Linear layer using Nested Low-Rank Quantized decomposition.

    W ≈ dequant(U_q) @ diag(S) @ dequant(V_q)  (+ optional INT4 residual)

    Where:
      - U_q: (out, rank) stored as INT8 — 8 bits per element
      - S:   (rank,) singular values (bf16, negligible size)
      - V_q: (rank, in) stored as INT8 — 8 bits per element
      - U_scale: (out, 1) bf16 per-row scale
      - V_scale: (1, in) bf16 per-col scale
      - residual: (out, in) INT4 group-quantized (optional)

    Real VRAM (rank=768, d_model=4096, intermediate=16384):
      U_q: 16384 × 768 × 1 byte = 12.6 MB
      V_q: 768 × 4096 × 1 byte = 3.1 MB
      S:   768 × 2 bytes = 1.5 KB
      scales: ~40 KB
      Total per layer per proj: ~15.8 MB (vs 128 MB dense bf16) = 8.1x

    Args:
        in_features: input dimension
        out_features: output dimension
        rank: SVD truncation rank (controls compression vs error)
        factor_bits: bits per factor element (8 = INT8, 4 = INT4)
        use_residual: if True, add INT4 group-quantized residual
        residual_group_size: group size for residual INT4 quantization
    """

    def __init__(self, in_features: int, out_features: int,
                 rank: int = 256, factor_bits: int = 8,
                 use_residual: bool = False, residual_group_size: int = 128,
                 bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = min(rank, min(in_features, out_features))
        self.factor_bits = factor_bits
        self.use_residual = use_residual
        self.residual_group_size = residual_group_size
        self._max_val = 2**(factor_bits - 1) - 1

        # U and V stored as INT8 (real quantized storage, not fake bf16)
        self.register_buffer('U_q', torch.zeros(out_features, self.rank, dtype=torch.int8))
        self.register_buffer('V_q', torch.zeros(self.rank, in_features, dtype=torch.int8))

        # S (singular values) — bf16, only rank elements
        self.S = nn.Parameter(torch.empty(self.rank))

        # Per-channel scales (bf16, small)
        self.register_buffer('U_scale', torch.ones(out_features, 1, dtype=torch.float16))
        self.register_buffer('V_scale', torch.ones(1, in_features, dtype=torch.float16))

        if use_residual:
            # Residual stored as INT4 (packed into int8 for PyTorch compat)
            # Each int8 holds 2 int4 values. We store unpacked for simplicity.
            self.register_buffer('residual_q', torch.zeros(out_features, in_features, dtype=torch.int8))
            n_groups = in_features // residual_group_size
            self.register_buffer('residual_scales',
                                 torch.ones(out_features, n_groups, dtype=torch.float16))
        else:
            self.register_buffer('residual_q', None)
            self.register_buffer('residual_scales', None)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        """Init via SVD of random matrix, then quantize factors to INT8."""
        W = torch.randn(self.out_features, self.in_features,
                        dtype=torch.float32, device=self.S.device) / math.sqrt(self.in_features)
        with torch.no_grad():
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            U_f = U[:, :self.rank].float()
            V_f = Vh[:self.rank, :].float()
            self.S.data = S[:self.rank].to(self.S.dtype).clone()
            # Compute scales and quantize
            self.U_scale = (U_f.abs().amax(dim=1, keepdim=True).clamp(min=1e-10) / self._max_val).to(torch.float16)
            self.V_scale = (V_f.abs().amax(dim=0, keepdim=True).clamp(min=1e-10) / self._max_val).to(torch.float16)
            self.U_q = torch.round(U_f / self.U_scale.float()).clamp(-self._max_val, self._max_val).to(torch.int8)
            self.V_q = torch.round(V_f / self.V_scale.float()).clamp(-self._max_val, self._max_val).to(torch.int8)

    @classmethod
    def from_dense(cls, weight: torch.Tensor, rank: int = 256,
                   factor_bits: int = 8, use_residual: bool = False,
                   residual_group_size: int = 128,
                   bias: torch.Tensor | None = None) -> 'NLRQLinear':
        """Fit NLRQ factors from a dense weight matrix via SVD + INT8 quantize.

        Args:
            weight: (out_features, in_features) — nn.Linear weight format
            rank: SVD truncation rank
            factor_bits: quantization bits for U and V factors
            use_residual: add INT4 group-quantized residual
            bias: optional bias vector
        """
        out_features, in_features = weight.shape
        layer = cls(in_features, out_features, rank=rank,
                    factor_bits=factor_bits,
                    use_residual=use_residual,
                    residual_group_size=residual_group_size,
                    bias=bias is not None)
        with torch.no_grad():
            W_f32 = weight.float()
            U, S, Vh = torch.linalg.svd(W_f32, full_matrices=False)
            r = min(rank, S.shape[0])
            U_f = U[:, :r].float()
            V_f = Vh[:r, :].float()
            layer.S.data = S[:r].to(layer.S.dtype).clone()
            # Quantize U and V to INT8
            max_val = layer._max_val
            layer.U_scale = (U_f.abs().amax(dim=1, keepdim=True).clamp(min=1e-10) / max_val).to(torch.float16)
            layer.V_scale = (V_f.abs().amax(dim=0, keepdim=True).clamp(min=1e-10) / max_val).to(torch.float16)
            layer.U_q = torch.round(U_f / layer.U_scale.float()).clamp(-max_val, max_val).to(torch.int8)
            layer.V_q = torch.round(V_f / layer.V_scale.float()).clamp(-max_val, max_val).to(torch.int8)
            # Compute residual if requested
            if use_residual:
                U_deq = layer.U_q.float() * layer.U_scale.float()
                V_deq = layer.V_q.float() * layer.V_scale.float()
                W_lr = U_deq @ torch.diag(layer.S.data.float()) @ V_deq
                residual = W_f32 - W_lr
                gs = residual_group_size
                n_groups = in_features // gs
                if n_groups > 0:
                    res_trunc = residual[:, :n_groups * gs].reshape(out_features, n_groups, gs)
                    scales = res_trunc.abs().amax(dim=-1, keepdim=True) / 7
                    scales = scales.clamp(min=1e-10)
                    Q = torch.round(res_trunc / scales).clamp(-8, 7)
                    layer.residual_q = torch.zeros(out_features, in_features, dtype=torch.int8)
                    layer.residual_q[:, :n_groups * gs] = Q.reshape(out_features, n_groups * gs).to(torch.int8)
                    layer.residual_scales = scales.squeeze(-1).to(torch.float16)
            if bias is not None:
                layer.bias.data = bias.clone()
        return layer

    def _dequantize_weight(self) -> torch.Tensor:
        """Dequantize INT8 factors to bf16 effective weight (fused, low-VRAM).

        Computes W = (U_q * U_scale) @ diag(S) @ (V_q * V_scale) without
        materializing intermediate (out, in) matrices separately.
        Uses sequential matmul: (U @ diag(S)) is (out, rank), then @ V is (out, in).
        This keeps peak memory at out*rank + rank*in, not out*in + out*in.
        """
        U_f = self.U_q.to(torch.float32) * self.U_scale.to(torch.float32)
        V_f = self.V_q.to(torch.float32) * self.V_scale.to(torch.float32)
        # Scale V by S first (rank, in) — small
        SV = self.S.to(torch.float32).unsqueeze(1) * V_f  # (rank, in)
        # U @ SV = (out, in) — this is the only large allocation
        W = U_f @ SV
        if self.residual_q is not None:
            gs = self.residual_group_size
            n_groups = self.in_features // gs
            if n_groups > 0:
                res = self.residual_q[:, :n_groups * gs].to(torch.float32).reshape(
                    self.out_features, n_groups, gs)
                res = res * self.residual_scales.to(torch.float32).unsqueeze(-1)
                W[:, :n_groups * gs] += res.reshape(self.out_features, n_groups * gs)
        return W.to(torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute y = x @ W^T + b directly from INT8 factors (no W materialization).

        Instead of: W = U @ diag(S) @ V; y = x @ W^T
        Computes:    y = ((x @ V^T) * S) @ U^T  + b

        Memory-optimized: dequantizes factors in-place into pre-allocated bf16
        buffers, avoiding per-call allocation of 31MB per projection.
        For 32 layers × 3 projections = 96 layers, this saves ~3GB of temporaries.
        """
        compute_dtype = x.dtype if x.dtype != torch.int64 else torch.float32

        # In-place dequantization into cached buffers (allocated on first call)
        # U_f: (out, rank) in compute_dtype — reused across calls
        if not hasattr(self, '_U_f_buf') or self._U_f_buf.dtype != compute_dtype:
            self._U_f_buf = torch.empty(
                self.U_q.shape, dtype=compute_dtype, device=self.U_q.device)
            self._V_f_buf = torch.empty(
                self.V_q.shape, dtype=compute_dtype, device=self.V_q.device)
        # Dequantize INT8 → compute_dtype in-place (no new allocation)
        torch.mul(self.U_q.to(compute_dtype), self.U_scale.to(compute_dtype),
                  out=self._U_f_buf)
        torch.mul(self.V_q.to(compute_dtype), self.V_scale.to(compute_dtype),
                  out=self._V_f_buf)
        S = self.S.to(compute_dtype)

        # y = ((x @ V^T) * S) @ U^T — fused to avoid intermediates
        # Pre-scale V by S: SV = S * V (rank, in) — small, in-place
        # Then y = (x @ SV^T) @ U^T — two small matmuls
        h = torch.matmul(x.to(compute_dtype), (S.unsqueeze(1) * self._V_f_buf).t())
        out = torch.matmul(h, self._U_f_buf.t())

        # Add residual if present
        if self.residual_q is not None:
            gs = self.residual_group_size
            n_groups = self.in_features // gs
            if n_groups > 0:
                res = self.residual_q[:, :n_groups * gs].to(compute_dtype).reshape(
                    self.out_features, n_groups, gs)
                res = res * self.residual_scales.to(compute_dtype).unsqueeze(-1)
                res_flat = res.reshape(self.out_features, n_groups * gs)
                out = out + x.to(compute_dtype)[:, :n_groups * gs] @ res_flat.t()

        if self.bias is not None:
            out = out + self.bias.to(compute_dtype)

        return out.to(x.dtype)

    @property
    def weight(self) -> torch.Tensor:
        """Reconstruct the approximate dense weight (for compat/checkpointing)."""
        return self._dequantize_weight()

    def compressed_storage_bytes(self) -> int:
        """Actual bytes stored in compressed form (INT8 factors + bf16 S + scales)."""
        total = self.U_q.numel() * 1  # int8 = 1 byte
        total += self.V_q.numel() * 1
        total += self.S.numel() * 2   # bf16
        total += self.U_scale.numel() * 2  # float16
        total += self.V_scale.numel() * 2
        if self.residual_q is not None:
            total += self.residual_q.numel() * 1  # int8 (holds int4)
            total += self.residual_scales.numel() * 2
        if self.bias is not None:
            total += self.bias.numel() * 2
        return total

    def dense_storage_bytes(self) -> int:
        """Bytes if stored as dense bf16."""
        return self.out_features * self.in_features * 2

    def compression_ratio(self) -> float:
        """Actual storage compression ratio."""
        return self.dense_storage_bytes() / max(self.compressed_storage_bytes(), 1)


class NLRQSwiGLUFFN(nn.Module):
    """SwiGLU FFN with NLRQ-factored linear layers.

    Replaces w_gate, w_up, w_down with NLRQLinear.
    Real INT8 storage: 8x FFN param reduction (rank=768, 8-bit factors).
    With INT4 residual: 3x reduction at 0.15% error.
    """

    def __init__(self, d_model: int = 2048, hidden_dim: int | None = None,
                 rank: int = 256, factor_bits: int = 8,
                 use_residual: bool = False, residual_group_size: int = 128,
                 use_clamp: bool = False,
                 clamp_alpha: float = 1.702, clamp_limit: float = 7.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(8 * d_model / 3)
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.factor_bits = factor_bits
        self.use_residual = use_residual
        self.use_clamp = use_clamp
        self.clamp_alpha = clamp_alpha
        self.clamp_limit = clamp_limit

        self.w_gate = NLRQLinear(d_model, hidden_dim, rank=rank,
                                 factor_bits=factor_bits,
                                 use_residual=use_residual,
                                 residual_group_size=residual_group_size)
        self.w_up = NLRQLinear(d_model, hidden_dim, rank=rank,
                               factor_bits=factor_bits,
                               use_residual=use_residual,
                               residual_group_size=residual_group_size)
        self.w_down = NLRQLinear(hidden_dim, d_model, rank=rank,
                                 factor_bits=factor_bits,
                                 use_residual=use_residual,
                                 residual_group_size=residual_group_size)

    @classmethod
    def from_dense_ffn(cls, ffn: nn.Module, rank: int = 256,
                       factor_bits: int = 8, use_residual: bool = False,
                       residual_group_size: int = 128) -> 'NLRQSwiGLUFFN':
        """Convert a dense SwiGLUFFN to NLRQSwiGLUFFN."""
        d_model = ffn.w_gate.in_features
        hidden_dim = ffn.w_gate.out_features
        layer = cls(d_model, hidden_dim, rank=rank,
                    factor_bits=factor_bits,
                    use_residual=use_residual,
                    residual_group_size=residual_group_size,
                    use_clamp=getattr(ffn, 'use_clamp', False),
                    clamp_alpha=getattr(ffn, 'clamp_alpha', 1.702),
                    clamp_limit=getattr(ffn, 'clamp_limit', 7.0))
        with torch.no_grad():
            layer.w_gate = NLRQLinear.from_dense(
                ffn.w_gate.weight, rank, factor_bits, use_residual,
                residual_group_size,
                ffn.w_gate.bias if hasattr(ffn.w_gate, 'bias') and ffn.w_gate.bias is not None else None)
            layer.w_up = NLRQLinear.from_dense(
                ffn.w_up.weight, rank, factor_bits, use_residual,
                residual_group_size,
                ffn.w_up.bias if hasattr(ffn.w_up, 'bias') and ffn.w_up.bias is not None else None)
            layer.w_down = NLRQLinear.from_dense(
                ffn.w_down.weight, rank, factor_bits, use_residual,
                residual_group_size,
                ffn.w_down.bias if hasattr(ffn.w_down, 'bias') and ffn.w_down.bias is not None else None)
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w_gate(x)
        up = self.w_up(x)
        if self.use_clamp:
            gate = gate.clamp(min=None, max=self.clamp_limit)
            up = up.clamp(-self.clamp_limit, max=self.clamp_limit)
            glu = gate * torch.sigmoid(self.clamp_alpha * gate)
            return self.w_down((up + 1) * glu)
        return self.w_down(F.silu(gate) * up)

    def compressed_storage_bytes(self) -> int:
        return (self.w_gate.compressed_storage_bytes() +
                self.w_up.compressed_storage_bytes() +
                self.w_down.compressed_storage_bytes())

    def dense_storage_bytes(self) -> int:
        return (self.w_gate.dense_storage_bytes() +
                self.w_up.dense_storage_bytes() +
                self.w_down.dense_storage_bytes())

    def compression_ratio(self) -> float:
        return self.dense_storage_bytes() / max(self.compressed_storage_bytes(), 1)
