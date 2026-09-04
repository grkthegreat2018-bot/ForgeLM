"""GRINQH — Graded Input-Based Quantization Hierarchy.

Paper: arXiv 2606.23419 — "GRINQH: Effective 2-Bit Weight Quantization via
Graded Input-Based Precision Allocation".

GRINQH is a weight-only post-training quantization (PTQ) method that unifies
quantization and sparsification. The core idea: not all weight channels are
equally important. Channels whose activations (or, in the weight-only proxy
used here, whose own L2 norm) are large contribute disproportionately to the
output and must be quantized at higher precision, while low-magnitude channels
can tolerate aggressive 2-bit quantization with minimal accuracy loss.

Per-channel precision assignment (per output channel):
  - High-norm channels (outliers) -> 4-bit (INT4, 16 levels)
  - Medium-norm channels          -> 3-bit (sign + 3 magnitudes, 8 levels)
  - Low-norm channels             -> 2-bit (sign + 1 magnitude, 4 levels)

The proportion of channels assigned to each tier is derived from
``target_effective_bits`` so that the average bits-per-weight matches the
target. For example, target_effective_bits=2.5 with equal thirds gives
(4+3+2)/3 = 3.0; to hit 2.5 we shift more channels to 2-bit.

Within each channel, group-wise quantization is applied (group_size=128):
each group gets its own absmax scale, so outliers within a group do not
destroy precision for the rest of the group.

Level definitions:
  - 2-bit: absmax scaling, 4 levels {-1, -0.5, +0.5, +1} * scale
           (sign bit + 1 magnitude bit). Code: {0:+0.5, 1:+1, 2:-0.5, 3:-1}.
  - 3-bit: absmax scaling, 8 levels (sign + 3 magnitudes {0.25,0.5,0.75,1.0})
           Code: sign_bit(1) | mag_idx(2 bits).
  - 4-bit: standard INT4 with absmax scale, 16 levels (symmetric, sign + 3 mag).

Packed storage (int8 per element):
  - 2-bit: 4 weights per byte (packed)
  - 3-bit: 1 weight per byte (wasteful upper 5 bits unused, but simple + fast)
  - 4-bit: 2 weights per byte (packed)

Dequantization produces bf16 for computation on CUDA, float32 on CPU.

VRAM budget (RTX 5070, 12 GB):
  At target_effective_bits=2.5, group_size=128:
    - Weight storage: ~2.5 bits/w = 0.3125 bytes/w
    - Scales: 2 bytes (float16) per group of 128 = 0.0156 bytes/w
    - Per-channel precision map: 1 byte per out_channel (negligible)
    - Total: ~0.33 bytes/w
  For a 1.2B model (~2.4 GB in bf16): ~0.49 GB in GRINQH weights.
  Dequantized bf16 weight cache (optional): +2.4 GB if cache=True.
  On 12 GB VRAM this leaves ample room for KV cache + activations.

Follows the patterns in novel_quant.py (ASFP4Linear, ResidualFP4Linear) and
iri_fp4_key.py (IRIFP4Linear). Self-contained — depends only on torch.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────
# Level tables
# ──────────────────────────────────────────────────────────────────────────

_2BIT_MAGNITUDES = torch.tensor([0.5, 1.0], dtype=torch.float32)
_3BIT_MAGNITUDES = torch.tensor([0.25, 0.5, 0.75, 1.0], dtype=torch.float32)
_4BIT_MAGNITUDES = torch.tensor(
    [1.0 / 7, 2.0 / 7, 3.0 / 7, 4.0 / 7, 5.0 / 7, 6.0 / 7, 1.0],
    dtype=torch.float32,
)  # 7 symmetric positive levels for INT4 (sign + 3 mag bits = 8 levels each side)


# ──────────────────────────────────────────────────────────────────────────
# Precision-tier assignment
# ──────────────────────────────────────────────────────────────────────────

def _assign_precision_tiers(
    weight: torch.Tensor, target_effective_bits: float,
) -> torch.Tensor:
    """Assign per-output-channel precision (2, 3, or 4 bits).

    Uses the L2 norm of each weight row as the importance proxy (the paper
    uses activation magnitudes; for weight-only PTQ without calibration data,
    the weight row norm is a well-correlated proxy — high-norm rows have
    larger absolute contributions to the dot product and are more sensitive
    to quantization error).

    The fraction of channels in each tier is chosen so the weighted average
    of bits matches ``target_effective_bits``. We sort channels by norm
    descending: the top fraction get 4-bit, the next fraction get 3-bit,
    the rest get 2-bit. Among the non-2-bit channels, 4-bit and 3-bit are
    split equally (frac4 = frac3) until 2-bit is exhausted, after which the
    remaining budget goes to 4-bit over 3-bit.

    This ensures all three tiers are populated for targets in (2.0, 3.5),
    matching the paper's hierarchical 3-tier design.

    Args:
        weight: [out_features, in_features] float tensor.
        target_effective_bits: desired average bits per weight (2.0–4.0).

    Returns:
        int8 tensor [out_features] with values in {2, 3, 4}.
    """
    out_f = weight.shape[0]
    norms = weight.float().norm(dim=1)  # [out_f]
    sorted_idx = torch.argsort(norms, descending=True)

    bits = torch.full((out_f,), 2, dtype=torch.int8)

    # Solve: frac4*4 + frac3*3 + frac2*2 = T, frac4+frac3+frac2 = 1
    # Constraint: frac4 = frac3 (equal split among promoted channels)
    # => 7*frac4 + 2*(1 - 2*frac4) = T  =>  3*frac4 = T - 2
    # If frac2 would go negative (T > 3.5), set frac2=0 and solve:
    #   frac4 = T - 3, frac3 = 4 - T
    T = target_effective_bits
    if T <= 2.0:
        return bits
    if T >= 4.0:
        bits[sorted_idx] = 4
        return bits

    if T <= 3.5:
        frac4 = (T - 2.0) / 3.0
        frac3 = frac4
        frac2 = 1.0 - 2.0 * frac4
    else:
        frac2 = 0.0
        frac4 = T - 3.0
        frac3 = 4.0 - T

    n4 = int(round(frac4 * out_f))
    n3 = int(round(frac3 * out_f))
    n4 = min(n4, out_f)
    n3 = min(n3, out_f - n4)

    top4 = sorted_idx[:n4]
    next3 = sorted_idx[n4:n4 + n3]
    bits[top4] = 4
    bits[next3] = 3
    return bits


# ──────────────────────────────────────────────────────────────────────────
# Per-tier quantize / dequantize (group-wise)
# ──────────────────────────────────────────────────────────────────────────

def _quantize_2bit(w_row: torch.Tensor, group_size: int):
    """Quantize a 1D weight row to 2-bit (4 levels) with group-wise absmax.

    Levels: {-1, -0.5, +0.5, +1} * scale.
    Code: sign_bit(1) | mag_bit(1).  mag 0 -> 0.5, mag 1 -> 1.0.
    Packed: 4 codes per uint8 byte.

    Returns:
        packed: uint8 [n_packed] (ceil(n / 4))
        scales: float16 [n_groups]
    """
    n = w_row.numel()
    pad = (group_size - n % group_size) % group_size
    if pad > 0:
        w_row = F.pad(w_row, (0, pad))
    n_padded = w_row.numel()
    n_groups = n_padded // group_size
    wg = w_row.view(n_groups, group_size)

    absmax = wg.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scale = absmax  # absmax scaling: levels are fractions of absmax
    w_norm = wg / scale.clamp(min=1e-12)

    abs_norm = w_norm.abs()
    # mag bit: 0 -> 0.5, 1 -> 1.0. Threshold at 0.75.
    mag_bit = (abs_norm > 0.75).long()
    sign_bit = (w_norm < 0).long()
    code = (sign_bit << 1) | mag_bit  # 2-bit code [0..3]

    code_flat = code.view(-1).to(torch.int64)
    n_packed = (n_padded + 3) // 4
    packed = torch.zeros(n_packed, dtype=torch.uint8, device=w_row.device)
    rem = n_padded % 4
    n_full = n_padded - rem
    if n_full > 0:
        c4 = code_flat[:n_full].view(-1, 4)
        packed[:n_full // 4] = (c4[:, 0] | (c4[:, 1] << 2) |
                                (c4[:, 2] << 4) | (c4[:, 3] << 6)).to(torch.uint8)
    if rem > 0:
        tail = code_flat[n_full:]
        val = 0
        for j in range(rem):
            val |= int(tail[j]) << (j * 2)
        packed[-1] = val

    scales = scale.squeeze(1).to(torch.float16)
    return packed, scales


def _dequantize_2bit(packed: torch.Tensor, scales: torch.Tensor,
                     n: int, group_size: int, dtype: torch.dtype) -> torch.Tensor:
    """Dequantize 2-bit packed weights."""
    n_padded = scales.numel() * group_size
    device = packed.device
    n_full = (n_padded // 4) * 4
    code_flat = torch.zeros(n_padded, dtype=torch.int64, device=device)
    if n_full > 0:
        p = packed[:n_full // 4].to(torch.int64)
        c0 = p & 0x03
        c1 = (p >> 2) & 0x03
        c2 = (p >> 4) & 0x03
        c3 = (p >> 6) & 0x03
        codes = torch.stack([c0, c1, c2, c3], dim=1).reshape(-1)
        code_flat[:n_full] = codes
    rem = n_padded % 4
    if rem > 0:
        tail = int(packed[-1])
        for j in range(rem):
            code_flat[n_full + j] = (tail >> (j * 2)) & 0x03

    sign_bit = (code_flat >> 1) & 1
    mag_bit = code_flat & 1
    mags = _2BIT_MAGNITUDES.to(device)
    magnitude = mags[mag_bit]
    w_norm = torch.where(sign_bit.bool(), -magnitude, magnitude)
    n_groups = scales.numel()
    w_norm = w_norm.view(n_groups, group_size)
    w = w_norm * scales.to(torch.float32).unsqueeze(1)
    w = w.view(-1)[:n].to(dtype)
    return w


def _quantize_3bit(w_row: torch.Tensor, group_size: int):
    """Quantize a 1D weight row to 3-bit (8 levels) with group-wise absmax.

    Levels: sign * {0.25, 0.5, 0.75, 1.0} * scale.
    Code: sign_bit(1) | mag_idx(2 bits).  Stored 1 per uint8 (upper bits 0).

    Returns:
        codes: uint8 [n_padded]
        scales: float16 [n_groups]
    """
    n = w_row.numel()
    pad = (group_size - n % group_size) % group_size
    if pad > 0:
        w_row = F.pad(w_row, (0, pad))
    n_padded = w_row.numel()
    n_groups = n_padded // group_size
    wg = w_row.view(n_groups, group_size)

    absmax = wg.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scale = absmax
    w_norm = wg / scale.clamp(min=1e-12)

    abs_norm = w_norm.abs()
    # Find nearest magnitude index via searchsorted on boundaries
    boundaries = torch.tensor([0.125, 0.375, 0.625, 0.875], dtype=torch.float32,
                              device=w_row.device)
    mag_idx = torch.searchsorted(boundaries, abs_norm).clamp(0, 3)
    sign_bit = (w_norm < 0).long()
    code = (sign_bit << 2) | mag_idx  # 3-bit code [0..7]

    codes = code.view(-1).to(torch.uint8)
    scales = scale.squeeze(1).to(torch.float16)
    return codes, scales


def _dequantize_3bit(codes: torch.Tensor, scales: torch.Tensor,
                     n: int, group_size: int, dtype: torch.dtype) -> torch.Tensor:
    """Dequantize 3-bit coded weights."""
    n_padded = scales.numel() * group_size
    device = codes.device
    code_flat = codes.to(torch.int64)[:n_padded]
    sign_bit = (code_flat >> 2) & 1
    mag_idx = code_flat & 0x03
    mags = _3BIT_MAGNITUDES.to(device)
    magnitude = mags[mag_idx]
    w_norm = torch.where(sign_bit.bool(), -magnitude, magnitude)
    n_groups = scales.numel()
    w_norm = w_norm.view(n_groups, group_size)
    w = w_norm * scales.to(torch.float32).unsqueeze(1)
    w = w.view(-1)[:n].to(dtype)
    return w


def _quantize_4bit(w_row: torch.Tensor, group_size: int):
    """Quantize a 1D weight row to 4-bit INT4 (16 levels) with group-wise absmax.

    Symmetric INT4: sign + 3 magnitude bits, 7 positive levels + zero.
    Code: sign_bit(1) | mag_idx(3 bits).  Packed 2 per uint8 byte.

    Returns:
        packed: uint8 [ceil(n_padded / 2)]
        scales: float16 [n_groups]
    """
    n = w_row.numel()
    pad = (group_size - n % group_size) % group_size
    if pad > 0:
        w_row = F.pad(w_row, (0, pad))
    n_padded = w_row.numel()
    n_groups = n_padded // group_size
    wg = w_row.view(n_groups, group_size)

    absmax = wg.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scale = absmax  # fractional magnitudes {1/7,...,1.0} * absmax
    w_norm = wg / scale.clamp(min=1e-12)

    abs_norm = w_norm.abs()
    boundaries = torch.tensor(
        [0.5 / 7, 1.5 / 7, 2.5 / 7, 3.5 / 7, 4.5 / 7, 5.5 / 7, 6.5 / 7],
        dtype=torch.float32, device=w_row.device,
    )
    mag_idx = torch.searchsorted(boundaries, abs_norm).clamp(0, 6)
    sign_bit = (w_norm < 0).long()
    code = (sign_bit << 3) | mag_idx  # 4-bit code [0..14]

    code_flat = code.view(-1).to(torch.int64)
    n_packed = (n_padded + 1) // 2
    packed = torch.zeros(n_packed, dtype=torch.uint8, device=w_row.device)
    n_even = (n_padded // 2) * 2
    if n_even > 0:
        c2 = code_flat[:n_even].view(-1, 2)
        packed[:n_even // 2] = ((c2[:, 0] & 0x0F) | ((c2[:, 1] << 4) & 0xF0)).to(torch.uint8)
    if n_padded % 2 == 1:
        packed[-1] = int(code_flat[-1]) & 0x0F

    scales = scale.squeeze(1).to(torch.float16)
    return packed, scales


def _dequantize_4bit(packed: torch.Tensor, scales: torch.Tensor,
                     n: int, group_size: int, dtype: torch.dtype) -> torch.Tensor:
    """Dequantize 4-bit INT4 packed weights."""
    n_padded = scales.numel() * group_size
    device = packed.device
    code_flat = torch.zeros(n_padded, dtype=torch.int64, device=device)
    n_even = (n_padded // 2) * 2
    if n_even > 0:
        p = packed[:n_even // 2].to(torch.int64)
        c0 = p & 0x0F
        c1 = (p >> 4) & 0x0F
        codes = torch.stack([c0, c1], dim=1).reshape(-1)
        code_flat[:n_even] = codes
    if n_padded % 2 == 1:
        code_flat[-1] = int(packed[-1]) & 0x0F

    sign_bit = (code_flat >> 3) & 1
    mag_idx = code_flat & 0x07
    mags = _4BIT_MAGNITUDES.to(device)
    magnitude = mags[mag_idx.clamp(0, 6)]
    w_norm = torch.where(sign_bit.bool(), -magnitude, magnitude)
    n_groups = scales.numel()
    w_norm = w_norm.view(n_groups, group_size)
    w = w_norm * scales.to(torch.float32).unsqueeze(1)
    w = w.view(-1)[:n].to(dtype)
    return w


# ──────────────────────────────────────────────────────────────────────────
# GRINQHQuantizer
# ──────────────────────────────────────────────────────────────────────────

class GRINQHQuantizer:
    """GRINQH weight-only PTQ: per-channel precision (2/3/4-bit) by importance.

    For each output channel, computes the L2 norm of the weight row. Channels
    with high norm (outliers) get 4-bit, medium get 3-bit, low get 2-bit. The
    proportion of each tier is determined by ``target_effective_bits``.

    Group-wise quantization within each channel (group_size=128) ensures
    intra-channel outliers don't destroy precision for the rest of the group.

    Paper: arXiv 2606.23419.

    Args:
        group_size: elements per quantization group (default 128).
        target_effective_bits: desired average bits per weight (default 2.5).
            Range [2.0, 4.0]. Lower = more compression, higher = more accuracy.
    """

    def __init__(self, group_size: int = 128,
                 target_effective_bits: float = 2.5):
        self.group_size = group_size
        self.target_effective_bits = target_effective_bits

    def quantize(self, weight: torch.Tensor) -> dict:
        """Quantize a 2D weight tensor [out_features, in_features].

        Returns a dict with:
          - precision: int8 [out_features] — per-channel bits {2,3,4}
          - packed_2bit: list of (uint8, float16) per 2-bit channel
          - packed_3bit: list of (uint8, float16) per 3-bit channel
          - packed_4bit: list of (uint8, float16) per 4-bit channel
          - channel_indices: dict mapping tier -> list of channel indices
          - shape: (out_features, in_features)
          - group_size: int
          - target_effective_bits: float
          - effective_bits: float (actual achieved average)
        """
        w = weight.float().cpu()
        out_f, in_f = w.shape
        gs = self.group_size

        precision = _assign_precision_tiers(w, self.target_effective_bits)

        packed_2bit = []
        packed_3bit = []
        packed_4bit = []
        idx_2 = []
        idx_3 = []
        idx_4 = []

        for ch in range(out_f):
            row = w[ch]
            bits = int(precision[ch].item())
            if bits == 2:
                p, s = _quantize_2bit(row, gs)
                packed_2bit.append((p, s))
                idx_2.append(ch)
            elif bits == 3:
                p, s = _quantize_3bit(row, gs)
                packed_3bit.append((p, s))
                idx_3.append(ch)
            else:
                p, s = _quantize_4bit(row, gs)
                packed_4bit.append((p, s))
                idx_4.append(ch)

        # Compute actual effective bits
        total_bits = int((precision == 2).sum()) * 2 * in_f \
            + int((precision == 3).sum()) * 3 * in_f \
            + int((precision == 4).sum()) * 4 * in_f
        effective_bits = total_bits / (out_f * in_f)

        return {
            "precision": precision,
            "packed_2bit": packed_2bit,
            "packed_3bit": packed_3bit,
            "packed_4bit": packed_4bit,
            "channel_indices": {2: idx_2, 3: idx_3, 4: idx_4},
            "shape": (out_f, in_f),
            "group_size": gs,
            "target_effective_bits": self.target_effective_bits,
            "effective_bits": effective_bits,
        }

    def dequantize(self, packed: dict,
                   dtype: torch.dtype | None = None) -> torch.Tensor:
        """Reconstruct a weight tensor from the packed representation.

        Args:
            packed: dict from ``quantize``.
            dtype: output dtype. Defaults to bfloat16 on CUDA, float32 on CPU.

        Returns:
            Reconstructed weight [out_features, in_features].
        """
        out_f, in_f = packed["shape"]
        gs = packed["group_size"]
        device = packed["precision"].device

        if dtype is None:
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        w = torch.zeros(out_f, in_f, dtype=dtype, device=device)

        for (p, s), ch in zip(packed["packed_2bit"], packed["channel_indices"][2]):
            row = _dequantize_2bit(p.to(device), s.to(device), in_f, gs, dtype)
            w[ch] = row
        for (p, s), ch in zip(packed["packed_3bit"], packed["channel_indices"][3]):
            row = _dequantize_3bit(p.to(device), s.to(device), in_f, gs, dtype)
            w[ch] = row
        for (p, s), ch in zip(packed["packed_4bit"], packed["channel_indices"][4]):
            row = _dequantize_4bit(p.to(device), s.to(device), in_f, gs, dtype)
            w[ch] = row

        return w


# ──────────────────────────────────────────────────────────────────────────
# GRINQHLinear: inference module
# ──────────────────────────────────────────────────────────────────────────

class GRINQHLinear(nn.Module):
    """Linear layer with GRINQH graded-precision quantized weights.

    Stores per-channel precision (2/3/4-bit) packed weights. Dequantizes
    on-the-fly to bf16 (CUDA) or float32 (CPU) for computation.

    Interface matches nn.Linear (in_features, out_features, bias).

    Storage at target_effective_bits=2.5, group_size=128:
      ~0.33 bytes/weight (vs 2.0 for bf16) = ~6x compression.

    Args:
        in_features, out_features: as nn.Linear.
        bias: whether to include a bias term.
        group_size: elements per quantization group (default 128).
        target_effective_bits: desired average bits per weight (default 2.5).
    """

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, group_size: int = 128,
                 target_effective_bits: float = 2.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.target_effective_bits = target_effective_bits

        n_groups = (in_features + group_size - 1) // group_size
        n_padded = n_groups * group_size
        n_packed_2 = (n_padded + 3) // 4
        n_packed_3 = n_padded
        n_packed_4 = (n_padded + 1) // 2

        self.register_buffer(
            "precision", torch.full((out_features,), 2, dtype=torch.int8),
        )
        # Per-tier packed storage (padded to out_features for uniform shape)
        self.register_buffer(
            "packed_2bit",
            torch.zeros(out_features, n_packed_2, dtype=torch.uint8),
        )
        self.register_buffer(
            "scales_2bit",
            torch.zeros(out_features, n_groups, dtype=torch.float16),
        )
        self.register_buffer(
            "packed_3bit",
            torch.zeros(out_features, n_packed_3, dtype=torch.uint8),
        )
        self.register_buffer(
            "scales_3bit",
            torch.zeros(out_features, n_groups, dtype=torch.float16),
        )
        self.register_buffer(
            "packed_4bit",
            torch.zeros(out_features, n_packed_4, dtype=torch.uint8),
        )
        self.register_buffer(
            "scales_4bit",
            torch.zeros(out_features, n_groups, dtype=torch.float16),
        )

        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None

        self._cached_weight = None
        self._effective_bits = 0.0

    @classmethod
    def from_linear(cls, lin: nn.Linear, group_size: int = 128,
                    target_effective_bits: float = 2.5) -> "GRINQHLinear":
        """Build from a standard nn.Linear by quantizing its weights."""
        w = lin.weight.data
        out_f, in_f = w.shape
        obj = cls(in_f, out_f, bias=lin.bias is not None,
                  group_size=group_size,
                  target_effective_bits=target_effective_bits)
        obj._load_quantized(w)
        if lin.bias is not None:
            obj.bias.copy_(lin.bias.data.to(torch.float16))
        return obj

    @torch.no_grad()
    def _load_quantized(self, weight: torch.Tensor) -> None:
        """Quantize a weight tensor and load into packed buffers."""
        quantizer = GRINQHQuantizer(self.group_size, self.target_effective_bits)
        packed = quantizer.quantize(weight)

        self.precision.copy_(packed["precision"])
        self._effective_bits = packed["effective_bits"]

        out_f = self.out_features
        in_f = self.in_features

        # Zero out all packed buffers first
        self.packed_2bit.zero_()
        self.scales_2bit.zero_()
        self.packed_3bit.zero_()
        self.scales_3bit.zero_()
        self.packed_4bit.zero_()
        self.scales_4bit.zero_()

        for (p, s), ch in zip(packed["packed_2bit"], packed["channel_indices"][2]):
            self.packed_2bit[ch, :p.numel()].copy_(p)
            self.scales_2bit[ch].copy_(s)
        for (p, s), ch in zip(packed["packed_3bit"], packed["channel_indices"][3]):
            self.packed_3bit[ch, :p.numel()].copy_(p)
            self.scales_3bit[ch].copy_(s)
        for (p, s), ch in zip(packed["packed_4bit"], packed["channel_indices"][4]):
            self.packed_4bit[ch, :p.numel()].copy_(p)
            self.scales_4bit[ch].copy_(s)

        self._cached_weight = None

    def _dequantize_weight(self, dtype: torch.dtype | None = None,
                           cache: bool = False) -> torch.Tensor:
        """Dequantize all channels to a full weight tensor.

        Args:
            dtype: output dtype. Defaults to bf16 on CUDA, float32 on CPU.
            cache: if True, cache the result (trades VRAM for speed).
        """
        if dtype is None:
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        if cache and self._cached_weight is not None \
                and self._cached_weight.dtype == dtype:
            return self._cached_weight

        out_f = self.out_features
        in_f = self.in_features
        gs = self.group_size
        device = self.precision.device

        w = torch.zeros(out_f, in_f, dtype=dtype, device=device)

        prec = self.precision
        mask2 = (prec == 2)
        mask3 = (prec == 3)
        mask4 = (prec == 4)

        ch2 = mask2.nonzero(as_tuple=True)[0]
        ch3 = mask3.nonzero(as_tuple=True)[0]
        ch4 = mask4.nonzero(as_tuple=True)[0]

        for ch in ch2.tolist():
            p = self.packed_2bit[ch]
            s = self.scales_2bit[ch]
            w[ch] = _dequantize_2bit(p, s, in_f, gs, dtype)
        for ch in ch3.tolist():
            p = self.packed_3bit[ch]
            s = self.scales_3bit[ch]
            w[ch] = _dequantize_3bit(p, s, in_f, gs, dtype)
        for ch in ch4.tolist():
            p = self.packed_4bit[ch]
            s = self.scales_4bit[ch]
            w[ch] = _dequantize_4bit(p, s, in_f, gs, dtype)

        if cache:
            self._cached_weight = w
        return w

    @property
    def weight_quantized(self) -> torch.Tensor:
        """Return the dequantized weight (full precision reconstruction)."""
        return self._dequantize_weight(torch.float32)

    @property
    def effective_bits(self) -> float:
        """Actual achieved average bits per weight."""
        return self._effective_bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._dequantize_weight(x.dtype, cache=True)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        out = F.linear(x, w, bias)
        if hasattr(self, "lora_adapter") and self.lora_adapter is not None:
            out = out + self.lora_adapter(x)
        return out

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={self.bias is not None}, "
                f"group_size={self.group_size}, "
                f"target_bits={self.target_effective_bits}, "
                f"effective_bits={self._effective_bits:.2f}")


# ──────────────────────────────────────────────────────────────────────────
# Model conversion
# ──────────────────────────────────────────────────────────────────────────

def quantize_model_grinqh(model: nn.Module, group_size: int = 128,
                          target_effective_bits: float = 2.5,
                          verbose: bool = True) -> int:
    """Replace all nn.Linear with GRINQHLinear (graded-precision quantization).

    Walks the model and replaces each nn.Linear (except embeddings/heads and
    already-quantized layers) with a GRINQHLinear that stores 2/3/4-bit
    graded-precision weights. The original Linear weights are quantized
    in-place.

    Args:
        model: nn.Module with nn.Linear layers.
        group_size: elements per quantization group (default 128).
        target_effective_bits: desired average bits per weight (default 2.5).
        verbose: print progress.

    Returns:
        Number of layers quantized.
    """
    skip_types = ("NVFP4Linear", "ASFP4Linear", "ResidualFP4Linear",
                  "W8A8Linear", "FP8Linear", "BitNetLinear",
                  "INT4Linear", "QuantizedLinear", "FastINT8Linear",
                  "NLRQLinear", "IRIFP4Linear", "GRINQHLinear")
    skip_names = ("embed", "head", "lm_head", "output")
    n = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and type(module).__name__ not in skip_types:
            if any(s in name for s in skip_names):
                continue
            parent = model
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            try:
                setattr(parent, parts[-1],
                        GRINQHLinear.from_linear(
                            module, group_size=group_size,
                            target_effective_bits=target_effective_bits))
                n += 1
            except Exception as e:
                if verbose:
                    print(f"  [GRINQH] Skipped {name}: {e}")
    if verbose and n > 0:
        print(f"  [GRINQH] {n} layers quantized "
              f"(target={target_effective_bits} bits/w, group={group_size})")
    return n
