"""Quantization simulator functions — raw metric computation only.

Scoring is handled by RewardGuard + domain JSON specs, NOT here.
Each simulator returns a dict of raw metrics (sqnr, err, compression, etc.).

These are PORT-EXACT replicas of the original domain evaluate() methods,
separated into pure metric computation (here) + declarative scoring (JSON).
"""
from __future__ import annotations

import numpy as np
import torch

from . import register


def _quantize(tensor, n_bits, scheme="symmetric"):
    """Simulate quantization to n_bits (fallback when ForgeEngine quant unavailable)."""
    if scheme == "symmetric":
        max_val = tensor.abs().max()
        scale = max_val / (2 ** (n_bits - 1) - 1)
        q = torch.round(tensor / (scale + 1e-8)) * scale
    else:
        mn, mx = tensor.min(), tensor.max()
        scale = (mx - mn) / (2 ** n_bits - 1)
        zero = -mn / (scale + 1e-8)
        q = torch.round(tensor / (scale + 1e-8) + zero) * scale - zero * scale
    return q


@register("w8a8_simulate")
def w8a8_simulate(config: dict, domain=None) -> dict:
    """W8A8 quantization: smoothquant alpha + INT8/FP8 mode."""
    device = domain.device if domain is not None else torch.device("cpu")
    w = torch.randn(128, 256, device=device) * 0.02
    a = torch.randn(32, 256, device=device) * 0.01
    alpha = float(config.get("smoothquant_alpha", 0.0))
    if alpha > 0:
        s = (a.abs().max(dim=0).values ** alpha) * (w.abs().max(dim=0).values ** (1 - alpha)) + 1e-8
        w_s, a_s = w / s, a / s
    else:
        w_s, a_s = w, a
    mode = config.get("mode", "int8")
    try:
        from research.inference.quant.w8a8_quant import W8A8Linear
        lin = W8A8Linear(256, 128, bias=False, mode=mode)
        lin.weight.data.copy_(w_s.t().contiguous())
        y_ref = torch.nn.functional.linear(a_s, w_s)
        y_q = lin(a_s)
        noise = (y_ref - y_q).norm()
        signal = y_ref.norm()
        sqnr = 20 * torch.log10(signal / (noise + 1e-8))
    except (ImportError, Exception):
        bits = 8
        w_q = _quantize(w_s, bits)
        a_q = _quantize(a_s, bits)
        noise = ((w_s - w_q) @ (a_s - a_q).T).norm()
        signal = (w_s @ a_s.T).norm()
        sqnr = 20 * torch.log10(signal / (noise + 1e-8))
    sqnr = float(sqnr)
    compression = 2.0
    return {"sqnr": sqnr, "compression": compression,
            "behavioral_0": sqnr, "behavioral_1": compression}


@register("nvfp4_simulate")
def nvfp4_simulate(config: dict, domain=None) -> dict:
    """NVFP4 block quantization."""
    device = domain.device if domain is not None else torch.device("cpu")
    bs = int(config.get("block_size", 32))
    w = torch.randn(256, 512, device=device) * 0.02
    try:
        from research.inference.quant.nvfp4_quant import _quantize_to_fp4, _dequantize_fp4
        packed, scales, global_scale = _quantize_to_fp4(w, bs)
        w_dq = _dequantize_fp4(packed, scales, 256, 512, bs, torch.float32, global_scale)
        err = float((w - w_dq).norm().item() / (w.norm().item() + 1e-8))
    except (ImportError, Exception):
        scale_mode = config.get("scale_mode", "per_block")
        if scale_mode == "per_block":
            w_blocks = w.view(-1, bs)
            scales = w_blocks.abs().max(dim=1, keepdim=True).values / 7.0
            w_q = torch.round(w_blocks / (scales + 1e-8)) * scales
            w_q = w_q.view_as(w)
        else:
            scales = w.abs().max(dim=1, keepdim=True).values / 7.0
            w_q = torch.round(w / (scales + 1e-8)) * scales
        err = float((w - w_q).norm().item() / (w.norm().item() + 1e-8))
    compression = 8.0 if config.get("w4a8", False) else 3.8
    return {"err": err, "compression": compression,
            "behavioral_0": err, "behavioral_1": compression}


@register("bitnet_simulate")
def bitnet_simulate(config: dict, domain=None) -> dict:
    """BitNet b1.58 ternary / b1.0 binary quantization."""
    device = domain.device if domain is not None else torch.device("cpu")
    w = torch.randn(256, 512, device=device) * 0.02
    init_scale = float(config.get("init_scale", 1.0))
    scale = init_scale * w.abs().mean()
    quant_mode = config.get("quant_mode", "ternary")
    if quant_mode == "ternary":
        try:
            from research.keys.quantization.bitnet_b158_key import ternary_quantize
            q, qscale = ternary_quantize(w, scale)
            w_q = q.float() * qscale
        except (ImportError, Exception):
            w_q = torch.clamp(torch.round(w / (scale + 1e-8)), -1, 1) * scale
        compression = 8.0
    else:
        w_q = torch.sign(w) * scale
        compression = 16.0
    err = float((w - w_q).norm().item() / (w.norm().item() + 1e-8))
    return {"err": err, "compression": compression,
            "behavioral_0": err, "behavioral_1": compression}


@register("sharq_simulate")
def sharq_simulate(config: dict, domain=None) -> dict:
    """SharQ adaptive/non-adaptive quantization."""
    device = domain.device if domain is not None else torch.device("cpu")
    w = torch.randn(256, 512, device=device) * 0.02
    n_l = int(config.get("n_levels", 16))
    bits = float(np.log2(max(n_l, 2)))
    if config.get("adaptive", True):
        sorted_w = w.abs().flatten().sort().values
        boundaries = sorted_w[::max(1, len(sorted_w) // n_l)]
        w_q = torch.zeros_like(w)
        for i in range(len(boundaries) - 1):
            mask = (w.abs() >= boundaries[i]) & (w.abs() < boundaries[i + 1])
            w_q[mask] = boundaries[i] * w[mask].sign()
    else:
        scale = w.abs().max() / (n_l // 2 - 1)
        w_q = torch.round(w / (scale + 1e-8)) * scale
    err = float((w - w_q).norm().item() / (w.norm().item() + 1e-8))
    return {"err": err, "bits": bits,
            "behavioral_0": err, "behavioral_1": bits}


@register("mosaic_simulate")
def mosaic_simulate(config: dict, domain=None) -> dict:
    """MosaicQuant n_tiles, tile_dim, mix_ratio."""
    device = domain.device if domain is not None else torch.device("cpu")
    w = torch.randn(128, 128, device=device) * 0.02
    td = int(config.get("tile_dim", 256))
    mr = float(config.get("mix_ratio", 0.5))
    td_safe = min(td, 128)
    while 128 % td_safe != 0:
        td_safe -= 1
    n_cols = 128 // td_safe
    tiles = w.view(128, n_cols, td_safe).permute(1, 0, 2)
    err_sq = torch.tensor(0.0, device=device)
    for t in tiles:
        bits = 4 if torch.rand(1, device=device).item() < mr else 8
        t_q = _quantize(t, bits)
        err_sq += (t - t_q).norm() ** 2
    err = float((err_sq.sqrt() / (w.norm() + 1e-8)).item())
    mem_ratio = 0.5 + mr * 0.5
    return {"err": err, "mem_ratio": mem_ratio,
            "behavioral_0": err, "behavioral_1": mem_ratio}


@register("aaac_simulate")
def aaac_simulate(config: dict, domain=None) -> dict:
    """AAAC n_codebooks, codebook_size, n_bits."""
    device = domain.device if domain is not None else torch.device("cpu")
    w = torch.randn(64, 64, device=device) * 0.02
    nc = int(config.get("n_codebooks", 8))
    cs = min(int(config.get("codebook_size", 256)), 256)
    nb = int(config.get("n_bits", 3))
    residual = w.clone()
    flat = residual.flatten()
    for _ in range(nc):
        codebook = torch.randn(cs, device=device) * 0.01
        codes = (flat.unsqueeze(1) - codebook.unsqueeze(0)).abs().argmin(dim=1)
        flat = flat - codebook[codes]
    residual = flat.view_as(w)
    err = float(residual.norm().item() / (w.norm().item() + 1e-8))
    compression = 32 / (nc * nb)
    return {"err": err, "compression": compression,
            "behavioral_0": err, "behavioral_1": compression}


@register("offq_simulate")
def offq_simulate(config: dict, domain=None) -> dict:
    """OffQ offset_init, n_iter, learn_offset."""
    device = domain.device if domain is not None else torch.device("cpu")
    w = torch.randn(256, 512, device=device) * 0.02
    offset = float(config.get("offset_init", 0.5)) * w.std()
    if config.get("learn_offset", True):
        for _ in range(min(int(config.get("n_iter", 50)), 20)):
            offset = offset - 0.01 * (w + offset).abs().mean()
    w_shifted = w + offset
    w_q = _quantize(w_shifted, 4)
    w_recon = w_q - offset
    err = float((w - w_recon).norm().item() / (w.norm().item() + 1e-8))
    clipping = float((w_shifted.abs() > w_shifted.abs().max() * 0.99).float().mean().item())
    return {"err": err, "clipping": clipping,
            "behavioral_0": clipping, "behavioral_1": err}


@register("group_quant_simulate")
def group_quant_simulate(config: dict, domain=None) -> dict:
    """Group quantization: group_size, n_bits, scheme."""
    device = domain.device if domain is not None else torch.device("cpu")
    w = torch.randn(256, 512, device=device) * 0.02
    gs = int(config.get("group_size", 32))
    bits = int(config.get("n_bits", 4))
    scheme = config.get("scheme", "symmetric")
    w_groups = w.view(-1, gs)
    w_q = _quantize(w_groups, bits, scheme)
    w_q = w_q.view_as(w)
    err = float((w - w_q).norm().item() / (w.norm().item() + 1e-8))
    compression = 32 / bits
    return {"err": err, "compression": compression,
            "behavioral_0": err, "behavioral_1": compression}


@register("mixed_precision_simulate")
def mixed_precision_simulate(config: dict, domain=None) -> dict:
    """Mixed precision: n_levels, assignment, bits_base."""
    device = domain.device if domain is not None else torch.device("cpu")
    w = torch.randn(16, 64, 64, device=device) * 0.02
    nl = int(config.get("n_levels", 3))
    bb = int(config.get("bits_base", 4))
    assignment = config.get("assignment", "uniform")
    bits_list = [bb, bb + 2, bb + 4, bb + 6][:nl]
    if assignment == "importance":
        importance = w.view(16, -1).norm(dim=1)
        order = importance.argsort()
    elif assignment == "random":
        order = torch.randperm(16, device=device)
    else:
        order = torch.arange(16, device=device)
    err_sq = torch.tensor(0.0, device=device)
    bits_total = 0
    for i in range(16):
        bits = bits_list[i % nl]
        layer = int(order[i])
        w_q = _quantize(w[layer], bits)
        err_sq += (w[layer] - w_q).norm() ** 2
        bits_total += bits
    err = float((err_sq.sqrt() / (w.norm() + 1e-8)).item())
    avg_bits = bits_total / 16
    return {"err": err, "avg_bits": avg_bits,
            "behavioral_0": err, "behavioral_1": avg_bits}


@register("activation_quant_simulate")
def activation_quant_simulate(config: dict, domain=None) -> dict:
    """Activation quantization: calib_method, percentile, smooth_alpha."""
    device = domain.device if domain is not None else torch.device("cpu")
    a = torch.randn(256, 512, device=device) * 0.01
    a[0, 0] = 0.5  # outlier
    method = config.get("calib_method", "minmax")
    if method == "minmax":
        scale = a.abs().max() / 127
    elif method == "percentile":
        scale = torch.quantile(a.abs().flatten(), float(config.get("percentile", 0.99))) / 127
    else:
        scale = a.std() * 3 / 127
    a_q = torch.round(a / (scale + 1e-8)) * scale
    err = float((a - a_q).norm().item() / (a.norm().item() + 1e-8))
    outlier_handled = 1.0 - float((a_q[0, 0] - a[0, 0]).abs().item() / (a[0, 0].abs().item() + 1e-8))
    return {"err": err, "outlier_handled": outlier_handled,
            "behavioral_0": outlier_handled, "behavioral_1": err}


@register("quant_domain_simulate")
def quant_domain_simulate(config: dict, domain=None) -> dict:
    """Generic quantization domain (legacy QuantDomain)."""
    device = domain.device if domain is not None else torch.device("cpu")
    w = torch.randn(256, 512, device=device) * 0.02
    n_bits = int(config.get("n_bits", 4))
    scheme = config.get("scheme", "symmetric")
    w_q = _quantize(w, n_bits, scheme)
    noise = (w - w_q).norm()
    signal = w.norm()
    sqnr = float(20 * torch.log10(signal / (noise + 1e-8)))
    compression = 16.0 / n_bits
    return {"sqnr": sqnr, "compression": compression,
            "behavioral_0": sqnr, "behavioral_1": compression}
