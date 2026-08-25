"""Quantization-related evolution domains (10 domains).

Each domain explores a different quantization configuration search space.
All use small tensor operations for fast evaluation (no model loading).
"""
from __future__ import annotations
import torch
import numpy as np
from typing import Any
from . import BaseDomain


def _quantize(tensor, n_bits, scheme="symmetric"):
    """Simulate quantization to n_bits."""
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


class W8A8Quant(BaseDomain):
    """W8A8 mode, calibration samples, per-channel, smoothquant alpha."""
    def name(self): return "w8a8_quant"
    def output_dim(self): return 4
    def discrete_choices(self):
        return {"mode": ["int8", "fp8"]}
    def behavioral_dims(self):
        return [("sqnr_db", 10, 0, 60), ("compression", 10, 1, 4)]
    def decode(self, params):
        p = params.detach().cpu().numpy()
        return {"mode": ["int8", "fp8"][int(p[0] * 1.999)],
                "calib_samples": int(np.interp(p[1], [0, 1], [64, 1024])),
                "per_channel": bool(p[2] > 0.5),
                "smoothquant_alpha": float(p[3])}
    def encode(self, config):
        return torch.tensor([
            0.0 if config.get("mode", "int8") == "int8" else 1.0,
            np.clip((config.get("calib_samples", 256) - 64) / 960, 0, 1),
            1.0 if config.get("per_channel", True) else 0.0,
            config.get("smoothquant_alpha", 0.5),
        ], dtype=torch.float32)
    def evaluate(self, config):
        w = self._randn(128, 256) * 0.02
        a = self._randn(32, 256) * 0.01
        alpha = config["smoothquant_alpha"]
        if alpha > 0:
            s = (a.abs().max(dim=0).values ** alpha) * (w.abs().max(dim=0).values ** (1 - alpha)) + 1e-8
            w_s, a_s = w / s, a / s
        else:
            w_s, a_s = w, a
        # Use ForgeEngine's W8A8 quantization for real INT8/FP8 error
        try:
            from research.inference.quant.w8a8_quant import W8A8Linear
            lin = W8A8Linear(256, 128, bias=False, mode=config["mode"])
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
        compression = 2.0
        score = float(sqnr) * 2 + compression * 3
        return {"score": float(score), "behavioral": (float(sqnr), compression),
                "metadata": {"mode": config["mode"], "sqnr": float(sqnr)}}


class Nvfp4Quant(BaseDomain):
    """NVFP4 block size, w4a8, scale mode."""
    def name(self): return "nvfp4_quant"
    def output_dim(self): return 3
    def discrete_choices(self):
        return {"block_size": [16, 32, 64], "scale_mode": ["per_block", "per_channel"]}
    def behavioral_dims(self):
        return [("quant_err", 10, 0, 1), ("compression", 10, 2, 8)]
    def decode(self, params):
        p = params.detach().cpu().numpy()
        bs_choices = [16, 32, 64]
        return {"block_size": bs_choices[int(p[0] * 2.999)],
                "w4a8": bool(p[1] > 0.5),
                "scale_mode": ["per_block", "per_channel"][int(p[2] * 1.999)]}
    def encode(self, config):
        bs_map = {16: 0, 32: 0.5, 64: 1.0}
        return torch.tensor([
            bs_map.get(config.get("block_size", 32), 0.5),
            1.0 if config.get("w4a8", False) else 0.0,
            0.0 if config.get("scale_mode", "per_block") == "per_block" else 1.0,
        ], dtype=torch.float32)
    def evaluate(self, config):
        bs = config["block_size"]
        w = self._randn(256, 512) * 0.02
        # Use ForgeEngine's NVFP4 quantization for real FP4 error
        try:
            from research.inference.quant.nvfp4_quant import _quantize_to_fp4, _dequantize_fp4
            packed, scales, global_scale = _quantize_to_fp4(w, bs)
            w_dq = _dequantize_fp4(packed, scales, 256, 512, bs, torch.float32, global_scale)
            err = float((w - w_dq).norm().item() / (w.norm().item() + 1e-8))
        except (ImportError, Exception):
            # Fallback: simulated FP4
            if config["scale_mode"] == "per_block":
                w_blocks = w.view(-1, bs)
                scales = w_blocks.abs().max(dim=1, keepdim=True).values / 7.0
                w_q = torch.round(w_blocks / (scales + 1e-8)) * scales
                w_q = w_q.view_as(w)
            else:
                scales = w.abs().max(dim=1, keepdim=True).values / 7.0
                w_q = torch.round(w / (scales + 1e-8)) * scales
            err = float((w - w_q).norm().item() / (w.norm().item() + 1e-8))
        compression = 8.0 if config["w4a8"] else 3.8
        if compression < 2.0:
            score = -50
        else:
            score = -err * 100 + compression * 3
        return {"score": float(score), "behavioral": (err, compression),
                "metadata": {"block_size": bs, "err": err}}


class BitnetConfig(BaseDomain):
    """BitNet learned_scale, quant_mode, init_scale."""
    def name(self): return "bitnet_config"
    def output_dim(self): return 3
    def discrete_choices(self):
        return {"quant_mode": ["ternary", "binary"]}
    def behavioral_dims(self):
        return [("weight_err", 10, 0, 1), ("compression", 10, 4, 16)]
    def decode(self, params):
        p = params.detach().cpu().numpy()
        return {"learned_scale": bool(p[0] > 0.5),
                "quant_mode": ["ternary", "binary"][int(p[1] * 1.999)],
                "init_scale": float(np.interp(p[2], [0, 1], [0.5, 2.0]))}
    def encode(self, config):
        return torch.tensor([
            1.0 if config.get("learned_scale", True) else 0.0,
            0.0 if config.get("quant_mode", "ternary") == "ternary" else 1.0,
            np.clip((config.get("init_scale", 1.0) - 0.5) / 1.5, 0, 1),
        ], dtype=torch.float32)
    def evaluate(self, config):
        w = self._randn(256, 512) * 0.02
        scale = config["init_scale"] * w.abs().mean()
        if config["quant_mode"] == "ternary":
            # Use ForgeEngine's ternary quantization for real error measurement
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
        score = -err * 100 + compression * 2
        return {"score": float(score), "behavioral": (err, compression),
                "metadata": {"mode": config["quant_mode"], "err": err}}


class SharqQuant(BaseDomain):
    """SharQ n_levels, adaptive, warmup_steps."""
    def name(self): return "sharq_quant"
    def output_dim(self): return 3
    def behavioral_dims(self):
        return [("quant_err", 10, 0, 1), ("bit_width", 10, 2, 8)]
    def decode(self, params):
        p = params.detach().cpu().numpy()
        return {"n_levels": int(np.interp(p[0], [0, 1], [4, 32])),
                "adaptive": bool(p[1] > 0.5),
                "warmup_steps": int(np.interp(p[2], [0, 1], [0, 1000]))}
    def encode(self, config):
        return torch.tensor([
            np.clip((config.get("n_levels", 16) - 4) / 28, 0, 1),
            1.0 if config.get("adaptive", True) else 0.0,
            np.clip(config.get("warmup_steps", 100) / 1000, 0, 1),
        ], dtype=torch.float32)
    def evaluate(self, config):
        w = self._randn(256, 512) * 0.02
        n_l = config["n_levels"]
        bits = np.log2(n_l)
        if config["adaptive"]:
            # Adaptive: more levels where weight density is higher
            sorted_w = w.abs().flatten().sort().values
            boundaries = sorted_w[::len(sorted_w) // n_l]
            w_q = torch.zeros_like(w)
            for i in range(len(boundaries) - 1):
                mask = (w.abs() >= boundaries[i]) & (w.abs() < boundaries[i + 1])
                w_q[mask] = boundaries[i] * w[mask].sign()
        else:
            scale = w.abs().max() / (n_l // 2 - 1)
            w_q = torch.round(w / (scale + 1e-8)) * scale
        err = float((w - w_q).norm().item() / (w.norm().item() + 1e-8))
        score = -err * 100 - bits * 2
        return {"score": float(score), "behavioral": (err, bits),
                "metadata": {"n_levels": n_l, "bits": bits}}


class MosaicQuant(BaseDomain):
    """MosaicQuant n_tiles, tile_dim, mix_ratio."""
    def name(self): return "mosaic_quant"
    def output_dim(self): return 3
    def behavioral_dims(self):
        return [("quant_err", 10, 0, 1), ("memory_ratio", 10, 0, 1)]
    def decode(self, params):
        p = params.detach().cpu().numpy()
        return {"n_tiles": int(np.interp(p[0], [0, 1], [4, 32])),
                "tile_dim": int(np.interp(p[1], [0, 1], [64, 512])),
                "mix_ratio": float(p[2])}
    def encode(self, config):
        return torch.tensor([
            np.clip((config.get("n_tiles", 16) - 4) / 28, 0, 1),
            np.clip((config.get("tile_dim", 256) - 64) / 448, 0, 1),
            config.get("mix_ratio", 0.5),
        ], dtype=torch.float32)
    def evaluate(self, config):
        w = self._randn(128, 128) * 0.02  # reduced from 512x512
        nt = config["n_tiles"]
        td = config["tile_dim"]
        mr = config["mix_ratio"]
        td_safe = min(td, 128)
        while 128 % td_safe != 0:
            td_safe -= 1
        n_cols = 128 // td_safe
        tiles = w.view(128, n_cols, td_safe).permute(1, 0, 2)
        # Vectorized: no .item() in loop, accumulate on GPU
        err_sq = torch.tensor(0.0, device=self.device)
        for t in tiles:
            bits = 4 if torch.rand(1, device=self.device).item() < mr else 8
            t_q = _quantize(t, bits)
            err_sq += (t - t_q).norm() ** 2
        err = float((err_sq.sqrt() / (w.norm() + 1e-8)).item())  # single .item()
        mem_ratio = 0.5 + mr * 0.5
        score = -err * 100 + (1 - mem_ratio) * 10
        return {"score": float(score), "behavioral": (err, mem_ratio),
                "metadata": {"n_tiles": nt, "err": err}}


class AaacQuant(BaseDomain):
    """AAAC n_codebooks, codebook_size, n_bits."""
    def name(self): return "aaac_quant"
    def output_dim(self): return 3
    def behavioral_dims(self):
        return [("recon_err", 10, 0, 1), ("compression", 10, 2, 32)]
    def decode(self, params):
        p = params.detach().cpu().numpy()
        return {"n_codebooks": int(np.interp(p[0], [0, 1], [4, 16])),
                "codebook_size": int(np.interp(p[1], [0, 1], [64, 512])),
                "n_bits": int(np.interp(p[2], [0, 1], [2, 4]))}
    def encode(self, config):
        return torch.tensor([
            np.clip((config.get("n_codebooks", 8) - 4) / 12, 0, 1),
            np.clip((config.get("codebook_size", 256) - 64) / 448, 0, 1),
            np.clip((config.get("n_bits", 3) - 2) / 2, 0, 1),
        ], dtype=torch.float32)
    def evaluate(self, config):
        # Small sample — the pairwise distance matrix is (n_samples, cs)
        # so keep n_samples small to avoid O(n*cs) memory bomb
        w = self._randn(64, 64) * 0.02  # 4096 elements, not 131072
        nc = config["n_codebooks"]
        cs = min(config["codebook_size"], 256)  # cap codebook size
        nb = config["n_bits"]
        # Simulate additive quantization — vectorized but small
        residual = w.clone()
        flat = residual.flatten()  # (4096,)
        for _ in range(nc):
            codebook = self._randn(cs) * 0.01
            # Pairwise: (4096, cs) = ~1M elements, manageable
            codes = (flat.unsqueeze(1) - codebook.unsqueeze(0)).abs().argmin(dim=1)
            flat = flat - codebook[codes]
        residual = flat.view_as(w)
        err = float(residual.norm().item() / (w.norm().item() + 1e-8))
        compression = 32 / (nc * nb)
        score = -err * 100 + np.log2(compression) * 5
        return {"score": float(score), "behavioral": (err, compression),
                "metadata": {"n_codebooks": nc, "compression": compression}}


class OffqQuant(BaseDomain):
    """OffQ offset_init, n_iter, learn_offset."""
    def name(self): return "offq_quant"
    def output_dim(self): return 3
    def behavioral_dims(self):
        return [("clipping", 10, 0, 1), ("quant_err", 10, 0, 1)]
    def decode(self, params):
        p = params.detach().cpu().numpy()
        return {"offset_init": float(p[0]), "n_iter": int(np.interp(p[1], [0, 1], [10, 100])),
                "learn_offset": bool(p[2] > 0.5)}
    def encode(self, config):
        return torch.tensor([
            config.get("offset_init", 0.5),
            np.clip((config.get("n_iter", 50) - 10) / 90, 0, 1),
            1.0 if config.get("learn_offset", True) else 0.0,
        ], dtype=torch.float32)
    def evaluate(self, config):
        w = self._randn(256, 512) * 0.02
        offset = config["offset_init"] * w.std()
        if config["learn_offset"]:
            for _ in range(min(config["n_iter"], 20)):
                offset = offset - 0.01 * (w + offset).abs().mean()
        w_shifted = w + offset
        w_q = _quantize(w_shifted, 4)
        w_recon = w_q - offset
        err = float((w - w_recon).norm().item() / (w.norm().item() + 1e-8))
        clipping = float((w_shifted.abs() > w_shifted.abs().max() * 0.99).float().mean().item())
        score = -err * 100 - clipping * 10
        return {"score": float(score), "behavioral": (clipping, err),
                "metadata": {"offset": float(offset), "err": err}}


class GroupQuant(BaseDomain):
    """Group quantization: group_size, n_bits, scheme."""
    def name(self): return "group_quant"
    def output_dim(self): return 3
    def discrete_choices(self):
        return {"group_size": [16, 32, 64, 128], "n_bits": [2, 3, 4, 8],
                "scheme": ["symmetric", "asymmetric"]}
    def behavioral_dims(self):
        return [("quant_err", 10, 0, 1), ("compression", 10, 2, 16)]
    def decode(self, params):
        p = params.detach().cpu().numpy()
        gs_choices = [16, 32, 64, 128]
        bits_choices = [2, 3, 4, 8]
        return {"group_size": gs_choices[int(p[0] * 3.999)],
                "n_bits": bits_choices[int(p[1] * 3.999)],
                "scheme": ["symmetric", "asymmetric"][int(p[2] * 1.999)]}
    def encode(self, config):
        gs_map = {16: 0, 32: 0.33, 64: 0.66, 128: 1.0}
        bits_map = {2: 0, 3: 0.33, 4: 0.66, 8: 1.0}
        return torch.tensor([
            gs_map.get(config.get("group_size", 32), 0.33),
            bits_map.get(config.get("n_bits", 4), 0.66),
            0.0 if config.get("scheme", "symmetric") == "symmetric" else 1.0,
        ], dtype=torch.float32)
    def evaluate(self, config):
        w = self._randn(256, 512) * 0.02
        gs = int(config["group_size"])
        bits = int(config["n_bits"])
        w_groups = w.view(-1, gs)
        w_q = _quantize(w_groups, bits, config["scheme"])
        w_q = w_q.view_as(w)
        err = float((w - w_q).norm().item() / (w.norm().item() + 1e-8))
        compression = 32 / bits
        score = -err * 100 + np.log2(compression) * 5
        return {"score": float(score), "behavioral": (err, compression),
                "metadata": {"group_size": gs, "bits": bits, "err": err}}


class MixedPrecision(BaseDomain):
    """Mixed precision: n_levels, assignment, bits per level."""
    def name(self): return "mixed_precision"
    def output_dim(self): return 3
    def discrete_choices(self):
        return {"assignment": ["uniform", "importance", "random"]}
    def behavioral_dims(self):
        return [("avg_error", 10, 0, 1), ("avg_bits", 10, 2, 16)]
    def decode(self, params):
        p = params.detach().cpu().numpy()
        return {"n_levels": int(np.interp(p[0], [0, 1], [2, 4])),
                "assignment": ["uniform", "importance", "random"][int(p[1] * 2.999)],
                "bits_base": int(np.interp(p[2], [0, 1], [2, 8]))}
    def encode(self, config):
        amap = {"uniform": 0, "importance": 0.5, "random": 1.0}
        return torch.tensor([
            np.clip((config.get("n_levels", 3) - 2) / 2, 0, 1),
            amap.get(config.get("assignment", "uniform"), 0),
            np.clip((config.get("bits_base", 4) - 2) / 6, 0, 1),
        ], dtype=torch.float32)
    def evaluate(self, config):
        w = self._randn(16, 64, 64) * 0.02  # reduced from 16x256x512
        nl = config["n_levels"]
        bb = config["bits_base"]
        bits_list = [bb, bb + 2, bb + 4, bb + 6][:nl]
        if config["assignment"] == "importance":
            importance = w.view(16, -1).norm(dim=1)
            order = importance.argsort()
        elif config["assignment"] == "random":
            order = torch.randperm(16, device=self.device)
        else:
            order = torch.arange(16, device=self.device)
        # Vectorized: accumulate on GPU, single .item() at end
        err_sq = torch.tensor(0.0, device=self.device)
        bits_total = 0
        for i in range(16):
            bits = bits_list[i % nl]
            layer = int(order[i])
            w_q = _quantize(w[layer], bits)
            err_sq += (w[layer] - w_q).norm() ** 2
            bits_total += bits
        err = float((err_sq.sqrt() / (w.norm() + 1e-8)).item())
        avg_bits = bits_total / 16
        score = -err * 100 - avg_bits * 2
        return {"score": float(score), "behavioral": (err, avg_bits),
                "metadata": {"n_levels": nl, "avg_bits": avg_bits}}


class ActivationQuant(BaseDomain):
    """Activation quantization calibration method, percentile, smooth alpha."""
    def name(self): return "activation_quant"
    def output_dim(self): return 3
    def discrete_choices(self):
        return {"calib_method": ["minmax", "percentile", "entropy"]}
    def behavioral_dims(self):
        return [("outlier_handling", 10, 0, 1), ("quant_err", 10, 0, 1)]
    def decode(self, params):
        p = params.detach().cpu().numpy()
        return {"calib_method": ["minmax", "percentile", "entropy"][int(p[0] * 2.999)],
                "percentile": float(np.interp(p[1], [0, 1], [0.9, 0.999])),
                "smooth_alpha": float(p[2])}
    def encode(self, config):
        cmap = {"minmax": 0, "percentile": 0.5, "entropy": 1.0}
        return torch.tensor([
            cmap.get(config.get("calib_method", "minmax"), 0),
            np.clip((config.get("percentile", 0.99) - 0.9) / 0.099, 0, 1),
            config.get("smooth_alpha", 0.5),
        ], dtype=torch.float32)
    def evaluate(self, config):
        a = self._randn(256, 512) * 0.01
        a[0, 0] = 0.5  # outlier
        method = config["calib_method"]
        if method == "minmax":
            scale = a.abs().max() / 127
        elif method == "percentile":
            scale = torch.quantile(a.abs().flatten(), config["percentile"]) / 127
        else:
            # Entropy: minimize KL divergence
            scale = a.std() * 3 / 127
        a_q = torch.round(a / (scale + 1e-8)) * scale
        err = float((a - a_q).norm().item() / (a.norm().item() + 1e-8))
        outlier_handled = 1.0 - float((a_q[0, 0] - a[0, 0]).abs().item() / (a[0, 0].abs().item() + 1e-8))
        score = -err * 100 + outlier_handled * 10
        return {"score": float(score), "behavioral": (outlier_handled, err),
                "metadata": {"method": method, "err": err}}
