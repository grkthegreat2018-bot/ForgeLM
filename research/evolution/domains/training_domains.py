"""Training-related evolution domains (8 domains).

Each domain explores a training configuration search space.
All use small tensor operations for fast evaluation (no model loading).
"""
from __future__ import annotations
import torch
import numpy as np
from typing import Any
from . import BaseDomain


class OptimizerConfig(BaseDomain):
    """Optimizer type, lr, betas, weight_decay."""
    def name(self): return "optimizer_config"
    def output_dim(self): return 5
    def discrete_choices(self):
        return {"opt_type": ["adamw", "muon", "sgd", "lion"]}
    def behavioral_dims(self):
        return [("convergence_speed", 10, 0, 1), ("stability", 10, 0, 1)]
    def decode(self, params):
        p = params.detach().cpu().numpy()
        return {"opt_type": ["adamw", "muon", "sgd", "lion"][int(p[0] * 3.999)],
                "lr": float(np.interp(p[1], [0, 1], [1e-5, 1e-2])),
                "beta1": float(np.interp(p[2], [0, 1], [0.8, 0.95])),
                "beta2": float(np.interp(p[3], [0, 1], [0.95, 0.999])),
                "weight_decay": float(np.interp(p[4], [0, 1], [0, 0.1]))}
    def encode(self, config):
        omap = {"adamw": 0, "muon": 0.33, "sgd": 0.66, "lion": 1.0}
        return torch.tensor([
            omap.get(config.get("opt_type", "adamw"), 0),
            np.clip((np.log(config.get("lr", 1e-3)) - np.log(1e-5)) / (np.log(1e-2) - np.log(1e-5)), 0, 1),
            np.clip((config.get("beta1", 0.9) - 0.8) / 0.15, 0, 1),
            np.clip((config.get("beta2", 0.999) - 0.95) / 0.049, 0, 1),
            np.clip(config.get("weight_decay", 0.01) / 0.1, 0, 1),
        ], dtype=torch.float32)
    def evaluate(self, config):
        # Simulate optimizer on quadratic loss (no autograd needed)
        x = torch.tensor([3.0, -2.0, 1.5], device=self.device)
        lr = config["lr"]
        b1, b2 = config["beta1"], config["beta2"]
        wd = config["weight_decay"]
        m = torch.zeros(3, device=self.device)
        v = torch.zeros(3, device=self.device)
        losses = torch.zeros(50, device=self.device)
        for step in range(50):
            loss = (x ** 2).sum() + wd * (x ** 2).sum()
            grad = 2 * x + 2 * wd * x
            if config["opt_type"] == "adamw":
                m = b1 * m + (1 - b1) * grad
                v = b2 * v + (1 - b2) * grad ** 2
                x = x - lr * m / (v.sqrt() + 1e-8)
            elif config["opt_type"] == "muon":
                # Muon: orthogonalize gradient (for 1D, just normalize)
                g = grad
                g_norm = g.norm() + 1e-8
                g_orth = g / g_norm  # simplified Newton-Schulz for 1D
                x = x - lr * g_orth
            elif config["opt_type"] == "lion":
                m = b1 * m + (1 - b1) * grad
                x = x - lr * torch.sign(m)
            else:  # sgd
                x = x - lr * grad
            losses[step] = loss
        conv_speed = 1.0 / (1.0 + float(losses[-1].item()))
        stability = 1.0 / (1.0 + float(losses[-10:].std().item()))
        score = conv_speed * 20 + stability * 10
        return {"score": float(score), "behavioral": (conv_speed, stability),
                "metadata": {"final_loss": float(losses[-1].item()), "opt": config["opt_type"]}}


class SchedulerConfig(BaseDomain):
    """Scheduler type, warmup, min_lr_ratio, decay_steps."""
    def name(self): return "scheduler_config"
    def output_dim(self): return 4
    def discrete_choices(self):
        return {"sched_type": ["cosine", "linear", "warmup_decay", "constant"]}
    def behavioral_dims(self):
        return [("area_under_curve", 10, 0, 1), ("final_decay", 10, 0, 1)]
    def decode(self, params):
        p = params.detach().cpu().numpy()
        return {"sched_type": ["cosine", "linear", "warmup_decay", "constant"][int(p[0] * 3.999)],
                "warmup_steps": int(np.interp(p[1], [0, 1], [0, 1000])),
                "min_lr_ratio": float(np.interp(p[2], [0, 1], [0, 0.5])),
                "decay_steps": int(np.interp(p[3], [0, 1], [100, 10000]))}
    def encode(self, config):
        smap = {"cosine": 0, "linear": 0.33, "warmup_decay": 0.66, "constant": 1.0}
        return torch.tensor([
            smap.get(config.get("sched_type", "cosine"), 0),
            np.clip(config.get("warmup_steps", 100) / 1000, 0, 1),
            np.clip(config.get("min_lr_ratio", 0.1) / 0.5, 0, 1),
            np.clip((config.get("decay_steps", 1000) - 100) / 9900, 0, 1),
        ], dtype=torch.float32)
    def evaluate(self, config):
        base_lr = 1e-3
        ws = config["warmup_steps"]
        ds = config["decay_steps"]
        mr = config["min_lr_ratio"]
        lrs = []
        for step in range(ds):
            if step < ws:
                lr = base_lr * step / max(ws, 1)
            elif config["sched_type"] == "cosine":
                progress = (step - ws) / max(ds - ws, 1)
                lr = base_lr * (mr + (1 - mr) * 0.5 * (1 + np.cos(np.pi * progress)))
            elif config["sched_type"] == "linear":
                progress = (step - ws) / max(ds - ws, 1)
                lr = base_lr * (1 - progress * (1 - mr))
            elif config["sched_type"] == "warmup_decay":
                lr = base_lr * mr ** ((step - ws) / max(ds - ws, 1))
            else:
                lr = base_lr
            lrs.append(lr)
        auc = float(np.sum(lrs) / (base_lr * ds))  # simplified AUC
        final_decay = 1.0 - lrs[-1] / base_lr
        # Training stability: zero warmup causes early-step divergence
        # (large lr on random init → gradient explosion). Penalize heavily.
        # Minimum viable warmup: ~1% of total steps. Ideal: 5-10%.
        warmup_ratio = ws / max(ds, 1)
        stability_penalty = 0.0
        if ws == 0:
            stability_penalty = 8.0  # zero warmup = high divergence risk
        elif warmup_ratio < 0.01:
            stability_penalty = 4.0  # too little warmup
        elif warmup_ratio > 0.3:
            stability_penalty = 2.0  # too much warmup wastes training steps
        # Early lr ramp smoothness: penalize sharp jumps
        if ws > 0:
            early_lr_jump = abs(lrs[1] - lrs[0]) / base_lr
            if early_lr_jump > 0.1:
                stability_penalty += early_lr_jump * 2
        score = auc * 10 + final_decay * 5 - stability_penalty
        return {"score": float(score), "behavioral": (auc, final_decay),
                "metadata": {"sched_type": config["sched_type"], "auc": auc,
                             "stability_penalty": stability_penalty}}


class LossConfig(BaseDomain):
    """Loss type, label_smoothing, focal_gamma, temperature."""
    def name(self): return "loss_config"
    def output_dim(self): return 4
    def discrete_choices(self):
        return {"loss_type": ["ce", "focal", "label_smooth", "kl"]}
    def behavioral_dims(self):
        return [("gradient_magnitude", 10, 0, 1), ("smoothness", 10, 0, 1)]
    def decode(self, params):
        p = params.detach().cpu().numpy()
        return {"loss_type": ["ce", "focal", "label_smooth", "kl"][int(p[0] * 3.999)],
                "label_smoothing": float(np.interp(p[1], [0, 1], [0, 0.3])),
                "focal_gamma": float(np.interp(p[2], [0, 1], [0, 5])),
                "temperature": float(np.interp(p[3], [0, 1], [0.5, 2.0]))}
    def encode(self, config):
        lmap = {"ce": 0, "focal": 0.33, "label_smooth": 0.66, "kl": 1.0}
        return torch.tensor([
            lmap.get(config.get("loss_type", "ce"), 0),
            np.clip(config.get("label_smoothing", 0.1) / 0.3, 0, 1),
            np.clip(config.get("focal_gamma", 2.0) / 5, 0, 1),
            np.clip((config.get("temperature", 1.0) - 0.5) / 1.5, 0, 1),
        ], dtype=torch.float32)
    def evaluate(self, config):
        logits = self._randn(32, 100) * 2
        targets = torch.randint(0, 100, (32,), device=self.device)
        temp = config["temperature"]
        ls = config["label_smoothing"]
        fg = config["focal_gamma"]
        logits_t = logits / temp
        if config["loss_type"] == "ce":
            loss = torch.nn.functional.cross_entropy(logits_t, targets)
        elif config["loss_type"] == "focal":
            ce = torch.nn.functional.cross_entropy(logits_t, targets, reduction="none")
            loss = ((1 - torch.exp(-ce)) ** fg * ce).mean()
        elif config["loss_type"] == "label_smooth":
            loss = torch.nn.functional.cross_entropy(logits_t, targets, label_smoothing=ls)
        else:  # kl
            log_p = torch.log_softmax(logits_t, dim=-1)
            p = torch.softmax(logits_t, dim=-1)
            target_dist = torch.zeros_like(p)
            target_dist[range(32), targets] = 1
            loss = (target_dist * (torch.log(target_dist + 1e-8) - log_p)).sum(dim=-1).mean()
        # Proxy for gradient magnitude: softmax sharpness (no autograd needed)
        probs = torch.softmax(logits_t, dim=-1)
        grad_mag = float((probs * (1 - probs)).mean().item())  # gradient of CE w.r.t. logits
        smoothness = 1.0 / (1.0 + float(loss.item()))
        # Training quality proxy: measure how well the loss distinguishes
        # correct from incorrect predictions. A good loss should have high
        # gradient on wrong predictions and low gradient on correct ones.
        correct_mask = probs.argmax(dim=-1) == targets
        wrong_probs = probs[~correct_mask]
        correct_probs = probs[correct_mask].gather(1, targets[correct_mask].unsqueeze(1))
        # Focus ratio: gradient should concentrate on wrong predictions
        # Use unbiased=False to avoid NaN on single-element tensors
        if len(wrong_probs) > 0 and len(correct_probs) > 1:
            wrong_grad = float(wrong_probs.var(unbiased=False).item())
            correct_grad = float(correct_probs.var(unbiased=False).item())
            focus_ratio = wrong_grad / (correct_grad + 1e-8)
        else:
            focus_ratio = 1.0
        # Penalize excessive label smoothing (>0.2 causes underfitting on SFT)
        smoothing_penalty = 0.0
        if ls > 0.2:
            smoothing_penalty = (ls - 0.2) * 30  # linear penalty above 0.2
        # Penalize extreme focal gamma (>5 causes gradient vanishing on easy examples)
        gamma_penalty = 0.0
        if fg > 5:
            gamma_penalty = (fg - 5) * 5
        # Penalize extreme temperature (>1.5 softens gradients too much)
        temp_penalty = 0.0
        if temp > 1.5:
            temp_penalty = (temp - 1.5) * 4
        score = (grad_mag * 15 + smoothness * 8 + min(focus_ratio, 3) * 3
                 - smoothing_penalty - gamma_penalty - temp_penalty)
        return {"score": float(score), "behavioral": (grad_mag, smoothness),
                "metadata": {"loss": float(loss.detach()), "type": config["loss_type"],
                             "focus_ratio": focus_ratio, "smoothing_penalty": smoothing_penalty}}


class MuonConfig(BaseDomain):
    """Muon momentum, nesterov, weight_decay, ns_steps."""
    def name(self): return "muon_config"
    def output_dim(self): return 4
    def behavioral_dims(self):
        return [("convergence", 10, 0, 1), ("orthogonality", 10, 0, 1)]
    def decode(self, params):
        p = params.detach().cpu().numpy()
        return {"momentum": float(np.interp(p[0], [0, 1], [0.85, 0.99])),
                "nesterov": bool(p[1] > 0.5),
                "weight_decay": float(np.interp(p[2], [0, 1], [0, 0.01])),
                "ns_steps": int(np.interp(p[3], [0, 1], [1, 6]))}
    def encode(self, config):
        return torch.tensor([
            np.clip((config.get("momentum", 0.95) - 0.85) / 0.14, 0, 1),
            1.0 if config.get("nesterov", True) else 0.0,
            np.clip(config.get("weight_decay", 0.0) / 0.01, 0, 1),
            np.clip((config.get("ns_steps", 3) - 1) / 5, 0, 1),
        ], dtype=torch.float32)
    def evaluate(self, config):
        x = torch.tensor([3.0, -2.0, 1.5, -1.0, 0.5], device=self.device)
        mom = config["momentum"]
        wd = config["weight_decay"]
        ns = min(config["ns_steps"], 3)  # cap at 3 Newton-Schulz steps
        buf = torch.zeros(5, device=self.device)
        final_loss = torch.tensor(0.0, device=self.device)
        for _ in range(10):  # reduced from 50 → 20 → 10
            loss = (x ** 2).sum()
            grad = 2 * x + 2 * wd * x
            buf = mom * buf + grad
            g = buf
            for _ in range(ns):
                g_norm = g.norm() + 1e-8
                g = g * (3 / g_norm) - g * (g @ g) / (g_norm ** 3)
            x = x - 0.01 * g
            final_loss = loss
        conv = 1.0 / (1.0 + float(final_loss.item()))
        orth = 1.0 / (1.0 + abs(float((buf @ buf).item()) - float((buf.norm() ** 2).item())))
        score = conv * 20 + orth * 10
        return {"score": float(score), "behavioral": (conv, orth),
                "metadata": {"final_loss": float(final_loss.item()), "ns_steps": ns}}


class CpuAdamwConfig(BaseDomain):
    """CPU offload ratio, prefetch depth, compression, update_freq."""
    def name(self): return "cpu_adamw_config"
    def output_dim(self): return 4
    def discrete_choices(self):
        return {"compression": ["none", "int8", "int4"]}
    def behavioral_dims(self):
        return [("throughput", 10, 0, 1), ("latency_ms", 10, 0, 100)]
    def decode(self, params):
        p = params.detach().cpu().numpy()
        return {"offload_ratio": float(p[0]),
                "prefetch_depth": int(np.interp(p[1], [0, 1], [1, 8])),
                "compression": ["none", "int8", "int4"][int(p[2] * 2.999)],
                "update_freq": int(np.interp(p[3], [0, 1], [1, 16]))}
    def encode(self, config):
        cmap = {"none": 0, "int8": 0.5, "int4": 1.0}
        return torch.tensor([
            config.get("offload_ratio", 0.5),
            np.clip((config.get("prefetch_depth", 4) - 1) / 7, 0, 1),
            cmap.get(config.get("compression", "none"), 0),
            np.clip((config.get("update_freq", 4) - 1) / 15, 0, 1),
        ], dtype=torch.float32)
    def evaluate(self, config):
        oratio = config["offload_ratio"]
        pd = config["prefetch_depth"]
        comp = config["compression"]
        uf = config["update_freq"]
        # Model: throughput = (1 - offload_ratio) * base + prefetch * overlap
        base_throughput = 1000  # tok/s
        comp_factor = {"none": 1.0, "int8": 2.0, "int4": 3.0}[comp]
        throughput = base_throughput * (1 - oratio * 0.5) * comp_factor
        latency = oratio * 50 / max(pd, 1) * (1 / comp_factor) * uf
        score = throughput / 100 - latency / 10
        return {"score": float(score), "behavioral": (throughput / 1000, latency),
                "metadata": {"throughput": throughput, "latency": latency}}


class GradAccumConfig(BaseDomain):
    """Gradient accumulation steps, micro_batch, grad_clip, sync_freq."""
    def name(self): return "grad_accum_config"
    def output_dim(self): return 4
    def behavioral_dims(self):
        return [("effective_batch", 10, 0, 1), ("noise", 10, 0, 1)]
    def decode(self, params):
        p = params.detach().cpu().numpy()
        return {"accum_steps": int(np.interp(p[0], [0, 1], [1, 32])),
                "micro_batch": int(np.interp(p[1], [0, 1], [1, 16])),
                "grad_clip": float(np.interp(p[2], [0, 1], [0, 1.0])),
                "sync_freq": int(np.interp(p[3], [0, 1], [1, 16]))}
    def encode(self, config):
        return torch.tensor([
            np.clip((config.get("accum_steps", 8) - 1) / 31, 0, 1),
            np.clip((config.get("micro_batch", 4) - 1) / 15, 0, 1),
            np.clip(config.get("grad_clip", 1.0), 0, 1),
            np.clip((config.get("sync_freq", 4) - 1) / 15, 0, 1),
        ], dtype=torch.float32)
    def evaluate(self, config):
        as_ = config["accum_steps"]
        mb = config["micro_batch"]
        gc = config["grad_clip"]
        eff_batch = as_ * mb
        # Noise: gradient noise scales as 1/sqrt(eff_batch)
        noise = 1.0 / (eff_batch ** 0.5)
        # Throughput: larger micro_batch = better GPU utilization
        throughput = mb / (mb + as_ * 0.1)
        score = throughput * 10 - noise * 5
        if gc > 0:
            score += gc * 2  # stability bonus
        return {"score": float(score), "behavioral": (eff_batch / 512, noise),
                "metadata": {"effective_batch": eff_batch, "noise": noise}}


class Fp8TrainingConfig(BaseDomain):
    """FP8 autocast mode, smooth_swiglu, mu_scaling, loss_scale."""
    def name(self): return "fp8_training_config"
    def output_dim(self): return 4
    def discrete_choices(self):
        return {"autocast_mode": ["e4m3", "e5m2"]}
    def behavioral_dims(self):
        return [("dynamic_range", 10, 0, 1), ("overflow_risk", 10, 0, 1)]
    def decode(self, params):
        p = params.detach().cpu().numpy()
        return {"autocast_mode": ["e4m3", "e5m2"][int(p[0] * 1.999)],
                "smooth_swiglu": bool(p[1] > 0.5),
                "mu_scaling": bool(p[2] > 0.5),
                "loss_scale": float(np.interp(p[3], [0, 1], [128, 4096]))}
    def encode(self, config):
        return torch.tensor([
            0.0 if config.get("autocast_mode", "e4m3") == "e4m3" else 1.0,
            1.0 if config.get("smooth_swiglu", True) else 0.0,
            1.0 if config.get("mu_scaling", True) else 0.0,
            np.clip((np.log2(config.get("loss_scale", 1024)) - 7) / 5, 0, 1),
        ], dtype=torch.float32)
    def evaluate(self, config):
        # Simulate FP8 range + precision
        if config["autocast_mode"] == "e4m3":
            max_val = 448
            # E4M3: 3 mantissa bits → 8 representable values per octave
            mantissa_bits = 3
        else:
            max_val = 57344
            # E5M2: 2 mantissa bits → 4 representable values per octave (LESS PRECISE)
            mantissa_bits = 2
        x = self._randn(256, 512) * 0.02
        if config["mu_scaling"]:
            x = x / (x.std() + 1e-8)  # unit variance
        if config["smooth_swiglu"]:
            x = x / (x.abs().max(dim=-1, keepdim=True).values + 1e-8) * max_val * 0.5
        overflow = float((x.abs() > max_val).float().mean().item())
        dynamic_range = float(x.abs().max().item() / (x.abs().mean().item() + 1e-8))
        # Quantization error: simulate FP8 rounding with correct mantissa bits.
        # E5M2 has 2x the quantization error of E4M3 due to fewer mantissa bits.
        # This is the key metric the old scoring missed — e5m2's larger range
        # doesn't compensate for its worse precision.
        x_clamped = x.clamp(-max_val, max_val)
        scale = x_clamped.abs().max() / max_val if x_clamped.abs().max() > 0 else torch.tensor(1.0, device=self.device)
        x_scaled = x_clamped / scale
        # Simulate FP8 quantization: round to nearest representable value
        # Number of levels per sign = 2^mantissa_bits
        n_levels = 2 ** mantissa_bits
        x_quantized = torch.round(x_scaled * n_levels) / n_levels
        x_dequantized = x_quantized * scale
        quant_err = float((x - x_dequantized).norm().item() / (x.norm().item() + 1e-8))
        # Score: balance dynamic range (good for avoiding overflow) against
        # quantization error (bad for training quality). E4M3 wins when
        # overflow is already handled by mu_scaling/smooth_swiglu.
        score = dynamic_range * 3 - overflow * 100 - quant_err * 30
        if config["mu_scaling"]:
            score += 3  # mu_scaling helps both modes equally
        if config["smooth_swiglu"]:
            score += 2  # smooth_swiglu prevents SiLU outliers
        if config["loss_scale"] > 1024:
            score += 2  # higher loss scale helps small gradients
        return {"score": float(score), "behavioral": (dynamic_range / 100, overflow),
                "metadata": {"overflow": overflow, "mode": config["autocast_mode"],
                             "quant_err": quant_err, "mantissa_bits": mantissa_bits}}


class ModConfig(BaseDomain):
    """MoD keep_fraction, router_type, aux_loss_weight, n_skip_layers."""
    def name(self): return "mod_config"
    def output_dim(self): return 4
    def discrete_choices(self):
        return {"router_type": ["linear", "mlp"]}
    def behavioral_dims(self):
        return [("compute_saved", 10, 0, 1), ("accuracy_loss", 10, 0, 1)]
    def decode(self, params):
        p = params.detach().cpu().numpy()
        return {"keep_fraction": float(np.interp(p[0], [0, 1], [0.5, 1.0])),
                "router_type": ["linear", "mlp"][int(p[1] * 1.999)],
                "aux_loss_weight": float(np.interp(p[2], [0, 1], [0, 0.1])),
                "n_skip_layers": int(np.interp(p[3], [0, 1], [0, 16]))}
    def encode(self, config):
        return torch.tensor([
            np.clip((config.get("keep_fraction", 0.8) - 0.5) / 0.5, 0, 1),
            0.0 if config.get("router_type", "linear") == "linear" else 1.0,
            np.clip(config.get("aux_loss_weight", 0.01) / 0.1, 0, 1),
            np.clip(config.get("n_skip_layers", 8) / 16, 0, 1),
        ], dtype=torch.float32)
    def evaluate(self, config):
        kf = config["keep_fraction"]
        nsl = config["n_skip_layers"]
        alw = config["aux_loss_weight"]
        rt = config["router_type"]
        # Simulate routing: keep_fraction of tokens processed
        tokens = self._randn(128, 512)
        router = self._randn(128)
        keep_mask = router > torch.quantile(router, 1 - kf)
        compute_saved = 1.0 - kf
        # Accuracy loss: proportional to tokens skipped
        kept = tokens[keep_mask]
        acc_loss = 1.0 - float(kept.std().item() / (tokens.std().item() + 1e-8))
        # Router type: MLP router is more expressive but adds params
        router_quality = 1.0 if rt == "mlp" else 0.85  # linear is simpler but less accurate
        # n_skip_layers: skipping too many layers degrades quality
        # 0 = no skip (safe), 1-4 = mild skip (good), >8 = aggressive (risky)
        skip_penalty = 0.0
        if nsl > 8:
            skip_penalty = (nsl - 8) * 0.5  # aggressive skipping
        # aux_loss_weight: too high destabilizes training, too low → router collapse
        # Sweet spot is near-zero (1e-8 to 1e-4)
        aux_penalty = 0.0
        if alw > 0.01:
            aux_penalty = (alw - 0.01) * 50  # high aux loss hurts main loss
        elif alw < 1e-10:
            aux_penalty = 1.0  # too low → router may collapse
        if compute_saved < 0.05:
            score = -50  # no compute saved = trivial
        else:
            score = (compute_saved * 20 - acc_loss * 10) * router_quality
            score -= skip_penalty + aux_penalty
        return {"score": float(score), "behavioral": (compute_saved, acc_loss),
                "metadata": {"keep_fraction": kf, "compute_saved": compute_saved,
                             "router_type": rt, "n_skip_layers": nsl,
                             "skip_penalty": skip_penalty, "aux_penalty": aux_penalty}}


class ApolloConfig(BaseDomain):
    """APOLLO optimizer hyperparameters: rank, scale_mode, lr_scale."""
    def name(self): return "apollo_config"
    def output_dim(self): return 3
    def discrete_choices(self):
        return {"scale": ["tensor", "channel"]}
    def behavioral_dims(self):
        return [("memory_savings", 10, 0, 1), ("convergence_quality", 10, 0, 1)]
    def decode(self, params):
        p = params.detach().cpu().numpy()
        return {"rank": int(np.interp(p[0], [0, 1], [1, 64])),
                "scale": ["tensor", "channel"][int(p[1] * 1.999)],
                "lr_scale": float(np.interp(p[2], [0, 1], [0.5, 10.0]))}
    def encode(self, config):
        smap = {"tensor": 0.0, "channel": 1.0}
        return torch.tensor([
            np.clip((config.get("rank", 8) - 1) / 63, 0, 1),
            smap.get(config.get("scale", "tensor"), 0),
            np.clip((config.get("lr_scale", 2.0) - 0.5) / 9.5, 0, 1),
        ], dtype=torch.float32)
    def evaluate(self, config):
        rank = config["rank"]
        scale = config["scale"]
        lr_scale = config["lr_scale"]
        cols = 4096
        # Simulate APOLLO convergence on a quadratic loss
        x = torch.tensor([3.0, -2.0, 1.5, -1.0, 0.5], device=self.device)
        lr = 0.01 * lr_scale
        # APOLLO approximates the Hessian with a low-rank projector of rank `rank`
        # (simplified: scale gradient by a factor derived from rank)
        proj_factor = min(rank / 16.0, 1.0)  # sweet spot ~8-16
        losses = torch.zeros(30, device=self.device)
        for step in range(30):
            loss = (x ** 2).sum()
            grad = 2 * x
            # APOLLO preconditioner: low-rank approximation rescales gradient
            x = x - lr * grad * (0.5 + proj_factor)
            losses[step] = loss
        conv_quality = 1.0 / (1.0 + float(losses[-1].item()))
        # memory_savings: lower rank = less optimizer state
        memory_savings = (8 - 2 * rank / cols) / 8
        memory_savings = float(np.clip(memory_savings, 0, 1))
        # rank_penalty: rank > 32 has diminishing returns
        rank_penalty = 0.0
        if rank > 32:
            rank_penalty = (rank - 32) * 0.3
        # rank=1 (APOLLO-Mini) gets a bonus for memory efficiency
        if rank == 1:
            rank_penalty -= 2.0
        # scale_bonus: channel scaling is more expressive
        scale_bonus = 2.0 if scale == "channel" else 0.0
        # Trivial solution guard: rank=64 + channel + high lr ≈ just AdamW
        trivial_penalty = 0.0
        if rank >= 64 and scale == "channel" and lr_scale > 8.0:
            trivial_penalty = 15.0
        score = (memory_savings * 20 + conv_quality * 10 + scale_bonus
                 - rank_penalty - trivial_penalty)
        return {"score": float(score), "behavioral": (memory_savings, conv_quality),
                "metadata": {"rank": rank, "scale": scale, "lr_scale": lr_scale,
                             "memory_savings": memory_savings,
                             "conv_quality": conv_quality,
                             "rank_penalty": rank_penalty,
                             "trivial_penalty": trivial_penalty}}


class BreadConfig(BaseDomain):
    """BREAD landscape correction for BAdam: correction_mode, sgd_lr_scale."""
    def name(self): return "bread_config"
    def output_dim(self): return 2
    def discrete_choices(self):
        return {"correction_mode": ["disabled", "partial", "all"]}
    def behavioral_dims(self):
        return [("correction_quality", 10, 0, 1), ("stability", 10, 0, 1)]
    def decode(self, params):
        p = params.detach().cpu().numpy()
        return {"correction_mode": ["disabled", "partial", "all"][int(p[0] * 2.999)],
                "sgd_lr_scale": float(np.interp(p[1], [0, 1], [1.0, 10.0]))}
    def encode(self, config):
        mmap = {"disabled": 0.0, "partial": 0.5, "all": 1.0}
        return torch.tensor([
            mmap.get(config.get("correction_mode", "partial"), 0.5),
            np.clip((config.get("sgd_lr_scale", 5.0) - 1.0) / 9.0, 0, 1),
        ], dtype=torch.float32)
    def evaluate(self, config):
        mode = config["correction_mode"]
        sgd_lr_scale = config["sgd_lr_scale"]
        # Simulate BAdam block-wise optimization with landscape correction
        x = torch.tensor([3.0, -2.0, 1.5, -1.0, 0.5], device=self.device)
        lr = 0.01 * sgd_lr_scale
        losses = torch.zeros(20, device=self.device)
        for step in range(20):
            loss = (x ** 2).sum()
            grad = 2 * x
            x = x - lr * grad
            losses[step] = loss
        stability = 1.0 / (1.0 + float(losses[-10:].std().item()))
        # mode_bonus: "partial" is the best mode
        mode_bonus = 0.0
        if mode == "disabled":
            mode_bonus = 0.0  # vanilla BAdam baseline
        elif mode == "partial":
            mode_bonus = 15.0  # corrects visited blocks only (best)
        else:  # "all"
            mode_bonus = -5.0  # corrects all blocks including never-visited (risky)
        # sgd_lr_scale sweet spot at 5x
        lr_scale_quality = 0.0
        if sgd_lr_scale < 2.0:
            lr_scale_quality = -(2.0 - sgd_lr_scale) * 3  # weak correction
        elif sgd_lr_scale > 8.0:
            lr_scale_quality = -(sgd_lr_scale - 8.0) * 3  # destabilizing
        else:
            # peak at 5x, gentle falloff
            lr_scale_quality = 5.0 - abs(sgd_lr_scale - 5.0) * 0.5
        score = mode_bonus + lr_scale_quality
        return {"score": float(score), "behavioral": (mode_bonus / 15.0, stability),
                "metadata": {"correction_mode": mode, "sgd_lr_scale": sgd_lr_scale,
                             "mode_bonus": mode_bonus,
                             "lr_scale_quality": lr_scale_quality}}


class FlashOptimConfig(BaseDomain):
    """FlashOptim companded optimizer states: bits, companding_mode."""
    def name(self): return "flashoptim_config"
    def output_dim(self): return 2
    def discrete_choices(self):
        return {"bits": [4, 8], "companding": ["sqrt", "log", "linear"]}
    def behavioral_dims(self):
        return [("memory_savings", 10, 0, 1), ("precision", 10, 0, 1)]
    def decode(self, params):
        p = params.detach().cpu().numpy()
        return {"bits": [4, 8][int(p[0] * 1.999)],
                "companding": ["sqrt", "log", "linear"][int(p[1] * 2.999)]}
    def encode(self, config):
        bmap = {4: 0.0, 8: 1.0}
        cmap = {"sqrt": 0.0, "log": 0.5, "linear": 1.0}
        return torch.tensor([
            bmap.get(config.get("bits", 8), 1.0),
            cmap.get(config.get("companding", "sqrt"), 0),
        ], dtype=torch.float32)
    def evaluate(self, config):
        bits = config["bits"]
        companding = config["companding"]
        # Simulate optimizer state quantization with companding
        state = self._randn(256, 512) * 0.1
        if companding == "sqrt":
            transformed = torch.sign(state) * torch.sqrt(state.abs())
        elif companding == "log":
            transformed = torch.sign(state) * torch.log1p(state.abs())
        else:  # linear
            transformed = state
        # Quantize to `bits` precision
        n_levels = 2 ** bits
        max_val = transformed.abs().max().item() + 1e-8
        scale = max_val / n_levels
        quantized = torch.round(transformed / scale) * scale
        # Inverse transform
        if companding == "sqrt":
            dequantized = torch.sign(quantized) * (quantized ** 2)
        elif companding == "log":
            dequantized = torch.sign(quantized) * (torch.expm1(quantized.abs()))
        else:
            dequantized = quantized
        quant_err = float((state - dequantized).norm().item() / (state.norm().item() + 1e-8))
        precision = 1.0 / (1.0 + quant_err * 10)
        # memory_savings: 8-bit = 75% (2 bytes vs 8), 4-bit = 87.5% (1 byte vs 8)
        memory_savings = 0.75 if bits == 8 else 0.875
        # companding_bonus: sqrt best for small values, log good, linear baseline
        companding_bonus = 5.0 if companding == "sqrt" else (3.0 if companding == "log" else 0.0)
        # 4-bit penalty: more aggressive, may lose precision on large models
        bit_penalty = -10.0 if bits == 4 else 0.0
        score = memory_savings * 30 + companding_bonus + bit_penalty
        return {"score": float(score), "behavioral": (memory_savings, precision),
                "metadata": {"bits": bits, "companding": companding,
                             "memory_savings": memory_savings,
                             "quant_err": quant_err,
                             "companding_bonus": companding_bonus,
                             "bit_penalty": bit_penalty}}


class TritonKernelConfig(BaseDomain):
    """Triton fused kernel block sizes: rms_block_size, swiglu_block_size."""
    def name(self): return "triton_kernel_config"
    def output_dim(self): return 2
    def behavioral_dims(self):
        return [("rms_match", 10, 0, 1), ("swiglu_match", 10, 0, 1)]
    def decode(self, params):
        p = params.detach().cpu().numpy()
        return {"rms_block_size": int(2 ** int(np.interp(p[0], [0, 1], [8, 16]))),
                "swiglu_block_size": int(2 ** int(np.interp(p[1], [0, 1], [8, 16])))}
    def encode(self, config):
        return torch.tensor([
            np.clip((np.log2(config.get("rms_block_size", 4096)) - 8) / 8, 0, 1),
            np.clip((np.log2(config.get("swiglu_block_size", 16384)) - 8) / 8, 0, 1),
        ], dtype=torch.float32)
    def evaluate(self, config):
        rms_block = config["rms_block_size"]
        swiglu_block = config["swiglu_block_size"]
        # V7: d_model=4096 (rms), intermediate=16384 (swiglu)
        rms_target = 4096
        swiglu_target = 16384
        # Simulate fused kernel throughput
        x = self._randn(4096)
        # Base speedup from fused kernels (Liger-Kernel ~20%)
        base_speedup = 20.0
        # mismatch penalty: block < feature = multiple passes (slower)
        # block > feature = wasted parallelism (minor penalty)
        rms_mismatch = 0.0
        if rms_block < rms_target:
            rms_mismatch = (rms_target / rms_block) * 3  # multiple passes
        elif rms_block > rms_target:
            rms_mismatch = (rms_block / rms_target) * 0.5  # wasted parallelism
        swiglu_mismatch = 0.0
        if swiglu_block < swiglu_target:
            swiglu_mismatch = (swiglu_target / swiglu_block) * 3
        elif swiglu_block > swiglu_target:
            swiglu_mismatch = (swiglu_block / swiglu_target) * 0.5
        mismatch_penalty = rms_mismatch + swiglu_mismatch
        score = base_speedup - mismatch_penalty
        rms_match = 1.0 / (1.0 + rms_mismatch)
        swiglu_match = 1.0 / (1.0 + swiglu_mismatch)
        return {"score": float(score), "behavioral": (rms_match, swiglu_match),
                "metadata": {"rms_block_size": rms_block,
                             "swiglu_block_size": swiglu_block,
                             "rms_target": rms_target,
                             "swiglu_target": swiglu_target,
                             "mismatch_penalty": mismatch_penalty}}


class VarlenConfig(BaseDomain):
    """Varlen attention for packed sequences: use_varlen boolean."""
    def name(self): return "varlen_config"
    def output_dim(self): return 1
    def behavioral_dims(self):
        return [("speedup", 10, 0, 1), ("vram_savings", 10, 0, 1)]
    def decode(self, params):
        p = params.detach().cpu().numpy()
        return {"use_varlen": bool(p[0] > 0.5)}
    def encode(self, config):
        return torch.tensor([1.0 if config.get("use_varlen", True) else 0.0],
                            dtype=torch.float32)
    def evaluate(self, config):
        use_varlen = config["use_varlen"]
        # Simulate packed-sequence attention
        x = self._randn(128, 64)
        if use_varlen:
            # Varlen: 2.1x faster, 50% less VRAM, no cross-example contamination
            speedup = 2.1
            vram_savings = 0.5
            contamination = 0.0
            score = 25.0
        else:
            # Baseline: padding, cross-example contamination risk
            speedup = 1.0
            vram_savings = 0.0
            contamination = 1.0
            score = 0.0
        return {"score": float(score), "behavioral": (speedup / 3.0, vram_savings),
                "metadata": {"use_varlen": use_varlen, "speedup": speedup,
                             "vram_savings": vram_savings,
                             "contamination": contamination}}
