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
                "metadata": {"final_loss": losses[-1], "opt": config["opt_type"]}}


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
        score = auc * 10 + final_decay * 5
        return {"score": float(score), "behavioral": (auc, final_decay),
                "metadata": {"sched_type": config["sched_type"], "auc": auc}}


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
        score = grad_mag * 20 + smoothness * 10
        return {"score": float(score), "behavioral": (grad_mag, smoothness),
                "metadata": {"loss": float(loss.detach()), "type": config["loss_type"]}}


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
        # Simulate FP8 range
        if config["autocast_mode"] == "e4m3":
            max_val, min_val = 448, -448
        else:
            max_val, min_val = 57344, -57344
        x = self._randn(256, 512) * 0.02
        if config["mu_scaling"]:
            x = x / (x.std() + 1e-8)  # unit variance
        if config["smooth_swiglu"]:
            x = x / (x.abs().max(dim=-1, keepdim=True).values + 1e-8) * max_val * 0.5
        overflow = float((x.abs() > max_val).float().mean().item())
        dynamic_range = float(x.abs().max().item() / (x.abs().mean().item() + 1e-8))
        score = dynamic_range * 5 - overflow * 100
        if config["loss_scale"] > 1024:
            score += 2  # higher loss scale helps small gradients
        return {"score": float(score), "behavioral": (dynamic_range / 100, overflow),
                "metadata": {"overflow": overflow, "mode": config["autocast_mode"]}}


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
        # Simulate routing: keep_fraction of tokens processed
        tokens = self._randn(128, 512)
        router = self._randn(128)
        keep_mask = router > torch.quantile(router, 1 - kf)
        compute_saved = 1.0 - kf
        # Accuracy loss: proportional to tokens skipped
        kept = tokens[keep_mask]
        acc_loss = 1.0 - float(kept.std().item() / (tokens.std().item() + 1e-8))
        if compute_saved < 0.05:
            score = -50  # no compute saved = trivial
        else:
            score = compute_saved * 20 - acc_loss * 10
        return {"score": float(score), "behavioral": (compute_saved, acc_loss),
                "metadata": {"keep_fraction": kf, "compute_saved": compute_saved}}
