"""HqeKV hybrid quantization domain: search tiered KV cache quantization.

Search space (5 params):
  - budget_full: [0.05, 0.25] (continuous) — fraction at bf16
  - budget_int8: [0.1, 0.5] (continuous) — fraction at INT8
  - budget_int4: [0.2, 0.7] (continuous) — fraction at INT4
  - group_size: [32, 128] (continuous) — INT4 group size
  - recency_decay: [0.8, 1.0] (continuous) — importance recency weighting

Constraint: budget_full + budget_int8 + budget_int4 <= 1.0 (rest evicted)

Evaluation: simulate tiered quantization on synthetic KV, measure reconstruction
error + memory savings. Score = -error * 100 + compression * 5.
"""
from __future__ import annotations

import torch
import numpy as np
import time
from typing import Any
from . import BaseDomain
from .kv_utils import (
    generate_synthetic_kv, generate_synthetic_q,
    full_attention_output, measure_speed,
)


def _quantize_int8(x: torch.Tensor, group_size: int = 64) -> torch.Tensor:
    """Quantize tensor to INT8 with per-group scaling, return dequantized."""
    orig_shape = x.shape
    x_flat = x.reshape(-1, group_size)
    scale = x_flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    x_q = (x_flat / scale).round().clamp(-127, 127).to(torch.int8)
    x_dq = x_q.to(torch.float32) * scale
    return x_dq.reshape(orig_shape)


def _quantize_int4(x: torch.Tensor, group_size: int = 64) -> torch.Tensor:
    """Quantize tensor to INT4 with per-group scaling, return dequantized."""
    orig_shape = x.shape
    x_flat = x.reshape(-1, group_size)
    scale = x_flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 7.0
    x_q = (x_flat / scale).round().clamp(-7, 7).to(torch.int8)
    x_dq = x_q.to(torch.float32) * scale
    return x_dq.reshape(orig_shape)


class HqeKVDomain(BaseDomain):
    """Search HqeKV tiered quantization parameters."""

    N_KV_HEADS = 8
    HEAD_DIM = 64
    N_HEADS = 32

    def __init__(self, seq_len: int = 4096, seed: int = 42,
                 device: torch.device = None):
        self.seq_len = seq_len
        self.device = device or (torch.device("cuda") if torch.cuda.is_available()
                                 else torch.device("cpu"))

        self.k_full, self.v_full = generate_synthetic_kv(
            seq_len=seq_len, n_kv_heads=self.N_KV_HEADS,
            head_dim=self.HEAD_DIM, device=self.device, seed=seed,
        )
        self.q = generate_synthetic_q(
            q_len=1, n_heads=self.N_HEADS, head_dim=self.HEAD_DIM,
            device=self.device, seed=seed + 1,
        )

        with torch.no_grad():
            self.y_ref = full_attention_output(
                self.q, self.k_full, self.v_full, self.N_KV_HEADS,
            )

    def name(self) -> str:
        return "hqe_kv"

    def output_dim(self) -> int:
        return 5

    def decode(self, params: torch.Tensor) -> dict[str, Any]:
        p = params.detach().cpu().numpy()
        # Normalize budgets to sum <= 1.0
        bf = 0.05 + p[0] * 0.20       # [0.05, 0.25]
        i8 = 0.1 + p[1] * 0.40        # [0.1, 0.5]
        i4 = 0.2 + p[2] * 0.50        # [0.2, 0.7]
        total = bf + i8 + i4
        if total > 0.95:
            # Scale down proportionally
            scale = 0.95 / total
            bf, i8, i4 = bf * scale, i8 * scale, i4 * scale

        return {
            "budget_full": float(bf),
            "budget_int8": float(i8),
            "budget_int4": float(i4),
            "group_size": int(32 + p[3] * 96),    # [32, 128]
            "recency_decay": float(0.8 + p[4] * 0.2),  # [0.8, 1.0]
        }

    def encode(self, config: dict[str, Any]) -> torch.Tensor:
        return torch.tensor([
            (config.get("budget_full", 0.1) - 0.05) / 0.20,
            (config.get("budget_int8", 0.3) - 0.1) / 0.40,
            (config.get("budget_int4", 0.5) - 0.2) / 0.50,
            (config.get("group_size", 64) - 32) / 96,
            (config.get("recency_decay", 0.95) - 0.8) / 0.2,
        ], dtype=torch.float32)

    def evaluate(self, config: dict[str, Any]) -> dict:
        budget_full = config.get("budget_full", 0.1)
        budget_int8 = config.get("budget_int8", 0.3)
        budget_int4 = config.get("budget_int4", 0.5)
        group_size = config.get("group_size", 64)
        recency_decay = config.get("recency_decay", 0.95)

        try:
            seq_len = self.seq_len
            n_full = int(seq_len * budget_full)
            n_int8 = int(seq_len * budget_int8)
            n_int4 = int(seq_len * budget_int4)

            # Importance scoring: ||K|| * ||V|| * recency
            k_norms = self.k_full.norm(dim=-1).squeeze()  # (n_kv, seq_len)
            v_norms = self.v_full.norm(dim=-1).squeeze()
            importance = (k_norms * v_norms).mean(dim=0)  # (seq_len,)
            # Recency weighting
            pos_weights = torch.linspace(recency_decay, 1.0, seq_len,
                                         device=self.device)
            importance = importance * pos_weights

            # Sort by importance (descending)
            sorted_idx = importance.argsort(descending=True)

            # Assign tiers
            full_idx = sorted_idx[:n_full]
            int8_idx = sorted_idx[n_full:n_full + n_int8]
            int4_idx = sorted_idx[n_full + n_int8:n_full + n_int8 + n_int4]
            evicted_idx = sorted_idx[n_full + n_int8 + n_int4:]

            # Build compressed K, V
            k_comp = torch.zeros_like(self.k_full)
            v_comp = torch.zeros_like(self.v_full)

            # Full precision tier
            if len(full_idx) > 0:
                k_comp[:, :, full_idx, :] = self.k_full[:, :, full_idx, :]
                v_comp[:, :, full_idx, :] = self.v_full[:, :, full_idx, :]

            # INT8 tier
            if len(int8_idx) > 0:
                k_int8 = _quantize_int8(self.k_full[:, :, int8_idx, :], group_size)
                v_int8 = _quantize_int8(self.v_full[:, :, int8_idx, :], group_size)
                k_comp[:, :, int8_idx, :] = k_int8
                v_comp[:, :, int8_idx, :] = v_int8

            # INT4 tier
            if len(int4_idx) > 0:
                k_int4 = _quantize_int4(self.k_full[:, :, int4_idx, :], group_size)
                v_int4 = _quantize_int4(self.v_full[:, :, int4_idx, :], group_size)
                k_comp[:, :, int4_idx, :] = k_int4
                v_comp[:, :, int4_idx, :] = v_int4

            # Evicted tier: zeros (will be ignored by attention)

            # Reconstruction error
            with torch.no_grad():
                y_comp = full_attention_output(self.q, k_comp, v_comp, self.N_KV_HEADS)
                fwd_err = (self.y_ref - y_comp).norm().item() / self.y_ref.norm().item()

            # Memory savings (bytes)
            full_bytes = seq_len * self.N_KV_HEADS * self.HEAD_DIM * 2 * 2  # K+V bf16
            comp_bytes = (
                n_full * self.N_KV_HEADS * self.HEAD_DIM * 2 * 2 +     # bf16 K+V
                n_int8 * self.N_KV_HEADS * self.HEAD_DIM * 1 * 2 +     # int8 K+V
                n_int4 * self.N_KV_HEADS * self.HEAD_DIM * 0.5 * 2     # int4 K+V (4-bit)
            )
            compression = full_bytes / max(comp_bytes, 1)

            # Speed
            def _run_quant():
                k_c = torch.zeros_like(self.k_full)
                v_c = torch.zeros_like(self.v_full)
                if len(full_idx) > 0:
                    k_c[:, :, full_idx, :] = self.k_full[:, :, full_idx, :]
                    v_c[:, :, full_idx, :] = self.v_full[:, :, full_idx, :]
                if len(int8_idx) > 0:
                    k_c[:, :, int8_idx, :] = _quantize_int8(self.k_full[:, :, int8_idx, :], group_size)
                    v_c[:, :, int8_idx, :] = _quantize_int8(self.v_full[:, :, int8_idx, :], group_size)
                if len(int4_idx) > 0:
                    k_c[:, :, int4_idx, :] = _quantize_int4(self.k_full[:, :, int4_idx, :], group_size)
                    v_c[:, :, int4_idx, :] = _quantize_int4(self.v_full[:, :, int4_idx, :], group_size)
                return k_c, v_c

            # Speed estimate (no benchmarking — use compression as proxy)
            quant_ms = compression * 0.3

            # Score: reward compression + low error
            # Penalize trivial configs (compression < 1.5x)
            if compression < 1.5:
                score = -fwd_err * 100 - 50
            else:
                score = -fwd_err * 100 + compression * 5 - quant_ms * 0.01

            return {
                "score": float(score),
                "behavioral": (compression, fwd_err),
                "metadata": {
                    "fwd_err": fwd_err,
                    "compression": compression,
                    "quant_ms": quant_ms,
                    "n_full": n_full, "n_int8": n_int8,
                    "n_int4": n_int4, "n_evicted": len(evicted_idx),
                    "group_size": group_size,
                    "recency_decay": recency_decay,
                },
            }

        except Exception as e:
            return {
                "score": -1000.0,
                "behavioral": (1.0, 1.0),
                "metadata": {"error": str(e)},
            }

    def behavioral_dims(self) -> list[tuple[str, int, float, float]]:
        return [
            ("compression", 10, 1.0, 8.0),
            ("fwd_error", 10, 0.0, 0.3),
        ]

    def discrete_choices(self) -> dict[str, list] | None:
        return {
            "budget_full": [0.05, 0.10, 0.15, 0.20, 0.25],
            "budget_int8": [0.10, 0.20, 0.30, 0.40, 0.50],
            "budget_int4": [0.20, 0.35, 0.50, 0.65, 0.70],
            "group_size": [32, 64, 96, 128],
            "recency_decay": [0.80, 0.85, 0.90, 0.95, 1.0],
        }

    def seed_configs(self) -> list[dict[str, Any]]:
        seeds = []
        # Default
        seeds.append({"budget_full": 0.1, "budget_int8": 0.3, "budget_int4": 0.5,
                      "group_size": 64, "recency_decay": 0.95})
        # Aggressive quant
        seeds.append({"budget_full": 0.05, "budget_int8": 0.15, "budget_int4": 0.70,
                      "group_size": 32, "recency_decay": 0.90})
        # Conservative
        seeds.append({"budget_full": 0.20, "budget_int8": 0.40, "budget_int4": 0.30,
                      "group_size": 128, "recency_decay": 1.0})
        # No INT4 (INT8 only)
        seeds.append({"budget_full": 0.15, "budget_int8": 0.70, "budget_int4": 0.05,
                      "group_size": 64, "recency_decay": 0.95})
        # All INT4
        seeds.append({"budget_full": 0.05, "budget_int8": 0.10, "budget_int4": 0.75,
                      "group_size": 32, "recency_decay": 0.85})
        # Group size sweep
        for gs in [32, 64, 128]:
            seeds.append({"budget_full": 0.10, "budget_int8": 0.30, "budget_int4": 0.50,
                          "group_size": gs, "recency_decay": 0.95})
        return seeds

    def to_cpu(self) -> "HqeKVDomain":
        """Create CPU copy for parallel evaluation."""
        return HqeKVDomain(seq_len=self.seq_len, seed=43,
                          device=torch.device("cpu"))
