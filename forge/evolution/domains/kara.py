"""KARA sliding window domain: search KARA KV cache compression parameters.

Search space (4 params):
  - sink_size: [2, 8] (continuous) — initial sink tokens always kept
  - window_size: [128, 1024] (continuous) — sliding window of recent tokens
  - target_budget: [512, 4096] (continuous) — max compressed tokens
  - chunk_expand_size: [4, 16] (continuous) — Token2Chunk expansion size

Evaluation: simulate KARA compression on synthetic KV, measure reconstruction
error + compression + speed. Score = -error * 100 + compression * 5.
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


class KARADomain(BaseDomain):
    """Search KARA KV cache sliding window compression parameters."""

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
        return "kara"

    def output_dim(self) -> int:
        return 4

    def decode(self, params: torch.Tensor) -> dict[str, Any]:
        p = params.detach().cpu().numpy()
        # target_budget as fraction of seq_len (0.125 to 1.0)
        budget_frac = 0.125 + p[2] * 0.875
        return {
            "sink_size": int(2 + p[0] * 6),          # [2, 8]
            "window_size": int(128 + p[1] * 896),    # [128, 1024]
            "target_budget": int(self.seq_len * budget_frac),  # fraction of seq_len
            "chunk_expand_size": int(4 + p[3] * 12), # [4, 16]
        }

    def encode(self, config: dict[str, Any]) -> torch.Tensor:
        budget_frac = config.get("target_budget", 2048) / self.seq_len
        return torch.tensor([
            (config.get("sink_size", 4) - 2) / 6,
            (config.get("window_size", 512) - 128) / 896,
            (budget_frac - 0.125) / 0.875,
            (config.get("chunk_expand_size", 8) - 4) / 12,
        ], dtype=torch.float32)

    def evaluate(self, config: dict[str, Any]) -> dict:
        sink_size = config.get("sink_size", 4)
        window_size = config.get("window_size", 512)
        target_budget = config.get("target_budget", 2048)
        chunk_expand = config.get("chunk_expand_size", 8)

        try:
            # Simulate KARA: keep sinks + compressed middle + sliding window
            seq_len = self.seq_len

            # Sinks: first N tokens (always kept at full precision)
            k_sink = self.k_full[:, :, :sink_size, :]
            v_sink = self.v_full[:, :, :sink_size, :]

            # Window: last W tokens (always kept at full precision)
            k_window = self.k_full[:, :, -window_size:, :]
            v_window = self.v_full[:, :, -window_size:, :]

            # Middle: compress to target_budget tokens
            middle_start = sink_size
            middle_end = seq_len - window_size
            middle_len = middle_end - middle_start

            if middle_len > 0 and target_budget > 0:
                k_middle = self.k_full[:, :, middle_start:middle_end, :]
                v_middle = self.v_full[:, :, middle_start:middle_end, :]

                # Compress by importance-weighted sampling
                k_norms = k_middle.norm(dim=-1).squeeze()  # (n_kv, middle_len)
                v_norms = v_middle.norm(dim=-1).squeeze()
                importance = (k_norms * v_norms).mean(dim=0)  # (middle_len,)

                # Chunk-based compression (KARA Token2Chunk)
                chunk_size = max(1, middle_len // target_budget)
                n_chunks = min(target_budget, middle_len // chunk_size)

                # Score per chunk (average importance)
                chunk_scores = importance[:n_chunks * chunk_size].view(n_chunks, chunk_size).mean(dim=-1)

                # Select top chunks
                n_select = min(target_budget, n_chunks)
                top_chunks = chunk_scores.topk(n_select).indices.sort().values

                # Gather selected chunks (vectorized — no Python loop)
                # top_chunks: (n_select,) chunk indices → expand to token indices
                selected_idx = (
                    top_chunks.unsqueeze(1) * chunk_size
                    + torch.arange(chunk_size, device=self.device).unsqueeze(0)
                ).reshape(-1)
                k_compressed = k_middle[:, :, selected_idx, :]
                v_compressed = v_middle[:, :, selected_idx, :]
            else:
                k_compressed = self.k_full[:, :, :0, :]
                v_compressed = self.v_full[:, :, :0, :]

            # Concatenate: sink + compressed + window
            k_comp = torch.cat([k_sink, k_compressed, k_window], dim=2)
            v_comp = torch.cat([v_sink, v_compressed, v_window], dim=2)

            # Reconstruction error
            with torch.no_grad():
                y_comp = full_attention_output(self.q, k_comp, v_comp, self.N_KV_HEADS)
                fwd_err = (self.y_ref - y_comp).norm().item() / self.y_ref.norm().item()

            # Compression ratio
            compression = seq_len / max(k_comp.shape[2], 1)

            # Speed estimate (no benchmarking — use compression ratio as proxy)
            kara_ms = compression * 0.5  # rough estimate: more compression = more work

            # Score
            score = -fwd_err * 100 + compression * 5 - kara_ms * 0.01

            return {
                "score": float(score),
                "behavioral": (compression, fwd_err),
                "metadata": {
                    "fwd_err": fwd_err,
                    "compression": compression,
                    "kara_ms": kara_ms,
                    "sink_size": sink_size,
                    "window_size": window_size,
                    "target_budget": target_budget,
                    "chunk_expand_size": chunk_expand,
                    "comp_seq_len": k_comp.shape[2],
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
            "sink_size": [2, 4, 6, 8],
            "window_size": [128, 256, 512, 1024],
            "target_budget": [512, 1024, 2048, 3072, 4096],
            "chunk_expand_size": [4, 8, 12, 16],
        }

    def seed_configs(self) -> list[dict[str, Any]]:
        seeds = []
        sl = self.seq_len
        # Default
        seeds.append({"sink_size": 4, "window_size": 512, "target_budget": sl // 2, "chunk_expand_size": 8})
        # Aggressive compression
        seeds.append({"sink_size": 2, "window_size": 128, "target_budget": sl // 8, "chunk_expand_size": 4})
        # Conservative
        seeds.append({"sink_size": 8, "window_size": 1024, "target_budget": sl, "chunk_expand_size": 16})
        # Window sweep
        for ws in [128, 256, 512, 1024]:
            seeds.append({"sink_size": 4, "window_size": ws, "target_budget": sl // 2, "chunk_expand_size": 8})
        # Budget sweep (fractions of seq_len)
        for bf in [0.125, 0.25, 0.5, 0.75]:
            seeds.append({"sink_size": 4, "window_size": 512, "target_budget": int(sl * bf), "chunk_expand_size": 8})
        return seeds

    def to_cpu(self) -> "KARADomain":
        """Create CPU copy for parallel evaluation."""
        return KARADomain(seq_len=self.seq_len, seed=43,
                         device=torch.device("cpu"))
