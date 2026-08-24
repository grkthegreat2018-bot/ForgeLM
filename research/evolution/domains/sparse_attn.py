"""Sparse attention domain: search CompactAttention/CoSA/MoSA parameters.

Search space (5 params):
  - strategy: {compact, cosa, mosa} (discrete)
  - budget_ratio: [0.1, 0.9] (continuous) — fraction of blocks/tokens selected
  - block_size: [8, 32] (continuous) — KV block size
  - min_seq_len: [512, 4096] (continuous) — activation threshold
  - k_ratio: [0.2, 0.8] (continuous) — MoSA token selection ratio

Evaluation: run sparse attention on synthetic Q/K/V, measure output similarity
to full attention + speedup. Score = -error * 100 + speedup * 10.
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


class SparseAttentionDomain(BaseDomain):
    """Search sparse attention parameters across 3 strategies."""

    STRATEGIES = ["compact", "cosa", "mosa"]
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

        # Measure full attention speed as baseline
        self.full_attn_ms = measure_speed(
            full_attention_output, self.q, self.k_full, self.v_full,
            self.N_KV_HEADS, n_iters=20, device=self.device,
        )

    def name(self) -> str:
        return "sparse_attn"

    def output_dim(self) -> int:
        return 5

    def decode(self, params: torch.Tensor) -> dict[str, Any]:
        p = params.detach().cpu().numpy()
        strategy_idx = int(p[0] * len(self.STRATEGIES))
        strategy_idx = max(0, min(len(self.STRATEGIES) - 1, strategy_idx))

        return {
            "strategy": self.STRATEGIES[strategy_idx],
            "budget_ratio": float(0.1 + p[1] * 0.8),    # [0.1, 0.9]
            "block_size": int(8 + p[2] * 24),           # [8, 32]
            "min_seq_len": int(256 + p[3] * 768),       # [256, 1024]
            "k_ratio": float(0.2 + p[4] * 0.6),         # [0.2, 0.8]
        }

    def encode(self, config: dict[str, Any]) -> torch.Tensor:
        strategy = config.get("strategy", "compact")
        strat_idx = self.STRATEGIES.index(strategy) if strategy in self.STRATEGIES else 0
        return torch.tensor([
            strat_idx / len(self.STRATEGIES),
            (config.get("budget_ratio", 0.5) - 0.1) / 0.8,
            (config.get("block_size", 16) - 8) / 24,
            (config.get("min_seq_len", 512) - 256) / 768,
            (config.get("k_ratio", 0.5) - 0.2) / 0.6,
        ], dtype=torch.float32)

    def evaluate(self, config: dict[str, Any]) -> dict:
        strategy = config.get("strategy", "compact")
        budget_ratio = config.get("budget_ratio", 0.5)
        block_size = config.get("block_size", 16)
        min_seq_len = config.get("min_seq_len", 2048)
        k_ratio = config.get("k_ratio", 0.5)

        try:
            # If seq_len < min_seq_len, sparse attention doesn't activate → same as full
            if self.seq_len < min_seq_len:
                # No sparsification → perfect quality, no speedup
                return {
                    "score": 0.0 + self.full_attn_ms * 0.001,  # neutral score
                    "behavioral": (1.0, 0.0),
                    "metadata": {
                        "fwd_err": 0.0, "speedup": 1.0,
                        "strategy": strategy, "activated": False,
                        "budget_ratio": budget_ratio,
                    },
                }

            with torch.no_grad():
                if strategy == "compact":
                    from research.inference.attention.compact_attention import compact_attention
                    y_sparse = compact_attention(
                        self.q, self.k_full, self.v_full,
                        block_size=block_size, budget_ratio=budget_ratio,
                    )
                elif strategy == "cosa":
                    from research.inference.attention.cosa import cosa_attention
                    y_sparse = cosa_attention(
                        self.q, self.k_full, self.v_full,
                        block_size=block_size, budget_ratio=budget_ratio,
                    )
                else:  # mosa — approximate with top-k selection on K
                    # MoSA selects k_ratio of tokens per head
                    k_len = self.k_full.shape[2]
                    k_keep = max(1, int(k_len * k_ratio))
                    # Simple importance: L2 norm of K per position
                    k_norms = self.k_full.norm(dim=-1)  # (B, n_kv, seq_len)
                    topk_idx = k_norms.topk(k_keep, dim=-1).indices
                    # Gather selected K, V
                    k_sel = torch.gather(
                        self.k_full, 2,
                        topk_idx.unsqueeze(-1).expand(-1, -1, -1, self.HEAD_DIM),
                    )
                    v_sel = torch.gather(
                        self.v_full, 2,
                        topk_idx.unsqueeze(-1).expand(-1, -1, -1, self.HEAD_DIM),
                    )
                    y_sparse = full_attention_output(
                        self.q, k_sel, v_sel, self.N_KV_HEADS,
                    )

                # Reconstruction error
                fwd_err = (self.y_ref - y_sparse).norm().item() / self.y_ref.norm().item()

            # Speed measurement
            def _run_sparse():
                with torch.no_grad():
                    if strategy == "compact":
                        from research.inference.attention.compact_attention import compact_attention
                        return compact_attention(self.q, self.k_full, self.v_full,
                                                block_size=block_size, budget_ratio=budget_ratio)
                    elif strategy == "cosa":
                        from research.inference.attention.cosa import cosa_attention
                        return cosa_attention(self.q, self.k_full, self.v_full,
                                             block_size=block_size, budget_ratio=budget_ratio)
                    else:
                        k_len = self.k_full.shape[2]
                        k_keep = max(1, int(k_len * k_ratio))
                        k_norms = self.k_full.norm(dim=-1)
                        topk_idx = k_norms.topk(k_keep, dim=-1).indices
                        k_sel = torch.gather(self.k_full, 2,
                            topk_idx.unsqueeze(-1).expand(-1, -1, -1, self.HEAD_DIM))
                        v_sel = torch.gather(self.v_full, 2,
                            topk_idx.unsqueeze(-1).expand(-1, -1, -1, self.HEAD_DIM))
                        return full_attention_output(self.q, k_sel, v_sel, self.N_KV_HEADS)

            # Speed estimate (no benchmarking — use budget ratio as proxy)
            sparse_ms = self.full_attn_ms / max(budget_ratio, 0.1)
            speedup = self.full_attn_ms / max(sparse_ms, 0.001)

            # Effective compression (tokens attended to)
            if strategy == "mosa":
                eff_compression = 1.0 / k_ratio
            else:
                eff_compression = 1.0 / budget_ratio

            # Score: penalize trivial configs (high budget_ratio = nearly full attention)
            if budget_ratio >= 0.85:
                score = -fwd_err * 100 - 50  # penalty for near-full attention
            else:
                score = -fwd_err * 100 + speedup * 10 + eff_compression * 2

            return {
                "score": float(score),
                "behavioral": (eff_compression, fwd_err),
                "metadata": {
                    "fwd_err": fwd_err,
                    "speedup": speedup,
                    "sparse_ms": sparse_ms,
                    "full_ms": self.full_attn_ms,
                    "strategy": strategy,
                    "activated": True,
                    "budget_ratio": budget_ratio,
                    "block_size": block_size,
                    "k_ratio": k_ratio,
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
            ("compression", 10, 1.0, 10.0),
            ("fwd_error", 10, 0.0, 0.3),
        ]

    def discrete_choices(self) -> dict[str, list] | None:
        return {
            "strategy": self.STRATEGIES,
            "budget_ratio": [0.1, 0.25, 0.5, 0.75, 0.9],
            "block_size": [8, 16, 24, 32],
            "min_seq_len": [256, 512, 768, 1024],
            "k_ratio": [0.2, 0.35, 0.5, 0.65, 0.8],
        }

    def seed_configs(self) -> list[dict[str, Any]]:
        seeds = []
        for strat in self.STRATEGIES:
            for br in [0.25, 0.5, 0.75]:
                seeds.append({
                    "strategy": strat, "budget_ratio": br,
                    "block_size": 16, "min_seq_len": 512, "k_ratio": 0.5,
                })
        for bs in [8, 16, 32]:
            seeds.append({
                "strategy": "compact", "budget_ratio": 0.5,
                "block_size": bs, "min_seq_len": 512, "k_ratio": 0.5,
            })
        return seeds

    def to_cpu(self) -> "SparseAttentionDomain":
        """Create CPU copy for parallel evaluation."""
        return SparseAttentionDomain(seq_len=self.seq_len, seed=43,
                                     device=torch.device("cpu"))
