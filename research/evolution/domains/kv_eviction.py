"""KV cache eviction domain: search eviction strategies + parameters.

Self-contained simulation that doesn't depend on real cache classes (which need
attention weights from a model forward pass). Instead, we simulate the eviction
policies directly on synthetic KV data.

Search space (6 params):
  - strategy: {snapkv, streaming, paged} (discrete)
  - budget: [128, 2048] (continuous) — max tokens to keep
  - observation_window: [32, 512] (continuous) — SnapKV/Paged obs window
  - n_sinks: [2, 8] (continuous) — StreamingLLM sink tokens
  - window_size: [128, 1024] (continuous) — StreamingLLM sliding window
  - block_size: [8, 32] (continuous) — PagedEviction block size
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


def _snapkv_evict(k: torch.Tensor, v: torch.Tensor,
                  budget: int, obs_window: int,
                  q: torch.Tensor, n_kv_heads: int) -> tuple:
    """Simulate SnapKV: score tokens by attention from observation window."""
    seq_len = k.shape[2]
    if seq_len <= budget + obs_window:
        return k, v

    # Observation window = last obs_window tokens
    obs_k = k[:, :, -obs_window:, :]  # (B, n_kv, obs_window, hd)
    obs_v = v[:, :, -obs_window:, :]

    # Score all tokens by attention from obs window (per KV head, then average)
    # obs_k: (B, n_kv, obs, hd), k: (B, n_kv, seq, hd)
    scores = torch.matmul(obs_k, k.transpose(-2, -1))  # (B, n_kv, obs, seq)
    scores = scores.mean(dim=2)  # (B, n_kv, seq) — average over obs window
    scores = scores.mean(dim=1)  # (B, seq) — average over heads

    # Keep top-budget tokens (excluding obs window which is always kept)
    keep_len = seq_len - obs_window
    keep_scores = scores[0, :keep_len]
    top_idx = keep_scores.topk(min(budget, keep_len)).indices.sort().values

    # Concatenate: selected tokens + observation window
    k_comp = torch.cat([k[:, :, top_idx, :], obs_k], dim=2)
    v_comp = torch.cat([v[:, :, top_idx, :], obs_v], dim=2)
    return k_comp, v_comp


def _streaming_evict(k: torch.Tensor, v: torch.Tensor,
                     n_sinks: int, window_size: int) -> tuple:
    """Simulate StreamingLLM: keep sinks + sliding window."""
    seq_len = k.shape[2]
    k_sink = k[:, :, :n_sinks, :]
    v_sink = v[:, :, :n_sinks, :]
    k_window = k[:, :, -window_size:, :]
    v_window = v[:, :, -window_size:, :]
    return torch.cat([k_sink, k_window], dim=2), torch.cat([v_sink, v_window], dim=2)


def _paged_evict(k: torch.Tensor, v: torch.Tensor,
                 budget: int, block_size: int, obs_window: int,
                 q: torch.Tensor, n_kv_heads: int) -> tuple:
    """Simulate PagedEviction: block-level scoring + eviction (vectorized)."""
    seq_len = k.shape[2]
    if seq_len <= budget + obs_window:
        return k, v

    n_blocks = (seq_len + block_size - 1) // block_size
    keep_blocks = budget // block_size

    # Vectorized block scoring: single matmul, then reshape
    obs_k = k[:, :, -obs_window:, :]  # (B, n_kv, obs, hd)
    pad = (block_size - seq_len % block_size) % block_size
    if pad > 0:
        k_padded = torch.nn.functional.pad(k, (0, 0, 0, pad))
        v_padded = torch.nn.functional.pad(v, (0, 0, 0, pad))
    else:
        k_padded = k
        v_padded = v
    n_blocks_padded = k_padded.shape[2] // block_size

    # scores_flat: (B, n_kv, obs, n_blocks*bs) → reshape → mean
    scores_flat = torch.matmul(obs_k, k_padded.transpose(-2, -1))
    scores_4d = scores_flat.view(k.shape[0], k.shape[1], obs_window, n_blocks_padded, block_size)
    block_scores = scores_4d.mean(dim=(2, 4)).mean(dim=1)  # (B, n_blocks_padded)

    # Keep top blocks (excluding last obs_window/block_size which are always kept)
    n_obs_blocks = obs_window // block_size
    candidate_blocks = n_blocks_padded - n_obs_blocks
    keep_idx = block_scores[0, :candidate_blocks].topk(
        min(keep_blocks, candidate_blocks)
    ).indices.sort().values

    # Gather selected blocks + observation blocks (vectorized)
    block_starts = keep_idx * block_size
    selected_indices = torch.cat([
        torch.cat([torch.arange(s, s + block_size, device=k.device) for s in block_starts]),
        torch.arange(seq_len - obs_window, seq_len, device=k.device),
    ])
    k_comp = k_padded[:, :, selected_indices, :]
    v_comp = v_padded[:, :, selected_indices, :]

    return k_comp, v_comp


class KVEvictionDomain(BaseDomain):
    """Search KV cache eviction parameters across 3 strategies."""

    STRATEGIES = ["snapkv", "streaming", "paged"]
    N_KV_HEADS = 8
    HEAD_DIM = 64
    N_HEADS = 32

    def __init__(self, seq_len: int = 1024, seed: int = 42,
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
        return "kv_eviction"

    def output_dim(self) -> int:
        return 6

    def decode(self, params: torch.Tensor) -> dict[str, Any]:
        p = params.detach().cpu().numpy()
        strategy_idx = int(p[0] * len(self.STRATEGIES))
        strategy_idx = max(0, min(len(self.STRATEGIES) - 1, strategy_idx))

        return {
            "strategy": self.STRATEGIES[strategy_idx],
            "budget": int(128 + p[1] * 1920),
            "observation_window": int(32 + p[2] * 480),
            "n_sinks": int(2 + p[3] * 6),
            "window_size": int(128 + p[4] * 896),
            "block_size": int(8 + p[5] * 24),
        }

    def encode(self, config: dict[str, Any]) -> torch.Tensor:
        strategy = config.get("strategy", "snapkv")
        strat_idx = self.STRATEGIES.index(strategy) if strategy in self.STRATEGIES else 0
        return torch.tensor([
            strat_idx / len(self.STRATEGIES),
            (config.get("budget", 512) - 128) / 1920,
            (config.get("observation_window", 128) - 32) / 480,
            (config.get("n_sinks", 4) - 2) / 6,
            (config.get("window_size", 512) - 128) / 896,
            (config.get("block_size", 16) - 8) / 24,
        ], dtype=torch.float32)

    def _run_eviction(self, config: dict) -> tuple:
        strategy = config.get("strategy", "snapkv")
        budget = config.get("budget", 512)
        obs_window = config.get("observation_window", 128)
        n_sinks = config.get("n_sinks", 4)
        window_size = config.get("window_size", 512)
        block_size = config.get("block_size", 16)

        if strategy == "snapkv":
            return _snapkv_evict(self.k_full, self.v_full, budget, obs_window,
                                 self.q, self.N_KV_HEADS)
        elif strategy == "streaming":
            return _streaming_evict(self.k_full, self.v_full, n_sinks, window_size)
        else:  # paged
            return _paged_evict(self.k_full, self.v_full, budget, block_size,
                                obs_window, self.q, self.N_KV_HEADS)

    def evaluate(self, config: dict[str, Any]) -> dict:
        try:
            k_comp, v_comp = self._run_eviction(config)

            with torch.no_grad():
                y_comp = full_attention_output(self.q, k_comp, v_comp, self.N_KV_HEADS)
                fwd_err = (self.y_ref - y_comp).norm().item() / self.y_ref.norm().item()

            comp_ratio = self.seq_len / max(k_comp.shape[2], 1)

            # Speed estimate (no benchmarking — use compression as proxy)
            cache_ms = comp_ratio * 0.3

            # Score: penalize trivial configs (compression=1.0 means no eviction)
            if comp_ratio <= 1.01:
                score = -fwd_err * 100 - 50  # heavy penalty for no compression
            else:
                score = -fwd_err * 100 + comp_ratio * 5 - cache_ms * 0.01

            return {
                "score": float(score),
                "behavioral": (comp_ratio, fwd_err),
                "metadata": {
                    "fwd_err": fwd_err,
                    "compression": comp_ratio,
                    "cache_ms": cache_ms,
                    "strategy": config.get("strategy", "snapkv"),
                    "budget": config.get("budget", 512),
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
            ("compression", 10, 1.0, 32.0),
            ("fwd_error", 10, 0.0, 0.5),
        ]

    def discrete_choices(self) -> dict[str, list] | None:
        return {
            "strategy": self.STRATEGIES,
            "budget": [128, 256, 512, 1024, 2048],
            "observation_window": [32, 64, 128, 256, 512],
            "n_sinks": [2, 4, 6, 8],
            "window_size": [128, 256, 512, 1024],
            "block_size": [8, 16, 24, 32],
        }

    def seed_configs(self) -> list[dict[str, Any]]:
        seeds = []
        for strat in self.STRATEGIES:
            seeds.append({"strategy": strat, "budget": 512, "observation_window": 128,
                          "n_sinks": 4, "window_size": 512, "block_size": 16})
        for strat in self.STRATEGIES:
            seeds.append({"strategy": strat, "budget": 256, "observation_window": 64,
                          "n_sinks": 2, "window_size": 256, "block_size": 8})
        for strat in self.STRATEGIES:
            seeds.append({"strategy": strat, "budget": 1024, "observation_window": 256,
                          "n_sinks": 8, "window_size": 1024, "block_size": 32})
        return seeds

    def to_cpu(self) -> "KVEvictionDomain":
        """Create CPU copy for parallel evaluation."""
        cpu = KVEvictionDomain(seq_len=self.seq_len, seed=43,
                               device=torch.device("cpu"))
        return cpu
