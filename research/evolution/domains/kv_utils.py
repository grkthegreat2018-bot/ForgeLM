"""Shared utilities for KV cache and attention evolution domains.

Generates synthetic K/V tensors that mimic real LLM attention patterns:
- Normal weights with occasional outliers (like real attention heads)
- Position-dependent decay (recent tokens more important)
- Multi-head structure matching LFM2.5-1.2B (8 KV heads, head_dim=64)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import time


def generate_synthetic_kv(
    seq_len: int = 4096,
    n_kv_heads: int = 8,
    head_dim: int = 64,
    batch: int = 1,
    device: torch.device = None,
    seed: int = 42,
    outlier_frac: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic K, V tensors mimicking real LLM patterns.

    Returns:
        k: (batch, n_kv_heads, seq_len, head_dim)
        v: (batch, n_kv_heads, seq_len, head_dim)
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    k = torch.randn(batch, n_kv_heads, seq_len, head_dim, device=device) * 0.1
    v = torch.randn(batch, n_kv_heads, seq_len, head_dim, device=device) * 0.1

    # Add outliers (activation spikes in real models)
    outlier_mask = torch.rand(batch, n_kv_heads, seq_len, head_dim, device=device) < outlier_frac
    k[outlier_mask] *= 8.0
    v[outlier_mask] *= 5.0

    # Position-dependent magnitude decay (recent tokens have larger norms in real models)
    pos_decay = torch.linspace(0.7, 1.3, seq_len, device=device).view(1, 1, seq_len, 1)
    k = k * pos_decay
    v = v * pos_decay

    return k, v


def generate_synthetic_q(
    q_len: int = 1,
    n_heads: int = 32,
    head_dim: int = 64,
    batch: int = 1,
    device: torch.device = None,
    seed: int = 43,
) -> torch.Tensor:
    """Generate synthetic Q tensor for attention computation.

    Returns:
        q: (batch, n_heads, q_len, head_dim)
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    return torch.randn(batch, n_heads, q_len, head_dim, device=device) * 0.1


def full_attention_output(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    n_kv_heads: int = 8,
) -> torch.Tensor:
    """Compute full attention output (ground truth reference).

    Args:
        q: (batch, n_heads, q_len, head_dim)
        k: (batch, n_kv_heads, seq_len, head_dim)
        v: (batch, n_kv_heads, seq_len, head_dim)
        n_kv_heads: number of KV heads (for GQA replication)

    Returns:
        out: (batch, n_heads, q_len, head_dim)
    """
    n_heads = q.shape[1]
    head_dim = q.shape[-1]
    n_rep = n_heads // n_kv_heads

    # Replicate KV heads for GQA
    k_rep = k.repeat_interleave(n_rep, dim=1)  # (batch, n_heads, seq_len, head_dim)
    v_rep = v.repeat_interleave(n_rep, dim=1)

    # Scaled dot-product attention
    scale = 1.0 / (head_dim ** 0.5)
    scores = torch.matmul(q, k_rep.transpose(-2, -1)) * scale  # (B, H, q_len, seq_len)
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v_rep)  # (B, H, q_len, head_dim)

    return out


def compressed_attention_output(
    q: torch.Tensor, k_comp: torch.Tensor, v_comp: torch.Tensor,
    n_kv_heads: int = 8,
) -> torch.Tensor:
    """Compute attention with compressed/evicted KV cache.

    Same as full_attention_output but with potentially shorter KV cache.
    """
    return full_attention_output(q, k_comp, v_comp, n_kv_heads)


def reconstruction_error(
    q: torch.Tensor, k_full: torch.Tensor, v_full: torch.Tensor,
    k_comp: torch.Tensor, v_comp: torch.Tensor,
    n_kv_heads: int = 8,
) -> float:
    """Measure how well compressed KV preserves attention output.

    Returns: relative L2 error (0.0 = perfect, 1.0 = completely wrong)
    """
    y_ref = full_attention_output(q, k_full, v_full, n_kv_heads)
    y_comp = compressed_attention_output(q, k_comp, v_comp, n_kv_heads)
    err = (y_ref - y_comp).norm().item() / y_ref.norm().item()
    return err


def compression_ratio(
    k_full: torch.Tensor, k_comp: torch.Tensor,
) -> float:
    """Memory compression ratio: full_bytes / compressed_bytes."""
    full_elements = k_full.shape[2]  # seq_len
    comp_elements = k_comp.shape[2]  # compressed seq_len
    if comp_elements == 0:
        return 1.0
    return full_elements / comp_elements


def measure_speed(
    fn, *args, n_iters: int = 20, device: torch.device = None, **kwargs
) -> float:
    """Measure function execution time in milliseconds."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = fn(*args, **kwargs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iters * 1000
