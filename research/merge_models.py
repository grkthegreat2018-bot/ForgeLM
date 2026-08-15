"""Model merging — SLERP / TIES / DARE / SVD / Task Arithmetic / Linear.

Operates directly on state dicts (safetensors or .pt) so it works with
ForgeAI's custom LFM2.5 architecture without needing HF-format models
(mergekit's YAML pipeline expects HF-format models, incompatible here).

Methods:
  linear       — weighted average of N models (model soup).
  task_arith   — add/subtract task vectors (theta_ft - theta_base).
  slerp        — spherical linear interpolation between 2 models.
  ties         — Trim, Elect Sign, Disjoint Merge of N task vectors.
  dare         — random drop + rescale of task vectors (Yu et al. 2023).
  svd          — SVD-based low-rank merge of task vectors (Ainsworth et al.).

All methods are pure PyTorch. `torch.kthvalue` is used instead of
`torch.quantile` for magnitude thresholds (quantile has a 2^24 element
limit on CUDA; embeddings exceed it).

Usage:
    python -m research.merge_models --method slerp \\
        --model-a research/checkpoints/a.safetensors \\
        --model-b research/checkpoints/b.safetensors \\
        --t 0.5 --out research/checkpoints/merged.safetensors

    python -m research.merge_models --method ties \\
        --base research/checkpoints/base.safetensors \\
        --models a.safetensors b.safetensors c.safetensors \\
        --density 0.5 --out research/checkpoints/merged.safetensors
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import torch

from research.checkpoint_io import load_checkpoint, save_checkpoint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_state_dict(path: str) -> dict[str, torch.Tensor]:
    """Load a checkpoint and return only the tensor entries."""
    state = load_checkpoint(path, map_location="cpu")
    return {k: v for k, v in state.items() if isinstance(v, torch.Tensor)}


def _task_vectors(ft: dict[str, torch.Tensor],
                  base: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Compute per-parameter task vectors: ft - base."""
    out = {}
    for k, v in ft.items():
        if k in base and base[k].shape == v.shape:
            out[k] = (v.float() - base[k].float())
    return out


def _kth_threshold(values: torch.Tensor, density: float) -> torch.Tensor:
    """Per-row magnitude threshold keeping fraction `density` of elements.

    Uses torch.kthvalue (not torch.quantile) to avoid the 2^24 element
    CUDA limit on large tensors (e.g. embeddings).
    `values`: [N] magnitudes for one parameter flattened.
    Returns the scalar threshold below which elements are trimmed.
    """
    n = values.numel()
    n_keep = max(1, int(n * density))
    if n_keep >= n:
        return values.min()
    # kthvalue returns the k-th smallest; we want the n_keep-th largest,
    # i.e. the (n - n_keep + 1)-th smallest.
    k = n - n_keep + 1
    return values.kthvalue(k).values


# ---------------------------------------------------------------------------
# Merge methods
# ---------------------------------------------------------------------------

def merge_linear(models: list[dict[str, torch.Tensor]],
                 weights: list[float] | None = None
                 ) -> dict[str, torch.Tensor]:
    """Weighted average of N models (model soup)."""
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    assert len(weights) == len(models), "weights/models length mismatch"
    w_sum = sum(weights)
    weights = [w / w_sum for w in weights]
    out = {}
    keys = models[0].keys()
    for k in keys:
        acc = torch.zeros_like(models[0][k], dtype=torch.float32)
        for m, w in zip(models, weights):
            acc = acc + w * m[k].float()
        out[k] = acc.to(models[0][k].dtype)
    return out


def merge_task_arith(base: dict[str, torch.Tensor],
                     task_vectors: list[dict[str, torch.Tensor]],
                     scales: list[float] | None = None
                     ) -> dict[str, torch.Tensor]:
    """Task arithmetic: base + sum(scale_i * task_vector_i)."""
    if scales is None:
        scales = [1.0] * len(task_vectors)
    out = {}
    for k, v in base.items():
        acc = v.float().clone()
        for tv, s in zip(task_vectors, scales):
            if k in tv:
                acc = acc + s * tv[k]
        out[k] = acc.to(v.dtype)
    return out


def merge_slerp(a: dict[str, torch.Tensor],
                b: dict[str, torch.Tensor],
                t: float = 0.5) -> dict[str, torch.Tensor]:
    """Spherical linear interpolation between two models.

    Falls back to linear interpolation when vectors are nearly collinear
    (dot > 0.9995), matching the standard SLERP convention.
    """
    out = {}
    for k in a.keys():
        if k not in b or a[k].shape != b[k].shape:
            out[k] = a[k].clone()
            continue
        wa = a[k].flatten().float()
        wb = b[k].flatten().float()
        na = wa.norm() + 1e-8
        nb = wb.norm() + 1e-8
        ua = wa / na
        ub = wb / nb
        dot = (ua * ub).sum().clamp(-1.0, 1.0)
        if dot > 0.9995:
            result = (1 - t) * wa + t * wb
        else:
            theta = torch.arccos(dot)
            sin_theta = torch.sin(theta)
            result = (torch.sin((1 - t) * theta) / sin_theta) * wa + \
                     (torch.sin(t * theta) / sin_theta) * wb
        out[k] = result.reshape(a[k].shape).to(a[k].dtype)
    return out


def merge_ties(base: dict[str, torch.Tensor],
               task_vectors: list[dict[str, torch.Tensor]],
               density: float = 0.5) -> dict[str, torch.Tensor]:
    """TIES merging (Yadav et al. 2023).

    1. Trim: keep top-`density` fraction by magnitude per parameter per model.
    2. Elect Sign: per parameter, the consensus sign is the sign of the
       sum of trimmed vectors; only vectors matching the consensus sign
       contribute.
    3. Disjoint Merge: sum the sign-consistent vectors, divide by count.
    """
    out = {}
    keys = task_vectors[0].keys()
    for k in keys:
        if k not in base:
            continue
        # Stack trimmed task vectors: [N, *param_shape]
        stacked = []
        for tv in task_vectors:
            if k not in tv:
                continue
            v = tv[k]
            mag = v.abs().flatten()
            thresh = _kth_threshold(mag, density)
            mask = (v.abs() >= thresh).to(v.dtype)
            stacked.append(v * mask)
        if not stacked:
            out[k] = base[k].clone()
            continue
        stacked = torch.stack(stacked, dim=0)  # [N, *shape]
        # Elect sign: consensus = sign of sum across models.
        total = stacked.sum(dim=0)
        consensus = torch.sign(total)
        # Keep only vectors whose sign matches consensus; zero others.
        agree = (torch.sign(stacked) == consensus.unsqueeze(0)).to(stacked.dtype)
        pruned = stacked * agree
        # Disjoint merge: sum / count of contributors (avoid div-by-zero).
        count = agree.sum(dim=0).clamp(min=1.0)
        merged_delta = pruned.sum(dim=0) / count
        out[k] = (base[k].float() + merged_delta).to(base[k].dtype)
    return out


def merge_dare(base: dict[str, torch.Tensor],
               task_vectors: list[dict[str, torch.Tensor]],
               drop_rate: float = 0.1,
               seed: int = 0) -> dict[str, torch.Tensor]:
    """DARE merging (Yu et al. 2023).

    For each task vector, randomly drop `drop_rate` fraction of deltas and
    rescale survivors by 1/(1-drop_rate) to preserve expected magnitude,
    then average the rescaled deltas onto the base.
    """
    gen = torch.Generator().manual_seed(seed)
    scale = 1.0 / max(1.0 - drop_rate, 1e-6)
    out = {}
    keys = task_vectors[0].keys()
    for k in keys:
        if k not in base:
            continue
        acc = torch.zeros_like(base[k], dtype=torch.float32)
        n = 0
        for tv in task_vectors:
            if k not in tv:
                continue
            v = tv[k]
            mask = (torch.rand(v.shape, generator=gen) >= drop_rate).to(v.dtype)
            acc = acc + (v * mask * scale)
            n += 1
        if n == 0:
            out[k] = base[k].clone()
            continue
        out[k] = (base[k].float() + acc / n).to(base[k].dtype)
    return out


def merge_svd(base: dict[str, torch.Tensor],
              task_vectors: list[dict[str, torch.Tensor]],
              rank_ratio: float = 0.5,
              scales: list[float] | None = None) -> dict[str, torch.Tensor]:
    """SVD-based low-rank merge of task vectors.

    For each parameter, stack the N task vectors into a matrix [N, numel]
    (or [numel, N] for 1-D params), compute SVD, truncate to rank
    floor(rank_ratio * min(N, numel)), reconstruct, and apply weighted sum
    onto the base. This compresses the task-vector space to its principal
    components, reducing interference between conflicting updates
    (Ainsworth et al., "SVDETFix").
    """
    if scales is None:
        scales = [1.0] * len(task_vectors)
    out = {}
    keys = task_vectors[0].keys()
    for k in keys:
        if k not in base:
            continue
        # Collect available task vectors for this param.
        vecs = [tv[k].flatten().float() for tv in task_vectors if k in tv]
        if not vecs:
            out[k] = base[k].clone()
            continue
        M = torch.stack(vecs, dim=0)  # [N, numel]
        n_models, numel = M.shape
        # SVD of the task-vector matrix.
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        k_rank = max(1, int(rank_ratio * min(n_models, numel)))
        k_rank = min(k_rank, S.numel())
        U_k = U[:, :k_rank]
        S_k = S[:k_rank]
        Vh_k = Vh[:k_rank, :]
        # Weighted reconstruction: scale per-model contribution.
        w = torch.tensor(scales[:n_models], dtype=M.dtype, device=M.device)
        # Reconstruct a single consensus delta: weighted combination of
        # the top-k right singular directions, scaled by singular values.
        weighted = (U_k * w.unsqueeze(1)) @ torch.diag(S_k)  # [N, k]
        # Sum across models into the principal subspace, then project back.
        consensus = weighted.sum(dim=0) @ Vh_k  # [numel]
        # Normalize by number of models so magnitude stays comparable.
        consensus = consensus / n_models
        out[k] = (base[k].float() + consensus.reshape(base[k].shape)).to(base[k].dtype)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Model merging for ForgeAI (LFM2.5)")
    p.add_argument("--method", required=True,
                   choices=["linear", "task_arith", "slerp", "ties", "dare", "svd"])
    p.add_argument("--base", default=None,
                   help="Base model checkpoint (required for task_arith/ties/dare/svd)")
    p.add_argument("--model-a", default=None, help="First model (slerp/linear 2-model)")
    p.add_argument("--model-b", default=None, help="Second model (slerp/linear 2-model)")
    p.add_argument("--models", nargs="*", default=None,
                   help="List of model checkpoints (linear/task_arith/ties/dare/svd)")
    p.add_argument("--weights", nargs="*", type=float, default=None,
                   help="Per-model weights (linear/task_arith/svd)")
    p.add_argument("--t", type=float, default=0.5, help="SLERP interpolation factor")
    p.add_argument("--density", type=float, default=0.5,
                   help="TIES: fraction of magnitudes to keep (0.5 = keep top 50%)")
    p.add_argument("--drop-rate", type=float, default=0.1, help="DARE: drop fraction")
    p.add_argument("--rank-ratio", type=float, default=0.5,
                   help="SVD: rank truncation ratio (0.5 = keep top half of singular values)")
    p.add_argument("--seed", type=int, default=0, help="DARE: RNG seed")
    p.add_argument("--out", required=True, help="Output checkpoint path")
    args = p.parse_args()

    out_path = args.out
    if not out_path.endswith(".safetensors") and not out_path.endswith(".pt"):
        out_path = out_path + ".safetensors"

    # --- 2-model methods ---
    if args.method == "slerp":
        if not args.model_a or not args.model_b:
            p.error("--model-a and --model-b required for slerp")
        a = _load_state_dict(args.model_a)
        b = _load_state_dict(args.model_b)
        print(f"SLERP(t={args.t}): {args.model_a} <-> {args.model_b}")
        merged = merge_slerp(a, b, t=args.t)

    elif args.method == "linear" and args.model_a and args.model_b and not args.models:
        a = _load_state_dict(args.model_a)
        b = _load_state_dict(args.model_b)
        w = args.weights or [0.5, 0.5]
        print(f"Linear soup: {args.model_a} ({w[0]}) + {args.model_b} ({w[1]})")
        merged = merge_linear([a, b], w)

    # --- N-model methods requiring base ---
    else:
        if not args.base:
            p.error(f"--base required for {args.method}")
        if not args.models:
            p.error(f"--models required for {args.method}")
        base = _load_state_dict(args.base)
        ft_models = [_load_state_dict(m) for m in args.models]
        task_vectors = [_task_vectors(m, base) for m in ft_models]
        print(f"{args.method}: base={args.base} + {len(ft_models)} task vectors")

        if args.method == "linear":
            merged = merge_linear([base] + ft_models, args.weights)
        elif args.method == "task_arith":
            merged = merge_task_arith(base, task_vectors, args.weights)
        elif args.method == "ties":
            merged = merge_ties(base, task_vectors, density=args.density)
        elif args.method == "dare":
            merged = merge_dare(base, task_vectors, drop_rate=args.drop_rate,
                                seed=args.seed)
        elif args.method == "svd":
            merged = merge_svd(base, task_vectors, rank_ratio=args.rank_ratio,
                               scales=args.weights)
        else:
            p.error(f"unknown method: {args.method}")

    save_checkpoint(merged, out_path)
    print(f"Saved merged model to {out_path} ({len(merged)} tensors)")


if __name__ == "__main__":
    main()
