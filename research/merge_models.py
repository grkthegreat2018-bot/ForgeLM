"""Model merging for the ForgeAI research model.

Implements the core MergeKit algorithms (SLERP, TIES, DARE) directly on our
state dicts, since MergeKit's YAML pipeline expects HF-format models with
config.json and our architecture is custom.

Algorithms:
- SLERP: spherical linear interpolation between two models' tensors.
- TIES: sum of (sign-gated, magnitude-pruned) task vectors, applied to base.
- DARE: drop-and-rescale task vectors before merging.

Usage:
    python -m research.merge_models --method slerp --alpha 0.5 \
        --model-a research/checkpoints/pretrained_llm.safetensors \
        --model-b research/checkpoints/pruned_llm.safetensors \
        --out research/checkpoints/merged_llm.safetensors
"""
import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import torch

from research.checkpoint_io import load_checkpoint, save_checkpoint


def slerp(t, v0, v1):
    """Spherical linear interpolation between two tensors.

    Falls back to linear interpolation when vectors are nearly collinear.
    """
    v0_n = v0 / (v0.norm() + 1e-8)
    v1_n = v1 / (v1.norm() + 1e-8)
    dot = (v0_n * v1_n).sum().clamp(-1.0, 1.0)
    if dot.abs() > 0.9995:
        # Nearly collinear: linear interpolation is fine.
        return v0 * (1.0 - t) + v1 * t
    omega = dot.acos()
    so = torch.sin(omega)
    return torch.sin((1.0 - t) * omega) / so * v0 + torch.sin(t * omega) / so * v1


def merge_slerp(state_a, state_b, alpha=0.5):
    """SLERP each tensor. alpha=0 -> A, alpha=1 -> B."""
    merged = {}
    for k in state_a:
        if k not in state_b:
            merged[k] = state_a[k]
            continue
        a, b = state_a[k], state_b[k]
        if not isinstance(a, torch.Tensor) or a.shape != b.shape or a.numel() < 4:
            merged[k] = a * (1.0 - alpha) + b * alpha
        else:
            merged[k] = slerp(alpha, a.float(), b.float()).to(a.dtype)
    return merged


def task_vector(base, tuned):
    """task_vector = tuned - base."""
    return {k: (tuned[k] - base[k]) if isinstance(tuned[k], torch.Tensor) and k in base else tuned[k] for k in tuned}


def ties_merge(base, task_vectors, density=0.5):
    """TIES: prune task vectors to top-k magnitude, resolve sign conflicts,
    average non-conflicting, add to base.
    """
    # Prune: keep top (density) fraction by magnitude per task vector.
    pruned = []
    for tv in task_vectors:
        p = {}
        for k, v in tv.items():
            if not isinstance(v, torch.Tensor):
                p[k] = v
                continue
            # Use kthvalue (no size limit) instead of quantile (capped at 2^24 on CUDA).
            flat_abs = v.abs().flatten()
            kth = max(1, int(flat_abs.numel() * (1.0 - density)))
            thresh = torch.kthvalue(flat_abs, kth).values
            mask = v.abs() >= thresh
            p[k] = v * mask.to(v.dtype)
        pruned.append(p)

    # Sign election: per-element, the sign chosen by the majority of task vectors
    # (by total magnitude). Conflicting-sign entries are zeroed.
    merged = {}
    keys = pruned[0].keys() if pruned else []
    for k in keys:
        if not isinstance(pruned[0][k], torch.Tensor):
            merged[k] = pruned[0][k]
            continue
        stacked = torch.stack([p[k].float() for p in pruned])  # [N, ...]
        sign_sum = stacked.sum(0)
        elected_sign = sign_sum.sign()
        # Keep only entries that agree with the elected sign.
        agree = (stacked.sign() == elected_sign.unsqueeze(0)).float()
        # Average over agreeing task vectors only.
        denom = agree.sum(0).clamp(min=1.0)
        merged_tv = (stacked * agree).sum(0) / denom
        # Add to base.
        if k in base and isinstance(base[k], torch.Tensor):
            merged[k] = base[k].to(merged_tv.dtype) + merged_tv.to(base[k].dtype)
        else:
            merged[k] = merged_tv
    return merged


def dare_merge(base, task_vectors, drop_rate=0.1):
    """DARE: randomly drop task-vector deltas, rescale survivors by 1/(1-drop),
    then merge via union (sum with rescale)."""
    import math
    scale = 1.0 / max(1.0 - drop_rate, 1e-6)
    merged = dict(base)
    for tv in task_vectors:
        for k, v in tv.items():
            if not isinstance(v, torch.Tensor) or k not in merged:
                continue
            mask = (torch.rand_like(v.float()) >= drop_rate).to(v.dtype)
            delta = v * mask * scale
            merged[k] = merged[k].to(delta.dtype) + delta.to(merged[k].dtype)
    return merged


def differentiable_dare_ties(base, task_vectors, eval_fn, n_iters=100,
                              lr=0.01, density=0.5, drop_rate=0.1):
    """Differentiable DARE-TIES merging via gradient descent.

    Makes the merge parameters (drop masks, scaling weights) differentiable
    and optimizes them to minimize eval loss. 10x faster convergence than
    evolutionary approaches (NeurIPS 2024 Competition, 4th place).

    Args:
        base: base model state dict
        task_vectors: list of task vector state dicts
        eval_fn: callable(merged_state_dict) -> loss (lower is better)
        n_iters: optimization iterations
        lr: learning rate for merge parameters
        density: TIES density (fraction of mass to keep)
        drop_rate: initial DARE drop rate

    Returns:
        merged state dict
    """
    # Learnable parameters: per-task-vector scaling weights + drop logits.
    n_tasks = len(task_vectors)
    # Initialize scaling weights uniformly.
    scales = torch.ones(n_tasks, requires_grad=True)
    # Initialize drop logits (sigmoid → drop probability).
    drop_logits = torch.full((n_tasks,), -2.0, requires_grad=True)  # sigmoid(-2)≈0.12

    optimizer = torch.optim.AdamW([scales, drop_logits], lr=lr)

    print(f"[Diff DARE-TIES] optimizing merge of {n_tasks} task vectors, {n_iters} iters")
    for it in range(n_iters):
        optimizer.zero_grad()

        # Build merged model from differentiable parameters.
        drop_probs = torch.sigmoid(drop_logits)
        merged = {k: v.clone() for k, v in base.items()}

        for i, tv in enumerate(task_vectors):
            scale_i = scales[i]
            # Soft mask (differentiable approximation of Bernoulli).
            for k, v in tv.items():
                if not isinstance(v, torch.Tensor) or k not in merged:
                    continue
                # TIES: prune to top-density by magnitude (differentiable via soft threshold).
                magnitude = v.abs()
                threshold = torch.quantile(magnitude.float(), 1.0 - density)
                soft_mask = torch.sigmoid((magnitude - threshold) * 10.0)  # sharp sigmoid
                # DARE: soft drop.
                keep_prob = 1.0 - drop_probs[i]
                delta = v * soft_mask * scale_i * keep_prob
                merged[k] = merged[k].to(delta.dtype) + delta.to(merged[k].dtype)

        # Evaluate merged model.
        loss = eval_fn(merged)
        loss.backward()
        optimizer.step()

        if (it + 1) % 10 == 0:
            print(f"  [Diff DARE-TIES] iter {it+1}/{n_iters} | loss: {loss.item():.4f} | "
                  f"scales: {[f'{s.item():.2f}' for s in scales]} | "
                  f"drop: {[f'{torch.sigmoid(d).item():.2f}' for d in drop_logits]}")

    # Final merge with optimized parameters (hard masks).
    drop_probs = torch.sigmoid(drop_logits).detach()
    scales_final = scales.detach()
    merged = dict(base)
    for i, tv in enumerate(task_vectors):
        for k, v in tv.items():
            if not isinstance(v, torch.Tensor) or k not in merged:
                continue
            magnitude = v.abs()
            threshold = torch.quantile(magnitude.float(), 1.0 - density)
            mask = (magnitude >= threshold).to(v.dtype)
            keep = (1.0 - drop_probs[i]).item()
            delta = v * mask * scales_final[i].item() * keep
            merged[k] = merged[k].to(delta.dtype) + delta.to(merged[k].dtype)

    return merged


def main():
    p = argparse.ArgumentParser(description="Merge two ForgeAI checkpoints")
    p.add_argument("--method", choices=["slerp", "ties", "dare", "diff-dare-ties"], default="slerp")
    p.add_argument("--model-a", required=True, help="First model (base for TIES/DARE)")
    p.add_argument("--model-b", required=True, help="Second model (task vector source for TIES/DARE)")
    p.add_argument("--model-c", default=None, help="Optional third model for TIES/DARE multi-way merge")
    p.add_argument("--alpha", type=float, default=0.5, help="SLERP mixing weight (0=A, 1=B)")
    p.add_argument("--density", type=float, default=0.5, help="TIES: fraction of task-vector mass to keep")
    p.add_argument("--drop-rate", type=float, default=0.1, help="DARE: fraction of deltas to drop")
    p.add_argument("--out", default="research/checkpoints/merged_llm.safetensors")
    args = p.parse_args()

    print(f"Loading model A: {args.model_a}")
    a = load_checkpoint(args.model_a, map_location="cpu")
    print(f"Loading model B: {args.model_b}")
    b = load_checkpoint(args.model_b, map_location="cpu")
    c = None
    if args.model_c:
        print(f"Loading model C: {args.model_c}")
        c = load_checkpoint(args.model_c, map_location="cpu")

    if args.method == "slerp":
        print(f"SLERP merge (alpha={args.alpha})...")
        merged = merge_slerp(a, b, alpha=args.alpha)
    elif args.method == "ties":
        tvs = [task_vector(a, b)]
        if c is not None:
            tvs.append(task_vector(a, c))
        print(f"TIES merge (density={args.density}, {len(tvs)} task vectors)...")
        merged = ties_merge(a, tvs, density=args.density)
    elif args.method == "dare":
        tvs = [task_vector(a, b)]
        if c is not None:
            tvs.append(task_vector(a, c))
        print(f"DARE merge (drop_rate={args.drop_rate}, {len(tvs)} task vectors)...")
        merged = dare_merge(a, tvs, drop_rate=args.drop_rate)

    out_path = args.out
    if not out_path.endswith(".safetensors") and not out_path.endswith(".pt"):
        out_path = out_path + ".safetensors"
    save_checkpoint(merged, out_path)
    print(f"Saved merged model to {out_path}")


if __name__ == "__main__":
    main()
