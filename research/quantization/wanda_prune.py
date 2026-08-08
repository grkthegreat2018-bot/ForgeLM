"""Wanda pruning (Sun et al. 2023) for the ForgeAI research model.

Wanda prunes weights by the product of weight magnitude and the L2 norm of
corresponding input activations, measured on a small calibration set. No
retraining is required. The result is a structurally sparse model (zeroed
weights) that can be saved and served as-is.

Usage:
    python -m research.wanda_prune --config 360m_mla \
        --checkpoint research/checkpoints/pretrained_llm.safetensors \
        --sparsity 0.2 --n-samples 64 --seq-len 256 \
        --out research/checkpoints/pruned_llm_safetensors
"""
import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn

from research.config import get_config
from research.model_loader import ModelLoader
from research.training.training_utils import BinaryDataset


def _capture_layer_inputs(model, layers, calibration_loader, device):
    """Run calibration data through the model and record input activations
    to each target Linear layer. Returns {layer_name: [activations, ...]}.
    """
    captured = {name: [] for name in layers}
    hooks = []

    def make_hook(name):
        def hook(module, inputs, _output):
            # inputs[0] shape: [B, T, in_features] (or [B*T, in_features])
            x = inputs[0].detach()
            # Flatten to [N, in_features] and subsample to keep memory bounded.
            x = x.reshape(-1, x.shape[-1])
            if x.shape[0] > 512:
                x = x[torch.randperm(x.shape[0])[:512]]
            captured[name].append(x.cpu())
        return hook

    modules = dict(model.named_modules())
    for name in layers:
        if name in modules:
            hooks.append(modules[name].register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        for batch in calibration_loader:
            x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                _ = model(x)

    for h in hooks:
        h.remove()

    # Concatenate per-layer activations and compute the per-input-feature L2 norm.
    norms = {}
    for name, acts in captured.items():
        if not acts:
            continue
        all_acts = torch.cat(acts, dim=0).float()  # [N, in_features]
        norms[name] = all_acts.norm(dim=0)  # [in_features]
    return norms


def wanda_prune(model, sparsity, activation_norms, device):
    """Zero out the lowest (|w| * ||act||)-scoring weights per output row,
    per Linear layer, to reach the target sparsity fraction.
    """
    total_zeroed = 0
    total_params = 0
    modules = dict(model.named_modules())
    for name, norm in activation_norms.items():
        if name not in modules:
            continue
        layer = modules[name]
        if not isinstance(layer, nn.Linear):
            continue
        W = layer.weight.data  # [out, in]
        norm = norm.to(W.device).to(W.dtype)
        # Score per element: |W| * ||act_in||. Broadcast norm over output rows.
        score = W.abs() * norm.unsqueeze(0)  # [out, in]
        # Per-output-row threshold: keep top (1-sparsity) fraction by score.
        n_keep = int(W.shape[1] * (1.0 - sparsity))
        if n_keep >= W.shape[1] or n_keep < 1:
            continue
        # Find the threshold per row.
        topk_vals, _ = score.topk(n_keep, dim=1)
        thresh = topk_vals[:, -1:].to(W.dtype)  # [out, 1]
        mask = score >= thresh  # [out, in] - True where we keep
        zeroed = (~mask).sum().item()
        W.mul_(mask.to(W.dtype))
        total_zeroed += zeroed
        total_params += W.numel()
    actual_sparsity = total_zeroed / max(total_params, 1)
    print(f"Wanda pruning: zeroed {total_zeroed}/{total_params} weights ({actual_sparsity:.2%} actual sparsity, target {sparsity:.2%})")
    return actual_sparsity


def main():
    p = argparse.ArgumentParser(description="Wanda pruning for ForgeAI model")
    p.add_argument("--config", default="360m_mla")
    p.add_argument("--checkpoint", default="research/checkpoints/pretrained_llm.safetensors")
    p.add_argument("--sparsity", type=float, default=0.2, help="Fraction of weights to zero per layer (0.2 = 20%)")
    p.add_argument("--n-samples", type=int, default=64, help="Calibration samples")
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--out", default="research/checkpoints/pruned_llm.safetensors")
    p.add_argument("--data", default="research/data/train.bin")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = get_config(args.config)
    cfg.seq_len = args.seq_len

    print(f"Loading model from {args.checkpoint}...")
    model = ModelLoader.build_model(cfg, checkpoint_path=args.checkpoint).to(device).eval()

    # Identify all Linear layers to prune (skip embeddings, norms).
    target_layers = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and mod.weight.ndim == 2:
            # Skip the embedding-like projections (very wide input dim is a hint).
            target_layers.append(name)
    print(f"Target Linear layers: {len(target_layers)}")

    # Build calibration batches from the pretraining data.
    ds = BinaryDataset(args.data, args.seq_len, cfg.vocab_size)
    batches = [ds.get_batch(args.batch_size, device) for _ in range(args.n_samples)]
    calibration_loader = [b[0] for b in batches]  # just input ids

    print(f"Capturing activations on {args.n_samples} calibration batches...")
    norms = _capture_layer_inputs(model, target_layers, calibration_loader, device)
    print(f"Captured norms for {len(norms)} layers.")

    print(f"Pruning to {args.sparsity:.0%} sparsity...")
    actual = wanda_prune(model, args.sparsity, norms, device)

    # Save the pruned model.
    from research.checkpoint_io import save_checkpoint
    out_path = args.out
    if not out_path.endswith(".safetensors") and not out_path.endswith(".pt"):
        out_path = out_path + ".safetensors"
    save_checkpoint(model.state_dict(), out_path)
    print(f"Saved pruned model to {out_path}")

    # Quick perplexity check on a few val batches.
    val_ds = BinaryDataset("research/data/val.bin", args.seq_len, cfg.vocab_size)
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for _ in range(10):
            x, y = val_ds.get_batch(args.batch_size, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                out = model(x)
                logits = out[0] if isinstance(out, tuple) else out
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)).float(),
                    y.view(-1),
                )
            total_loss += loss.item()
            n += 1
    import math
    avg_loss = total_loss / n
    print(f"Pruned model val loss: {avg_loss:.4f} | ppl: {math.exp(avg_loss):.2f}")


if __name__ == "__main__":
    main()
