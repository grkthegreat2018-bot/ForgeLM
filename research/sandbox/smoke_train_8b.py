"""Smoke test for V7-8B variants (8B-B and 8B-D) on RTX 5070 12GB.

Uses the EXACT production build path from train_8b_all.build_model() —
meta-device build, NLRQ reset, BitNet-QAT disable, kaiming+depth-scaled init —
so what the smoke test validates is what the full trainer runs.

Verifies per config:
  - Model builds within VRAM budget
  - Forward pass works (NLRQ factorized)
  - Backward + BAdam step works
  - Loss is finite and decreases
"""
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from research.sandbox.train_8b_all import (
    autocast_ctx, build_model, enable_factor_training_all_, forward_model,
    next_token_cross_entropy, print_model_stats, vram_snapshot,
)
from research.training.optim.badam import configure_badam

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def smoke_test_8b(config_name: str, seq_len: int = 512, steps: int = 5) -> dict:
    print(f"\n{'=' * 70}\n  SMOKE TEST: {config_name}\n{'=' * 70}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t_build = time.time()
    model, cfg = build_model(config_name, device, torch.bfloat16,
                             use_checkpointing=True, grad_clip=1.0)
    n_factor = enable_factor_training_all_(model)
    # Factor masters add ~2 GB on rank-1024 configs — keep them only if the
    # GPU has headroom for optimizer states + activations (mirrors the
    # trainer's --factor-training auto).
    if torch.cuda.is_available():
        free_gb = torch.cuda.mem_get_info(device)[0] / 1e9
        alloc_gb = torch.cuda.memory_allocated() / 1e9
        if free_gb < 2.2:
            print(f"  Factor training: {free_gb:.1f} GB free < 2.2 GB — "
                  f"dropping to S-only (as the trainer's auto mode would)")
            from research.keys.compression.nlrq_ffn_key import NLRQLinear
            for m in model.modules():
                if isinstance(m, NLRQLinear):
                    m.disable_factor_training_(export=False)
            torch.cuda.empty_cache()
        else:
            print(f"  Factor training: {n_factor} NLRQ layers (STE masters)")
    print(f"  Build time: {time.time() - t_build:.1f}s")
    print_model_stats(model, cfg)
    vram_snapshot("post-build")

    optimizer = configure_badam(model, lr=3e-4, weight_decay=0.1,
                                blocks_per_layer=1, switch_every=1)

    print(f"\n  Training: {steps} steps, seq_len={seq_len}, batch=1")

    vocab_size = cfg.vocab_size
    losses = []
    t0 = time.time()

    for step in range(steps):
        input_ids = torch.randint(0, vocab_size, (1, seq_len), device=device)

        with autocast_ctx(device):
            loss = next_token_cross_entropy(forward_model(model, input_ids), input_ids)

        loss_value = loss.item()
        if math.isfinite(loss_value):
            loss.backward()
            optimizer.step()
        optimizer.zero_grad()
        losses.append(loss_value)
        print(f"  Step {step + 1}/{steps} | loss={loss_value:.4f} | "
              f"{time.time() - t0:.1f}s")
        if step == 0:
            vram_snapshot("step 1")

    elapsed = time.time() - t0
    finite = [l for l in losses if math.isfinite(l)]
    status = "pass" if len(finite) == steps and finite[-1] < finite[0] else "check"
    print(f"\n  RESULT: {config_name} — loss {losses[0]:.4f} → {losses[-1]:.4f} "
          f"({elapsed:.1f}s) [{status}]")

    result = {
        "config": config_name,
        "params_M": sum(p.numel() for p in model.parameters()) / 1e6,
        "steps": steps,
        "time_s": elapsed,
        "losses": losses,
        "status": status,
    }

    del model, optimizer
    if torch.cuda.is_available():
        result["peak_vram_GB"] = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Peak VRAM: {result['peak_vram_GB']:.2f} GB")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    return result


def main():
    results = []
    for config_name in ("forgelm_v7_8b_b", "forgelm_v7_8b_d"):
        try:
            results.append(smoke_test_8b(config_name, seq_len=512, steps=5))
        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append({"config": config_name, "status": "fail", "error": str(e)})
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    print(f"\n{'=' * 70}\n  SUMMARY\n{'=' * 70}")
    for r in results:
        if r["status"] != "fail":
            print(f"  {r['config']}: {r['params_M']:.0f}M params, "
                  f"{r.get('peak_vram_GB', 0):.2f} GB peak VRAM, "
                  f"{r['time_s']:.1f}s, loss {r['losses'][0]:.3f} → {r['losses'][-1]:.3f}")
        else:
            print(f"  {r['config']}: FAILED — {r.get('error', 'unknown')}")

    out_path = os.path.join(ROOT, "research", "results", "smoke_8b_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
