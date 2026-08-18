"""Validate Muon-SF (plain) + 3-way grad mixup on REAL ForgeLM V3 (1.2B params).

The decisive production test. Sister agent validated Muon-SF-Blockwise alone.
This tests the OPTIMAL stack: Muon-SF (no blockwise) + grad mixup 3.

Variants (20 steps each, same data, same init):
1. fused AdamW (baseline)
2. muon_sf_plain (Muon+SF, no blockwise) — no mixup
3. muon_sf_plain + mixup3 — THE OPTIMAL STACK
4. muon_sf (Muon+SF+Blockwise) — no mixup (sister's config)
5. muon_sf + mixup3 — blockwise + mixup (should be worse per toy test)

Measures: convergence (loss reduction), time/step, peak memory.
"""
import os, sys, torch, gc, time, math
from pathlib import Path
sys.stdout.reconfigure(line_buffering=True)
for line in Path("D:/windsurf/ForgeAI/.env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

from research.model_loader import load_default_model
from research.checkpoint_io import load_checkpoint
from research.training.training_utils import configure_optimizer
import torch.nn.functional as F

print("=== Building ForgeLM V3 (1.2B) ===", flush=True)
model, tok = load_default_model("forgelm_v3")
ckpt_path = "research/checkpoints/ForgeLM_V3_Base.safetensors"
if not os.path.exists(ckpt_path):
    ckpt_path = "research/checkpoints/ForgeLM_V2_BSP.safetensors"
sd = load_checkpoint(ckpt_path)
model.load_state_dict({k: v for k, v in sd.items()}, strict=False)
model = model.to("cuda").to(torch.bfloat16)
config = model.config
total_params = sum(p.numel() for p in model.parameters())
print(f"Model: {total_params/1e6:.1f}M params, {len(model.blocks)} layers", flush=True)

initial_state = {k: v.clone() for k, v in model.state_dict().items()}

torch.manual_seed(42)
batch_size, seq_len = 2, 256
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device="cuda")
labels = torch.cat([input_ids[:, 1:], input_ids[:, 0:1]], dim=1)

# Pre-generate mixup batches (different seeds)
mixup_ids = []
for i in range(4):
    torch.manual_seed(1000 + i)
    mixup_ids.append(torch.randint(0, config.vocab_size, (batch_size, seq_len), device="cuda"))
torch.manual_seed(42)  # reset


def run_training(model, optimizer_name, n_steps=20, lr=3e-4, grad_mixup=1):
    model.load_state_dict(initial_state)
    model.train()
    optimizer = configure_optimizer(model, lr, 0.1, optimizer_name=optimizer_name)

    # ScheduleFree needs .train()
    if hasattr(optimizer, 'train'):
        optimizer.train()

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    losses = []
    times = []

    for step in range(n_steps):
        optimizer.zero_grad()
        t0 = time.time()

        # First batch
        out = model(input_ids, targets=labels)
        loss = out[1] if isinstance(out, tuple) and len(out) > 1 else None
        if loss is None:
            logits = out[0] if isinstance(out, tuple) else out
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        if grad_mixup > 1:
            # Save first batch grads
            loss.backward()
            saved_grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

            # Additional batches
            for mi in range(grad_mixup - 1):
                mi_ids = mixup_ids[mi]
                mi_labels = torch.cat([mi_ids[:, 1:], mi_ids[:, 0:1]], dim=1)
                optimizer.zero_grad()
                out2 = model(mi_ids, targets=mi_labels)
                loss2 = out2[1] if isinstance(out2, tuple) and len(out2) > 1 else None
                if loss2 is None:
                    logits2 = out2[0] if isinstance(out2, tuple) else out2
                    loss2 = F.cross_entropy(logits2.reshape(-1, logits2.size(-1)), mi_labels.reshape(-1))
                loss2.backward()
                # Average grads
                for n, p in model.named_parameters():
                    if p.grad is not None and n in saved_grads:
                        saved_grads[n] = (saved_grads[n] * (mi + 1) + p.grad) / (mi + 2)

            # Restore averaged grads
            optimizer.zero_grad()
            for n, p in model.named_parameters():
                if n in saved_grads:
                    p.grad = saved_grads[n]
        else:
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.time()
        losses.append(loss.item())
        times.append(t1 - t0)
        if step % 5 == 0 or step == n_steps - 1:
            mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"  [{optimizer_name:15s} mixup={grad_mixup}] step {step:3d} loss {loss.item():.4f} "
                  f"time {t1-t0:.3f}s peak_mem {mem:.2f}GB", flush=True)

    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    avg_time = sum(times[3:]) / len(times[3:])
    return {
        "losses": losses,
        "peak_mem": peak_mem,
        "avg_time": avg_time,
        "final_loss": losses[-1],
        "first_loss": losses[0],
        "loss_reduction": losses[0] - losses[-1],
    }


N_STEPS = 20
LR = 3e-4

print(f"\n=== Benchmark: {N_STEPS} steps, batch={batch_size}, seq={seq_len}, lr={LR} ===\n", flush=True)

results = {}

print("--- 1. Fused AdamW (baseline) ---", flush=True)
results["adamw"] = run_training(model, "fused", N_STEPS, LR, grad_mixup=1)
torch.cuda.empty_cache(); gc.collect()

print("\n--- 2. Muon-SF-Plain (no blockwise, no mixup) ---", flush=True)
results["muon_sf_plain"] = run_training(model, "muon_sf_plain", N_STEPS, LR, grad_mixup=1)
torch.cuda.empty_cache(); gc.collect()

print("\n--- 3. Muon-SF-Plain + 3-way mixup (THE OPTIMAL STACK) ---", flush=True)
results["muon_sf_plain_mix3"] = run_training(model, "muon_sf_plain", N_STEPS, LR, grad_mixup=3)
torch.cuda.empty_cache(); gc.collect()

print("\n--- 4. Muon-SF-Blockwise (sister's config, no mixup) ---", flush=True)
results["muon_sf_bw"] = run_training(model, "muon_sf", N_STEPS, LR, grad_mixup=1)
torch.cuda.empty_cache(); gc.collect()

print("\n--- 5. Muon-SF-Blockwise + 3-way mixup ---", flush=True)
results["muon_sf_bw_mix3"] = run_training(model, "muon_sf", N_STEPS, LR, grad_mixup=3)
torch.cuda.empty_cache(); gc.collect()

# Summary
print("\n" + "=" * 80)
print("SUMMARY: Real V3 1.2B — Muon-SF + grad mixup validation")
print("=" * 80)
print(f"{'Variant':<30s} {'Final':>8s} {'Reduction':>10s} {'AvgTime':>8s} {'PeakMem':>8s}")
print("-" * 80)
for name, r in sorted(results.items(), key=lambda x: x[1]["final_loss"]):
    print(f"{name:<30s} {r['final_loss']:8.3f} {r['loss_reduction']:10.3f} "
          f"{r['avg_time']:8.3f}s {r['peak_mem']:7.2f}GB")

baseline = results["adamw"]
optimal = results["muon_sf_plain_mix3"]
print(f"\n  Optimal stack (muon_sf_plain + mixup3):")
print(f"    vs AdamW:     {baseline['final_loss']/optimal['final_loss']:.2f}x better loss, "
      f"{baseline['avg_time']/optimal['avg_time']:.2f}x time, "
      f"{optimal['peak_mem']/baseline['peak_mem']:.2f}x mem")
print(f"    vs Muon-SF-BW: {results['muon_sf_bw']['final_loss']/optimal['final_loss']:.2f}x better loss")
