"""Benchmark Muon-SF-Blockwise vs fused AdamW on ForgeLM V3 (1.2B params).

Measures: convergence speed, memory, wall-clock time per step.
"""
import os, sys, torch, gc, time, math
from pathlib import Path
sys.stdout.reconfigure(line_buffering=True)  # unbuffered output
for line in Path("D:/windsurf/ForgeAI/.env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

from research.model_loader import load_default_model
from research.checkpoint_io import load_checkpoint
from research.training.training_utils import configure_optimizer
import torch.nn.functional as F

# ── Build V3 model ONCE, save initial state ──
print("=== Building ForgeLM V3 ===", flush=True)
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

# Save initial state for resetting between optimizer runs
initial_state = {k: v.clone() for k, v in model.state_dict().items()}

# ── Synthetic training data ──
# Use a fixed pattern so we can measure convergence deterministically
torch.manual_seed(42)
batch_size, seq_len = 2, 256
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device="cuda")
labels = torch.cat([input_ids[:, 1:], input_ids[:, 0:1]], dim=1)  # shifted

def run_training(model, optimizer_name, n_steps=20, lr=3e-4):
    """Run N training steps, return losses, times, peak memory."""
    # Reset model weights to initial state
    model.load_state_dict(initial_state)
    model.train()
    optimizer = configure_optimizer(model, lr, 0.1, optimizer_name=optimizer_name)

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    losses = []
    times = []

    for step in range(n_steps):
        optimizer.zero_grad()
        t0 = time.time()
        out = model(input_ids, targets=labels)
        loss = out[1] if isinstance(out, tuple) and len(out) > 1 else None
        if loss is None:
            logits = out[0] if isinstance(out, tuple) else out
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.time()
        losses.append(loss.item())
        times.append(t1 - t0)
        if step % 5 == 0 or step == n_steps - 1:
            mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"  [{optimizer_name:12s}] step {step:3d} loss {loss.item():.4f} "
                  f"time {t1-t0:.3f}s peak_mem {mem:.2f}GB", flush=True)

    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    avg_time = sum(times[3:]) / len(times[3:])  # skip warmup
    return {
        "losses": losses,
        "peak_mem": peak_mem,
        "avg_time": avg_time,
        "final_loss": losses[-1],
        "first_loss": losses[0],
    }

# ── Run benchmarks ──
N_STEPS = 20
LR = 3e-4

print(f"\n=== Benchmark: {N_STEPS} steps, batch={batch_size}, seq={seq_len}, lr={LR} ===\n", flush=True)

print("--- Fused AdamW (baseline) ---", flush=True)
result_adamw = run_training(model, "fused", N_STEPS, LR)
torch.cuda.empty_cache()
gc.collect()

print("\n--- Muon-SF-Blockwise ---", flush=True)
result_muon = run_training(model, "muon_sf", N_STEPS, LR)
torch.cuda.empty_cache()
gc.collect()

# ── Summary ──
print(f"\n{'='*70}")
print(f"SUMMARY ({N_STEPS} steps, batch={batch_size}, seq={seq_len})")
print(f"{'='*70}")
print(f"{'Metric':<25} {'AdamW':>15} {'Muon-SF-BW':>15} {'Delta':>10}")
print(f"{'-'*65}")
print(f"{'Final loss':<25} {result_adamw['final_loss']:>15.4f} {result_muon['final_loss']:>15.4f} "
      f"{result_muon['final_loss']-result_adamw['final_loss']:>+10.4f}")
print(f"{'Loss reduction':<25} {result_adamw['first_loss']-result_adamw['final_loss']:>15.4f} "
      f"{result_muon['first_loss']-result_muon['final_loss']:>15.4f} "
      f"{(result_muon['first_loss']-result_muon['final_loss'])-(result_adamw['first_loss']-result_adamw['final_loss']):>+10.4f}")
print(f"{'Avg time/step (s)':<25} {result_adamw['avg_time']:>15.3f} {result_muon['avg_time']:>15.3f} "
      f"{result_muon['avg_time']/result_adamw['avg_time']:>9.2f}x")
print(f"{'Peak memory (GB)':<25} {result_adamw['peak_mem']:>15.2f} {result_muon['peak_mem']:>15.2f} "
      f"{result_muon['peak_mem']-result_adamw['peak_mem']:>+9.2f}")

# Convergence speed: which reaches X loss first?
target_losses = [5.0, 4.5, 4.0, 3.5]
print(f"\nSteps to reach target loss:")
for target in target_losses:
    adamw_step = next((i for i, l in enumerate(result_adamw["losses"]) if l <= target), None)
    muon_step = next((i for i, l in enumerate(result_muon["losses"]) if l <= target), None)
    a_str = str(adamw_step) if adamw_step is not None else "N/A"
    m_str = str(muon_step) if muon_step is not None else "N/A"
    print(f"  loss <= {target}: AdamW={a_str}, Muon-SF-BW={m_str}")

print("\n=== DONE ===")
