"""Updated V8-8B training cost with R22 speedups applied.

R22 speedups (measured, not theoretical):
  - DataDedup:         1.25x  (20% near-dup removal, realistic corpus)
  - TokenImportance:   1.08x  (skip 7.7% low-loss tokens)
  - ProgressiveUnfreeze: 1.98x (avg over 3 phases, early steps faster)
  - GradCompression:   1.94x  (4-bit grad transfer for BAdam)
  - AsyncPipeline:     1.74x  (I/O hidden behind compute, 95.6% overlap)
  - DeltaCheckpoint:   6.00x  (save/resume only, not training time)

Combined training speedup: 9.09x (multiplicative, different bottlenecks)
  BUT: ProgressiveUnfreeze only helps early phases. Realistic combined
  for full training: ~5-6x (unfreeze diminishes over training).

This script computes the realistic training time with R22 applied.
"""
import math

# ── V8-8B baseline (from calc_v8_8b_scratch_train.py) ──
BASELINE = {
    "total_tokens": 32.1e9,       # 10.7B × 3 epochs
    "true_params": 1.60e9,
    "step_time_ms": 530,          # BAdam + GradTopK 10%
    "total_steps": 1_960_000,
    "total_hours": 289,
    "gpu_vram_gb": 6.67,
    "ram_gb": 18.70,
    "cost_electricity": 12.12,
}

# ── R22 measured speedups ──
R22_SPEEDUPS = {
    "data_dedup": 1.25,        # 20% dup removal
    "token_importance": 1.08,  # 7.7% low-loss skip
    "progressive_unfreeze": 1.98,  # avg over phases (optimistic for full train)
    "grad_compression": 1.94,  # 4-bit BAdam transfer
    "async_pipeline": 1.74,    # I/O overlap
    "delta_checkpoint": 6.00,  # save/resume only
}

# ── Realistic combined speedup ──
# ProgressiveUnfreeze only helps the first ~30% of training (phases 1-2).
# After that, all layers are active and it provides no speedup.
# Weighted average: 30% of training at 1.98x, 70% at 1.0x
# → effective unfreeze speedup = 0.3 * 1.98 + 0.7 * 1.0 = 1.29x
EFFECTIVE_UNFREEZE = 0.3 * R22_SPEEDUPS["progressive_unfreeze"] + 0.7 * 1.0

# Combined (multiplicative, different bottlenecks)
combined = (R22_SPEEDUPS["data_dedup"] *
            R22_SPEEDUPS["token_importance"] *
            EFFECTIVE_UNFREEZE *
            R22_SPEEDUPS["grad_compression"] *
            R22_SPEEDUPS["async_pipeline"])

# ── Updated training cost ──
updated_hours = BASELINE["total_hours"] / combined
updated_step_time = BASELINE["step_time_ms"] / combined
updated_cost = BASELINE["cost_electricity"] / combined

print("=" * 75)
print("  V8-8B Training Cost: Baseline vs R22-Applied")
print("=" * 75)

print(f"\n  {'Metric':<35} {'Baseline':>12} {'R22 Applied':>12} {'Speedup':>8}")
print(f"  {'-'*75}")
print(f"  {'Step time (ms)':<35} {BASELINE['step_time_ms']:>12.0f} {updated_step_time:>12.1f} {combined:>7.2f}x")
print(f"  {'Total hours':<35} {BASELINE['total_hours']:>12.0f} {updated_hours:>12.1f} {combined:>7.2f}x")
print(f"  {'Total days':<35} {BASELINE['total_hours']/24:>12.1f} {updated_hours/24:>12.1f} {combined:>7.2f}x")
print(f"  {'Cost (electricity)':<35} {'$' + str(round(BASELINE['cost_electricity'], 2)):>12} {'$' + str(round(updated_cost, 2)):>12} {combined:>7.2f}x")

print(f"\n  R22 Speedup Breakdown:")
print(f"  {'-'*75}")
print(f"  {'DataDedup (20% dup removal)':<40} {R22_SPEEDUPS['data_dedup']:>6.2f}x")
print(f"  {'TokenImportance (7.7% skip)':<40} {R22_SPEEDUPS['token_importance']:>6.2f}x")
print(f"  {'ProgressiveUnfreeze (effective)':<40} {EFFECTIVE_UNFREEZE:>6.2f}x")
print(f"  {'GradCompression (4-bit BAdam)':<40} {R22_SPEEDUPS['grad_compression']:>6.2f}x")
print(f"  {'AsyncPipeline (I/O overlap)':<40} {R22_SPEEDUPS['async_pipeline']:>6.2f}x")
print(f"  {'DeltaCheckpoint (save/resume only)':<40} {R22_SPEEDUPS['delta_checkpoint']:>6.2f}x")
print(f"  {'─'*40} {'─'*6}")
print(f"  {'COMBINED (training)':<40} {combined:>6.2f}x")

print(f"\n  {'='*75}")
print(f"  {'SESSION FEASIBILITY (1.5 hr sessions)':^75}")
print(f"  {'='*75}")

session_hr = 1.5
baseline_sessions = BASELINE["total_hours"] / session_hr
r22_sessions = updated_hours / session_hr

print(f"\n  {'Scenario':<30} {'Hours':>8} {'Sessions':>10} {'Calendar':>10}")
print(f"  {'-'*65}")
print(f"  {'V8-8B baseline (3 epoch)':<30} {BASELINE['total_hours']:>8.0f} {baseline_sessions:>10.0f} {baseline_sessions/30:>9.1f}mo")
print(f"  {'V8-8B + R22 (3 epoch)':<30} {updated_hours:>8.1f} {r22_sessions:>10.0f} {r22_sessions/30:>9.1f}mo")

# 1 epoch
baseline_1ep = BASELINE["total_hours"] / 3
r22_1ep = updated_hours / 3
print(f"  {'V8-8B baseline (1 epoch)':<30} {baseline_1ep:>8.0f} {baseline_1ep/session_hr:>10.0f} {baseline_1ep/session_hr/30:>9.1f}mo")
print(f"  {'V8-8B + R22 (1 epoch)':<30} {r22_1ep:>8.1f} {r22_1ep/session_hr:>10.0f} {r22_1ep/session_hr/30:>9.1f}mo")

# Fine-tune 1.2B (10K steps, ~80ms/step)
ft_seconds_baseline = 10000 * 0.08  # 80ms/step for 1.2B
ft_hours_baseline = ft_seconds_baseline / 3600  # 0.22 hours
ft_hours_r22 = ft_hours_baseline / (R22_SPEEDUPS["token_importance"] * R22_SPEEDUPS["async_pipeline"])
print(f"  {'Fine-tune 1.2B (10K steps)':<30} {ft_hours_baseline:>8.1f} {ft_hours_baseline/session_hr:>10.0f} {ft_hours_baseline/session_hr/30:>9.1f}mo")
print(f"  {'Fine-tune 1.2B + R22 (10K)':<30} {ft_hours_r22:>8.1f} {ft_hours_r22/session_hr:>10.0f} {ft_hours_r22/session_hr/30:>9.1f}mo")

print(f"\n  {'='*75}")
print(f"  {'CLOUD COMPARISON (V8-8B + R22, 1 epoch)':^75}")
print(f"  {'='*75}")

# 1-epoch times (from session feasibility above)
local_1ep_r22 = updated_hours / 3  # 16.3 hours

# Cloud: H100 doesn't need BAdam (80GB VRAM fits full model + optimizer).
# R22 speedups that apply to cloud: dedup (1.25x), importance (1.08x),
# unfreeze (1.29x effective), pipeline (1.74x). No grad compression.
cloud_speedup = (R22_SPEEDUPS["data_dedup"] * R22_SPEEDUPS["token_importance"] *
                 EFFECTIVE_UNFREEZE * R22_SPEEDUPS["async_pipeline"])

# H100 raw throughput: ~100K tok/s for 1.6B model (no BAdam, 80GB VRAM, ~40% MFU)
# H100 bf16 ~1000 TFLOPS, 1.6B params × 6 FLOP/token = 9.6 GFLOP/token
# 1e15 / 9.6e9 = 104K tok/s at 100% MFU, ~42K at 40% MFU
# With R22 data reduction (1.25 × 1.08 = 1.35x fewer tokens):
h100_base_tps = 42000  # 40% MFU, no BAdam
h100_1ep_hr = (10.7e9 / h100_base_tps / 3600) / (R22_SPEEDUPS["data_dedup"] *
                  R22_SPEEDUPS["token_importance"] * EFFECTIVE_UNFREEZE *
                  R22_SPEEDUPS["async_pipeline"])

cloud_options = [
    ("RTX 5070 (local)", local_1ep_r22, 0.0, "All R22 speedups + BAdam"),
    ("H100 80GB", h100_1ep_hr, 4.0, "No BAdam, data+importance+unfreeze+pipeline"),
    ("4x H100 80GB", h100_1ep_hr / 3.5, 16.0, "4x scaling ~3.5x"),
    ("8x H100 80GB", h100_1ep_hr / 6.3, 32.0, "8x scaling ~6.3x"),
]

print(f"\n  {'GPU':<25} {'Time (1ep)':>10} {'Cost':>8}  {'Note'}")
print(f"  {'-'*75}")
for name, time_hr, price, note in cloud_options:
    cost = time_hr * price
    if time_hr < 1:
        time_str = f"{time_hr*60:.0f}min"
    else:
        time_str = f"{time_hr:.1f}hr"
    cost_str = f"${cost:.2f}" if price > 0 else "$0"
    print(f"  {name:<25} {time_str:>10} {cost_str:>8}  {note}")

print(f"\n  {'='*75}")
print(f"  {'VERDICT':^75}")
print(f"  {'='*75}")

print(f"""
  V8-8B + R22 full 3-epoch training:
    Local:  {updated_hours:.0f} hours = {updated_hours/session_hr:.0f} sessions = {updated_hours/session_hr/30:.1f} months
    → Still NOT feasible for 1-2hr session pattern

  V8-8B + R22 1-epoch training:
    Local:  {r22_1ep:.0f} hours = {r22_1ep/session_hr:.0f} sessions = {r22_1ep/session_hr/30:.1f} months
    → Still NOT feasible for 1-2hr session pattern

  Fine-tune 1.2B + R22:
    Local:  {ft_hours_r22:.1f} hours = {ft_hours_r22/session_hr:.0f} sessions
    → FEASIBLE for 1-2hr sessions (3-4 sessions)

  Cloud H100 + R22 (1 epoch):
    {h100_1ep_hr:.1f} hours, ${h100_1ep_hr * 4:.2f}
    → FEASIBLE as 1-2 overnight rentals

  R22 speedups help but don't change the fundamental picture:
  full 8B pretraining needs continuous multi-day runs.
  Local is best for fine-tuning; cloud is best for pretraining.
""")
