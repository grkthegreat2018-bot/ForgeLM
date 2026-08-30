"""Best cost/power cloud GPU for V8-8B training.

Uses live Runpod pricing (fetched 2026-08-29) and accurate FLOPS estimates.
V8-8B needs only 6.67GB VRAM with BAdam, but on cloud with 24GB+ VRAM
we can skip BAdam and use standard AdamW (faster, no NVMe bottleneck).

Key insight: the cheapest GPU that fits the workload wins, because V8-8B
is small (1.6B true params). We don't need 80GB H100s — that's overkill.
"""
import math

# ── V8-8B training requirements ──
TRUE_PARAMS = 1.60e9      # 1.6B true trainable params
DENSE_PARAMS = 5.81e9     # 8B dense equivalent
TOKENS_1EPOCH = 10.7e9    # 1 epoch
TOKENS_3EPOCH = 32.1e9    # 3 epochs
FLOP_PER_TOKEN = 6 * TRUE_PARAMS  # 9.6 GFLOP/token

# Without BAdam (cloud, 24GB+ VRAM): standard AdamW
# Optimizer states: 2 × params × 4 bytes (fp32 momentum + variance) = 12.8 GB
# Model: 1.6B × 2 bytes (bf16) = 3.2 GB
# Gradients: 1.6B × 2 bytes (bf16) = 3.2 GB
# Activations: ~2 GB (with FP8 + gradient checkpointing)
# Total without BAdam: ~21 GB → needs 24GB+ VRAM
VRAM_NO_BADAM = 21.0  # GB

# With BAdam (cloud, <24GB VRAM): 4-bit streamed optimizer
# Same as local: 6.67 GB VRAM
VRAM_WITH_BADAM = 6.67  # GB

# ── R22 speedups ──
# Data reduction (fewer tokens to process)
R22_DATA = 1.25 * 1.08  # dedup × importance = 1.35x fewer tokens
# Compute reduction (faster steps)
R22_COMPUTE = 1.29 * 1.74  # unfreeze × pipeline = 2.24x faster steps
# BAdam overhead factor (BAdam is ~1.5x slower than standard AdamW due to
# CPU↔GPU transfer, even with R22 grad compression)
BADAM_OVERHEAD = 1.5

# ── Live Runpod GPU pricing (2026-08-29, secure cloud) ──
# TFLOPS are bf16 dense (no sparsity), approximate
GPUS = [
    # (name, VRAM_GB, secure_$/hr, community_$/hr, bf16_TFLOPS, availability)
    ("RTX 3080 Ti",      12,  0.00,  0.18,   136, "LOW"),     # community only
    ("RTX PRO 4000 BW",  24,  0.57,  0.00,   200, "MEDIUM"),  # secure only
    ("PRO 6000 MIG 24",  24,  0.59,  0.50,   200, "MEDIUM"),
    ("A40",              48,  0.44,  0.35,   150, "HIGH"),
    ("RTX PRO 4500 BW",  32,  0.72,  0.34,   240, "HIGH"),
    ("RTX 5090",         32,  0.99,  0.69,   420, "MEDIUM"),
    ("L40S",             48,  0.99,  0.79,   362, "MEDIUM"),
    ("RTX PRO 6000 BW",  96,  2.09,  1.69,   500, "MEDIUM"),
    ("A100 PCIe 80GB",   80,  1.39,  1.19,   312, "LOW"),
    ("A100 SXM 80GB",    80,  1.59,  1.39,   312, "MEDIUM"),
    ("H100 SXM 80GB",    80,  3.29,  2.69,   989, "HIGH"),
    ("H200 SXM 141GB",  141,  4.59,  3.59,   989, "HIGH"),
    ("B200 180GB",      180,  6.79,  5.98,  1800, "LOW"),
    ("B300 288GB",      288,  7.89,  6.94,  2200, "LOW"),
]

def compute_training(name, vram, sec_price, comm_price, tflops, avail):
    """Compute training time and cost for 1 and 3 epochs."""
    # Determine if we need BAdam (VRAM < 24GB)
    needs_badam = vram < VRAM_NO_BADAM
    use_price = comm_price if comm_price > 0 else sec_price
    cloud_type = "community" if comm_price > 0 and comm_price < sec_price else "secure"

    # Throughput: TFLOPS × MFU / FLOP_per_token
    # MFU varies by GPU: consumer GPUs ~35%, data center ~45%
    if "RTX" in name and "PRO" not in name:
        mfu = 0.35  # consumer GPU
    elif "RTX PRO" in name or "PRO 6000" in name:
        mfu = 0.40  # pro GPU
    else:
        mfu = 0.45  # data center GPU

    effective_tflops = tflops * mfu
    base_tps = effective_tflops * 1e12 / FLOP_PER_TOKEN

    # Apply R22 speedups
    # Data reduction: fewer tokens to process
    effective_tokens_1ep = TOKENS_1EPOCH / R22_DATA
    effective_tokens_3ep = TOKENS_3EPOCH / R22_DATA

    # Compute speedup: faster steps (unfreeze + pipeline)
    # BAdam overhead if needed
    if needs_badam:
        compute_speedup = R22_COMPUTE / BADAM_OVERHEAD  # BAdam slows things down
    else:
        compute_speedup = R22_COMPUTE

    effective_tps = base_tps * compute_speedup

    # Time
    time_1ep_sec = effective_tokens_1ep / effective_tps
    time_3ep_sec = effective_tokens_3ep / effective_tps
    time_1ep_hr = time_1ep_sec / 3600
    time_3ep_hr = time_3ep_sec / 3600

    # Cost (use cheaper of community/secure)
    cost_1ep = time_1ep_hr * use_price
    cost_3ep = time_3ep_hr * use_price

    # Cost efficiency: tokens per dollar
    tok_per_dollar_1ep = TOKENS_1EPOCH / max(cost_1ep, 0.01)

    return {
        "name": name,
        "vram": vram,
        "price": use_price,
        "cloud": cloud_type,
        "tflops": tflops,
        "mfu": mfu,
        "needs_badam": needs_badam,
        "base_tps": base_tps,
        "effective_tps": effective_tps,
        "time_1ep_hr": time_1ep_hr,
        "time_3ep_hr": time_3ep_hr,
        "cost_1ep": cost_1ep,
        "cost_3ep": cost_3ep,
        "tok_per_dollar": tok_per_dollar_1ep,
        "availability": avail,
    }

# ── Compute for all GPUs ──
results = []
for gpu in GPUS:
    r = compute_training(*gpu)
    results.append(r)

# Sort by cost for 1 epoch (cheapest first)
results.sort(key=lambda x: x["cost_1ep"])

print("=" * 90)
print("  Cloud GPU Cost/Power Analysis for V8-8B Training (with R22 speedups)")
print("=" * 90)
print(f"\n  V8-8B: 1.6B true params, 10.7B tokens/epoch, R22 combined ~3.0x speedup")
print(f"  VRAM without BAdam: {VRAM_NO_BADAM:.0f} GB | with BAdam: {VRAM_WITH_BADAM:.1f} GB")
print(f"  Pricing: live Runpod rates (2026-08-29), cheaper of community/secure")

print(f"\n  {'GPU':<20} {'VRAM':>5} {'$/hr':>6} {'TFLOPS':>7} {'tok/s':>8} "
      f"{'1ep(hr)':>8} {'1ep($)':>8} {'3ep($)':>8} {'tok/$':>10} {'BAdam':>6}")
print(f"  {'-'*90}")

for r in results:
    badam = "YES" if r["needs_badam"] else "no"
    print(f"  {r['name']:<20} {r['vram']:>4}G ${r['price']:>5.2f} "
          f"{r['tflops']:>6.0f}T {r['effective_tps']:>7.0f} "
          f"{r['time_1ep_hr']:>7.1f}h ${r['cost_1ep']:>7.2f} "
          f"${r['cost_3ep']:>7.2f} {r['tok_per_dollar']:>10.0f} {badam:>6}")

# ── Top 5 by cost efficiency ──
print(f"\n  {'='*90}")
print(f"  {'TOP 5 BY COST EFFICIENCY (tokens per dollar)':^90}")
print(f"  {'='*90}")

by_efficiency = sorted(results, key=lambda x: x["tok_per_dollar"], reverse=True)
for i, r in enumerate(by_efficiency[:5], 1):
    print(f"\n  #{i}: {r['name']} — {r['tok_per_dollar']:.0f} tokens/$")
    print(f"      ${r['price']}/hr ({r['cloud']}), {r['vram']}GB VRAM, "
          f"{r['tflops']:.0f} TFLOPS, {r['effective_tps']:.0f} tok/s")
    print(f"      1 epoch: {r['time_1ep_hr']:.1f}hr = ${r['cost_1ep']:.2f}")
    print(f"      3 epochs: {r['time_3ep_hr']:.1f}hr = ${r['cost_3ep']:.2f}")
    badam = "needs BAdam (slower)" if r["needs_badam"] else "standard AdamW (fast)"
    print(f"      Optimizer: {badam}")

# ── Best overall recommendation ──
print(f"\n  {'='*90}")
print(f"  {'RECOMMENDATION':^90}")
print(f"  {'='*90}")

# Filter: must be available (not LOW) and fit without BAdam (24GB+)
viable = [r for r in results if r["availability"] != "LOW" and not r["needs_badam"]]
viable.sort(key=lambda x: x["cost_1ep"])

if viable:
    best = viable[0]
    print(f"""
  BEST COST/POWER: {best['name']}
  ─────────────────────────────────────────────────────────────────────────────
    Price:        ${best['price']}/hr ({best['cloud']} cloud)
    VRAM:         {best['vram']} GB (fits without BAdam)
    Throughput:   {best['effective_tps']:.0f} tok/s (with R22)
    1 epoch:      {best['time_1ep_hr']:.1f} hours = ${best['cost_1ep']:.2f}
    3 epochs:     {best['time_3ep_hr']:.1f} hours = ${best['cost_3ep']:.2f}
    Efficiency:   {best['tok_per_dollar']:.0f} tokens/$
    Availability: {best['availability']}

  WHY: {best['name']} has the best $/TFLOP ratio among GPUs that:
    - Have enough VRAM (24GB+) to skip BAdam (standard AdamW is 1.5x faster)
    - Are readily available (HIGH/MEDIUM stock)
    - Are cheap enough for multi-hour runs

  COMPARISON TO H100:
    H100 costs ${next(r for r in results if 'H100' in r['name'])['price']}/hr
    H100 1 epoch: ${next(r for r in results if 'H100' in r['name'])['cost_1ep']:.2f}
    {best['name']} 1 epoch: ${best['cost_1ep']:.2f}
    {best['name']} is {next(r for r in results if 'H100' in r['name'])['cost_1ep'] / best['cost_1ep']:.1f}x cheaper
    H100 is {next(r for r in results if 'H100' in r['name'])['time_1ep_hr'] / best['time_1ep_hr']:.1f}x faster

  FOR YOUR 1-2 HOUR SESSION PATTERN:
    Rent {best['name']} for {best['time_1ep_hr']:.1f} hours (${best['cost_1ep']:.2f}) = 1 overnight rental
    Or rent for 3 epochs: {best['time_3ep_hr']:.1f} hours (${best['cost_3ep']:.2f}) = 1 weekend rental
""")

# ── Multi-GPU scaling ──
print(f"  {'='*90}")
print(f"  {'MULTI-GPU SCALING (1 epoch, 4x GPUs)':^90}")
print(f"  {'='*90}")
print(f"\n  {'GPU':<20} {'4x $/hr':>8} {'4x tok/s':>10} {'Time':>8} {'Cost':>8} {'Scaling':>8}")
print(f"  {'-'*65}")

multi_gpu = [r for r in results if r["availability"] != "LOW" and not r["needs_badam"]]
multi_gpu.sort(key=lambda x: x["vram"], reverse=True)  # data center GPUs first

for r in multi_gpu[:6]:
    # Multi-GPU scaling: ~3.5x for 4 GPUs (DDP overhead)
    tps_4x = r["effective_tps"] * 3.5
    price_4x = r["price"] * 4
    time_4x = (TOKENS_1EPOCH / R22_DATA) / tps_4x / 3600
    cost_4x = time_4x * price_4x
    scaling = 3.5
    print(f"  {r['name']:<20} ${price_4x:>6.2f} {tps_4x:>9.0f} "
          f"{time_4x:>7.1f}h ${cost_4x:>7.2f} {scaling:>7.1f}x")
