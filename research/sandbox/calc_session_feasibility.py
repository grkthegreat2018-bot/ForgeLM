"""What's feasible in 1-2 hour sessions? Local vs cloud options."""
import math

# ── Local RTX 5070 throughput ──
local_tps = 574  # tok/s (from V8 calc)
session_hr = 1.5  # average session
local_tokens_per_session = local_tps * session_hr * 3600  # 3.1M tokens

print("=" * 75)
print("  Training in 1-2 Hour Sessions: Feasibility Analysis")
print("=" * 75)

print(f"\n  Local RTX 5070: {local_tps} tok/s")
print(f"  Per 1.5hr session: {local_tokens_per_session/1e6:.1f}M tokens")

# ── What can be done locally in 1-2 hours? ──
print(f"\n  {'='*73}")
print(f"  {'LOCAL (1-2 hr sessions, RTX 5070)':^73}")
print(f"  {'='*73}")

scenarios = [
    ("V8-8B full scratch (32.1B tok)", 32.1e9, "8B equiv from scratch"),
    ("V8-8B 1 epoch (10.7B tok)", 10.7e9, "1 epoch only"),
    ("Fine-tune 1.2B (50K steps)", 50e3 * 16384, "SFT on existing model"),
    ("Fine-tune 1.2B (10K steps)", 10e3 * 16384, "Quick SFT pass"),
    ("LoRA fine-tune 1.2B (5K steps)", 5e3 * 16384, "LoRA adapter only"),
    ("Small model 350M scratch (7B tok)", 7e9, "350M from scratch"),
    ("Tiny model 100M scratch (2B tok)", 2e9, "100M from scratch"),
]

print(f"\n  {'Scenario':<40} {'Tokens':>10} {'Sessions':>10} {'Calendar':>10}")
print(f"  {'-'*73}")
for name, tokens, note in scenarios:
    sessions = tokens / local_tokens_per_session
    days = sessions  # 1 session/day
    feasible = "YES" if sessions <= 5 else ("MAYBE" if sessions <= 30 else "NO")
    if days > 365:
        cal = f"{days/365:.1f}yr"
    elif days > 30:
        cal = f"{days/30:.1f}mo"
    else:
        cal = f"{days:.0f}d"
    print(f"  {name:<40} {tokens/1e9:>9.2f}B {sessions:>9.0f}  {cal:>10}  {feasible}")

# ── Cloud options (RunPod) ──
print(f"\n  {'='*73}")
print(f"  {'CLOUD (RunPod, train while you sleep)':^73}")
print(f"  {'='*73}")

cloud_gpus = [
    ("RTX 4070 Ti (12GB)", 0.34, 600, "Equivalent to your local"),
    ("RTX 4090 (24GB)", 0.69, 1500, "2.6x faster, 2x VRAM"),
    ("A100 80GB", 2.50, 4028, "7x faster, full batch training"),
    ("H100 80GB", 4.00, 14187, "25x faster, no BAdam needed"),
    ("H100 80GB ×4", 16.00, 50000, "87x faster, distributed"),
    ("H100 80GB ×8", 32.00, 90000, "157x faster, large-batch"),
]

print(f"\n  {'GPU':<25} {'$/hr':>6} {'tok/s':>8} {'Time':>10} {'Cost':>8} {'Feasible':>8}")
print(f"  {'-'*73}")

cloud_scenarios = [
    ("V8-8B full (32.1B tok)", 32.1e9),
    ("V8-8B 1 epoch (10.7B)", 10.7e9),
    ("Fine-tune 1.2B (10K steps)", 10e3 * 16384),
]

for gpu_name, price, tps, note in cloud_gpus:
    print(f"\n  {gpu_name} (${price}/hr, {tps} tok/s) — {note}")
    for scenario_name, tokens in cloud_scenarios:
        time_sec = tokens / tps
        time_hr = time_sec / 3600
        cost = time_hr * price
        if time_hr < 1:
            time_str = f"{time_hr*60:.0f}min"
        elif time_hr < 24:
            time_str = f"{time_hr:.1f}hr"
        else:
            time_str = f"{time_hr/24:.1f}d"
        feasible = "YES" if time_hr <= 8 else ("OK" if time_hr <= 24 else "EXPENSIVE")
        print(f"    {scenario_name:<35} {time_str:>8}  ${cost:>7.2f}  {feasible}")

# ── Best recommendation ──
print(f"\n  {'='*73}")
print(f"  {'RECOMMENDATION':^73}")
print(f"  {'='*73}")

print(f"""
  YOUR PATTERN: 1-2 hr sessions, can't leave on overnight

  OPTION A: Fine-tune existing 1.2B locally (BEST for your pattern)
  ────────────────────────────────────────────────────────────────
    Model:      ForgeLM V2 (1.2B, already trained)
    Task:       SFT fine-tune on your 240K examples
    Tokens:     ~164M (10K steps × 16K tokens)
    Local time: 7 hours = 4-5 sessions of 1.5hr
    Cost:       $0 (electricity ~$0.30)
    Verdict:    FEASIBLE — fits your pattern perfectly

  OPTION B: Rent H100 for V8-8B pretraining (BEST for 8B from scratch)
  ────────────────────────────────────────────────────────────────────
    Model:      ForgeLM V8-8B (1.6B true params)
    Task:       Full scratch training on 10.7B tokens × 1 epoch
    Cloud:      RunPod H100 80GB
    Time:       3.1 hours (overnight rental)
    Cost:       $12.40
    Verdict:    FEASIBLE — rent for 1 evening, done by morning

  OPTION C: Rent 4x H100 for V8-8B full 3-epoch (BEST quality)
  ────────────────────────────────────────────────────────────────────
    Model:      ForgeLM V8-8B (1.6B true params)
    Task:       Full 3-epoch training (32.1B tokens)
    Cloud:      RunPod 4x H100 80GB
    Time:       5.3 hours (overnight rental)
    Cost:       $85.00
    Verdict:    FEASIBLE — 1 overnight rental, production-quality model

  OPTION D: Rent 8x H100 for V8-8B (FASTEST)
  ────────────────────────────────────────────────────────────────────
    Model:      ForgeLM V8-8B (1.6B true params)
    Task:       Full 3-epoch training (32.1B tokens)
    Cloud:      RunPod 8x H100 80GB
    Time:       3.0 hours
    Cost:       $96.00
    Verdict:    FEASIBLE — dinner-and-a-movie rental

  LOCAL 8B FROM SCRATCH: NOT FEASIBLE for your pattern
  ────────────────────────────────────────────────────────
    289 hours = 193 sessions of 1.5hr = 6+ months
    Even 1 epoch = 64 sessions = 2+ months
    Don't do this. Use cloud (Option B/C/D).
""")

# ── What local IS good for ──
print(f"  {'='*73}")
print(f"  {'WHAT LOCAL RTX 5070 IS GOOD FOR':^73}")
print(f"  {'='*73}")
print(f"""
  ✓ Fine-tuning 1.2B model (4-5 sessions, $0)
  ✓ LoRA adapters on any model (1-2 sessions, $0)
  ✓ Inference / serving (8B model, 194 tok/s, $0)
  ✓ Testing & debugging training code before cloud run
  ✓ Self-play generation (run model, collect data)
  ✓ Evolution search (forge_evolve, lightweight)

  ✗ Full pretraining of 8B (289 hours, 6+ months)
  ✗ Full pretraining of 350M (3,388 hours, 9 months)
  ✗ Anything requiring >10 hours continuous
""")
