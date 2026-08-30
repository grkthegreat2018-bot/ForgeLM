"""Quick Vast.ai offer search — shows best perf/$ GPUs for 8B training."""
import json, subprocess, sys, os

API_KEY = os.environ.get("VAST_API_KEY", "")
VASTAI = r"D:\windsurf\ForgeAI\venv\Scripts\vastai.exe"

cmd = [
    VASTAI, "search", "offers",
    "direct_port_count>=1 rentable=true reliability>=0.9",
    "--raw", "--limit", "20", "-o", "dlperf_usd-",
    "-n",  # no default query
]
env = dict(os.environ)
if API_KEY:
    env["VAST_API_KEY"] = API_KEY

proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=60)
if proc.returncode != 0:
    print(f"ERROR (rc={proc.returncode}): {proc.stderr[:500]}")
    sys.exit(1)

offers = json.loads(proc.stdout)
print(f"Found {len(offers)} offers (sorted by dlperf/$ descending):\n")
print(f"{'ID':>10} {'GPU':>20} {'xN':>3} {'VRAM':>6} {'$/hr':>8} {'dlperf':>7} {'perf/$':>8} {'rel':>5} {'Location':>20}")
print("-" * 100)
for o in offers[:20]:
    gpu = o.get("gpu_name", "?")
    ng = o.get("num_gpus", 1)
    vram = o.get("gpu_total_ram", 0) / 1024
    dph = o.get("dph_total", 0)
    dlperf = o.get("dlperf", 0)
    ppd = o.get("dlperf_per_dphtotal", 0)
    rel = o.get("reliability", 0)
    loc = o.get("geolocation", "?")
    oid = o.get("id", "?")
    print(f"{oid:>10} {gpu:>20} x{ng:<2} {vram:>5.0f}GB ${dph:>7.3f}/h {dlperf:>7.0f} {ppd:>8.0f} {rel:>5.2f} {loc:>20}")

# Estimate cost for 100-step smoke test
print("\n--- Cost estimate for 100-step smoke test (est 5s/step = 8.3 min) ---")
est_hours = 100 * 5.0 / 3600.0
for o in offers[:5]:
    cost = o["dph_total"] * est_hours
    print(f"  #{o['id']} {o['gpu_name']}: ${cost:.3f} (budget $10)")
