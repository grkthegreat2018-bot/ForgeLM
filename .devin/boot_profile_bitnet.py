"""Clean profile: BitNet vs other keys, with warmup + median of 3 runs."""
import os, sys, time, statistics
sys.path.insert(0, "D:/windsurf/ForgeAI")
from pathlib import Path
for line in Path("D:/windsurf/ForgeAI/.env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())
import torch
from research.config import get_config, ModelConfig
from research.model_loader import ConfigurableResearchLLM, ModelLoader

def t_meta(overrides=None):
    cfg = get_config("forgelm_v3", device="meta")
    if overrides:
        cfg = ModelConfig(**{**cfg.__dict__, **overrides})
    ModelLoader.clear_cache()
    t0 = time.perf_counter()
    with torch.device("meta"):
        m = ConfigurableResearchLLM(cfg)
    t = time.perf_counter() - t0
    del m
    return t

# Warmup imports
_ = t_meta({"use_bitnet": False})

configs = [
    ({}, "full V3"),
    ({"use_bitnet": False}, "no BitNet (other keys on)"),
    ({"use_bitnet": False, "use_titan_memory": False, "use_mod": False,
      "use_mhc": False, "use_attn_residual": False, "attn_type": "gqa",
      "use_qk_norm": False}, "no keys at all"),
]
print("\n=== Clean Profile (median of 3 runs, after warmup) ===\n")
for ov, label in configs:
    times = [t_meta(ov) for _ in range(3)]
    med = statistics.median(times)
    runs = ", ".join(f"{t*1000:.1f}" for t in times)
    print(f"  {label:30s}: {med*1000:7.1f} ms  (runs: {runs})")

# Now measure BitNet conversion cost specifically
print("\n=== BitNet Linear conversion cost ===\n")
from research.training.bitnet_lora import convert_to_bitnet_everywhere

cfg = get_config("forgelm_v3", device="meta")
cfg_no_bitnet = ModelConfig(**{**cfg.__dict__, "use_bitnet": False, "device": "meta"})
ModelLoader.clear_cache()
with torch.device("meta"):
    t0 = time.perf_counter()
    m = ConfigurableResearchLLM(cfg_no_bitnet)
    t_build = time.perf_counter() - t0
    print(f"  Build without BitNet:        {t_build*1000:7.1f} ms")
    t0 = time.perf_counter()
    m = convert_to_bitnet_everywhere(m)
    t_conv = time.perf_counter() - t0
    print(f"  convert_to_bitnet_everywhere: {t_conv*1000:7.1f} ms")
    print(f"  Total (build + convert):     {(t_build+t_conv)*1000:7.1f} ms")
