"""Simulate benchmark import state, then measure ConfigurableResearchLLM init."""
import time, os, sys
sys.path.insert(0, "D:/windsurf/ForgeAI")
from pathlib import Path
for line in Path("D:/windsurf/ForgeAI/.env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

# Simulate what the benchmark imports before calling run_variation
import torch
from research.config import get_config, ModelConfig
from research.model_loader import ConfigurableResearchLLM, ModelLoader

# Now measure ConfigurableResearchLLM init on meta device
ModelLoader.clear_cache()
cfg = get_config("forgelm_v3", device="meta")

t0 = time.perf_counter()
with torch.device("meta"):
    model = ConfigurableResearchLLM(cfg)
t1 = time.perf_counter()
print(f"ConfigurableResearchLLM (cold, after torch+model_loader): {(t1-t0)*1000:.1f} ms")

# Second run (warm)
ModelLoader.clear_cache()
t0 = time.perf_counter()
with torch.device("meta"):
    model2 = ConfigurableResearchLLM(cfg)
t1 = time.perf_counter()
print(f"ConfigurableResearchLLM (warm):                          {(t1-t0)*1000:.1f} ms")

# Now check what was imported during the cold run
import importlib
key_mods = [m for m in sys.modules if "research.keys" in m]
print(f"\nKey modules in sys.modules after cold run: {len(key_mods)}")
for m in sorted(key_mods):
    print(f"  {m}")

# Check triton
triton_mods = [m for m in sys.modules if "triton" in m]
print(f"\nTriton modules in sys.modules: {len(triton_mods)}")
