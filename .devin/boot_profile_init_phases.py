"""Profile what takes time in ConfigurableResearchLLM.__init__ (cold vs warm)."""
import time, os, sys
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

# Key modules are pre-imported by model_loader.py
key_mods_before = len([m for m in sys.modules if "research.keys" in m])
print(f"Key modules before init: {key_mods_before}")

cfg = get_config("forgelm_v3", device="meta")

# Instrument: measure each phase of ConfigurableResearchLLM.__init__
# by building with different config overrides to isolate costs

# 1. No keys at all
ModelLoader.clear_cache()
cfg_none = ModelConfig(**{**cfg.__dict__, "use_titan_memory": False,
    "use_mod": False, "use_mhc": False, "use_attn_residual": False,
    "attn_type": "gqa", "use_bitnet": False, "use_qk_norm": False,
    "device": "meta"})
t0 = time.perf_counter()
with torch.device("meta"):
    m = ConfigurableResearchLLM(cfg_none)
t1 = time.perf_counter()
print(f"No keys (cold):           {(t1-t0)*1000:7.1f} ms")
del m

# 2. Only TITAN
ModelLoader.clear_cache()
cfg_titan = ModelConfig(**{**cfg.__dict__, "use_titan_memory": True,
    "use_mod": False, "use_mhc": False, "use_attn_residual": False,
    "attn_type": "gqa", "use_bitnet": False, "use_qk_norm": False,
    "device": "meta"})
t0 = time.perf_counter()
with torch.device("meta"):
    m = ConfigurableResearchLLM(cfg_titan)
t1 = time.perf_counter()
print(f"Only TITAN (cold):        {(t1-t0)*1000:7.1f} ms")
del m

# 3. Only BitNet
ModelLoader.clear_cache()
cfg_bitnet = ModelConfig(**{**cfg.__dict__, "use_titan_memory": False,
    "use_mod": False, "use_mhc": False, "use_attn_residual": False,
    "attn_type": "gqa", "use_bitnet": True, "use_qk_norm": False,
    "device": "meta"})
t0 = time.perf_counter()
with torch.device("meta"):
    m = ConfigurableResearchLLM(cfg_bitnet)
t1 = time.perf_counter()
print(f"Only BitNet (cold):       {(t1-t0)*1000:7.1f} ms")
del m

# 4. Only DiffAttn
ModelLoader.clear_cache()
cfg_diff = ModelConfig(**{**cfg.__dict__, "use_titan_memory": False,
    "use_mod": False, "use_mhc": False, "use_attn_residual": False,
    "attn_type": "diff", "use_bitnet": False, "use_qk_norm": False,
    "device": "meta"})
t0 = time.perf_counter()
with torch.device("meta"):
    m = ConfigurableResearchLLM(cfg_diff)
t1 = time.perf_counter()
print(f"Only DiffAttn (cold):     {(t1-t0)*1000:7.1f} ms")
del m

# 5. Full V3
ModelLoader.clear_cache()
t0 = time.perf_counter()
with torch.device("meta"):
    m = ConfigurableResearchLLM(cfg)
t1 = time.perf_counter()
print(f"Full V3 (cold):           {(t1-t0)*1000:7.1f} ms")
del m

# 6. Full V3 again (warm)
ModelLoader.clear_cache()
t0 = time.perf_counter()
with torch.device("meta"):
    m = ConfigurableResearchLLM(cfg)
t1 = time.perf_counter()
print(f"Full V3 (warm):           {(t1-t0)*1000:7.1f} ms")
del m

# Check what new modules were imported during cold runs
key_mods_after = len([m for m in sys.modules if "research.keys" in m])
triton_mods_after = len([m for m in sys.modules if "triton" in m])
print(f"\nKey modules after all runs: {key_mods_after}")
print(f"Triton modules after all runs: {triton_mods_after}")
