"""Measure import cost of each key module."""
import time, sys

modules = [
    "research.keys.architecture.titan_memory_key",
    "research.keys.architecture.mod_router_key",
    "research.keys.architecture.mhc_key",
    "research.keys.architecture.attn_residual_key",
    "research.keys.attention.differential_attn_key",
    "research.keys.quantization.bitnet_b158_key",
    "research.keys.misc.pit_key",
    "research.training.bitnet_lora",
]

# Clear any cached imports
for m in list(sys.modules):
    if "research.keys" in m or "research.training.bitnet" in m:
        del sys.modules[m]

total = 0.0
for mod in modules:
    t0 = time.perf_counter()
    __import__(mod)
    t = time.perf_counter() - t0
    total += t
    print(f"  {mod:55s}: {t*1000:7.1f} ms")
print(f"  {'TOTAL':55s}: {total*1000:7.1f} ms")
