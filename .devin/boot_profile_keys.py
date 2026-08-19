"""Profile meta init: per-key-module cost breakdown.

Measures how much time each key module (TITAN, MoD, MHC, AttnRes, DiffAttn,
BitNet) adds to ConfigurableResearchLLM.__init__ on meta device. This tells
us which modules to prioritize for lazy instantiation.
"""
import os
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

for line in Path("D:/windsurf/ForgeAI/.env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

import torch
from research.config import get_config, ModelConfig


def time_meta_init(config_name: str, label: str, overrides: dict = None) -> float:
    """Time meta-device init with given config overrides."""
    cfg = get_config(config_name, device="meta")
    if overrides:
        cfg_dict = {**cfg.__dict__, **overrides}
        cfg = ModelConfig(**cfg_dict)
    # Force GC of any cached models
    from research.model_loader import ModelLoader
    ModelLoader.clear_cache()
    t0 = time.perf_counter()
    with torch.device("meta"):
        from research.model_loader import ConfigurableResearchLLM
        model = ConfigurableResearchLLM(cfg)
    t = time.perf_counter() - t0
    del model
    return t


def main():
    print("\n=== Meta Init Profile: per-key-module cost ===\n")

    # Baseline: full V3 (all keys on)
    t_full = time_meta_init("forgelm_v3", "full V3 (all keys)")
    print(f"  full V3 (all keys):           {t_full*1000:7.1f} ms")

    # Strip each key individually to measure its cost
    keys_to_strip = [
        ("use_titan_memory", False, "TITAN memory"),
        ("use_mod", False, "MoD router"),
        ("use_mhc", False, "MHC hyper-connections"),
        ("use_attn_residual", False, "AttnRes cross-layer"),
        ("attn_type", "gqa", "DiffAttn (-> GQA)"),
        ("use_bitnet", False, "BitNet b1.58"),
        ("use_qk_norm", False, "QK-Norm"),
        ("use_pit", False, "PIT (already off)"),
    ]

    print(f"\n  Stripping each key individually:")
    costs = {}
    for attr, val, label in keys_to_strip:
        t = time_meta_init("forgelm_v3", label, {attr: val})
        saved = t_full - t
        costs[label] = saved
        print(f"    without {label:25s}: {t*1000:7.1f} ms  (saves {saved*1000:6.1f} ms)")

    # Sort by cost
    print(f"\n  Ranked by cost (most expensive first):")
    for label, saved in sorted(costs.items(), key=lambda x: -x[1]):
        pct = saved / t_full * 100
        print(f"    {label:25s}: {saved*1000:6.1f} ms  ({pct:4.1f}% of total)")

    # No keys at all
    t_nokeys = time_meta_init("forgelm_v3", "no keys", {
        "use_titan_memory": False, "use_mod": False, "use_mhc": False,
        "use_attn_residual": False, "attn_type": "gqa", "use_bitnet": False,
        "use_qk_norm": False,
    })
    print(f"\n  no keys at all:               {t_nokeys*1000:7.1f} ms")
    print(f"  total key overhead:           {(t_full - t_nokeys)*1000:7.1f} ms")

    # Also time lfm25_1.2b (no keys by default)
    t_lfm = time_meta_init("lfm25_1.2b", "lfm25_1.2b")
    print(f"  lfm25_1.2b (reference):       {t_lfm*1000:7.1f} ms")


if __name__ == "__main__":
    main()
