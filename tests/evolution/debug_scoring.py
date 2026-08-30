"""Debug cross_layer_kv and gla_attention metrics."""
import sys, os, json, random
sys.path.insert(0, r"D:\windsurf\ForgeAI")
os.environ["PYTHONUTF8"] = "1"

import torch
from research.evolution.domain_spec import JSONSpecDomain

for name in ["cross_layer_kv", "gla_attention"]:
    domain = JSONSpecDomain(spec_name=name, device=torch.device("cpu"))
    print(f"\n=== {name} ===")

    # Test seed configs
    seeds = domain.seed_configs()
    for config in seeds[:5]:
        metrics = domain.simulate(config)
        result = domain.evaluate(config)
        print(f"  config={config}")
        print(f"  metrics={metrics}")
        print(f"  score={result['score']:.2f}")
        print()

    # Test optimized configs
    if name == "cross_layer_kv":
        test_configs = [
            {"share_ratio": 1.0, "n_share_groups": 8, "share_mode": "learned"},
            {"share_ratio": 1.0, "n_share_groups": 4, "share_mode": "avg"},
            {"share_ratio": 0.5, "n_share_groups": 8, "share_mode": "learned"},
            {"share_ratio": 1.0, "n_share_groups": 2, "share_mode": "avg"},
        ]
    else:
        test_configs = [
            {"latent_dim": 64, "n_heads": 4, "compression_ratio": 8.0},
            {"latent_dim": 128, "n_heads": 8, "compression_ratio": 4.0},
            {"latent_dim": 256, "n_heads": 16, "compression_ratio": 2.0},
        ]

    print("  --- Optimized configs ---")
    for config in test_configs:
        metrics = domain.simulate(config)
        result = domain.evaluate(config)
        print(f"  config={config}")
        print(f"  metrics={metrics}")
        print(f"  score={result['score']:.2f}")
        print()
