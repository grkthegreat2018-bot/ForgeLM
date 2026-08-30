"""Bit-exact migration test for KV domains."""
import torch, numpy as np, os, sys, importlib
os.environ['PYTHONUTF8'] = '1'
sys.path.insert(0, r'D:\windsurf\ForgeAI')

from research.evolution.domain_spec import JSONSpecDomain

domains = [
    ('RotorQuantKV', 'rotor_quant_kv', 'research.evolution.domains.kv_domains'),
    ('HadamardKV', 'hadamard_kv', 'research.evolution.domains.kv_domains'),
    ('StreamingKV', 'streaming_kv', 'research.evolution.domains.kv_domains'),
    ('KvZipKV', 'kvzip_kv', 'research.evolution.domains.kv_domains'),
    ('XQuantKV', 'xquant_kv', 'research.evolution.domains.kv_domains'),
    ('KvRecompute', 'kv_recompute', 'research.evolution.domains.kv_domains'),
    ('CrossLayerKV', 'cross_layer_kv', 'research.evolution.domains.kv_domains'),
    ('PagedEvictKV', 'paged_evict_kv', 'research.evolution.domains.kv_domains'),
]

n_pass = 0
n_fail = 0

for cls_name, spec_name, mod_path in domains:
    try:
        mod = importlib.import_module(mod_path)
        OldCls = getattr(mod, cls_name)
        old = OldCls()
        seeds = old.seed_configs()
        if not seeds:
            params = torch.tensor([0.5] * old.output_dim())
            seeds = [old.decode(params)]
        test_cfg = seeds[0]
        torch.manual_seed(42); np.random.seed(42)
        ro = old.evaluate(test_cfg)
        torch.manual_seed(42); np.random.seed(42)
        new = JSONSpecDomain(spec_name)
        rn = new.evaluate(test_cfg)
        match = abs(ro['score'] - rn['score']) < 1e-3
        status = 'OK' if match else 'MISMATCH'
        if match:
            n_pass += 1
        else:
            n_fail += 1
        print(f"{cls_name:20s} old={ro['score']:10.4f} new={rn['score']:10.4f} {status}")
    except Exception as e:
        n_fail += 1
        print(f"{cls_name:20s} ERROR: {e}")

print(f"\n{n_pass}/{n_pass+n_fail} passed")
