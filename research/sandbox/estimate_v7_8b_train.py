"""Estimate V7 8B training time with FreeToken improvements."""
import os, sys, math
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
sys.path.insert(0, r"D:\windsurf\ForgeAI")
from research.config import get_config

configs = ['forgelm_v7_8b_b', 'forgelm_v7_8b_d', 'forgelm_v7_moe', 'forgelm_v7']
for name in configs:
    try:
        c = get_config(name)
        d = c.d_model
        L = c.n_layers
        inter = getattr(c, 'intermediate_size', 4 * d)
        vocab = c.vocab_size
        attn_type = getattr(c, 'attn_type', 'gqa')
        use_moe = getattr(c, 'use_moe', False)
        n_experts = getattr(c, 'moe_num_experts', 0)
        print(f"\n{name}:")
        print(f"  d_model={d}, n_layers={L}, vocab={vocab}, intermediate={inter}")
        print(f"  attn_type={attn_type}, moe={use_moe}, n_experts={n_experts}")

        # Rough param estimate
        # Embedding (tied)
        embed = vocab * d
        # Per layer: attention (qkv + o) + FFN
        kv_heads = getattr(c, 'n_kv_heads', d // 64)
        n_heads = getattr(c, 'n_heads', d // 64)
        head_dim = d // n_heads
        qkv = d * (n_heads * head_dim) + 2 * (kv_heads * head_dim) * d  # q + k + v
        o_proj = d * d
        attn = qkv + o_proj

        if use_moe and n_experts > 0:
            # MoE: router + n_experts * (gate + up + down)
            router = d * n_experts
            ffn_per_expert = 3 * d * inter  # gate + up + down
            ffn = router + n_experts * ffn_per_expert
        else:
            ffn = 3 * d * inter  # SwiGLU: gate + up + down

        total = embed + L * (attn + ffn) + d * 2  # norms
        print(f"  ~{total/1e9:.2f}B params (rough)")
        print(f"  bf16 weights: {total * 2 / 1e9:.2f} GB")
        print(f"  fp32 optimizer (CPU): {total * 12 / 1e9:.2f} GB")
        print(f"  double-buffer extra: {total * 4 / 1e9:.2f} GB")
    except Exception as e:
        print(f"\n{name}: {e}")
