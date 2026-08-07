"""Test all new keys."""
import sys
sys.path.insert(0, '.')

from research.keys.qk_norm_mla_key import QKNormMLAKey
from research.keys.wq_elim_key import WQElimKey
from research.keys.denseformer_key import DenseFormerKey
from research.keys.logit_cap_key import LogitCapKey
from research.keys.swiglu_clamp_key import SwiGLUClampKey
from research.keys.sandwich_norm_key import SandwichNormKey

# Test QK-Norm MLA
k = QKNormMLAKey()
r = k.forward({"n_layers": 28, "head_dim": 128})
print(f"QK-Norm MLA: {r.success}, {len(r.weights)} tensors, {r.metadata['total_params']} params")

# Test WQ Elim
k = WQElimKey()
r = k.forward({"n_layers": 28, "d_model": 1536, "has_bias": True})
print(f"WQ Elim: {r.success}, {len(r.weights)} tensors, saves {r.metadata['params_saved']/1e6:.1f}M params")

# Test DenseFormer
k = DenseFormerKey()
r = k.forward({"n_layers": 28, "dilation": 1})
print(f"DenseFormer: {r.success}, {r.metadata['total_params']} DWA params")

# Test Logit Cap
k = LogitCapKey()
print(f"Logit Cap: {k.name}, class={k.key_class().value}")

# Test SwiGLU Clamp
k = SwiGLUClampKey()
print(f"SwiGLU Clamp: {k.name}, class={k.key_class().value}")

# Test Sandwich Norm
k = SandwichNormKey()
r = k.forward({"d_model": 1536, "n_layers": 28})
print(f"SandwichNorm: {r.success}, {len(r.weights['post_attn_norms'])}+{len(r.weights['post_ffn_norms'])} norms")

# Verify keystack
from research.keys.keystack import build_xp_keystack
s = build_xp_keystack()
print(f"\nKeyStack: {len(s.keys)} keys total")
for key in s.keys:
    print(f"  {key.name:25s} {key.key_class().value}")
