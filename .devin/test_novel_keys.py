"""Test all 4 novel keys."""
import sys
sys.path.insert(0, '.')
import torch

print("=" * 60)
print("Testing 4 Novel Keys")
print("=" * 60)

# 1. Norm Folding
print("\n[1] Norm Folding Key...")
from research.keys.norm_folding_key import NormFoldingKey
key = NormFoldingKey()
state = {
    "blocks.0.ln1.weight": torch.ones(128, dtype=torch.bfloat16) * 2.0,
    "blocks.0.ln2.weight": torch.ones(128, dtype=torch.bfloat16) * 3.0,
    "blocks.0.attn.q_proj.weight": torch.randn(128, 128, dtype=torch.bfloat16),
    "blocks.0.attn.kv_down_proj.weight": torch.randn(128, 128, dtype=torch.bfloat16),
    "blocks.0.ffn.w_gate.weight": torch.randn(256, 128, dtype=torch.bfloat16),
    "blocks.0.ffn.w_up.weight": torch.randn(256, 128, dtype=torch.bfloat16),
    "ln_f.weight": torch.ones(128, dtype=torch.bfloat16) * 1.5,
    "head.weight": torch.randn(1000, 128, dtype=torch.bfloat16),
}
r = key.forward(state)
print(f"  Success: {r.success}, norms_folded: {r.metadata.get('norms_folded', 'N/A')}")
assert "blocks.0.ln1.weight" not in r.weights, "ln1 not folded!"
assert "ln_f.weight" not in r.weights, "ln_f not folded!"
print("  Norm Folding verified ✓")

# 2. Expert Consolidation
print("\n[2] Expert Consolidation Key...")
from research.keys.expert_consolidation_key import ExpertConsolidationKey
key = ExpertConsolidationKey(threshold=0.9, min_experts=1)
state = {}
base_w = torch.randn(256, 128, dtype=torch.bfloat16)
for i in range(2):
    for ei in range(4):
        if ei < 2:
            w = base_w.clone()  # experts 0,1 are identical
        else:
            w = torch.randn(256, 128, dtype=torch.bfloat16)
        state[f"blocks.{i}.ffn.experts.{ei}.w_gate.weight"] = w
        state[f"blocks.{i}.ffn.experts.{ei}.w_up.weight"] = w.clone()
        state[f"blocks.{i}.ffn.experts.{ei}.w_down.weight"] = w.t().contiguous()
    state[f"blocks.{i}.ln1.weight"] = torch.ones(128, dtype=torch.bfloat16)
r = key.forward(state)
print(f"  Success: {r.success}, merged: {r.metadata['total_merged']} pairs")
assert r.metadata["total_merged"] > 0, "Should merge identical experts!"
print("  Expert Consolidation verified ✓")

# 3. GRAIL Compensation
print("\n[3] GRAIL Compensation Key...")
from research.keys.grail_key import GRAILKey
key = GRAILKey()
d, N = 64, 256
x_orig = torch.randn(N, d)
Q = torch.linalg.qr(torch.randn(d, d))[0]
x_trans = x_orig @ Q + 0.01 * torch.randn(N, d)
R = key.compute_reconstruction_map(x_orig, x_trans)
x_recon = x_trans @ R
err_before = (x_orig - x_trans).pow(2).mean().item()
err_after = (x_orig - x_recon).pow(2).mean().item()
print(f"  Error before: {err_before:.6f}, after: {err_after:.6f}")
assert err_after < err_before, "GRAIL should reduce error!"
print("  GRAIL verified ✓")

# 4. Activation Transmutation
print("\n[4] Activation Transmutation Key...")
from research.keys.activation_transmute_key import ActivationTransmuteKey
key = ActivationTransmuteKey(target="reglu")
g = torch.randn(1024, 256) * 3.0
alpha, beta = key.solve_per_channel(g)
source = key.source_activation(g)
target = key.target_activation(alpha * g + beta)
mse = (source - target).pow(2).mean().item()
rel_err = mse / source.pow(2).mean().item()
print(f"  SwiGLU → ReGLU relative error: {rel_err:.4f} ({rel_err*100:.1f}%)")
print(f"  alpha: mean={alpha.mean():.4f}, range=[{alpha.min():.4f}, {alpha.max():.4f}]")
assert rel_err < 0.3, "Error too high!"
print("  Activation Transmutation verified ✓")

print("\n" + "=" * 60)
print("All 4 novel keys verified ✓")
print("=" * 60)
