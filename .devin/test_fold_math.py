"""Direct test: norm folding math on actual model weights."""
import sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from safetensors.torch import load_file

orig = load_file('research/checkpoints/forgelm_v2.safetensors')
opt = load_file('research/checkpoints/forgelm_v2_opt.safetensors')

# Test layer 0: ln1 -> q_proj
i = 0
ln1_orig = orig[f'blocks.{i}.ln1.weight'].float()
ln1_opt = opt[f'blocks.{i}.ln1.weight'].float()
q_orig = orig[f'blocks.{i}.attn.q_proj.weight'].float()
q_opt = opt[f'blocks.{i}.attn.q_proj.weight'].float()
q_bias = orig[f'blocks.{i}.attn.q_proj.bias'].float()

print(f"ln1 orig: max_dev from 1.0 = {(ln1_orig - 1).abs().max():.4f}")
print(f"ln1 opt:  max_dev from 1.0 = {(ln1_opt - 1).abs().max():.4f}")
print(f"q_proj diff: {(q_orig - q_opt).abs().max():.4f}")

# Simulate RMSNorm + q_proj
x = torch.randn(1, 4, 1536)
eps = 1e-6

# Original: RMSNorm(x) * ln1, then q_proj
rms = x.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
x_normed_orig = x * rms * ln1_orig  # RMSNorm with real gamma
q_out_orig = x_normed_orig @ q_orig.T + q_bias

# Folded: RMSNorm(x) * 1.0, then q_proj_folded
x_normed_folded = x * rms * ln1_opt  # RMSNorm with gamma=1.0
q_out_folded = x_normed_folded @ q_opt.T + q_bias

diff = (q_out_orig - q_out_folded).abs().max().item()
cos = F.cosine_similarity(q_out_orig.flatten().unsqueeze(0), q_out_folded.flatten().unsqueeze(0)).item()
print(f"\nq_proj output: diff={diff:.8f}, cos={cos:.8f}")

# Check: is q_opt = q_orig * ln1?
q_expected = q_orig * ln1_orig.unsqueeze(0)
q_diff = (q_expected - q_opt).abs().max().item()
print(f"q_opt vs q_orig*ln1: diff={q_diff:.8f}")

# Also test kv_down_proj
kv_orig = orig[f'blocks.{i}.attn.kv_down_proj.weight'].float()
kv_opt = opt[f'blocks.{i}.attn.kv_down_proj.weight'].float()
kv_out_orig = x_normed_orig @ kv_orig.T
kv_out_folded = x_normed_folded @ kv_opt.T
kv_diff = (kv_out_orig - kv_out_folded).abs().max().item()
kv_cos = F.cosine_similarity(kv_out_orig.flatten().unsqueeze(0), kv_out_folded.flatten().unsqueeze(0)).item()
print(f"kv_down output: diff={kv_diff:.8f}, cos={kv_cos:.8f}")

# Test ln2 -> expert w1
ln2_orig = orig[f'blocks.{i}.ln2.weight'].float()
ln2_opt = opt[f'blocks.{i}.ln2.weight'].float()
w1_orig = orig[f'blocks.{i}.ffn.experts.0.w1.weight'].float()
w1_opt = opt[f'blocks.{i}.ffn.experts.0.w1.weight'].float()

print(f"\nln2 orig: max_dev from 1.0 = {(ln2_orig - 1).abs().max():.4f}")
print(f"ln2 opt:  max_dev from 1.0 = {(ln2_opt - 1).abs().max():.4f}")
print(f"w1 diff: {(w1_orig - w1_opt).abs().max():.4f}")

# Simulate: x after attention residual
x2 = torch.randn(1, 4, 1536)
rms2 = x2.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
x2_normed_orig = x2 * rms2 * ln2_orig
x2_normed_folded = x2 * rms2 * ln2_opt

w1_out_orig = x2_normed_orig @ w1_orig.T
w1_out_folded = x2_normed_folded @ w1_opt.T
w1_diff = (w1_out_orig - w1_out_folded).abs().max().item()
w1_cos = F.cosine_similarity(w1_out_orig.flatten().unsqueeze(0), w1_out_folded.flatten().unsqueeze(0)).item()
print(f"w1 output: diff={w1_diff:.8f}, cos={w1_cos:.8f}")

# Check: is w1_opt = w1_orig * ln2?
w1_expected = w1_orig * ln2_orig.unsqueeze(0)
w1_check = (w1_expected - w1_opt).abs().max().item()
print(f"w1_opt vs w1_orig*ln2: diff={w1_check:.8f}")

# Test ln_f -> head
lnf_orig = orig['ln_f.weight'].float()
lnf_opt = opt['ln_f.weight'].float()
head_orig = orig['head.weight'].float()
head_opt = opt['head.weight'].float()

print(f"\nln_f orig: max_dev from 1.0 = {(lnf_orig - 1).abs().max():.4f}")
print(f"ln_f opt:  max_dev from 1.0 = {(lnf_opt - 1).abs().max():.4f}")
print(f"head diff: {(head_orig - head_opt).abs().max():.4f}")

x3 = torch.randn(1, 4, 1536)
rms3 = x3.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
x3_normed_orig = x3 * rms3 * lnf_orig
x3_normed_folded = x3 * rms3 * lnf_opt

head_out_orig = x3_normed_orig @ head_orig.T
head_out_folded = x3_normed_folded @ head_opt.T
head_diff = (head_out_orig - head_out_folded).abs().max().item()
head_cos = F.cosine_similarity(head_out_orig.flatten().unsqueeze(0), head_out_folded.flatten().unsqueeze(0)).item()
print(f"head output: diff={head_diff:.8f}, cos={head_cos:.8f}")
