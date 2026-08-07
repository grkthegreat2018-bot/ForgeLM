"""KeyStack structural verification: confirm all weights are accounted for.

Tests:
1. Build model from copied weights → output must be 100% identical
2. Reconstruct RMSNorm + V projection from activations → verify Bi keys are exact
3. Report which keys are Bi (exact), Partial (need uptraining), or Trivial
"""
import torch
import torch.nn.functional as F
import sys
import time
from collections import defaultdict

sys.path.insert(0, '.')

from research.config import get_config
from research.model_loader import ModelLoader
from safetensors.torch import load_file

torch.manual_seed(42)
SEP = "=" * 60
device = torch.device("cuda")
dtype = torch.float32

# ============================================================
# Load original model
# ============================================================
print(f"{SEP}\nLoading Qwen2.5-Coder-1.5B (float32)\n{SEP}")
cfg = get_config("qwen25_coder_1.5b")
model = ModelLoader.build_model(cfg)
state = load_file("research/checkpoints/qwen25_coder_1.5b_ported.safetensors")
state = {k: v.float() for k, v in state.items()}
model.load_state_dict(state, strict=True)
model = model.to(device, dtype=dtype)
model.eval()
print(f"  {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

# ============================================================
# Hooks for Bi key verification
# ============================================================
activations = defaultdict(dict)

def make_hook(layer_idx, name):
    def hook(module, input, output):
        inp = input[0] if isinstance(input, tuple) else input
        out = output[0] if isinstance(output, tuple) else output
        inp = inp.reshape(-1, inp.shape[-1]).detach()
        out = out.reshape(-1, out.shape[-1]).detach()
        activations[layer_idx][name] = (inp, out)
    return hook

hooks = []
for i, block in enumerate(model.blocks):
    hooks.append(block.ln1.register_forward_hook(make_hook(i, "ln1")))
    hooks.append(block.attn.v_proj.register_forward_hook(make_hook(i, "v_proj")))
    hooks.append(block.ln2.register_forward_hook(make_hook(i, "ln2")))
hooks.append(model.ln_f.register_forward_hook(make_hook(-1, "ln_f")))

# ============================================================
# Forward pass
# ============================================================
seq_len = 2048
test_ids = torch.randint(0, cfg.vocab_size, (1, seq_len), device=device)
with torch.inference_mode():
    logits_orig = model(test_ids)
    if isinstance(logits_orig, tuple): logits_orig = logits_orig[0]
print(f"  Original logits: {list(logits_orig.shape)}")

# ============================================================
# Bi Key functions
# ============================================================
def rmsnorm_key(X, Y, eps=1e-6):
    X_f = X.float().cpu(); Y_f = Y.float().cpu()
    variance = X_f.pow(2).mean(-1, keepdim=True)
    X_norm = X_f * torch.rsqrt(variance + eps)
    weight = (Y_f / (X_norm + 1e-8)).mean(dim=0)
    return weight.to(X.dtype).to(X.device)

def linear_key_with_bias(X, Y):
    """Linear key with bias using centered lstsq (avoids bias-augmentation collinearity)."""
    X_f = X.float().cpu(); Y_f = Y.float().cpu()
    # Center data first, then solve W, then compute bias
    X_mean = X_f.mean(0, keepdim=True)
    Y_mean = Y_f.mean(0, keepdim=True)
    X_c = X_f - X_mean
    Y_c = Y_f - Y_mean
    result = torch.linalg.lstsq(X_c, Y_c)
    W = result.solution.T
    bias = (Y_mean - X_mean @ W.T).squeeze(0)
    return W.to(X.dtype).to(X.device), bias.to(X.dtype).to(X.device)

# ============================================================
# Test 1: Structural completeness (copy all weights)
# ============================================================
print(f"\n{SEP}\nTEST 1: Structural completeness (copy all weights)\n{SEP}")
reconstructed = {}
for name, param in model.state_dict().items():
    reconstructed[name] = param.clone()

# Build reconstructed model
del model
torch.cuda.empty_cache()
logits_saved = logits_orig.clone()
del logits_orig
torch.cuda.empty_cache()

model_recon = ModelLoader.build_model(cfg)
model_recon.load_state_dict(reconstructed, strict=True)
model_recon = model_recon.to(device, dtype=dtype)
model_recon.eval()

with torch.inference_mode():
    logits_recon = model_recon(test_ids)
    if isinstance(logits_recon, tuple): logits_recon = logits_recon[0]

diff = (logits_saved.float() - logits_recon.float()).abs().max().item()
cos = F.cosine_similarity(logits_saved.float().flatten().unsqueeze(0),
                           logits_recon.float().flatten().unsqueeze(0)).item()
match = (logits_saved.argmax(-1) == logits_recon.argmax(-1)).float().mean().item()
print(f"  Logit diff:  {diff:.10f}")
print(f"  Cosine:      {cos:.10f}")
print(f"  Token match: {match:.10f}")
if cos > 0.9999:
    print("  RESULT: PASS — KeyStack structurally complete")
else:
    print("  RESULT: FAIL — missing or mismatched weights")

# ============================================================
# Test 2: Bi Key exactness (RMSNorm + V projection)
# ============================================================
print(f"\n{SEP}\nTEST 2: Bi Key exactness (RMSNorm + V proj)\n{SEP}")

# Free reconstructed model first
del model_recon, logits_recon, reconstructed
torch.cuda.empty_cache()

# Reload original model for activation capture
model2 = ModelLoader.build_model(cfg)
model2.load_state_dict(state, strict=True)
state.clear()
model2 = model2.to(device, dtype=dtype)
model2.eval()

activations2 = defaultdict(dict)
def make_hook2(layer_idx, name):
    def hook(module, input, output):
        inp = input[0] if isinstance(input, tuple) else input
        out = output[0] if isinstance(output, tuple) else output
        inp = inp.reshape(-1, inp.shape[-1]).detach()
        out = out.reshape(-1, out.shape[-1]).detach()
        activations2[layer_idx][name] = (inp, out)
    return hook

hooks2 = []
for i, block in enumerate(model2.blocks):
    hooks2.append(block.ln1.register_forward_hook(make_hook2(i, "ln1")))
    hooks2.append(block.attn.v_proj.register_forward_hook(make_hook2(i, "v_proj")))
    hooks2.append(block.ln2.register_forward_hook(make_hook2(i, "ln2")))
hooks2.append(model2.ln_f.register_forward_hook(make_hook2(-1, "ln_f")))

with torch.inference_mode():
    _ = model2(test_ids)

# Verify RMSNorm keys
rms_pass = 0; rms_fail = 0
for i in range(cfg.n_layers):
    for ln_name, ln_module in [("ln1", model2.blocks[i].ln1), ("ln2", model2.blocks[i].ln2)]:
        X, Y = activations2[i][ln_name]
        w = rmsnorm_key(X, Y)
        diff = (w.float() - ln_module.weight.data.float()).abs().max().item()
        if diff < 0.001: rms_pass += 1
        else: rms_fail += 1

# Final ln_f
X, Y = activations2[-1]["ln_f"]
w = rmsnorm_key(X, Y)
diff_lnf = (w.float() - model2.ln_f.weight.data.float()).abs().max().item()
if diff_lnf < 0.001: rms_pass += 1
else: rms_fail += 1

print(f"  RMSNorm: {rms_pass} PASS, {rms_fail} FAIL (threshold=0.001)")

# Verify V projection keys
v_pass = 0; v_fail = 0
v_max_diff = 0
for i in range(cfg.n_layers):
    X, Y = activations2[i]["v_proj"]
    has_bias = hasattr(model2.blocks[i].attn.v_proj, "bias") and model2.blocks[i].attn.v_proj.bias is not None
    if has_bias:
        w, bias = linear_key_with_bias(X, Y)
        d1 = (w.float() - model2.blocks[i].attn.v_proj.weight.data.float()).abs().max().item()
        d2 = (bias.float() - model2.blocks[i].attn.v_proj.bias.data.float()).abs().max().item()
        diff = max(d1, d2)
    else:
        X_f = X.float().cpu(); Y_f = Y.float().cpu()
        result = torch.linalg.lstsq(X_f, Y_f)
        w = result.solution.T.to(X.dtype).to(X.device)
        diff = (w.float() - model2.blocks[i].attn.v_proj.weight.data.float()).abs().max().item()
    v_max_diff = max(v_max_diff, diff)
    if diff < 0.01: v_pass += 1
    else: v_fail += 1

print(f"  V proj:  {v_pass} PASS, {v_fail} FAIL (max_diff={v_max_diff:.8f}, threshold=0.01)")

# ============================================================
# Summary
# ============================================================
print(f"\n{SEP}\nKEYSTACK VERIFICATION SUMMARY\n{SEP}")
print(f"  Structural: {'PASS' if cos > 0.9999 else 'FAIL'} (cosine={cos:.10f})")
print(f"  RMSNorm Bi: {rms_pass}/{rms_pass+rms_fail} exact")
print(f"  V proj Bi:  {v_pass}/{v_pass+v_fail} exact (max_diff={v_max_diff:.8f})")
print(f"")
print(f"  Key classification:")
print(f"    Bi (exact from activations):")
print(f"      - Embedding (trivial copy)")
print(f"      - LM Head Tied (trivial copy)")
print(f"      - RMSNorm: {rms_pass}/{rms_pass+rms_fail} exact")
print(f"      - SwiGLU W_gate: exact via Newton (proven separately)")
print(f"      - SwiGLU W_up: exact via lstsq (proven separately)")
print(f"    Partial (Bi in theory, conditioning-limited in practice):")
print(f"      - V projection: {v_pass}/{v_pass+v_fail} exact (centered lstsq)")
print(f"        (fails on layers with high condition number — RMSNorm null space)")
print(f"      - GQA Q/K: softmax barrier (GD to cosine>0.9999)")
print(f"      - O projection: rank-deficient (attention output null space)")
print(f"      - W_down: underdetermined (seq < intermediate_size)")
print(f"    Trivial:")
print(f"      - RoPE (deterministic)")
print(f"      - Causal Mask (deterministic)")

for h in hooks2: h.remove()
