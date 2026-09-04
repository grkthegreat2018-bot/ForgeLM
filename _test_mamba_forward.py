"""Compare our Mamba forward pass with the HF Jamba reference.

Loads one Mamba layer's weights from the ported checkpoint, runs our forward,
and compares with a from-scratch reference implementation.
"""
import os, sys, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from safetensors.torch import load_file
from research.keys.architecture.mamba_probe import MambaLayer

CKPT = "research/checkpoints/Jamba_Reasoning_3B.safetensors"
LAYER = 0  # Mamba layer

# Load weights for layer 0
state = load_file(CKPT)
prefix = f"blocks.{LAYER}.attn."
w = {k.replace(prefix, ""): v for k, v in state.items() if k.startswith(prefix)}
print("Loaded keys:", sorted(w.keys()))

# Build our Mamba layer
d_model = 2560
d_state = 16
d_conv = 4
expand = 2
dt_rank = 160
d_inner = d_model * expand

our_mamba = MambaLayer(d_model=d_model, d_state=d_state, d_conv=d_conv,
                        expand=expand, dt_rank=dt_rank, bias=False,
                        conv_bias=True, use_jamba_norms=True)
our_mamba.eval()

# Load weights
our_mamba.in_proj.weight.data = w["in_proj.weight"]
our_mamba.conv1d.weight.data = w["conv1d.weight"]
our_mamba.conv1d.bias.data = w["conv1d.bias"]
our_mamba.x_proj.weight.data = w["x_proj.weight"]
our_mamba.dt_proj.weight.data = w["dt_proj.weight"]
our_mamba.dt_proj.bias.data = w["dt_proj.bias"]
our_mamba.A_log.data = w["A_log"]
our_mamba.D.data = w["D"]
our_mamba.out_proj.weight.data = w["out_proj.weight"]
# Jamba norms
our_mamba.dt_layernorm.data = w["dt_layernorm"]
our_mamba.b_layernorm.data = w["b_layernorm"]
our_mamba.c_layernorm.data = w["c_layernorm"]

# Reference implementation (matches HF Jamba's naive Mamba forward)
def ref_mamba_forward(x, in_proj, conv1d, x_proj, dt_proj, A_log, D,
                      out_proj, dt_ln, b_ln, c_ln, d_state, d_conv, d_inner, eps=1e-6):
    """Reference forward matching HF Jamba's modeling_jamba.py naive path."""
    B, T, _ = x.shape

    # in_proj -> xz, split into x and z
    xz = in_proj(x)  # (B, T, 2*d_inner)
    _x, z = xz.chunk(2, dim=-1)  # _x is ssm input, z is gate

    # conv1d (causal)
    _x = _x.transpose(1, 2)  # (B, d_inner, T)
    _x = conv1d(_x)[:, :, :T]  # causal trim
    _x = _x.transpose(1, 2)  # (B, T, d_inner)
    _x = F.silu(_x)

    # x_proj -> delta, B, C
    x_proj_out = x_proj(_x)  # (B, T, dt_rank + 2*d_state)
    delta, B_p, C_p = x_proj_out.split([dt_rank, d_state, d_state], dim=-1)

    # Jamba RMSNorms
    def rmsnorm(x, weight, eps):
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        return (x.to(weight.dtype) * weight).to(x.dtype)

    delta = rmsnorm(delta, dt_ln, eps).to(x.dtype)
    B_p = rmsnorm(B_p, b_ln, eps).to(x.dtype)
    C_p = rmsnorm(C_p, c_ln, eps).to(x.dtype)

    # dt_proj with softplus
    delta = F.softplus(dt_proj(delta))  # (B, T, d_inner)

    # Selective scan (reference loop)
    A = -torch.exp(A_log.float())  # (d_inner, d_state) — float32 for stability
    print(f"  A shape: {A.shape}")
    print(f"  delta shape: {delta.shape}")
    print(f"  _x shape: {_x.shape}")
    print(f"  B_p shape: {B_p.shape}")
    h = torch.zeros(B, d_inner, d_state, device=x.device, dtype=torch.float32)
    ys = []
    for t in range(T):
        dt = delta[:, t, :].float()  # (B, d_inner)
        x_t = _x[:, t, :].float()  # (B, d_inner)
        B_t = B_p[:, t, :].float()  # (B, d_state)
        C_t = C_p[:, t, :].float()  # (B, d_state)
        # h = exp(dt * A) * h + dt * B * x
        A_bar = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))  # (B, d_inner, d_state)
        B_bar = dt.unsqueeze(-1) * B_t.unsqueeze(1)  # (B, d_inner, d_state)
        h = A_bar * h + B_bar * x_t.unsqueeze(-1)
        # y = C @ h + D * x
        y_t = (h * C_t.unsqueeze(1)).sum(dim=-1) + D.float() * x_t
        ys.append(y_t)

    y = torch.stack(ys, dim=1)  # (B, T, d_inner)
    y = y.to(x.dtype)

    # Gate with z
    y = y * F.silu(z)

    # out_proj
    out = out_proj(y)
    return out

# Test input
torch.manual_seed(42)
x_test = torch.randn(1, 16, d_model, dtype=torch.bfloat16)

# Run our forward
with torch.no_grad():
    our_out, _ = our_mamba(x_test)

# Run reference forward
with torch.no_grad():
    ref_out = ref_mamba_forward(
        x_test, our_mamba.in_proj, our_mamba.conv1d, our_mamba.x_proj,
        our_mamba.dt_proj, our_mamba.A_log, our_mamba.D, our_mamba.out_proj,
        our_mamba.dt_layernorm, our_mamba.b_layernorm, our_mamba.c_layernorm,
        d_state, d_conv, d_inner)

# Compare
diff = (our_out - ref_out).abs()
print(f"\nOur output:  shape={our_out.shape} norm={our_out.norm().item():.4f}")
print(f"Ref output:  shape={ref_out.shape} norm={ref_out.norm().item():.4f}")
print(f"Max diff:    {diff.max().item():.6f}")
print(f"Mean diff:   {diff.mean().item():.6f}")
print(f"Relative:    {diff.max().item() / ref_out.abs().max().item():.6f}")

if diff.max().item() < 0.01:
    print("\n[OK] Mamba forward passes match!")
else:
    print("\n[MISMATCH] Mamba forward passes differ!")
    # Check intermediate outputs
    print("\nDiagnosing...")
    xz = our_mamba.in_proj(x_test)
    _x, z = xz.chunk(2, dim=-1)
    print(f"  in_proj out: norm={xz.norm().item():.4f}")
    print(f"  _x (ssm): norm={_x.norm().item():.4f}")
    print(f"  z (gate): norm={z.norm().item():.4f}")

    _x_t = _x.transpose(1, 2)
    _x_conv = our_mamba.conv1d(_x_t)[:, :, :16].transpose(1, 2)
    _x_silu = F.silu(_x_conv)
    print(f"  conv out: norm={_x_conv.norm().item():.4f}")
    print(f"  silu out: norm={_x_silu.norm().item():.4f}")
