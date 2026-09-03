"""Mamba probe — build a fake Mamba model, inspect weight structure.

This script creates a minimal Mamba block (no training, random init),
passes text through it, and dumps the weight names + shapes so we can
reverse-engineer the KeyStack key for lossless checkpoint conversion.

Mamba architecture (S6 selective SSM):
  in_proj:  Linear(d_model → 2*d_inner)     # split into z (gate) + x (ssm)
  conv1d:   Conv1d(d_inner, d_inner, k, groups=d_inner)  # depthwise causal
  x_proj:   Linear(d_inner → dt_rank + 2*d_state)       # projects to Δ, B, C
  dt_proj:  Linear(dt_rank → d_inner, bias=True)         # discretization
  A_log:    Parameter(d_inner, d_state)                   # log(-A) for stability
  D:        Parameter(d_inner)                            # skip connection
  out_proj: Linear(d_inner → d_model)
  norm:     RMSNorm (Mamba2 adds this before out_proj)

Forward:
  z, x = split(in_proj(x))
  x = silu(conv1d(x))
  Δ, B, C = split(x_proj(x))
  Δ = softplus(dt_proj(Δ))
  y = selective_scan(x, Δ, A, B, C, D)
  y = y * silu(z)           # gate
  out = out_proj(y)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))


class MambaLayer(nn.Module):
    """Minimal Mamba (S6) layer — pure PyTorch, no mamba-ssm dependency.

    Implements the selective scan as a simple Python loop for correctness,
    not speed. This is the reference implementation for weight structure
    reverse engineering.

    Supports Jamba's dt/b/c RMSNorm layers (loaded from checkpoint, applied
    before the selective scan). Also supports the ModularBlock interface
    (past_key_value, use_cache, returns (out, present) tuple).
    """

    def __init__(
        self,
        d_model: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        bias: bool = False,
        conv_bias: bool = True,
        layer_idx: int = 0,
        norm_eps: float = 1e-6,
        use_jamba_norms: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = expand * d_model
        self.layer_idx = layer_idx
        self.norm_eps = norm_eps

        # dt_rank: "auto" = ceil(d_model / 16)
        if dt_rank == "auto":
            self.dt_rank = max(1, (d_model + 15) // 16)
        else:
            self.dt_rank = int(dt_rank)

        # in_proj: d_model → 2*d_inner (split into z and x)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=bias)

        # Depthwise causal conv
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, d_conv,
            groups=self.d_inner, bias=conv_bias,
            padding=d_conv - 1,
        )

        # x_proj: d_inner → dt_rank + 2*d_state (Δ, B, C)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)

        # dt_proj: dt_rank → d_inner (with bias for softplus init)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A: log(-A) parameter (S4D real init)
        # A_init = -arange(1, d_state+1) → A_log = log(-A_init)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))  # (d_inner, d_state)

        # D: skip connection (init to 1)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # out_proj: d_inner → d_model
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

        # Jamba-specific RMSNorm layers (applied to dt, B, C before scan)
        # dt_layernorm: (dt_rank,) — applied to delta BEFORE dt_proj
        # b/c_layernorm: (d_state,) — applied to B, C before scan
        # These are loaded from checkpoint; init to ones (identity)
        if use_jamba_norms:
            self.dt_layernorm = nn.Parameter(torch.ones(self.dt_rank))
            self.b_layernorm = nn.Parameter(torch.ones(self.d_state))
            self.c_layernorm = nn.Parameter(torch.ones(self.d_state))
        else:
            self.dt_layernorm = None
            self.b_layernorm = None
            self.c_layernorm = None

        # SSM recurrent state (for incremental decoding)
        self._ssm_state = None
        # Conv state (for incremental decoding of the causal conv)
        self._conv_state = None

    def _rmsnorm(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """RMSNorm: x / rms(x) * weight"""
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.norm_eps).rsqrt()
        return x * rms * weight

    def reset_state(self):
        """Reset SSM and conv state (call at start of new generation)."""
        self._ssm_state = None
        self._conv_state = None

    def _selective_scan_ref(
        self, x: torch.Tensor, delta: torch.Tensor,
        A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor,
        h_init: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reference selective scan (Python loop, slow but correct).

        Args:
            x: (B, d_inner, L)
            delta: (B, d_inner, L)
            A: (d_inner, d_state) — negative (we use -exp(A_log))
            B: (B, d_state, L)
            C: (B, d_state, L)
            D: (d_inner,) — skip connection
            h_init: (B, d_inner, d_state) — initial SSM state (for incremental)
        Returns:
            y: (B, d_inner, L)
            h_final: (B, d_inner, d_state) — final SSM state
        """
        B_b, d_inner, L = x.shape
        d_state = A.shape[1]

        A_neg = -torch.exp(A)  # (d_inner, d_state) — ensure negative
        if h_init is not None:
            h = h_init
        else:
            h = torch.zeros(B_b, d_inner, d_state, device=x.device, dtype=x.dtype)
        ys = []

        for t in range(L):
            dt = delta[:, :, t:t+1]  # (B, d_inner, 1)
            # A_bar = exp(dt * A_neg) — (B, d_inner, d_state)
            A_bar = torch.exp(dt * A_neg.unsqueeze(0))
            # B_bar = dt * B_t — (B, d_inner, d_state)
            B_t = B[:, :, t]  # (B, d_state)
            B_bar = dt * B_t.unsqueeze(1)  # (B, d_inner, d_state)
            # h_t = A_bar * h + B_bar * x_t
            x_t = x[:, :, t:t+1]  # (B, d_inner, 1)
            h = A_bar * h + B_bar * x_t
            # y_t = C_t @ h + D * x_t
            C_t = C[:, :, t]  # (B, d_state)
            y_t = (h * C_t.unsqueeze(1)).sum(dim=-1) + D * x_t.squeeze(-1)
            ys.append(y_t)

        return torch.stack(ys, dim=-1), h  # (B, d_inner, L), (B, d_inner, d_state)

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: dict | None = None,
        use_cache: bool = False,
        attention_bias: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, dict | None]:
        """Forward pass.

        Args:
            x: (B, T, d_model)
            past_key_value: dict with 'ssm_state' and 'conv_state' (for incremental)
            use_cache: if True, return state for incremental decoding
            attention_bias: ignored (Mamba has no attention)
            position_ids: ignored (Mamba is position-agnostic)
        Returns:
            (out, present) — out is (B, T, d_model), present is state dict or None
        """
        B, T, D = x.shape

        # in_proj → split into x (ssm path) and z (gate)
        # HuggingFace Mamba: x, z = in_proj(x).chunk(2, dim=-1)
        xz = self.in_proj(x)  # (B, T, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # x first (ssm), z second (gate)

        # Conv1d (depthwise causal)
        # For incremental decoding (T=1), use conv state
        if T == 1 and past_key_value is not None and 'conv_state' in past_key_value:
            # Incremental: roll the conv state buffer
            conv_state = past_key_value['conv_state']  # (B, d_inner, d_conv-1)
            x_t = x.transpose(1, 2)  # (B, d_inner, 1)
            # New conv state = shift left, append new input
            new_conv_state = torch.cat([conv_state[:, :, 1:], x_t], dim=-1)
            # Compute conv output manually: sum(weight * state) + bias
            # conv1d.weight shape: (d_inner, 1, d_conv)
            w = self.conv1d.weight.squeeze(1)  # (d_inner, d_conv)
            conv_out = (w[:, :d_conv-1] * new_conv_state).sum(dim=-1, keepdim=True)
            if self.conv1d.bias is not None:
                conv_out = conv_out + self.conv1d.bias.unsqueeze(0).unsqueeze(-1)
            x = conv_out.transpose(1, 2)  # (B, 1, d_inner)
            present_conv_state = new_conv_state
        else:
            # Full sequence
            x = x.transpose(1, 2)  # (B, d_inner, T)
            x = self.conv1d(x)[:, :, :T]  # causal: trim right padding
            x = x.transpose(1, 2)  # (B, T, d_inner)
            # Save conv state (last d_conv-1 inputs) for incremental
            if use_cache:
                # Save conv state (last d_conv-1 inputs to conv) for incremental
                # x is the first half of in_proj output (ssm path, before conv)
                x_pre_conv, _ = xz.chunk(2, dim=-1)
                x_pre_conv_t = x_pre_conv.transpose(1, 2)  # (B, d_inner, T)
                if T >= self.d_conv - 1:
                    present_conv_state = x_pre_conv_t[:, :, -(self.d_conv - 1):]
                else:
                    # Pad if sequence is shorter than conv kernel
                    pad = self.d_conv - 1 - T
                    present_conv_state = torch.cat([
                        torch.zeros(B, self.d_inner, pad, device=x.device, dtype=x.dtype),
                        x_pre_conv_t,
                    ], dim=-1)
            else:
                present_conv_state = None

        x = F.silu(x)

        # x_proj → Δ, B, C
        x_proj_out = self.x_proj(x)  # (B, T, dt_rank + 2*d_state)
        delta, B, C = x_proj_out.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1)

        # Jamba RMSNorms on dt, B, C (all before dt_proj and scan)
        # dt_layernorm: (dt_rank,) — normalizes delta before dt_proj
        # b/c_layernorm: (d_state,) — normalizes B, C before scan
        if self.dt_layernorm is not None:
            delta = self._rmsnorm(delta, self.dt_layernorm)
        if self.b_layernorm is not None:
            B = self._rmsnorm(B, self.b_layernorm)
        if self.c_layernorm is not None:
            C = self._rmsnorm(C, self.c_layernorm)

        # dt_proj → Δ (with softplus)
        delta = F.softplus(self.dt_proj(delta))  # (B, T, d_inner)

        # Transpose for scan: (B, d_inner, L) / (B, d_state, L)
        x_scan = x.transpose(1, 2)  # (B, d_inner, T)
        delta_scan = delta.transpose(1, 2)  # (B, d_inner, T)
        B_scan = B.transpose(1, 2)  # (B, d_state, T)
        C_scan = C.transpose(1, 2)  # (B, d_state, T)

        # Selective scan (with optional initial state for incremental)
        h_init = None
        if past_key_value is not None and 'ssm_state' in past_key_value:
            h_init = past_key_value['ssm_state']

        y, h_final = self._selective_scan_ref(
            x_scan, delta_scan, self.A_log.data, B_scan, C_scan, self.D,
            h_init=h_init)

        # Gate with z
        y = y.transpose(1, 2)  # (B, T, d_inner)
        y = y * F.silu(z)

        # out_proj
        out = self.out_proj(y)  # (B, T, d_model)

        # Build present state for incremental decoding
        present = None
        if use_cache:
            present = {
                'ssm_state': h_final,
                'conv_state': present_conv_state,
            }

        return out, present


def inspect_weights():
    """Build a Mamba layer and dump all weight names + shapes."""
    print("=" * 70)
    print("Mamba Layer Weight Inspection")
    print("=" * 70)

    layer = MambaLayer(d_model=64, d_state=16, d_conv=4, expand=2)
    layer.eval()

    print(f"\nConfig: d_model=64, d_state=16, d_conv=4, expand=2")
    print(f"  d_inner = {layer.d_inner}")
    print(f"  dt_rank = {layer.dt_rank}")
    print(f"\nParameters ({sum(p.numel() for p in layer.parameters())} total):")

    for name, param in layer.named_parameters():
        print(f"  {name:30s} {str(tuple(param.shape)):20s} {param.numel():>8d} params")

    print(f"\nBuffers:")
    for name, buf in layer.named_buffers():
        print(f"  {name:30s} {str(tuple(buf.shape)):20s}")

    # Test forward pass
    print("\n--- Forward Pass Test ---")
    x = torch.randn(1, 8, 64)  # (B=1, T=8, d_model=64)
    with torch.no_grad():
        y = layer(x)
    print(f"Input:  {tuple(x.shape)}")
    print(f"Output: {tuple(y.shape)}")
    print(f"Output stats: mean={y.mean():.4f}, std={y.std():.4f}")

    # Test incremental decoding (T=1)
    print("\n--- Incremental Decode Test (T=1) ---")
    x1 = torch.randn(1, 1, 64)
    with torch.no_grad():
        y1 = layer(x1)
    print(f"Input:  {tuple(x1.shape)}")
    print(f"Output: {tuple(y1.shape)}")

    # Dump weight dict in ForgeAI checkpoint format
    print("\n--- ForgeAI Checkpoint Key Mapping ---")
    state = layer.state_dict()
    # Map to ForgeAI block format: blocks.{i}.attn.{name}
    forge_keys = {}
    for name, tensor in state.items():
        forge_key = f"blocks.0.attn.{name}"
        forge_keys[forge_key] = tuple(tensor.shape)
        print(f"  {forge_key:45s} {str(tuple(tensor.shape)):20s}")

    return layer, forge_keys


def test_text_to_weights():
    """Pass text through the model and see how it becomes weights.

    This demonstrates the KeyStack 'forward' direction: given some data
    (text), what weights does the model produce?

    For Mamba, the weights are NOT derived from text — they are learned
    parameters. The KeyStack key for Mamba is a STRUCTURAL key: it maps
    between checkpoint formats (e.g. HuggingFace Mamba → ForgeAI internal).

    The 'lossless' aspect: at init time, Mamba weights are deterministic
    (S4D init for A, ones for D, random for projections). The key converts
    between parameterizations without loss.
    """
    print("\n" + "=" * 70)
    print("Text -> Weights KeyStack Test")
    print("=" * 70)

    layer = MambaLayer(d_model=64, d_state=16, d_conv=4, expand=2)
    layer.eval()

    # The "data" for a Mamba key is the raw parameter tensors
    # The "weights" are the ForgeAI-internal format
    # Forward: HuggingFace Mamba checkpoint → ForgeAI format
    # Reverse: ForgeAI format → HuggingFace Mamba checkpoint

    # Simulate a HuggingFace-style checkpoint
    hf_state = {}
    for name, param in layer.named_parameters():
        # HuggingFace naming: mixer.{name}
        hf_key = f"mixer.{name}"
        hf_state[hf_key] = param.data.clone()

    print(f"\nHuggingFace-style keys ({len(hf_state)}):")
    for k, v in hf_state.items():
        print(f"  {k:45s} {str(tuple(v.shape)):20s}")

    # Convert to ForgeAI format
    forge_state = {}
    for hf_key, tensor in hf_state.items():
        # mixer.in_proj.weight → blocks.0.attn.in_proj.weight
        name = hf_key.replace("mixer.", "")
        forge_key = f"blocks.0.attn.{name}"
        forge_state[forge_key] = tensor.clone()

    print(f"\nForgeAI-style keys ({len(forge_state)}):")
    for k, v in forge_state.items():
        print(f"  {k:45s} {str(tuple(v.shape)):20s}")

    # Verify round-trip (lossless)
    rt_state = {}
    for forge_key, tensor in forge_state.items():
        name = forge_key.replace("blocks.0.attn.", "")
        hf_key = f"mixer.{name}"
        rt_state[hf_key] = tensor.clone()

    all_match = True
    for k in hf_state:
        if not torch.equal(hf_state[k], rt_state[k]):
            print(f"  MISMATCH: {k}")
            all_match = False

    print(f"\nRound-trip lossless: {all_match}")
    return all_match


def test_mamba2_differences():
    """Inspect Mamba2 differences (scalar A, shared B/C, norm)."""
    print("\n" + "=" * 70)
    print("Mamba2 Architecture Differences")
    print("=" * 70)

    # Mamba2 key differences:
    # 1. A is scalar-diagonal: (d_inner, 1) instead of (d_inner, d_state)
    #    Actually: A_log shape (n_heads, 1) where n_heads = d_inner / head_dim
    # 2. B, C can be shared across heads (like grouped-value attention)
    # 3. Adds RMSNorm before out_proj
    # 4. d_state typically 64 or 128 (vs 16 in Mamba1)
    # 5. Uses SSD algorithm (matmul-based, not scan)

    d_model = 64
    d_state = 64  # Mamba2 uses larger state
    expand = 2
    d_inner = expand * d_model
    head_dim = 64
    n_heads = d_inner // head_dim  # = 2

    print(f"Mamba2 config: d_model={d_model}, d_state={d_state}, "
          f"expand={expand}, head_dim={head_dim}")
    print(f"  d_inner = {d_inner}")
    print(f"  n_heads (SSD) = {n_heads}")
    print(f"  A_log shape (Mamba1): ({d_inner}, {d_state})")
    print(f"  A_log shape (Mamba2): ({n_heads}, 1) — scalar per head")
    print(f"  B shape (Mamba1): (B, d_state, L) — per-channel")
    print(f"  B shape (Mamba2): (B, n_heads, d_state, L) — per-head")
    print(f"\n  Extra Mamba2 components:")
    print(f"    - RMSNorm before out_proj (dt_norm + A_norm + B_norm + C_norm)")
    print(f"    - norm.weight: ({d_inner},)")
    print(f"\n  Weight count comparison:")
    print(f"    Mamba1: in_proj, conv1d, x_proj, dt_proj, A_log, D, out_proj")
    print(f"    Mamba2: in_proj, conv1d, x_proj, dt_proj, A_log, D,")
    print(f"            dt_norm, A_norm, B_norm, C_norm, out_proj")
    print(f"    (Mamba2 has 4 extra norm weights)")


def test_jamba_weight_mapping():
    """Inspect Jamba's specific weight naming (HuggingFace format)."""
    print("\n" + "=" * 70)
    print("Jamba HuggingFace Weight Mapping")
    print("=" * 70)

    # Jamba uses: model.layers.{i}.mixer.{name}
    # where mixer is a Mamba block
    # Attention layers use: model.layers.{i}.self_attn.{name}

    layer = MambaLayer(d_model=64, d_state=16, d_conv=4, expand=2)

    print("\nJamba Mamba layer → ForgeAI mapping:")
    for name, param in layer.named_parameters():
        jamba_key = f"model.layers.0.mixer.{name}"
        forge_key = f"blocks.0.attn.{name}"
        print(f"  {jamba_key:50s} → {forge_key}")

    print("\nJamba Attention layer → ForgeAI mapping (for reference):")
    attn_names = ["q_proj.weight", "k_proj.weight", "v_proj.weight",
                  "out_proj.weight"]
    for name in attn_names:
        jamba_key = f"model.layers.1.self_attn.{name}"
        forge_key = f"blocks.1.attn.{name}"
        print(f"  {jamba_key:50s} → {forge_key}")

    # Jamba also has: model.layers.{i}.input_layernorm.weight
    #                 model.layers.{i}.post_attention_layernorm.weight
    # → blocks.{i}.ln1.weight, blocks.{i}.ln2.weight


if __name__ == "__main__":
    inspect_weights()
    test_text_to_weights()
    test_mamba2_differences()
    test_jamba_weight_mapping()
    print("\n✓ Mamba probe complete — weight structure documented")
