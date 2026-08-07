"""Activation Transmutation Key — swap activation functions via weight transform.

Novel key: no published method for closed-form activation function swapping.

The idea: if we want to replace activation A with activation B, we find
a weight transform W' such that:
  B(W' · x) ≈ A(W · x)  for all x in the activation distribution

For SwiGLU: output = silu(W_gate · x) * (W_up · x)
  = silu(g) * u  where g = W_gate·x, u = W_up·x

Target activations:
  - ReGLU: relu(g) * u  (faster, no sigmoid)
  - GeGLU: gelu(g) * u  (different gradient)
  - SwiMeLU: max(0, g) * u  (simplest, fastest)

Method: collect (g, u) pairs from calibration data, then solve for W'
such that the new activation produces similar outputs.

For SwiGLU → ReGLU:
  silu(g) = g * sigmoid(g)
  relu(g) = max(0, g)

  We need: relu(g') * u' ≈ silu(g) * u
  where g' = W'_gate · x, u' = W'_up · x

  Approach: find scaling factors α, β such that:
    relu(α * g) * (β * u) ≈ silu(g) * u

  This is a per-channel scaling: W'_gate = α * W_gate, W'_up = β * W_up
  where α, β are found via least-squares on calibration data.

  For positive g: relu(α*g) = α*g, silu(g) ≈ g (for large g)
    So α ≈ 1, β ≈ 1 for large positive g
  For negative g: relu(α*g) = 0, but silu(g) ≈ g*sigmoid(g) ≈ 0
    So the error is small for negative g
  For small positive g: silu(g) ≈ g/2, relu(α*g) = α*g
    So α ≈ 0.5 to match, but then large g is off by 2x

  Better approach: per-channel affine transform on gate:
    g' = α * g + β
    relu(g') = max(0, α*g + β)
    Find (α, β) per channel to minimize |relu(α*g + β) - silu(g)|²

  This is a 1D least-squares problem per channel — very fast.

Key class: PARTIAL — requires calibration data, approximate (not exact).

Usage:
    from research.keys.activation_transmute_key import ActivationTransmuteKey, apply_activation_transmute
    state = apply_activation_transmute(state, model, calib_tokens, n_layers=28,
                                        target="reglu")
"""
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple
from .base import Key, KeyClass, KeyResult


class ActivationTransmuteKey(Key):
    """Activation Transmutation key — swap activation via weight transform.

    Finds per-channel affine transform (α, β) such that:
      target_act(α * g + β) ≈ source_act(g)

    Then: W'_gate = α * W_gate, bias'_gate = β + α * bias_gate
          W'_up stays the same (or scaled to compensate)

    Key class: PARTIAL — requires calibration, approximate.
    """

    def __init__(self, target: str = "reglu"):
        self.target = target  # "reglu", "geglu", "swimelu"

    @property
    def name(self) -> str:
        return "activation_transmute"

    @property
    def description(self) -> str:
        return f"Transmute SwiGLU → {self.target.upper()} via per-channel weight scaling"

    def key_class(self) -> KeyClass:
        return KeyClass.PARTIAL

    def target_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the target activation function."""
        if self.target == "reglu":
            return F.relu(x)
        elif self.target == "geglu":
            return F.gelu(x)
        elif self.target == "swimelu":
            # SwiMeLU: max(0, x) — simplest possible
            return torch.clamp(x, min=0)
        else:
            raise ValueError(f"Unknown target activation: {self.target}")

    def source_activation(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU's gate activation: silu(x) = x * sigmoid(x)."""
        return F.silu(x)

    def solve_per_channel(self, g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find per-channel (α, β) to match source → target activation.

        Minimizes: |target(α*g + β) - source(g)|² per channel.

        Uses grid search + refinement (the problem is non-convex due to relu).

        Args:
            g: (N, d_ff) gate pre-activations from calibration

        Returns:
            alpha: (d_ff,) scaling per channel
            beta: (d_ff,) bias per channel
        """
        N, d_ff = g.shape
        device = g.device

        source = self.source_activation(g)  # (N, d_ff) — silu(g)
        alpha = torch.ones(d_ff, device=device)
        beta = torch.zeros(d_ff, device=device)

        # Grid search over α (β=0 initially since silu(0)=0 and relu(0)=0)
        best_alpha = torch.ones(d_ff, device=device)
        best_loss = torch.full((d_ff,), float('inf'), device=device)

        for a in torch.linspace(0.3, 2.0, 35, device=device):
            # target(a * g) vs source(g)
            target_vals = self.target_activation(a * g)  # (N, d_ff)
            loss = (target_vals - source).pow(2).mean(dim=0)  # (d_ff,)
            improved = loss < best_loss
            best_loss = torch.where(improved, loss, best_loss)
            best_alpha = torch.where(improved, a.expand_as(best_alpha), best_alpha)

        # Refine with per-channel β (shift to align zero crossings)
        # For relu: relu(α*g + β) = max(0, α*g + β)
        # The optimal β shifts the "knee" of relu to match silu's soft knee
        for b in torch.linspace(-1.0, 1.0, 21, device=device):
            target_vals = self.target_activation(best_alpha * g + b)
            loss = (target_vals - source).pow(2).mean(dim=0)
            improved = loss < best_loss
            best_loss = torch.where(improved, loss, best_loss)
            beta = torch.where(improved, b.expand_as(beta), beta)

        alpha = best_alpha
        return alpha, beta

    def forward(self, data: Dict[str, torch.Tensor]) -> KeyResult:
        """Apply transmutation using pre-computed (alpha, beta) per channel.

        Args:
            data: {"state": state_dict, "alpha": (d_ff,), "beta": (d_ff,),
                   "n_layers": int}

        Returns:
            modified state dict with scaled gate weights
        """
        try:
            state = dict(data.get("state", data))
            alpha = data["alpha"]
            beta = data["beta"]
            n_layers = data["n_layers"]

            for i in range(n_layers):
                # Scale w_gate rows by alpha, add beta to bias
                gate_key = f"blocks.{i}.ffn.w_gate.weight"
                if gate_key in state:
                    w = state[gate_key].float()
                    state[gate_key] = (w * alpha.unsqueeze(1)).to(state[gate_key].dtype)

                gate_bias = f"blocks.{i}.ffn.w_gate.bias"
                if gate_bias in state:
                    state[gate_bias] = (state[gate_bias].float() * alpha + beta
                                       ).to(state[gate_bias].dtype)
                elif beta.abs().max() > 1e-6:
                    # Add bias if it doesn't exist but we need it
                    state[gate_bias] = beta.to(state[gate_key].dtype)

                # Also handle MoE experts
                for ei in range(10):
                    ek = f"blocks.{i}.ffn.experts.{ei}.w_gate.weight"
                    if ek in state:
                        w = state[ek].float()
                        state[ek] = (w * alpha.unsqueeze(1)).to(state[ek].dtype)

            return KeyResult(
                success=True, weights=state,
                metadata={
                    "n_layers": n_layers, "target": self.target,
                    "alpha_mean": alpha.mean().item(),
                    "beta_mean": beta.mean().item(),
                },
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: Dict[str, torch.Tensor]) -> KeyResult:
        return KeyResult(success=True, data=weights,
                         metadata={"reversible": False})


def apply_activation_transmute(state: Dict[str, torch.Tensor],
                                model: torch.nn.Module,
                                calib_tokens: torch.Tensor,
                                n_layers: int,
                                target: str = "reglu",
                                device: str = "cuda") -> Dict[str, torch.Tensor]:
    """Transmute SwiGLU → target activation via weight scaling.

    Args:
        state: model state dict
        model: loaded model (for collecting calibration activations)
        calib_tokens: (N,) token IDs for calibration
        n_layers: number of layers
        target: target activation ("reglu", "geglu", "swimelu")
        device: compute device

    Returns:
        modified state dict with transmuted activation
    """
    key = ActivationTransmuteKey(target=target)

    # Collect gate pre-activations from calibration data
    seq_len = min(128, calib_tokens.shape[0])
    input_ids = calib_tokens[:seq_len].unsqueeze(0).to(device)

    model.eval()
    print(f"  [Act Transmute] Collecting gate activations from {seq_len} tokens...")

    all_gates = []
    with torch.inference_mode():
        x = model.embed(input_ids)
        for block in model.blocks:
            # Get pre-activation gate values
            x_normed = block.ln1(x)
            attn_out, _ = block.attn(x_normed)
            x = x + attn_out
            x_normed2 = block.ln2(x)

            # FFN gate pre-activation
            if hasattr(block.ffn, 'w_gate'):
                g = block.ffn.w_gate(x_normed2)  # (1, T, d_ff)
                all_gates.append(g.flatten(0, 1))  # (T, d_ff)
            elif hasattr(block.ffn, 'experts'):
                # MoE: collect from all experts
                for ei, expert in enumerate(block.ffn.experts):
                    if hasattr(expert, 'w_gate'):
                        g = expert.w_gate(x_normed2)
                        all_gates.append(g.flatten(0, 1))

            ffn_out = block.ffn(x_normed2)
            if isinstance(ffn_out, tuple):
                ffn_out = ffn_out[0]
            x = x + ffn_out

    # Concatenate all gate activations
    gates = torch.cat(all_gates, dim=0)  # (N_total, d_ff)
    print(f"  [Act Transmute] Collected {gates.shape[0]} gate vectors, d_ff={gates.shape[1]}")

    # Solve per-channel (alpha, beta)
    print(f"  [Act Transmute] Solving per-channel transform (SwiGLU → {target.upper()})...")
    alpha, beta = key.solve_per_channel(gates)

    print(f"  [Act Transmute] alpha: mean={alpha.mean():.4f}, std={alpha.std():.4f}")
    print(f"  [Act Transmute] beta:  mean={beta.mean():.4f}, std={beta.std():.4f}")

    # Verify quality
    source = key.source_activation(gates)
    target_vals = key.target_activation(alpha * gates + beta)
    mse = (source - target_vals).pow(2).mean().item()
    source_energy = source.pow(2).mean().item()
    relative_error = mse / source_energy
    print(f"  [Act Transmute] Relative error: {relative_error:.4f} ({relative_error*100:.1f}%)")

    # Apply to state dict
    result = key.forward({
        "state": state, "alpha": alpha.cpu(), "beta": beta.cpu(),
        "n_layers": n_layers,
    })

    if not result.success:
        raise RuntimeError(f"Activation transmute failed: {result.error}")

    print(f"  [Act Transmute] Applied {target.upper()} weights to {n_layers} layers")
    return result.weights


if __name__ == "__main__":
    key = ActivationTransmuteKey(target="reglu")
    print(f"Key: {key.name}, class: {key.key_class().value}")

    # Test per-channel solving
    d_ff = 256
    N = 1024
    g = torch.randn(N, d_ff) * 3.0  # realistic gate pre-activations

    alpha, beta = key.solve_per_channel(g)
    source = key.source_activation(g)
    target = key.target_activation(alpha * g + beta)

    mse = (source - target).pow(2).mean().item()
    source_energy = source.pow(2).mean().item()
    print(f"  Relative error: {mse/source_energy:.4f} ({mse/source_energy*100:.1f}%)")
    print(f"  alpha: mean={alpha.mean():.4f}, range=[{alpha.min():.4f}, {alpha.max():.4f}]")
    assert mse / source_energy < 0.3, "Error too high!"
    print("  Transmutation verified ✓")
