"""Sparse Mixture-of-Experts (MoE) FFN layer.

Replaces dense FFN with top-k routed experts. For small models, we use
2-4 experts with top-1 or top-2 routing (vs large models using 8-64).

Key features:
- Router with load balancing loss (auxiliary loss)
- Top-k gating with noisy top-k (for exploration during training)
- Expert capacity factor (drop tokens if expert is overloaded)
- Shared expert (always-active, like DeepSeek-V3) for better quality

Memory: with 4 experts at 2x expansion, FFN params = 4x dense.
But only 1-2 experts are active per token, so FLOPs stay similar.

Usage:
    from research.moe import MoELayer, replace_ffn_with_moe

    # Replace FFN in existing model
    replace_ffn_with_moe(model, n_experts=4, top_k=2, d_model=1024)
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Router(nn.Module):
    """Top-k router for MoE. Outputs expert assignments + gating weights.

    Args:
        d_model: input dimension
        n_experts: number of experts
        top_k: number of experts to route each token to
        noisy_gating: if True, add noise during training (exploration)
        load_balance_loss_weight: weight of auxiliary load balancing loss
    """

    def __init__(self, d_model, n_experts, top_k=2, noisy_gating=True,
                 load_balance_loss_weight=0.01):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = min(top_k, n_experts)
        self.noisy_gating = noisy_gating
        self.load_balance_loss_weight = load_balance_loss_weight

        self.gate = nn.Linear(d_model, n_experts, bias=False)
        if noisy_gating:
            self.noise = nn.Linear(d_model, n_experts, bias=False)
            self.noise_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        """Route tokens to experts.

        Args:
            x: (B, T, d_model) or (N, d_model) flattened tokens

        Returns:
            dispatch_mask: (N, n_experts) — 1.0 if token routed to expert
            gating_weights: (N, n_experts) — gating weights (0 for non-routed)
            aux_loss: scalar load balancing loss
        """
        orig_shape = x.shape
        if x.dim() == 3:
            x = x.view(-1, self.d_model)  # (N, d_model)
        N = x.shape[0]

        # Compute routing logits.
        clean_logits = self.gate(x)  # (N, n_experts)

        if self.noisy_gating and self.training:
            noise = torch.randn_like(clean_logits) * self.noise_scale
            logits = clean_logits + noise * F.softplus(self.noise(x))
        else:
            logits = clean_logits

        # Top-k selection.
        top_k_logits, top_k_indices = logits.topk(self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)  # (N, top_k)

        # Build dispatch mask and gating weights.
        dispatch_mask = torch.zeros(N, self.n_experts, device=x.device, dtype=x.dtype)
        gating_weights = torch.zeros(N, self.n_experts, device=x.device, dtype=x.dtype)

        for k in range(self.top_k):
            expert_idx = top_k_indices[:, k]  # (N,)
            weight = top_k_weights[:, k]  # (N,)
            dispatch_mask.scatter_(1, expert_idx.unsqueeze(1), 1.0)
            gating_weights.scatter_(1, expert_idx.unsqueeze(1), weight.unsqueeze(1).to(gating_weights.dtype))

        # Load balancing loss (encourages uniform expert utilization).
        # Following Switch Transformer: loss = n_experts * sum(mean_gate * mean_dispatch)
        with torch.no_grad():
            # Fraction of tokens routed to each expert.
            tokens_per_expert = dispatch_mask.sum(dim=0)  # (n_experts,)
            fraction_per_expert = tokens_per_expert / (N * self.top_k)
            # Mean gating weight per expert.
            mean_gate = gating_weights.mean(dim=0)  # (n_experts,)
        # Aux loss = n_experts * sum(fraction * mean_gate)
        aux_loss = self.n_experts * (fraction_per_expert * mean_gate).sum()

        return dispatch_mask, gating_weights, aux_loss * self.load_balance_loss_weight


class Expert(nn.Module):
    """A single FFN expert (SwiGLU-style)."""

    def __init__(self, d_model, d_ff=None, activation="swiglu"):
        super().__init__()
        d_ff = d_ff or d_model * 2  # smaller than dense (4x) since we have multiple experts
        if activation == "swiglu":
            self.w1 = nn.Linear(d_model, d_ff, bias=False)
            self.w2 = nn.Linear(d_ff, d_model, bias=False)
            self.w3 = nn.Linear(d_model, d_ff, bias=False)
        else:
            self.fc1 = nn.Linear(d_model, d_ff)
            self.fc2 = nn.Linear(d_ff, d_model)
        self.activation = activation

    def forward(self, x):
        if self.activation == "swiglu":
            return self.w2(F.silu(self.w1(x)) * self.w3(x))
        else:
            return self.fc2(F.gelu(self.fc1(x)))


class MoELayer(nn.Module):
    """Mixture-of-Experts FFN layer with optional shared expert.

    Args:
        d_model: model dimension
        n_experts: number of routed experts
        top_k: experts activated per token
        d_ff: hidden dim per expert (default 2*d_model)
        shared_expert: if True, add an always-active shared expert (DeepSeek-V3 style)
        capacity_factor: max tokens per expert (drop excess). None = no limit.
        noisy_gating: add noise during training for exploration
    """

    def __init__(self, d_model, n_experts=4, top_k=2, d_ff=None,
                 shared_expert=True, capacity_factor=None, noisy_gating=True,
                 dense_bypass=False):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.has_shared = shared_expert
        # dense_bypass: skip router, run ALL experts with equal weight.
        # Used when router is untrained (uniform init) to reproduce dense FFN.
        # With w2 scaled by n_experts, equal weight 1/n gives exact dense output.
        self.dense_bypass = dense_bypass

        self.router = Router(d_model, n_experts, top_k, noisy_gating)
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff) for _ in range(n_experts)
        ])
        if shared_expert:
            self.shared = Expert(d_model, d_ff)

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, T, d_model)

        Returns:
            output: (B, T, d_model)
            aux_loss: load balancing loss
        """
        B, T, D = x.shape
        N = B * T
        x_flat = x.view(N, D)

        # Dense bypass: skip router, run all experts with equal weight.
        # This reproduces the original dense FFN when the router is untrained.
        if self.dense_bypass:
            # Batched: stack all expert weights into batched matmuls instead of
            # looping over experts (n_experts Python iterations → 1 batched call).
            # Handle both plain nn.Linear (.weight) and DoRA-wrapped (.base_weight).
            def _get_weight(mod):
                return getattr(mod, 'base_weight', getattr(mod, 'weight', None))
            w1 = torch.stack([_get_weight(e.w1) for e in self.experts])
            w3 = torch.stack([_get_weight(e.w3) for e in self.experts])
            w2 = torch.stack([_get_weight(e.w2) for e in self.experts])
            # x_flat: (N, D) → (n_experts, N, d_ff) via batched matmul
            x_exp = x_flat.unsqueeze(0).expand(self.n_experts, -1, -1)
            h1 = torch.bmm(x_exp, w1.transpose(1, 2))
            h3 = torch.bmm(x_exp, w3.transpose(1, 2))
            expert_out = torch.bmm(F.silu(h1) * h3, w2.transpose(1, 2))  # (n_experts, N, D)
            output = expert_out.mean(dim=0)  # equal weight = 1/n_experts → (N, D)
            if self.has_shared:
                output = output + self.shared(x_flat)
            aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            self._last_aux_loss = aux_loss
            return output.view(B, T, D), aux_loss

        # Normal routed MoE path.
        dispatch_mask, gating_weights, aux_loss = self.router(x_flat)

        # Compute expert outputs.
        # For efficiency, process each expert's assigned tokens.
        output = torch.zeros(N, D, device=x.device, dtype=x.dtype)

        for i, expert in enumerate(self.experts):
            # Find tokens routed to this expert.
            token_indices = dispatch_mask[:, i].nonzero(as_tuple=True)[0]
            if len(token_indices) == 0:
                continue

            # Capacity check (drop excess tokens).
            if self.capacity_factor is not None:
                capacity = int(self.capacity_factor * N / self.n_experts)
                if len(token_indices) > capacity:
                    token_indices = token_indices[:capacity]

            # Process tokens.
            expert_input = x_flat[token_indices]
            expert_output = expert(expert_input)
            weights = gating_weights[token_indices, i].unsqueeze(-1)
            output[token_indices] += expert_output * weights

        # Add shared expert output (always active, weight=1).
        if self.has_shared:
            shared_output = self.shared(x_flat)
            output = output + shared_output

        # Store aux loss for collection by collect_aux_loss().
        self._last_aux_loss = aux_loss

        return output.view(B, T, D), aux_loss


def replace_ffn_with_moe(model, n_experts=4, top_k=2, d_model=None,
                         shared_expert=True, capacity_factor=None, d_ff=None,
                         dense_bypass=False):
    """Replace FFN layers in a model with MoE layers.

    Args:
        model: the model (must have .blocks with .ffn)
        n_experts: number of experts per MoE layer
        top_k: experts activated per token
        d_model: model dimension (auto-detected if None)
        shared_expert: add shared expert
        capacity_factor: expert capacity limit
        d_ff: hidden dim per expert (default: d_model*2)
        dense_bypass: if True, skip router and run all experts (for untrained router)

    Returns:
        number of layers replaced
    """
    n_replaced = 0
    for block in model.blocks:
        d = d_model or block.ffn[-1].out_features if hasattr(block, 'ffn') else d_model
        if d is None:
            # Try to infer from the block.
            if hasattr(block, 'ffn') and hasattr(block.ffn, 'w2'):
                d = block.ffn.w2.out_features
            elif hasattr(block, 'ffn') and hasattr(block.ffn, 'fc2'):
                d = block.ffn.fc2.out_features

        if d is None:
            print("  [MoE] WARNING: could not infer d_model for block, skipping")
            continue

        moe = MoELayer(d, n_experts=n_experts, top_k=top_k, d_ff=d_ff,
                       shared_expert=shared_expert, capacity_factor=capacity_factor,
                       dense_bypass=dense_bypass)
        block.ffn = moe
        n_replaced += 1

    mode = "dense-bypass" if dense_bypass else f"top-{top_k}"
    print(f"  [MoE] replaced {n_replaced} FFN layers with {n_experts}-expert MoE ({mode})")
    return n_replaced


def collect_aux_loss(model):
    """Collect auxiliary (load balancing) losses from all MoE layers.

    Call this after forward pass to get the total aux loss for backprop.
    """
    total_aux = torch.tensor(0.0, device=next(model.parameters()).device)
    for block in model.blocks:
        if hasattr(block, 'ffn') and isinstance(block.ffn, MoELayer):
            # Store last aux loss on the layer for collection.
            if hasattr(block.ffn, '_last_aux_loss'):
                total_aux = total_aux + block.ffn._last_aux_loss
    return total_aux
