"""DiffusionBlocks: Block-wise training via diffusion interpretation.

Partitions a model with residual connections into B blocks, each trained
independently as a denoising step in a continuous-time diffusion process.

Based on Sakana AI's DiffusionBlocks (ICLR 2026):
- Paper: https://arxiv.org/abs/2506.14202
- Code: https://github.com/SakanaAI/DiffusionBlocks

Key benefits:
- B× memory reduction (only L/B layers need gradients per step)
- B× larger batch sizes or B× longer sequences
- Competitive quality at B=2-4, moderate drop at B=6

Usage:
    from research.diffusion_blocks import DiffusionBlocks, get_block_sigmas

    # Compute noise ranges for 4 blocks
    sigmas = get_block_sigmas(num_blocks=4)

    # Wrap model for block-wise training
    dblock = DiffusionBlocks(model, num_blocks=4, d_model=2048)

    # Training step (trains only one block)
    loss = dblock.train_step(input_ids, labels, optimizer)
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm


# ── Noise partitioning ──────────────────────────────────────────────────

def get_block_sigmas(
    num_blocks: int,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    p_mean: float = -1.2,
    p_std: float = 1.2,
) -> list[float]:
    """Compute noise level boundaries for B blocks using equi-probability partitioning.

    Each block gets an equal probability mass under the log-normal noise
    distribution, ensuring balanced learning difficulty across blocks.

    Args:
        num_blocks: Number of blocks B
        sigma_min: Minimum noise level (default 0.002)
        sigma_max: Maximum noise level (default 80.0)
        p_mean: Log-normal mean (default -1.2, from Karras et al. 2022)
        p_std: Log-normal std (default 1.2)

    Returns:
        List of B+1 sigma values defining block boundaries [σ_0, σ_1, ..., σ_B]
    """
    cdf_min = norm.cdf((np.log(sigma_min) - p_mean) / p_std)
    cdf_max = norm.cdf((np.log(sigma_max) - p_mean) / p_std)
    block_sigmas = []
    for i in range(num_blocks + 1):
        p = cdf_min + (cdf_max - cdf_min) * (i / num_blocks)
        sigma = float(np.exp(p_mean + p_std * norm.ppf(p)))
        block_sigmas.append(sigma)
    return block_sigmas


def sample_block_sigma(
    block_sigmas: list[float],
    block_idx: int,
    n_samples: int = 1,
    gamma: float = 0.0,
    p_mean: float = -1.2,
    p_std: float = 1.2,
) -> torch.Tensor:
    """Sample noise levels for a specific block.

    Args:
        block_sigmas: Output of get_block_sigmas()
        block_idx: Which block to sample for (0-indexed)
        n_samples: Number of sigma values to sample
        gamma: Overlap factor (0 = no overlap, 0.1 = 10% overlap with neighbors)
        p_mean, p_std: Log-normal parameters

    Returns:
        Tensor of shape (n_samples,) with sampled sigma values
    """
    sigma_min_block = block_sigmas[block_idx]
    sigma_max_block = block_sigmas[block_idx + 1]

    # Extend range with overlap
    if gamma > 0.0:
        log_min = np.log(sigma_min_block)
        log_max = np.log(sigma_max_block)
        log_range = log_max - log_min
        sigma_min_block = np.exp(log_min - gamma * log_range)
        sigma_max_block = np.exp(log_max + gamma * log_range)
        sigma_min_block = max(sigma_min_block, block_sigmas[0])
        sigma_max_block = min(sigma_max_block, block_sigmas[-1])

    cdf_min_block = norm.cdf((np.log(sigma_min_block) - p_mean) / p_std)
    cdf_max_block = norm.cdf((np.log(sigma_max_block) - p_mean) / p_std)

    rand = np.random.uniform(cdf_min_block, cdf_max_block, n_samples)
    sigma = np.exp(p_mean + p_std * norm.ppf(rand))
    return torch.from_numpy(sigma).float()


def get_loss_weights(sigmas: torch.Tensor, sigma_data: float = 0.5,
                     max_weight: float = 10.0) -> torch.Tensor:
    """Compute EDM-style loss weights for noise levels.

    w(σ) = (σ² + σ_data²) / (σ · σ_data)²

    Clamped to max_weight to prevent explosion at low sigma.
    For AR models, we recommend max_weight=10.0 (vs unbounded for diffusion).
    """
    weights = (sigmas ** 2 + sigma_data ** 2) / (sigmas * sigma_data).clamp(min=1e-4) ** 2
    return weights.clamp(max=max_weight)


# ── AdaLN noise conditioning ────────────────────────────────────────────

class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into a vector representation using sinusoidal
    positional embeddings + 2-layer MLP."""

    def __init__(self, hidden_size: int, freq_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = freq_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(freq_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32) / half
        ).to(t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = F.pad(embedding, (0, 1))
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_freq = t_freq.to(dtype=next(self.parameters()).dtype)
        return self.mlp(t_freq)


class AdaLN(nn.Module):
    """Adaptive Layer Normalization: produces (shift, scale) modulation
    from a conditioning vector.

    Outputs 4 values per layer: shift_msa, scale_msa, shift_mlp, scale_mlp.
    No gates — gates with zero-init would zero out block outputs and prevent
    gradient flow through block parameters. Shift/scale with zero-init is
    identity (lossless) AND allows gradients to flow.
    """

    def __init__(self, cond_dim: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(cond_dim, 4 * hidden_size, bias)
        # Zero-init for lossless start (shift=0, scale=0 → identity)
        nn.init.constant_(self.linear.weight, 0)
        if bias:
            nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.silu(self.linear(x))


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive modulation: x * (1 + scale) + shift"""
    # shift, scale: (B, C) → broadcast to (B, T, C)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ── DiffusionBlocks wrapper ─────────────────────────────────────────────

@dataclass
class DiffusionBlockConfig:
    """Configuration for DiffusionBlocks training."""
    num_blocks: int = 4
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    p_mean: float = -1.2
    p_std: float = 1.2
    sigma_data: float = 0.5
    gamma: float = 0.1  # overlap factor for smooth block transitions
    cond_dim: int = 256  # timestep embedding dimension
    use_noise_conditioning: bool = True  # add AdaLN to blocks


class DiffusionBlocks:
    """Wraps a ConfigurableResearchLLM for block-wise diffusion training.

    The model is partitioned into B blocks. Each training step:
    1. Randomly selects one block
    2. Adds noise to the target embeddings
    3. Runs only that block's layers (others skipped)
    4. Computes weighted loss
    5. Backprops only through that block

    This gives B× memory reduction since only L/B layers need gradients.
    """

    def __init__(
        self,
        model: nn.Module,
        config: DiffusionBlockConfig,
        d_model: int,
        num_layers: int,
    ):
        self.model = model
        self.config = config
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_blocks = config.num_blocks

        # Compute block boundaries
        self.block_sigmas = get_block_sigmas(
            num_blocks=self.num_blocks,
            sigma_min=config.sigma_min,
            sigma_max=config.sigma_max,
            p_mean=config.p_mean,
            p_std=config.p_std,
        )

        # Assign layers to blocks (evenly distributed)
        layers_per_block = num_layers // self.num_blocks
        self.block_layers: list[list[int]] = []
        for b in range(self.num_blocks):
            start = b * layers_per_block
            end = (b + 1) * layers_per_block if b < self.num_blocks - 1 else num_layers
            self.block_layers.append(list(range(start, end)))

        # Noise conditioning components (added to model)
        if config.use_noise_conditioning:
            self.timestep_embedder = TimestepEmbedder(config.cond_dim)
            # One AdaLN per block (shared across layers in the block)
            self.adalns = nn.ModuleList([
                AdaLN(config.cond_dim, d_model)
                for _ in range(self.num_blocks)
            ])
            # Move to model's device
            device = next(model.parameters()).device
            self.timestep_embedder = self.timestep_embedder.to(device)
            self.adalns = self.adalns.to(device)
        else:
            self.timestep_embedder = None
            self.adalns = None

        print(f"[DiffusionBlocks] {self.num_blocks} blocks, "
              f"{num_layers} layers → {[len(b) for b in self.block_layers]} layers/block")
        print(f"[DiffusionBlocks] Sigma ranges: "
              + ", ".join(f"[{self.block_sigmas[i]:.4f}, {self.block_sigmas[i+1]:.4f}]"
                         for i in range(self.num_blocks)))

    def get_block_sigma(self, block_idx: int, n_samples: int = 1) -> torch.Tensor:
        """Sample sigma values for a specific block."""
        return sample_block_sigma(
            self.block_sigmas, block_idx, n_samples,
            gamma=self.config.gamma,
            p_mean=self.config.p_mean,
            p_std=self.config.p_std,
        )

    def get_loss_weights(self, sigmas: torch.Tensor) -> torch.Tensor:
        """Compute EDM loss weights."""
        return get_loss_weights(sigmas, self.config.sigma_data)

    def estimate_block_idx(self, sigma: torch.Tensor) -> int:
        """Determine which block a sigma value belongs to."""
        block_sigmas = torch.tensor(self.block_sigmas, device=sigma.device)
        idx = torch.bucketize(sigma, block_sigmas, right=True) - 1
        idx = (self.num_blocks - 1) - idx
        idx = torch.clamp(idx, 0, self.num_blocks - 1).long()
        values, counts = idx.unique(return_counts=True)
        return values[counts.argmax()].item()

    def get_block_parameters(self, block_idx: int) -> list[nn.Parameter]:
        """Get parameters for a specific block (for selective optimization)."""
        layers = self.block_layers[block_idx]
        params = []
        for layer_idx in layers:
            block = self.model.blocks[layer_idx]
            params.extend(block.parameters())
        # Include AdaLN parameters for this block
        if self.adalns is not None:
            params.extend(self.adalns[block_idx].parameters())
        # Include embedding/head if first/last block
        if block_idx == 0:
            if hasattr(self.model, 'embed'):
                params.extend(self.model.embed.parameters())
        if block_idx == self.num_blocks - 1:
            if hasattr(self.model, 'head'):
                params.extend(self.model.head.parameters())
        return params

    def freeze_all_except_block(self, block_idx: int):
        """Freeze all parameters except those in the specified block."""
        # First freeze everything
        for p in self.model.parameters():
            p.requires_grad = False
        # Unfreeze the target block
        for p in self.get_block_parameters(block_idx):
            p.requires_grad = True
        # Unfreeze timestep embedder and block's AdaLN
        if self.timestep_embedder is not None:
            for p in self.timestep_embedder.parameters():
                p.requires_grad = True
        if self.adalns is not None:
            for p in self.adalns[block_idx].parameters():
                p.requires_grad = True

    def unfreeze_all(self):
        """Unfreeze all parameters (for eval or standard training)."""
        for p in self.model.parameters():
            p.requires_grad = True

    def forward_block(
        self,
        input_ids: torch.Tensor,
        block_idx: int,
        noisy_embeds: Optional[torch.Tensor] = None,
        sigma: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass through only one block's layers.

        Args:
            input_ids: Input token IDs (B, T)
            block_idx: Which block to run (0-indexed)
            noisy_embeds: Noisy target embeddings (B, T, d_model)
            sigma: Noise level (B,) for conditioning
            **kwargs: Additional args passed to model forward

        Returns:
            Logits (B, T, vocab_size)
        """
        layers = self.block_layers[block_idx]

        # Compute conditioning from sigma
        conditioning = None
        if self.timestep_embedder is not None and sigma is not None:
            c_noise = 0.25 * sigma.log()
            conditioning = self.timestep_embedder(c_noise)

        # Get AdaLN modulation for this block
        modulation = None
        if self.adalns is not None and conditioning is not None:
            modulation = self.adalns[block_idx](conditioning)

        # Run model with only the specified layers
        # The model needs to support layer_indices in forward
        return self.model(
            input_ids,
            layer_indices=layers,
            noisy_embeds=noisy_embeds,
            modulation=modulation,
            **kwargs,
        )

    def train_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        block_idx: Optional[int] = None,
        noise_dropout: float = 0.1,
    ) -> dict[str, float]:
        """One DiffusionBlocks training step.

        1. Sample a block (or use provided block_idx)
        2. Get target embeddings (from labels)
        3. Add noise to target embeddings
        4. Forward through only that block
        5. Compute weighted CE loss
        6. Backprop only through that block

        Args:
            input_ids: Input token IDs (B, T)
            labels: Target token IDs (B, T)
            optimizer: Optimizer (should only have block params)
            block_idx: Force specific block (default: random)
            noise_dropout: Probability of skipping noisy embeds (CFG-style,
                ensures model works without noise at inference)

        Returns:
            Dict with loss, block_idx, sigma, etc.
        """
        self.model.train()
        device = input_ids.device
        batch_size = input_ids.shape[0]

        # 1. Select block
        if block_idx is None:
            block_idx = random.randint(0, self.num_blocks - 1)

        # 2. Get target embeddings (at natural scale, NOT normalized)
        with torch.no_grad():
            target_embeds = self.model.embed(labels)  # (B, T, d_model)
            # Get input embedding scale for proper noise scaling
            input_embeds = self.model.embed(input_ids)
            embed_scale = input_embeds.norm(dim=-1).mean().item()

        # 3. Sample noise level for this block
        sigmas = self.get_block_sigma(block_idx, n_samples=batch_size).to(device)

        # 4. Create noisy embeddings: z = y + σ * ε
        # Scale noise to match embedding scale (AR adaptation — not normalized)
        noise = torch.randn_like(target_embeds) * embed_scale
        sigma_expanded = sigmas[:, None, None]
        noisy_embeds = target_embeds + sigma_expanded * noise

        # 5. Scale noisy embeds to be a reasonable addition to input
        # For AR: noisy embeds should be ~10% of input scale
        # (enough signal for block specialization, doesn't overwhelm input)
        noise_scale = 0.1
        scaled_noisy = noisy_embeds * noise_scale

        # Noise dropout: occasionally skip noisy embeds so model learns
        # to work WITHOUT them at inference (classifier-free guidance style)
        if random.random() < noise_dropout:
            scaled_noisy = None
            # Use small epsilon instead of zero (log(0) = -inf → NaN)
            sigmas_for_cond = torch.full_like(sigmas, 0.001)
        else:
            sigmas_for_cond = sigmas

        logits = self.forward_block(
            input_ids=input_ids,
            block_idx=block_idx,
            noisy_embeds=scaled_noisy,
            sigma=sigmas_for_cond,
        )
        # forward returns tuple (logits, loss) — extract logits
        if isinstance(logits, tuple):
            logits = logits[0]

        # 7. Compute denoised output
        # D(z, σ) = c_skip * z + c_out * model_output
        # But for LM, we work in logit space, so we use CE loss directly
        # The model predicts logits, and we compute CE against the true labels

        # 8. Cross-entropy loss (uniform weight for AR — EDM weights unstable)
        ce_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            reduction="none",
        )
        ce_loss = ce_loss.reshape(batch_size, -1)  # (B, T)

        # For AR models, use uniform weight=1 (EDM weights explode at low sigma)
        # The noise conditioning via AdaLN still provides block specialization
        weighted_loss = ce_loss.mean()

        # 9. Backprop only through this block
        optimizer.zero_grad()
        weighted_loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()

        return {
            "loss": weighted_loss.item(),
            "ce_loss": ce_loss.mean().item(),
            "block_idx": block_idx,
            "sigma_mean": sigmas.mean().item(),
            "weight_mean": 1.0,
            "noise_dropped": scaled_noisy is None,
        }

    def diffuse_step(
        self,
        input_ids: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Full diffusion inference: denoise from noise to output.

        Runs B denoising steps, each using the appropriate block.
        This is for generation (not standard autoregressive decoding).

        Args:
            input_ids: Context/prompt tokens (B, T_ctx)

        Returns:
            Logits for next token prediction (B, vocab_size)
        """
        self.model.eval()
        device = input_ids.device
        batch_size = input_ids.shape[0]

        # Start from pure noise
        z = torch.randn(batch_size, self.d_model, device=device)
        z *= math.sqrt(1.0 + self.block_sigmas[-1] ** 2)

        # Denoise through all blocks
        with torch.no_grad():
            for i in range(self.num_blocks):
                sigma = torch.full(
                    (batch_size,), self.block_sigmas[self.num_blocks - i],
                    device=device,
                )
                next_sigma = torch.full(
                    (batch_size,), self.block_sigmas[self.num_blocks - i - 1]
                    if i < self.num_blocks - 1 else self.block_sigmas[0],
                    device=device,
                )

                # Denoise
                logits = self.forward_block(
                    input_ids=input_ids,
                    block_idx=self.num_blocks - i - 1,
                    noisy_embeds=z.unsqueeze(1).expand(-1, input_ids.shape[1], -1),
                    sigma=sigma,
                    **kwargs,
                )
                if isinstance(logits, tuple):
                    logits = logits[0]

                # Get denoised embedding from logits
                probs = F.softmax(logits, dim=-1)
                denoised = F.linear(probs, self.model.embed.weight.t())
                denoised = F.normalize(denoised, p=2, dim=-1)

                # Euler step
                d = (z - denoised.mean(dim=1)) / sigma[0]
                dt = next_sigma[0] - sigma[0]
                z = z + dt * d

        # Final prediction
        sigma_final = torch.full(
            (batch_size,), self.block_sigmas[0], device=device
        )
        logits = self.forward_block(
            input_ids=input_ids,
            block_idx=0,
            noisy_embeds=z.unsqueeze(1).expand(-1, input_ids.shape[1], -1),
            sigma=sigma_final,
            **kwargs,
        )
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits[:, -1]  # Return last position logits
