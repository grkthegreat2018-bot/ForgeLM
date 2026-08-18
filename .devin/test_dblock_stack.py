"""Isolated test: DiffusionBlocks + muon_sf_blockwise + grad mixup (3-way stack).

Sister agent says DiffusionBlocks should stack with the optimizer + grad mixup
winners. This tests that hypothesis on a toy model.

Targets from randomizer:
  T1: DiffusionBlocks + Muon-SF-Blockwise + Grad mixup (3-way stack)
  T2: DiffusionBlocks + sigma-as-LR (free per-block LR signal from noise level)
  T3: DiffusionBlocks + Muon with sigma replacing Fisher EMA as sharpness

The synergy hypothesis: DiffusionBlocks frees B× VRAM → enables more mixup
batches → better gradient → muon_sf_blockwise applies it better. All three
are orthogonal: memory, data, optimizer.
"""
import math
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from muon import SingleDeviceMuonWithAuxAdam, muon_update
from schedulefree import AdamWScheduleFree
import numpy as np
from scipy.stats import norm


# ── Toy model with DiffusionBlocks support ───────────────────────────────

class TinyBlockDB(nn.Module):
    """TinyBlock with AdaLN modulation support for DiffusionBlocks."""
    def __init__(self, d=128, n_heads=4, n_kv=2, d_ff=512):
        super().__init__()
        self.ln1 = nn.RMSNorm(d)
        self.q = nn.Linear(d, n_heads * (d // n_heads), bias=False)
        self.k = nn.Linear(d, n_kv * (d // n_heads), bias=False)
        self.v = nn.Linear(d, n_kv * (d // n_heads), bias=False)
        self.o = nn.Linear(n_heads * (d // n_heads), d, bias=False)
        self.hd = d // n_heads
        self.nh = n_heads
        self.nkv = n_kv
        self.ln2 = nn.RMSNorm(d)
        self.w_gate = nn.Linear(d, d_ff, bias=False)
        self.w_up = nn.Linear(d, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d, bias=False)

    def forward(self, x, modulation=None):
        # AdaLN: shift/scale modulation (4 chunks: shift_msa, scale_msa, shift_mlp, scale_mlp)
        shift_msa = scale_msa = shift_mlp = scale_mlp = None
        if modulation is not None:
            mod = modulation.to(x.dtype)
            chunks = mod.chunk(4, dim=-1)
            shift_msa, scale_msa, shift_mlp, scale_mlp = chunks

        B, T, D = x.shape
        attn_in = self.ln1(x)
        if shift_msa is not None:
            attn_in = attn_in * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        q = self.q(attn_in).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.k(attn_in).view(B, T, self.nkv, self.hd).transpose(1, 2)
        v = self.v(attn_in).view(B, T, self.nkv, self.hd).transpose(1, 2)
        k = k.repeat_interleave(self.nh // self.nkv, dim=1)
        v = v.repeat_interleave(self.nh // self.nkv, dim=1)
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        att = att.transpose(1, 2).contiguous().view(B, T, -1)
        x = x + self.o(att)

        ffn_in = self.ln2(x)
        if shift_mlp is not None:
            ffn_in = ffn_in * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        g = F.silu(self.w_gate(ffn_in))
        u = self.w_up(ffn_in)
        x = x + self.w_down(g * u)
        return x


class TinyLMDB(nn.Module):
    """TinyLM with DiffusionBlocks interface: layer_indices, noisy_embeds, modulation."""
    def __init__(self, vocab=256, d=128, n_layers=6, n_heads=4, n_kv=2, d_ff=512):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([TinyBlockDB(d, n_heads, n_kv, d_ff) for _ in range(n_layers)])
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.embed.weight
        self.d = d

    def forward(self, ids, layer_indices=None, noisy_embeds=None, modulation=None, **kwargs):
        # If noisy_embeds provided, use them instead of standard embedding
        # (DiffusionBlocks adds noise to target embeddings and feeds through subset of layers)
        if noisy_embeds is not None:
            x = noisy_embeds
        else:
            x = self.embed(ids)

        if layer_indices is not None:
            # Run only specified layers (DiffusionBlocks mode)
            for i in layer_indices:
                x = self.blocks[i](x, modulation=modulation)
        else:
            for b in self.blocks:
                x = b(x, modulation=modulation)
        return self.head(x)


# ── DiffusionBlocks components (simplified from research/diffusion_blocks.py) ──

def get_block_sigmas(num_blocks, sigma_min=0.002, sigma_max=80.0, p_mean=-1.2, p_std=1.2):
    cdf_min = norm.cdf((np.log(sigma_min) - p_mean) / p_std)
    cdf_max = norm.cdf((np.log(sigma_max) - p_mean) / p_std)
    sigmas = []
    for i in range(num_blocks + 1):
        p = cdf_min + (cdf_max - cdf_min) * (i / num_blocks)
        sigmas.append(float(np.exp(p_mean + p_std * norm.ppf(p))))
    return sigmas


def sample_block_sigma(block_sigmas, block_idx, n_samples=1, gamma=0.1, p_mean=-1.2, p_std=1.2):
    sigma_min_block = block_sigmas[block_idx]
    sigma_max_block = block_sigmas[block_idx + 1]
    if gamma > 0.0:
        log_min = np.log(sigma_min_block)
        log_max = np.log(sigma_max_block)
        log_range = log_max - log_min
        sigma_min_block = np.exp(log_min - gamma * log_range)
        sigma_max_block = np.exp(log_max + gamma * log_range)
        sigma_min_block = max(sigma_min_block, block_sigmas[0])
        sigma_max_block = min(sigma_max_block, block_sigmas[-1])
    cdf_min = norm.cdf((np.log(sigma_min_block) - p_mean) / p_std)
    cdf_max = norm.cdf((np.log(sigma_max_block) - p_mean) / p_std)
    rand = np.random.uniform(cdf_min, cdf_max, n_samples)
    sigma = np.exp(p_mean + p_std * norm.ppf(rand))
    return torch.from_numpy(sigma).float()


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size=128, freq_embedding_size=64):
        super().__init__()
        self.frequency_embedding_size = freq_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(freq_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half).to(t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = F.pad(embedding, (0, 1))
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_freq = t_freq.to(dtype=next(self.parameters()).dtype)
        return self.mlp(t_freq)


class AdaLN(nn.Module):
    def __init__(self, cond_dim, hidden_size):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(cond_dim, 4 * hidden_size, bias=True)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        return self.silu(self.linear(x))


class DiffusionBlocksToy:
    """Simplified DiffusionBlocks for toy model testing."""
    def __init__(self, model, num_blocks=3, d_model=128, num_layers=6, cond_dim=64,
                 sigma_data=0.5, gamma=0.1):
        self.model = model
        self.num_blocks = num_blocks
        self.d_model = d_model
        self.sigma_data = sigma_data
        self.gamma = gamma

        self.block_sigmas = get_block_sigmas(num_blocks)
        layers_per_block = num_layers // num_blocks
        self.block_layers = []
        for b in range(num_blocks):
            start = b * layers_per_block
            end = (b + 1) * layers_per_block if b < num_blocks - 1 else num_layers
            self.block_layers.append(list(range(start, end)))

        # AdaLN conditioning
        self.timestep_embedder = TimestepEmbedder(hidden_size=cond_dim, freq_embedding_size=64)
        self.adalns = nn.ModuleList([AdaLN(cond_dim, d_model) for _ in range(num_blocks)])
        device = next(model.parameters()).device
        self.timestep_embedder = self.timestep_embedder.to(device)
        self.adalns = self.adalns.to(device)

        print(f"[DiffusionBlocks] {num_blocks} blocks, {num_layers} layers -> "
              f"{[len(b) for b in self.block_layers]} layers/block")

    def get_block_sigma(self, block_idx, n_samples=1):
        return sample_block_sigma(self.block_sigmas, block_idx, n_samples, gamma=self.gamma)

    def get_block_parameters(self, block_idx):
        layers = self.block_layers[block_idx]
        params = []
        for layer_idx in layers:
            params.extend(self.model.blocks[layer_idx].parameters())
        params.extend(self.adalns[block_idx].parameters())
        if block_idx == 0:
            params.extend(self.model.embed.parameters())
        if block_idx == self.num_blocks - 1:
            params.extend(self.model.head.parameters())
        return params

    def freeze_all_except_block(self, block_idx):
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.get_block_parameters(block_idx):
            p.requires_grad = True
        for p in self.timestep_embedder.parameters():
            p.requires_grad = True
        for p in self.adalns[block_idx].parameters():
            p.requires_grad = True

    def unfreeze_all(self):
        for p in self.model.parameters():
            p.requires_grad = True

    def forward_block(self, input_ids, block_idx, noisy_embeds=None, sigma=None):
        layers = self.block_layers[block_idx]
        conditioning = None
        if sigma is not None:
            c_noise = 0.25 * sigma.log()
            conditioning = self.timestep_embedder(c_noise)
        modulation = None
        if conditioning is not None:
            modulation = self.adalns[block_idx](conditioning)
        return self.model(input_ids, layer_indices=layers, noisy_embeds=noisy_embeds,
                          modulation=modulation)

    def train_step(self, input_ids, labels, optimizer, block_idx=None, grad_mixup=1):
        """One DiffusionBlocks training step with optional grad mixup."""
        self.model.train()
        device = input_ids.device
        batch_size = input_ids.shape[0]

        if block_idx is None:
            block_idx = random.randint(0, self.num_blocks - 1)

        # Get target embeddings
        with torch.no_grad():
            target_embeds = self.model.embed(labels)
            target_embeds = F.normalize(target_embeds, p=2, dim=-1)

        # Sample noise level
        sigmas = self.get_block_sigma(block_idx, n_samples=batch_size).to(device)
        noise = torch.randn_like(target_embeds)
        noisy_embeds = target_embeds + sigmas[:, None, None] * noise

        # EDM scaling
        sigma_data = self.sigma_data
        c_in = 1 / (sigmas ** 2 + sigma_data ** 2) ** 0.5
        scaled_noisy = noisy_embeds * c_in[:, None, None]

        def compute_ce(ids_batch, labels_batch, noisy_batch, sigmas_batch):
            logits = self.forward_block(ids_batch, block_idx, noisy_embeds=noisy_batch, sigma=sigmas_batch)
            if isinstance(logits, tuple):
                logits = logits[0]
            return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels_batch.reshape(-1))

        ce_loss = compute_ce(input_ids, labels, scaled_noisy, sigmas)

        if grad_mixup > 1:
            # Grad mixup: average gradients from N batches
            optimizer.zero_grad()
            ce_loss.backward()
            saved_grads = {}
            for n, p in self.model.named_parameters():
                if p.grad is not None and p.requires_grad:
                    saved_grads[n] = p.grad.clone()
            # Also save AdaLN/timestep grads
            for n, p in self.timestep_embedder.named_parameters():
                if p.grad is not None:
                    saved_grads[f"ts_{n}"] = p.grad.clone()
            for n, p in self.adalns[block_idx].named_parameters():
                if p.grad is not None:
                    saved_grads[f"adaln_{n}"] = p.grad.clone()

            for mixup_i in range(grad_mixup - 1):
                # New noise for same data
                noise2 = torch.randn_like(target_embeds)
                sigmas2 = self.get_block_sigma(block_idx, n_samples=batch_size).to(device)
                noisy2 = target_embeds + sigmas2[:, None, None] * noise2
                c_in2 = 1 / (sigmas2 ** 2 + sigma_data ** 2) ** 0.5
                scaled2 = noisy2 * c_in2[:, None, None]
                ce2 = compute_ce(input_ids, labels, scaled2, sigmas2)
                optimizer.zero_grad()
                ce2.backward()
                for n, p in self.model.named_parameters():
                    if p.grad is not None and n in saved_grads:
                        saved_grads[n] = (saved_grads[n] * (mixup_i + 1) + p.grad) / (mixup_i + 2)
                for n, p in self.timestep_embedder.named_parameters():
                    if p.grad is not None and f"ts_{n}" in saved_grads:
                        saved_grads[f"ts_{n}"] = (saved_grads[f"ts_{n}"] * (mixup_i + 1) + p.grad) / (mixup_i + 2)
                for n, p in self.adalns[block_idx].named_parameters():
                    if p.grad is not None and f"adaln_{n}" in saved_grads:
                        saved_grads[f"adaln_{n}"] = (saved_grads[f"adaln_{n}"] * (mixup_i + 1) + p.grad) / (mixup_i + 2)

            # Restore averaged grads
            optimizer.zero_grad()
            for n, p in self.model.named_parameters():
                if n in saved_grads and p.requires_grad:
                    p.grad = saved_grads[n]
            for n, p in self.timestep_embedder.named_parameters():
                if f"ts_{n}" in saved_grads:
                    p.grad = saved_grads[f"ts_{n}"]
            for n, p in self.adalns[block_idx].named_parameters():
                if f"adaln_{n}" in saved_grads:
                    p.grad = saved_grads[f"adaln_{n}"]
        else:
            optimizer.zero_grad()
            ce_loss.backward()

        optimizer.step()
        return {"loss": ce_loss.item(), "block_idx": block_idx, "sigma_mean": sigmas.mean().item()}


# ── Muon-SF-Blockwise (from previous tests) ──────────────────────────────

class MuonScheduleFree(SingleDeviceMuonWithAuxAdam):
    def __init__(self, param_groups):
        adam_params = []
        muon_groups = []
        adam_lr = 3e-4
        for g in param_groups:
            if g["use_muon"]:
                muon_groups.append(g)
            else:
                adam_params.extend(g["params"])
                adam_lr = g["lr"]
        self._sf = AdamWScheduleFree(adam_params, lr=adam_lr, betas=(0.9, 0.95), weight_decay=0.0)
        super().__init__(muon_groups)
        self._adam_params = adam_params
        self._sf.train()

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])
        self._sf.step()

    def zero_grad(self, set_to_none=True):
        super().zero_grad(set_to_none=set_to_none)
        self._sf.zero_grad(set_to_none=set_to_none)

    def eval(self):
        self._sf.eval()

    def train(self):
        self._sf.train()


class MuonSFBlockwise(MuonScheduleFree):
    def __init__(self, param_groups, n_blocks=3, refresh_every=16, sharp_beta=0.9,
                 lr_min_ratio=0.3, lr_max_ratio=2.5):
        super().__init__(param_groups)
        self._n_blocks = n_blocks
        self._refresh_every = refresh_every
        self._sharp_beta = sharp_beta
        self._lr_min_ratio = lr_min_ratio
        self._lr_max_ratio = lr_max_ratio
        self._step_count = 0
        self._block_sharp_ema = [0.0] * n_blocks
        self._base_muon_lr = next(g["lr"] for g in self.param_groups if g["use_muon"])
        self._initial_sharp = None

    @torch.no_grad()
    def step(self, closure=None):
        self._step_count += 1
        if self._step_count % self._refresh_every == 0:
            total_grad_sq = 0.0
            n_params = 0
            for g in self.param_groups:
                if g["use_muon"]:
                    for p in g["params"]:
                        if p.grad is not None:
                            total_grad_sq += p.grad.float().pow(2).mean().item()
                            n_params += 1
            avg_sharp = total_grad_sq / max(1, n_params)
            for i in range(self._n_blocks):
                self._block_sharp_ema[i] = (
                    self._sharp_beta * self._block_sharp_ema[i]
                    + (1 - self._sharp_beta) * avg_sharp
                )
            sharp = sum(self._block_sharp_ema) / max(1, len(self._block_sharp_ema))
            if self._initial_sharp is None:
                self._initial_sharp = sharp
            ratio = sharp / max(1e-8, self._initial_sharp)
            ratio = max(self._lr_min_ratio, min(self._lr_max_ratio, ratio))
            for g in self.param_groups:
                if g["use_muon"]:
                    g["lr"] = self._base_muon_lr * ratio
        super().step(closure)


# ── Sigma-as-LR optimizer (Target 2: use DiffusionBlocks sigma as per-block LR) ──

class MuonSFSigmaLR(MuonScheduleFree):
    """Target 2: Use DiffusionBlocks' sigma as per-block LR multiplier.

    Instead of Fisher EMA sharpness, use the block's noise level sigma:
    high sigma (hard, noisy) → high LR (explore)
    low sigma (easy, clean) → low LR (refine)

    This is a FREE signal — DiffusionBlocks already computes sigma.
    """
    def __init__(self, param_groups, block_sigmas, n_blocks=3, lr_min_ratio=0.3, lr_max_ratio=2.5):
        super().__init__(param_groups)
        self._block_sigmas = block_sigmas
        self._n_blocks = n_blocks
        self._lr_min_ratio = lr_min_ratio
        self._lr_max_ratio = lr_max_ratio
        self._base_muon_lr = next(g["lr"] for g in self.param_groups if g["use_muon"])
        self._current_block = 0
        self._current_sigma = block_sigmas[0]
        # Normalize: use log-sigma ratio relative to median sigma
        sigmas_sorted = sorted(block_sigmas)
        self._sigma_median = sigmas_sorted[len(sigmas_sorted) // 2]

    def set_current_block(self, block_idx, sigma):
        """Called before step to inform optimizer which block/sigma is active."""
        self._current_block = block_idx
        self._current_sigma = sigma

    @torch.no_grad()
    def step(self, closure=None):
        # Scale LR by sigma ratio (high sigma = high LR)
        sigma_ratio = math.log(max(1e-8, self._current_sigma)) / math.log(max(1e-8, self._sigma_median))
        sigma_ratio = max(self._lr_min_ratio, min(self._lr_max_ratio, sigma_ratio))
        for g in self.param_groups:
            if g["use_muon"]:
                g["lr"] = self._base_muon_lr * sigma_ratio
        super().step(closure)


# ── Data ─────────────────────────────────────────────────────────────────

def make_batch(batch_size, seq_len, vocab, seed):
    g = torch.Generator().manual_seed(seed)
    pos = torch.arange(seq_len).float()
    base = (pos.pow(1.7).long() * 11 + 7) % vocab
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    batch_primes = torch.tensor([primes[s % len(primes)] for s in range(batch_size)])
    cross = ((pos.unsqueeze(0) * batch_primes.unsqueeze(1)).long() ^ (pos.pow(2).long() // 3)) % vocab
    seq = (base.unsqueeze(0) + cross + torch.randint(0, 5, (batch_size, seq_len), generator=g)) % vocab
    return seq


def split_param_groups(model, lr_muon, lr_adam):
    muon_p, adam_p = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and "embed" not in name and "head" not in name:
            muon_p.append(p)
        else:
            adam_p.append(p)
    return [
        dict(params=muon_p, lr=lr_muon, momentum=0.95, weight_decay=0.0, use_muon=True),
        dict(params=adam_p, lr=lr_adam, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0, use_muon=False),
    ]


def cosine_lr(step, total, warmup, max_lr, min_lr):
    if step < warmup:
        return max_lr * step / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * prog))


# ── Training loops ───────────────────────────────────────────────────────

def train_standard(model, opt, n_steps, batch_size, seq_len, vocab, variant_name,
                   grad_mixup=1):
    """Standard (non-DiffusionBlocks) training with optional grad mixup."""
    model.train()
    losses = []
    t0 = time.time()
    for step in range(n_steps):
        ids = make_batch(batch_size, seq_len, vocab, seed=step * 7 + 1).cuda()
        x, y = ids[:, :-1], ids[:, 1:]
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))

        if grad_mixup > 1:
            opt.zero_grad()
            loss.backward()
            saved = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
            for mi in range(grad_mixup - 1):
                seed2 = (step + (mi + 1) * n_steps // 2) * 7 + 1
                ids2 = make_batch(batch_size, seq_len, vocab, seed=seed2).cuda()
                x2, y2 = ids2[:, :-1], ids2[:, 1:]
                loss2 = F.cross_entropy(model(x2).reshape(-1, vocab), y2.reshape(-1))
                opt.zero_grad()
                loss2.backward()
                for n, p in model.named_parameters():
                    if p.grad is not None and n in saved:
                        saved[n] = (saved[n] * (mi + 1) + p.grad) / (mi + 2)
            opt.zero_grad()
            for n, p in model.named_parameters():
                if n in saved:
                    p.grad = saved[n]
        else:
            opt.zero_grad()
            loss.backward()

        if variant_name == "adamw_cosine":
            lr = cosine_lr(step, n_steps, 10, 3e-3, 3e-4)
            for g in opt.param_groups:
                g["lr"] = lr
        opt.step()
        losses.append(loss.item())
        if step % 100 == 0 or step == n_steps - 1:
            print(f"  [{variant_name:30s}] step {step:4d} loss {loss.item():.4f}")
    dt = time.time() - t0
    return losses, dt


def train_diffusion_blocks(dblock, opt, n_steps, batch_size, seq_len, vocab,
                           variant_name, grad_mixup=1, sigma_lr=False, block_schedule="random"):
    """DiffusionBlocks training with optional grad mixup and sigma-as-LR.

    block_schedule: 'random' (default), 'round_robin' (deterministic cycle),
        'loss_adaptive' (pick block with highest recent loss), 'hardest_first'
        (cycle high→low sigma blocks).
    """
    model = dblock.model
    losses = []
    block_losses_ema = [float('inf')] * dblock.num_blocks  # untrained = inf → picked first
    t0 = time.time()
    for step in range(n_steps):
        ids = make_batch(batch_size, seq_len, vocab, seed=step * 7 + 1).cuda()
        x, y = ids[:, :-1], ids[:, 1:]

        # Block scheduling
        if block_schedule == "round_robin":
            block_idx = step % dblock.num_blocks
        elif block_schedule == "loss_adaptive":
            # Pick block with highest EMA loss (hardest block gets more training)
            # BUG FIX: untrained blocks have EMA=0, so they'd never get picked.
            # Solution: initialize EMA to infinity so untrained blocks go first.
            # Also track training count to ensure all blocks get minimum coverage.
            if not hasattr(dblock, "_adapt_train_count"):
                dblock._adapt_train_count = [0] * dblock.num_blocks
            # If any block has < 5 training steps, pick the least-trained one
            min_trained = min(dblock._adapt_train_count)
            if min_trained < 5:
                block_idx = dblock._adapt_train_count.index(min_trained)
            else:
                # All blocks have minimum coverage — pick highest EMA loss
                block_idx = max(range(dblock.num_blocks), key=lambda b: block_losses_ema[b])
            dblock._adapt_train_count[block_idx] += 1
        elif block_schedule == "hardest_first":
            # Cycle from high-sigma (hard) to low-sigma (easy) blocks
            order = list(range(dblock.num_blocks - 1, -1, -1))
            block_idx = order[step % dblock.num_blocks]
        elif block_schedule == "easy_first":
            # Cycle from low-sigma (easy) to high-sigma (hard) — curriculum style
            block_idx = (step // dblock.num_blocks) % dblock.num_blocks
        else:
            block_idx = random.randint(0, dblock.num_blocks - 1)

        dblock.freeze_all_except_block(block_idx)

        # For sigma-LR: set current block/sigma before step
        if sigma_lr and hasattr(opt, "set_current_block"):
            sigmas = dblock.get_block_sigma(block_idx, n_samples=1)
            opt.set_current_block(block_idx, sigmas[0].item())

        result = dblock.train_step(x, y, opt, block_idx=block_idx, grad_mixup=grad_mixup)
        losses.append(result["loss"])

        # Update block loss EMA for loss_adaptive
        beta_ema = 0.9
        if block_losses_ema[block_idx] == float('inf'):
            block_losses_ema[block_idx] = result["loss"]  # first observation
        else:
            block_losses_ema[block_idx] = (
                beta_ema * block_losses_ema[block_idx] + (1 - beta_ema) * result["loss"]
            )

        if step % 100 == 0 or step == n_steps - 1:
            print(f"  [{variant_name:30s}] step {step:4d} loss {result['loss']:.4f} "
                  f"block={result['block_idx']} sigma={result['sigma_mean']:.3f}")
    dt = time.time() - t0
    dblock.unfreeze_all()
    return losses, dt


def run_variant(name, n_steps, mode="standard", optimizer_name="adamw",
                grad_mixup=1, num_blocks=3, sigma_lr=False, block_schedule="random"):
    """Run one variant. mode = 'standard' or 'diffusion'."""
    torch.manual_seed(42)
    d, n_layers = 128, 6
    vocab = 256
    model = TinyLMDB(vocab=vocab, d=d, n_layers=n_layers).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n=== {name} === ({n_params/1e6:.2f}M params, mode={mode}, opt={optimizer_name}, mixup={grad_mixup})")

    LR_MUON = 2e-3
    LR_ADAM = 3e-3

    if mode == "diffusion":
        dblock = DiffusionBlocksToy(model, num_blocks=num_blocks, d_model=d, num_layers=n_layers)

        # Build optimizer for ALL params (freeze/unfreeze handles which get grads)
        dblock.unfreeze_all()
        if optimizer_name == "adamw":
            opt = AdamW(model.parameters(), lr=LR_ADAM, fused=True)
            # Also optimize timestep embedder and adalns
            opt.add_param_group({"params": list(dblock.timestep_embedder.parameters()) + 
                                 [p for adaln in dblock.adalns for p in adaln.parameters()]})
        elif optimizer_name == "muon_sf_bw":
            pg = split_param_groups(model, LR_MUON, LR_ADAM)
            aux_params = list(dblock.timestep_embedder.parameters()) + \
                         [p for adaln in dblock.adalns for p in adaln.parameters()]
            pg[1]["params"].extend(aux_params)
            if sigma_lr:
                opt = MuonSFSigmaLR(pg, block_sigmas=dblock.block_sigmas, n_blocks=num_blocks)
            else:
                opt = MuonSFBlockwise(pg, n_blocks=num_blocks)
        elif optimizer_name == "muon_plain":
            # Plain Muon (no SF, no blockwise) — isolate Muon effect
            pg = split_param_groups(model, LR_MUON, LR_ADAM)
            aux_params = list(dblock.timestep_embedder.parameters()) + \
                         [p for adaln in dblock.adalns for p in adaln.parameters()]
            pg[1]["params"].extend(aux_params)
            opt = SingleDeviceMuonWithAuxAdam(pg)
        elif optimizer_name == "muon_sf":
            # Muon + SF (no blockwise) — isolate SF effect
            pg = split_param_groups(model, LR_MUON, LR_ADAM)
            aux_params = list(dblock.timestep_embedder.parameters()) + \
                         [p for adaln in dblock.adalns for p in adaln.parameters()]
            pg[1]["params"].extend(aux_params)
            opt = MuonScheduleFree(pg)
        losses, dt = train_diffusion_blocks(dblock, opt, n_steps, 16, 64, vocab,
                                            variant_name=name, grad_mixup=grad_mixup,
                                            sigma_lr=sigma_lr, block_schedule=block_schedule)
    else:
        if optimizer_name == "adamw":
            opt = AdamW(model.parameters(), lr=LR_ADAM, fused=True)
        elif optimizer_name == "muon_sf_bw":
            opt = MuonSFBlockwise(split_param_groups(model, LR_MUON, LR_ADAM), n_blocks=num_blocks)
        elif optimizer_name == "muon_plain":
            opt = SingleDeviceMuonWithAuxAdam(split_param_groups(model, LR_MUON, LR_ADAM))
        elif optimizer_name == "muon_sf":
            opt = MuonScheduleFree(split_param_groups(model, LR_MUON, LR_ADAM))
        losses, dt = train_standard(model, opt, n_steps, 16, 64, vocab,
                                    variant_name=name, grad_mixup=grad_mixup)

    final = sum(losses[-20:]) / 20
    print(f"  final_20avg={final:.4f} time={dt:.2f}s")
    return {"name": name, "final_loss": final, "time": dt, "losses": losses}


def main():
    print("=" * 70)
    print("ROUND 5: Block scheduling strategies + variance measurement")
    print("=" * 70)
    N = 400

    results = []
    # Baseline
    results.append(run_variant("adamw_cosine", N, mode="standard", optimizer_name="adamw"))

    # Round 4b winner: random scheduling + muon_sf + mixup3 (run 3x for variance)
    for run in range(3):
        results.append(run_variant(f"dblock_rand_muon_sf_mix3_r{run}",
            N, mode="diffusion", optimizer_name="muon_sf", grad_mixup=3,
            block_schedule="random"))

    # Round-robin (deterministic) — the main hypothesis
    for run in range(3):
        results.append(run_variant(f"dblock_robin_muon_sf_mix3_r{run}",
            N, mode="diffusion", optimizer_name="muon_sf", grad_mixup=3,
            block_schedule="round_robin"))

    # Loss-adaptive (train hardest block more)
    for run in range(3):
        results.append(run_variant(f"dblock_adapt_muon_sf_mix3_r{run}",
            N, mode="diffusion", optimizer_name="muon_sf", grad_mixup=3,
            block_schedule="loss_adaptive"))

    # Hardest-first (cycle high→low sigma)
    results.append(run_variant("dblock_hard_first_muon_sf_mix3",
        N, mode="diffusion", optimizer_name="muon_sf", grad_mixup=3,
        block_schedule="hardest_first"))

    # Easy-first (curriculum: low→high sigma)
    results.append(run_variant("dblock_easy_first_muon_sf_mix3",
        N, mode="diffusion", optimizer_name="muon_sf", grad_mixup=3,
        block_schedule="easy_first"))

    # Also test round-robin with AdamW (is the scheduling win optimizer-independent?)
    results.append(run_variant("dblock_robin_adamw_mix3",
        N, mode="diffusion", optimizer_name="adamw", grad_mixup=3,
        block_schedule="round_robin"))

    print("\n" + "=" * 70)
    print("SUMMARY (sorted by final loss, lower=better)")
    print("=" * 70)
    results.sort(key=lambda r: r["final_loss"])
    for i, r in enumerate(results):
        print(f"  {i+1:2d}. {r['name']:35s} loss={r['final_loss']:.4f}  time={r['time']:.2f}s")

    # Variance analysis for the 3x runs
    print("\n" + "=" * 70)
    print("VARIANCE ANALYSIS (3 runs each)")
    print("=" * 70)
    for prefix in ["dblock_rand_muon_sf_mix3", "dblock_robin_muon_sf_mix3", "dblock_adapt_muon_sf_mix3"]:
        runs = [r for r in results if r["name"].startswith(prefix)]
        if len(runs) >= 2:
            losses = [r["final_loss"] for r in runs]
            mean = sum(losses) / len(losses)
            variance = sum((l - mean) ** 2 for l in losses) / len(losses)
            std = variance ** 0.5
            print(f"  {prefix}: mean={mean:.4f} std={std:.4f} range=[{min(losses):.4f}, {max(losses):.4f}]")

    best = results[0]
    baseline = next(r for r in results if r["name"] == "adamw_cosine")
    print(f"\n  Best: {best['name']} ({best['final_loss']:.4f})")
    print(f"  vs AdamW cosine ({baseline['final_loss']:.4f}): {baseline['final_loss']/best['final_loss']:.2f}x better loss")


if __name__ == "__main__":
    main()
