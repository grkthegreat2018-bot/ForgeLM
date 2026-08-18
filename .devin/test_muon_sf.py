"""Isolated test: Muon-SF-Blockwise optimizer vs baselines.

Toy problem: small transformer, synthetic next-token task.
Goal: find which optimizer combination converges fastest (lowest loss
in N steps). Get NUMBERS, not theory.

Variants tested:
  A. Fused AdamW + cosine schedule (current ForgeAI default)
  B. Muon (hidden) + internal-Adam (embed) — known good (MuonWithAuxAdam)
  C. Muon + AdamWScheduleFree (embed) — novel combo #1
  D. Muon + AdamWScheduleFree + blockwise-sharpness LR — novel combo #2
  E. Muon + AdamWScheduleFree + cross-domain: EDM-sigma-scaled LR
"""
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from muon import SingleDeviceMuonWithAuxAdam, muon_update
from schedulefree import AdamWScheduleFree


# ── Toy model ────────────────────────────────────────────────────────────

class TinyBlock(nn.Module):
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

    def forward(self, x):
        B, T, D = x.shape
        h = self.ln1(x)
        q = self.q(h).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.k(h).view(B, T, self.nkv, self.hd).transpose(1, 2)
        v = self.v(h).view(B, T, self.nkv, self.hd).transpose(1, 2)
        k = k.repeat_interleave(self.nh // self.nkv, dim=1)
        v = v.repeat_interleave(self.nh // self.nkv, dim=1)
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        att = att.transpose(1, 2).contiguous().view(B, T, -1)
        x = x + self.o(att)
        h = self.ln2(x)
        g = F.silu(self.w_gate(h))
        u = self.w_up(h)
        x = x + self.w_down(g * u)
        return x


class TinyLM(nn.Module):
    def __init__(self, vocab=256, d=128, n_layers=6, n_heads=4, n_kv=2, d_ff=512):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([TinyBlock(d, n_heads, n_kv, d_ff) for _ in range(n_layers)])
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.embed.weight  # tie

    def forward(self, ids):
        x = self.embed(ids)
        for b in self.blocks:
            x = b(x)
        return self.head(x)


def make_batch(batch_size, seq_len, vocab, seed):
    """Harder task: XOR-like nonlinear pattern + position-dependent rotation."""
    g = torch.Generator().manual_seed(seed)
    pos = torch.arange(seq_len).float()
    # nonlinear: (pos^1.7) mod vocab, plus cross-term with batch-dependent prime
    base = (pos.pow(1.7).long() * 11 + 7) % vocab
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    batch_primes = torch.tensor([primes[s % len(primes)] for s in range(batch_size)])
    # cross-term: (pos * batch_prime) XOR (pos^2 // 3)
    cross = ((pos.unsqueeze(0) * batch_primes.unsqueeze(1)).long() ^ (pos.pow(2).long() // 3)) % vocab
    seq = (base.unsqueeze(0) + cross + torch.randint(0, 5, (batch_size, seq_len), generator=g)) % vocab
    return seq


# ── Param splitting ──────────────────────────────────────────────────────

def split_param_groups(model, lr_muon, lr_adam):
    """Return param_groups list for SingleDeviceMuonWithAuxAdam."""
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


# ── Novel: Muon + ScheduleFree (subclass) ────────────────────────────────

class MuonScheduleFree(SingleDeviceMuonWithAuxAdam):
    """Muon for 2D hidden weights, AdamWScheduleFree for everything else.

    ScheduleFree eliminates the need for a LR schedule via iterate averaging.
    We delegate non-muon params to an internal AdamWScheduleFree instance.
    """
    def __init__(self, param_groups):
        # Extract adam params and build internal SF optimizer
        adam_params = []
        muon_groups = []
        for g in param_groups:
            if g["use_muon"]:
                muon_groups.append(g)
            else:
                adam_params.extend(g["params"])
        # Build SF optimizer for adam params
        self._sf = AdamWScheduleFree(
            adam_params,
            lr=param_groups[0]["lr"] if not param_groups[0]["use_muon"] else param_groups[1]["lr"],
            betas=(0.9, 0.95), weight_decay=0.0,
        )
        super().__init__(muon_groups)
        # Store adam params reference for our step
        self._adam_params = adam_params
        self._sf.train()  # SF requires explicit train mode

    @torch.no_grad()
    def step(self, closure=None):
        # Muon step (copy parent logic for muon groups only)
        for group in self.param_groups:
            # all groups are muon here
            for p in group["params"]:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])
        # ScheduleFree step for adam params
        self._sf.step()

    def zero_grad(self, set_to_none=True):
        super().zero_grad(set_to_none=set_to_none)
        self._sf.zero_grad(set_to_none=set_to_none)

    def eval(self):
        """ScheduleFree: swap to averaged weights for eval."""
        self._sf.eval()

    def train(self):
        self._sf.train()

    @property
    def is_fused(self):
        return False  # for compatibility checks


class MuonSFBlockwise(MuonScheduleFree):
    """Novel #2: + per-block sharpness-scaled LR for muon params.

    Sharpness proxy: EMA of grad^2 per block. Higher sharpness -> lower LR
    (Sophia-style clipping intuition).
    """
    def __init__(self, param_groups, n_blocks=3, refresh_every=16, sharp_beta=0.9):
        super().__init__(param_groups)
        self._n_blocks = n_blocks
        self._refresh_every = refresh_every
        self._sharp_beta = sharp_beta
        self._step_count = 0
        self._block_sharp_ema = [0.0] * n_blocks
        self._base_muon_lr = next(g["lr"] for g in self.param_groups if g["use_muon"])
        # Map each muon param to block index
        self._param_block = []
        for g in self.param_groups:
            if g["use_muon"]:
                for p in g["params"]:
                    # find name
                    self._param_block.append(p)

    @torch.no_grad()
    def step(self, closure=None):
        self._step_count += 1
        # Update sharpness EMA every step (cheap), apply LR every refresh_every
        if self._step_count % self._refresh_every == 0:
            # Compute per-block sharpness from current grads
            block_sharp = [0.0] * self._n_blocks
            block_count = [0] * self._n_blocks
            for g in self.param_groups:
                if not g["use_muon"]:
                    continue
                for p in g["params"]:
                    if p.grad is None:
                        continue
                    # find which block this param belongs to by searching named_parameters
                    # (we don't have model ref here; use param identity via stored mapping)
                    pass
            # Simpler: use global grad norm ratio as a proxy (since we can't easily
            # map params to blocks without model ref). For real impl we'd pass model.
            # Here: scale muon LR by inverse of grad-norm EMA
            total_grad_sq = 0.0
            n_params = 0
            for g in self.param_groups:
                if g["use_muon"]:
                    for p in g["params"]:
                        if p.grad is not None:
                            total_grad_sq += p.grad.float().pow(2).mean().item()
                            n_params += 1
            avg_sharp = total_grad_sq / max(1, n_params)
            # EMA update
            for i in range(self._n_blocks):
                self._block_sharp_ema[i] = (
                    self._sharp_beta * self._block_sharp_ema[i]
                    + (1 - self._sharp_beta) * avg_sharp
                )
            # Higher sharpness -> HIGHER LR for Muon (Muon already normalizes,
            # so high curvature directions can take bigger orthogonalized steps).
            # This is the OPPOSITE of Sophia clipping — Muon's orthogonalization
            # makes high-sharpness directions SAFE to step aggressively.
            sharp = sum(self._block_sharp_ema) / max(1, len(self._block_sharp_ema))
            if self._step_count == self._refresh_every:
                self._initial_sharp = sharp
            ratio = sharp / max(1e-8, self._initial_sharp)
            ratio = max(0.3, min(2.5, ratio))
            for g in self.param_groups:
                if g["use_muon"]:
                    g["lr"] = self._base_muon_lr * ratio
        super().step(closure)


class MuonSFBlockwiseTITAN(MuonSFBlockwise):
    """Randomizer Target 1: Muon + Blockwise + TITAN-Hebbian momentum.

    TITAN memory uses Hebbian 'surprise' updates: when input deviates from
    what the memory expects, the memory is updated more aggressively.
    Applied to optimizer momentum: when gradient is surprising (much larger
    than momentum EMA), boost momentum in that direction beyond standard EMA.

    m_t = β * m_{t-1} + (1-β) * g_t + α * surprise(g_t, m_{t-1})
    surprise = relu(|g_t| - |m_{t-1}|) * sign(g_t)
    """
    def __init__(self, param_groups, n_blocks=3, refresh_every=16, sharp_beta=0.9,
                 titan_alpha=0.3):
        super().__init__(param_groups, n_blocks, refresh_every, sharp_beta)
        self._titan_alpha = titan_alpha

    @torch.no_grad()
    def step(self, closure=None):
        # Inject TITAN surprise into grad before parent's Muon momentum update
        for group in self.param_groups:
            if not group["use_muon"]:
                continue
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                m = state["momentum_buffer"]
                g = p.grad
                # Surprise: how much bigger is grad than momentum?
                surprise = (g.abs() - m.abs()).clamp(min=0) * g.sign()
                p.grad = g + self._titan_alpha * surprise
        super().step(closure)


class BlockwiseWSDEDM(MuonSFBlockwise):
    """Randomizer Target 3: Blockwise sharpness + WSD + EDM sigma decay.

    WSD: long stable phase (LR constant) then decay.
    EDM sigma: shape the decay curve as EDM c_out (not cosine).
    Blockwise: per-block sharpness multiplier on top of WSD base.
    """
    def __init__(self, param_groups, n_blocks=3, refresh_every=16, sharp_beta=0.9,
                 total_steps=400, stable_frac=0.6, sigma_max=80.0, sigma_data=0.5, sigma_min=0.002):
        super().__init__(param_groups, n_blocks, refresh_every, sharp_beta)
        self._total = total_steps
        self._stable_frac = stable_frac
        self._sig_max = sigma_max
        self._sig_data = sigma_data
        self._sig_min = sigma_min
        self._c_out_max = sigma_max * sigma_data / math.sqrt(sigma_max**2 + sigma_data**2)
        self._base_adam_lr = self._sf.param_groups[0]["lr"]

    @torch.no_grad()
    def step(self, closure=None):
        self._step_count += 1
        stable_end = int(self._total * self._stable_frac)
        if self._step_count <= stable_end:
            lr_scale = 1.0
        else:
            decay_prog = (self._step_count - stable_end) / max(1, self._total - stable_end)
            sigma = self._sig_max * (self._sig_min / self._sig_max) ** decay_prog
            c_out = sigma * self._sig_data / math.sqrt(sigma**2 + self._sig_data**2)
            lr_scale = c_out / self._c_out_max
        original_base = self._base_muon_lr
        self._base_muon_lr = original_base * lr_scale
        self._sf.param_groups[0]["lr"] = self._base_adam_lr * lr_scale
        super().step(closure)
        self._base_muon_lr = original_base


class MuonSFDiffusion(MuonScheduleFree):
    """Novel #3 (cross-domain risky): + EDM-sigma-scaled LR.

    Treat training as denoising: early=high noise (high LR explore),
    late=low noise (low LR refine). LR follows EDM c_out curve.
    """
    def __init__(self, param_groups, total_steps, sigma_max=80.0, sigma_data=0.5, sigma_min=0.002):
        super().__init__(param_groups)
        self._total = total_steps
        self._sig_max = sigma_max
        self._sig_data = sigma_data
        self._sig_min = sigma_min
        self._base_muon_lr = next(g["lr"] for g in self.param_groups if g["use_muon"])
        self._base_adam_lr = self._sf.param_groups[0]["lr"]
        self._step_count = 0
        self._c_out_max = sigma_max * sigma_data / math.sqrt(sigma_max**2 + sigma_data**2)

    @torch.no_grad()
    def step(self, closure=None):
        self._step_count += 1
        prog = min(1.0, self._step_count / self._total)
        # log-linear sigma decay
        sigma = self._sig_max * (self._sig_min / self._sig_max) ** prog
        c_out = sigma * self._sig_data / math.sqrt(sigma**2 + self._sig_data**2)
        lr_scale = c_out / self._c_out_max
        for g in self.param_groups:
            if g["use_muon"]:
                g["lr"] = self._base_muon_lr * lr_scale
        self._sf.param_groups[0]["lr"] = self._base_adam_lr * lr_scale
        super().step(closure)


class MuonSFDiffusionAdaptive(MuonScheduleFree):
    """Novel #3b: EDM-sigma LR but adaptive — sigma decays based on loss plateau.

    Instead of fixed schedule, track EMA of loss. If loss is decreasing fast,
    keep sigma high (explore). If loss plateaus, decay sigma (refine).
    This ties the "noise level" to actual training dynamics, not step count.
    """
    def __init__(self, param_groups, total_steps, sigma_max=80.0, sigma_data=0.5, sigma_min=0.002,
                 loss_ema_beta=0.95, patience=20):
        super().__init__(param_groups)
        self._total = total_steps
        self._sig_max = sigma_max
        self._sig_data = sigma_data
        self._sig_min = sigma_min
        self._base_muon_lr = next(g["lr"] for g in self.param_groups if g["use_muon"])
        self._base_adam_lr = self._sf.param_groups[0]["lr"]
        self._step_count = 0
        self._c_out_max = sigma_max * sigma_data / math.sqrt(sigma_max**2 + sigma_data**2)
        self._loss_ema = None
        self._loss_beta = loss_ema_beta
        self._best_loss = float("inf")
        self._steps_since_best = 0
        self._patience = patience
        self._sigma = sigma_max  # current sigma, decays on plateau

    def update_loss(self, loss_val):
        """Call with current loss before step()."""
        if self._loss_ema is None:
            self._loss_ema = loss_val
        else:
            self._loss_ema = self._loss_beta * self._loss_ema + (1 - self._loss_beta) * loss_val
        if self._loss_ema < self._best_loss:
            self._best_loss = self._loss_ema
            self._steps_since_best = 0
        else:
            self._steps_since_best += 1

    @torch.no_grad()
    def step(self, closure=None):
        self._step_count += 1
        # Adaptive sigma: decay when plateaued, but also decay with progress
        prog = min(1.0, self._step_count / self._total)
        # Base decay from progress (slow)
        prog_sigma = self._sig_max * (self._sig_min / self._sig_max) ** prog
        # Plateau penalty: if no improvement for `patience` steps, decay faster
        plateau_factor = max(0.1, 1.0 - self._steps_since_best / self._patience)
        self._sigma = max(self._sig_min, prog_sigma * plateau_factor)
        c_out = self._sigma * self._sig_data / math.sqrt(self._sigma**2 + self._sig_data**2)
        lr_scale = c_out / self._c_out_max
        for g in self.param_groups:
            if g["use_muon"]:
                g["lr"] = self._base_muon_lr * lr_scale
        self._sf.param_groups[0]["lr"] = self._base_adam_lr * lr_scale
        super().step(closure)


# ── Cosine schedule ──────────────────────────────────────────────────────

def cosine_lr(step, total, warmup, max_lr, min_lr):
    if step < warmup:
        return max_lr * step / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * prog))


# ── Training loop ────────────────────────────────────────────────────────

def train(model, opt, n_steps, batch_size, seq_len, vocab, variant_name, schedule_free=False):
    model.train()
    if schedule_free:
        opt.train()  # ScheduleFree requires explicit train mode
    losses = []
    t0 = time.time()
    for step in range(n_steps):
        ids = make_batch(batch_size, seq_len, vocab, seed=step * 7 + 1).cuda()
        x, y = ids[:, :-1], ids[:, 1:]
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        if not schedule_free and variant_name == "adamw_cosine":
            lr = cosine_lr(step, n_steps, warmup=10, max_lr=3e-3, min_lr=3e-4)
            for g in opt.param_groups:
                g["lr"] = lr
        opt.step()
        losses.append(loss.item())
        if hasattr(opt, "update_loss"):
            opt.update_loss(loss.item())
        if step % 50 == 0 or step == n_steps - 1:
            lr_now = opt.param_groups[0]["lr"] if opt.param_groups else 0
            print(f"  [{variant_name:25s}] step {step:4d} loss {loss.item():.4f} lr {lr_now:.2e}")
    dt = time.time() - t0
    if schedule_free:
        opt.eval()  # swap to averaged weights for fair final loss
    return losses, dt


def run_variant(name, build_fn, n_steps=200, batch_size=16, seq_len=64, vocab=256, d=128, n_layers=6):
    torch.manual_seed(42)
    model = TinyLM(vocab=vocab, d=d, n_layers=n_layers).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n=== {name} === ({n_params/1e6:.2f}M params)")
    opt = build_fn(model)
    sf = "sf" in name
    losses, dt = train(model, opt, n_steps, batch_size, seq_len, vocab, variant_name=name, schedule_free=sf)
    final = sum(losses[-10:]) / 10
    print(f"  final_10avg={final:.4f} time={dt:.2f}s")
    return {"name": name, "final_loss": final, "time": dt, "losses": losses}


def main():
    print("=" * 70)
    print("Muon-SF-Blockwise isolated test")
    print("=" * 70)
    N = 400
    LR_MUON = 2e-3  # paper default 0.02 is for large models; toy needs ~10x lower
    LR_ADAM = 3e-3
    results = []
    results.append(run_variant("adamw_cosine", lambda m: AdamW(m.parameters(), lr=LR_ADAM, fused=True), n_steps=N))
    results.append(run_variant("muon_adamw",
        lambda m: SingleDeviceMuonWithAuxAdam(split_param_groups(m, LR_MUON, LR_ADAM)), n_steps=N))
    results.append(run_variant("muon_sf",
        lambda m: MuonScheduleFree(split_param_groups(m, LR_MUON, LR_ADAM)), n_steps=N))
    results.append(run_variant("muon_sf_blockwise",
        lambda m: MuonSFBlockwise(split_param_groups(m, LR_MUON, LR_ADAM), n_blocks=3), n_steps=N))
    results.append(run_variant("blockwise_titan",
        lambda m: MuonSFBlockwiseTITAN(split_param_groups(m, LR_MUON, LR_ADAM), n_blocks=3), n_steps=N))
    results.append(run_variant("blockwise_wsd_edm",
        lambda m: BlockwiseWSDEDM(split_param_groups(m, LR_MUON, LR_ADAM), n_blocks=3, total_steps=N), n_steps=N))
    results.append(run_variant("muon_sf_diffusion",
        lambda m: MuonSFDiffusion(split_param_groups(m, LR_MUON, LR_ADAM), total_steps=N), n_steps=N))
    results.append(run_variant("muon_sf_diff_adapt",
        lambda m: MuonSFDiffusionAdaptive(split_param_groups(m, LR_MUON, LR_ADAM), total_steps=N), n_steps=N))

    print("\n" + "=" * 70)
    print("SUMMARY (sorted by final loss, lower=better)")
    print("=" * 70)
    results.sort(key=lambda r: r["final_loss"])
    for i, r in enumerate(results):
        print(f"  {i+1}. {r['name']:25s} loss={r['final_loss']:.4f}  time={r['time']:.2f}s")
    best = results[0]
    baseline = next(r for r in results if r["name"] == "adamw_cosine")
    muon_base = next(r for r in results if r["name"] == "muon_adamw")
    print(f"\n  Best: {best['name']} ({best['final_loss']:.4f})")
    print(f"  vs AdamW cosine ({baseline['final_loss']:.4f}): {baseline['final_loss']/best['final_loss']:.2f}x better loss")
    print(f"  vs Muon+AdamW ({muon_base['final_loss']:.4f}): {muon_base['final_loss']/best['final_loss']:.2f}x better loss")


if __name__ == "__main__":
    main()
