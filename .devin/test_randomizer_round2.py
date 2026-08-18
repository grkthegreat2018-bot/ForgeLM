"""Isolated test: untested randomizer combos (AGENTS.md #7).

Sister agent is testing muon_sf_blockwise on real model — DO NOT touch that.
This script tests OTHER randomizer-suggested combos on toy model:

  R1. Muon + ELO curriculum (difficulty-ordered batches)
      Randomizer: "Blockwise sharpness LR + ELO curriculum"
      Idea: order batches by difficulty, sharp blocks get hard examples first

  R2. Grad mixup (interpolate gradients from 2 batches)
      Randomizer: "Muon + Mixup (interpolate two batches' gradients)"
      Idea: smoother gradient landscape, less variance

  R3. Quantization noise injection on gradients
      Randomizer: "Muon + Quantization noise injection"
      Idea: inject QAT-style noise into grads for robustness/regularization

  R4. Sophia-lite: diagonal Hessian clipping (no full Sophia, just the clip)
      Randomizer: "Sophia + Blockwise sharpness LR"
      Idea: clip grad by EMA of grad² (cheap Sophia proxy)

  R5. Label smoothing (loosely related, from randomizer combo 6)
      Idea: soft targets reduce overconfidence, may help Muon converge

Baselines: adamw_cosine, muon_adamw (no SF, no blockwise — clean Muon baseline)
"""
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from muon import SingleDeviceMuonWithAuxAdam, muon_update


# ── Toy model (same as test_muon_sf.py) ──────────────────────────────────

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
        self.head.weight = self.embed.weight

    def forward(self, ids):
        x = self.embed(ids)
        for b in self.blocks:
            x = b(x)
        return self.head(x)


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


# ── ELO curriculum: precompute batch difficulty, order by it ─────────────

def precompute_difficulty(model, n_steps, batch_size, seq_len, vocab):
    """Score each batch seed by initial loss (proxy for difficulty)."""
    difficulties = []
    model.eval()
    with torch.no_grad():
        for step in range(n_steps):
            seed = step * 7 + 1
            ids = make_batch(batch_size, seq_len, vocab, seed=seed).cuda()
            x, y = ids[:, :-1], ids[:, 1:]
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
            difficulties.append((seed, loss.item()))
    model.train()
    return difficulties


def curriculum_order(difficulties, n_steps, mode="easy_to_hard"):
    """Order batch seeds by difficulty."""
    sorted_diff = sorted(difficulties, key=lambda x: x[1])
    if mode == "easy_to_hard":
        return [d[0] for d in sorted_diff[:n_steps]]
    elif mode == "goldilocks":
        # Start near median, expand outward (50% success zone)
        mid = len(sorted_diff) // 2
        order = []
        left, right = mid, mid + 1
        while left >= 0 or right < len(sorted_diff):
            if right < len(sorted_diff):
                order.append(sorted_diff[right][0])
                right += 1
            if left >= 0:
                order.append(sorted_diff[left][0])
                left -= 1
        return order[:n_steps]
    elif mode == "hard_to_easy":
        return [d[0] for d in reversed(sorted_diff[:n_steps])]
    else:
        return [d[0] for d in sorted_diff[:n_steps]]


# ── Sophia-lite: diagonal Hessian EMA clipping ───────────────────────────

class SophiaLiteAdamW(AdamW):
    """Cheap Sophia proxy: clip grad by EMA of grad² (diagonal Hessian approx).

    Sophia uses diagonal Hessian estimate h ≈ EMA(grad²) and clips:
        update = grad / max(h, eps)  then  clip(update, -τ, τ)
    We approximate h with EMA(grad²) and apply element-wise clip.
    """
    def __init__(self, params, lr=3e-3, betas=(0.9, 0.95), weight_decay=0.0,
                 hessian_beta=0.95, clip_tau=1.0, refresh_every=10):
        super().__init__(params, lr=lr, betas=betas, weight_decay=weight_decay)
        self._h_beta = hessian_beta
        self._tau = clip_tau
        self._refresh = refresh_every
        self._step = 0

    @torch.no_grad()
    def step(self, closure=None):
        self._step += 1
        if self._step % self._refresh == 0:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if "hessian_ema" not in state:
                        state["hessian_ema"] = torch.zeros_like(p)
                    h = state["hessian_ema"]
                    h.mul_(self._h_beta).add_(p.grad.detach().pow(2), alpha=1 - self._h_beta)
        # Clip grad by hessian before parent step
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "hessian_ema" in state:
                    h = state["hessian_ema"]
                    # Sophia: update = clip(g / max(h, eps), -tau, tau)
                    denom = h.clamp(min=1e-8)
                    clipped = (p.grad / denom).clamp(-self._tau, self._tau)
                    # Scale back by h to preserve magnitude
                    p.grad = clipped * denom.sqrt()
        super().step(closure)


# ── Training loop with optional curriculum/mixup/noise/smoothing ─────────

def train(model, opt, n_steps, batch_size, seq_len, vocab, variant_name,
          batch_order=None, grad_mixup=False, qat_noise=0.0, label_smoothing=0.0):
    model.train()
    losses = []
    t0 = time.time()
    for step in range(n_steps):
        seed = batch_order[step] if batch_order else step * 7 + 1
        ids = make_batch(batch_size, seq_len, vocab, seed=seed).cuda()
        x, y = ids[:, :-1], ids[:, 1:]
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1),
                               label_smoothing=label_smoothing)

        if grad_mixup:
            opt.zero_grad()
            loss.backward()
            saved_grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
            seed2 = (step + n_steps // 2) * 7 + 1
            ids2 = make_batch(batch_size, seq_len, vocab, seed=seed2).cuda()
            x2, y2 = ids2[:, :-1], ids2[:, 1:]
            logits2 = model(x2)
            loss2 = F.cross_entropy(logits2.reshape(-1, vocab), y2.reshape(-1),
                                    label_smoothing=label_smoothing)
            opt.zero_grad()
            loss2.backward()
            alpha = 0.5
            for n, p in model.named_parameters():
                if p.grad is not None and n in saved_grads:
                    p.grad = alpha * saved_grads[n] + (1 - alpha) * p.grad
        else:
            opt.zero_grad()
            loss.backward()

        if qat_noise > 0:
            for p in model.parameters():
                if p.grad is not None:
                    noise = torch.randn_like(p.grad) * qat_noise * p.grad.std()
                    p.grad.add_(noise)

        if variant_name == "adamw_cosine":
            lr = cosine_lr(step, n_steps, warmup=10, max_lr=3e-3, min_lr=3e-4)
            for g in opt.param_groups:
                g["lr"] = lr

        opt.step()
        losses.append(loss.item())
        if step % 100 == 0 or step == n_steps - 1:
            lr_now = opt.param_groups[0]["lr"] if opt.param_groups else 0
            print(f"  [{variant_name:25s}] step {step:4d} loss {loss.item():.4f} lr {lr_now:.2e}")
    dt = time.time() - t0
    return losses, dt


def run_variant(name, build_fn, n_steps, batch_size=16, seq_len=64, vocab=256,
                d=128, n_layers=6, batch_order=None, grad_mixup=False,
                qat_noise=0.0, label_smoothing=0.0):
    torch.manual_seed(42)
    model = TinyLM(vocab=vocab, d=d, n_layers=n_layers).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n=== {name} === ({n_params/1e6:.2f}M params)")
    opt = build_fn(model)
    losses, dt = train(model, opt, n_steps, batch_size, seq_len, vocab, variant_name=name,
                       batch_order=batch_order, grad_mixup=grad_mixup,
                       qat_noise=qat_noise, label_smoothing=label_smoothing)
    final = sum(losses[-20:]) / 20
    print(f"  final_20avg={final:.4f} time={dt:.2f}s")
    return {"name": name, "final_loss": final, "time": dt, "losses": losses}


def main():
    print("=" * 70)
    print("Randomizer combos round 2 (untested ideas)")
    print("=" * 70)
    N = 400
    LR_MUON = 2e-3
    LR_ADAM = 3e-3
    BS, SL, VOCAB = 16, 64, 256

    # Precompute difficulty for curriculum variants (using a fresh model)
    print("\nPrecomputing batch difficulty for curriculum...")
    torch.manual_seed(42)
    probe_model = TinyLM(vocab=VOCAB, d=128, n_layers=6).cuda()
    diffs = precompute_difficulty(probe_model, N, BS, SL, VOCAB)
    print(f"  Difficulty range: {min(d[1] for d in diffs):.2f} - {max(d[1] for d in diffs):.2f}")
    del probe_model
    torch.cuda.empty_cache()

    easy_to_hard = curriculum_order(diffs, N, mode="easy_to_hard")
    goldilocks = curriculum_order(diffs, N, mode="goldilocks")

    results = []
    # Baselines
    results.append(run_variant("adamw_cosine",
        lambda m: AdamW(m.parameters(), lr=LR_ADAM, fused=True), n_steps=N))
    results.append(run_variant("muon_adamw",
        lambda m: SingleDeviceMuonWithAuxAdam(split_param_groups(m, LR_MUON, LR_ADAM)), n_steps=N))

    # R1: ELO curriculum (easy→hard) with Muon
    results.append(run_variant("muon_elo_easy",
        lambda m: SingleDeviceMuonWithAuxAdam(split_param_groups(m, LR_MUON, LR_ADAM)),
        n_steps=N, batch_order=easy_to_hard))

    # R1b: ELO curriculum (goldilocks) with Muon
    results.append(run_variant("muon_elo_goldilocks",
        lambda m: SingleDeviceMuonWithAuxAdam(split_param_groups(m, LR_MUON, LR_ADAM)),
        n_steps=N, batch_order=goldilocks))

    # R1c: ELO curriculum (easy→hard) with AdamW — isolate curriculum effect
    results.append(run_variant("adamw_elo_easy",
        lambda m: AdamW(m.parameters(), lr=LR_ADAM, fused=True),
        n_steps=N, batch_order=easy_to_hard))

    # R2: Grad mixup with Muon
    results.append(run_variant("muon_grad_mixup",
        lambda m: SingleDeviceMuonWithAuxAdam(split_param_groups(m, LR_MUON, LR_ADAM)),
        n_steps=N, grad_mixup=True))

    # R2b: Grad mixup with AdamW — isolate mixup effect
    results.append(run_variant("adamw_grad_mixup",
        lambda m: AdamW(m.parameters(), lr=LR_ADAM, fused=True),
        n_steps=N, grad_mixup=True))

    # R3: QAT noise injection with Muon
    results.append(run_variant("muon_qat_noise",
        lambda m: SingleDeviceMuonWithAuxAdam(split_param_groups(m, LR_MUON, LR_ADAM)),
        n_steps=N, qat_noise=0.1))

    # R4: Sophia-lite with AdamW
    results.append(run_variant("sophia_lite_adamw",
        lambda m: SophiaLiteAdamW(m.parameters(), lr=LR_ADAM), n_steps=N))

    # R5: Label smoothing with Muon
    results.append(run_variant("muon_label_smooth",
        lambda m: SingleDeviceMuonWithAuxAdam(split_param_groups(m, LR_MUON, LR_ADAM)),
        n_steps=N, label_smoothing=0.1))

    # R6: Combo — Muon + ELO goldilocks + label smoothing (stack the winners)
    results.append(run_variant("muon_elo_smooth_combo",
        lambda m: SingleDeviceMuonWithAuxAdam(split_param_groups(m, LR_MUON, LR_ADAM)),
        n_steps=N, batch_order=goldilocks, label_smoothing=0.1))

    print("\n" + "=" * 70)
    print("SUMMARY (sorted by final loss, lower=better)")
    print("=" * 70)
    results.sort(key=lambda r: r["final_loss"])
    for i, r in enumerate(results):
        print(f"  {i+1:2d}. {r['name']:25s} loss={r['final_loss']:.4f}  time={r['time']:.2f}s")
    best = results[0]
    baseline = next(r for r in results if r["name"] == "adamw_cosine")
    muon_base = next(r for r in results if r["name"] == "muon_adamw")
    print(f"\n  Best: {best['name']} ({best['final_loss']:.4f})")
    print(f"  vs AdamW cosine ({baseline['final_loss']:.4f}): {baseline['final_loss']/best['final_loss']:.2f}x better loss")
    print(f"  vs Muon+AdamW ({muon_base['final_loss']:.4f}): {muon_base['final_loss']/best['final_loss']:.2f}x better loss")


if __name__ == "__main__":
    main()
