"""Isolated test: STACK the two winners — grad mixup + muon_sf_blockwise.

Round 1-3 winner: muon_sf_blockwise (1.05x vs AdamW)
Round 2 winner: grad mixup (1.16x vs AdamW)
Question: do they stack? If yes, expect ~1.2x or better.

Also test: grad mixup with MORE batches (3-way mixup) — does averaging
more gradients help further?
"""
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from muon import SingleDeviceMuonWithAuxAdam, muon_update
from schedulefree import AdamWScheduleFree


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


# ── MuonScheduleFree + Blockwise (from test_muon_sf.py, the round 1-3 winner) ──

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
    """Round 1-3 winner: Muon + SF + blockwise sharpness DIRECT scaling."""
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


# ── Training loop with grad mixup (2-way and 3-way) ──────────────────────

def compute_loss(model, ids, vocab):
    x, y = ids[:, :-1], ids[:, 1:]
    logits = model(x)
    return F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))


def train(model, opt, n_steps, batch_size, seq_len, vocab, variant_name,
          schedule_free=False, mixup_way=0):
    """mixup_way: 0=none, 2=two-batch mixup, 3=three-batch mixup"""
    model.train()
    if schedule_free:
        opt.train()
    losses = []
    t0 = time.time()
    for step in range(n_steps):
        seed = step * 7 + 1
        ids = make_batch(batch_size, seq_len, vocab, seed=seed).cuda()
        loss = compute_loss(model, ids, vocab)

        if mixup_way > 0:
            # First batch grads
            opt.zero_grad()
            loss.backward()
            saved_grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

            if mixup_way == 2:
                # 2-way: average with one other batch
                seed2 = (step + n_steps // 2) * 7 + 1
                ids2 = make_batch(batch_size, seq_len, vocab, seed=seed2).cuda()
                loss2 = compute_loss(model, ids2, vocab)
                opt.zero_grad()
                loss2.backward()
                alpha = 0.5
                for n, p in model.named_parameters():
                    if p.grad is not None and n in saved_grads:
                        p.grad = alpha * saved_grads[n] + (1 - alpha) * p.grad
            elif mixup_way == 3:
                # 3-way: average with two other batches
                seed2 = (step + n_steps // 2) * 7 + 1
                seed3 = (step + n_steps // 3) * 7 + 1
                ids2 = make_batch(batch_size, seq_len, vocab, seed=seed2).cuda()
                ids3 = make_batch(batch_size, seq_len, vocab, seed=seed3).cuda()
                loss2 = compute_loss(model, ids2, vocab)
                opt.zero_grad()
                loss2.backward()
                g2 = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
                loss3 = compute_loss(model, ids3, vocab)
                opt.zero_grad()
                loss3.backward()
                g3 = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
                # 3-way average
                for n, p in model.named_parameters():
                    if p.grad is not None and n in saved_grads and n in g2 and n in g3:
                        p.grad = (saved_grads[n] + g2[n] + g3[n]) / 3.0
        else:
            opt.zero_grad()
            loss.backward()

        if not schedule_free and variant_name == "adamw_cosine":
            lr = cosine_lr(step, n_steps, warmup=10, max_lr=3e-3, min_lr=3e-4)
            for g in opt.param_groups:
                g["lr"] = lr

        opt.step()
        losses.append(loss.item())
        if step % 100 == 0 or step == n_steps - 1:
            lr_now = opt.param_groups[0]["lr"] if opt.param_groups else 0
            print(f"  [{variant_name:30s}] step {step:4d} loss {loss.item():.4f} lr {lr_now:.2e}")
    dt = time.time() - t0
    if schedule_free:
        opt.eval()
    return losses, dt


def run_variant(name, build_fn, n_steps, mixup_way=0, schedule_free=False):
    torch.manual_seed(42)
    model = TinyLM(vocab=256, d=128, n_layers=6).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n=== {name} === ({n_params/1e6:.2f}M params, mixup={mixup_way})")
    opt = build_fn(model)
    sf = schedule_free
    losses, dt = train(model, opt, n_steps, 16, 64, 256, variant_name=name,
                       schedule_free=sf, mixup_way=mixup_way)
    final = sum(losses[-20:]) / 20
    print(f"  final_20avg={final:.4f} time={dt:.2f}s")
    return {"name": name, "final_loss": final, "time": dt, "losses": losses}


def main():
    print("=" * 70)
    print("STACK TEST: grad mixup + muon_sf_blockwise")
    print("=" * 70)
    N = 400
    LR_MUON = 2e-3
    LR_ADAM = 3e-3

    results = []
    # Baselines (no mixup)
    results.append(run_variant("adamw_cosine",
        lambda m: AdamW(m.parameters(), lr=LR_ADAM, fused=True), n_steps=N))
    results.append(run_variant("adamw_mixup2",
        lambda m: AdamW(m.parameters(), lr=LR_ADAM, fused=True), n_steps=N, mixup_way=2))
    results.append(run_variant("adamw_mixup3",
        lambda m: AdamW(m.parameters(), lr=LR_ADAM, fused=True), n_steps=N, mixup_way=3))

    # Muon baselines
    results.append(run_variant("muon_adamw",
        lambda m: SingleDeviceMuonWithAuxAdam(split_param_groups(m, LR_MUON, LR_ADAM)), n_steps=N))
    results.append(run_variant("muon_mixup2",
        lambda m: SingleDeviceMuonWithAuxAdam(split_param_groups(m, LR_MUON, LR_ADAM)), n_steps=N, mixup_way=2))

    # The winners stacked
    results.append(run_variant("muon_sf_blockwise",
        lambda m: MuonSFBlockwise(split_param_groups(m, LR_MUON, LR_ADAM), n_blocks=3),
        n_steps=N, schedule_free=True))
    results.append(run_variant("muon_sf_bw_mixup2",
        lambda m: MuonSFBlockwise(split_param_groups(m, LR_MUON, LR_ADAM), n_blocks=3),
        n_steps=N, schedule_free=True, mixup_way=2))
    results.append(run_variant("muon_sf_bw_mixup3",
        lambda m: MuonSFBlockwise(split_param_groups(m, LR_MUON, LR_ADAM), n_blocks=3),
        n_steps=N, schedule_free=True, mixup_way=3))

    print("\n" + "=" * 70)
    print("SUMMARY (sorted by final loss, lower=better)")
    print("=" * 70)
    results.sort(key=lambda r: r["final_loss"])
    for i, r in enumerate(results):
        print(f"  {i+1}. {r['name']:30s} loss={r['final_loss']:.4f}  time={r['time']:.2f}s")
    best = results[0]
    baseline = next(r for r in results if r["name"] == "adamw_cosine")
    print(f"\n  Best: {best['name']} ({best['final_loss']:.4f})")
    print(f"  vs AdamW cosine ({baseline['final_loss']:.4f}): {baseline['final_loss']/best['final_loss']:.2f}x better loss")


if __name__ == "__main__":
    main()
