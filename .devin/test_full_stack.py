"""Prove the FULL STACK works: V3 architecture + DiffusionBlocks + round-robin
+ muon_sf + grad mixup.

This is the decisive test. If this passes, we wire everything into production.

Stack components:
1. V3 architecture: diff attention, BitNet, TITAN, MoD, MHC, AttnRes, QK-norm
2. DiffusionBlocks with round-robin scheduling (proven 8.7x lower variance)
3. Muon-SF optimizer (no blockwise — proven best for DB regime)
4. 3-way grad mixup (proven 1.25x convergence boost)

Tests:
A. V3 model, standard training, muon_sf_blockwise + mixup3 (our proven standard winner)
B. V3 model, DiffusionBlocks + round-robin + muon_sf + mixup3 (the DB stack)
C. V3 model, DiffusionBlocks + round-robin + AdamW + mixup3 (cheaper DB variant)
D. V3 model, standard training, AdamW cosine (baseline)

Key question: does the full stack (B) beat the standard winner (A)?
And: do the cross-layer keys (MHC, AttnRes) break DiffusionBlocks?
"""
import math
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

# ForgeAI imports
from research.config import get_config, ModelConfig
from research.model_loader import ConfigurableResearchLLM
from research.diffusion_blocks import DiffusionBlocks, DiffusionBlockConfig
from muon import SingleDeviceMuonWithAuxAdam, muon_update
from schedulefree import AdamWScheduleFree


# ── Tiny V3 config (all keys enabled, toy scale) ─────────────────────────

def make_tiny_v3_config():
    """Tiny V3 model with ALL architectural keys enabled."""
    return ModelConfig(
        vocab_size=256,
        d_model=128,
        n_layers=8,  # 8 layers so we can do 4 blocks of 2
        n_heads=4,
        n_kv_heads=2,
        intermediate_size=512,
        attn_type="diff",           # V3: differential attention
        attn_bias=False,
        ffn_type="swiglu",
        norm_type="rmsnorm",
        rope_base=1_000_000.0,
        max_seq_len=128,
        conv_kernel_size=3,
        use_qk_norm=True,           # V3: QK-norm
        use_bitnet=True,            # V3: BitNet b1.58
        bitnet_learned_scale=True,
        layer_types=["conv", "conv", "attention", "conv", "conv",
                     "attention", "conv", "attention"],
        use_titan_memory=True,      # V3: TITAN memory
        titan_memory_rank=32,
        use_mod=True,               # V3: MoD router
        mod_keep_fraction=1.0,      # lossless (all tokens)
        use_mhc=True,               # V3: MHC hyper-connections
        mhc_rank=32,
        use_attn_residual=True,     # V3: AttnRes cross-layer retrieval
        attn_res_k=4,
        tie_word_embeddings=True,
        batch_size=4,
        seq_len=64,
        max_steps=400,
        warmup_steps=10,
    )


def make_tiny_v3_model(config=None):
    """Build a tiny V3 model with all keys."""
    if config is None:
        config = make_tiny_v3_config()
    model = ConfigurableResearchLLM(config)
    model = model.cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  V3 tiny model: {n_params/1e6:.2f}M params, {config.n_layers} layers")
    return model, config


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


# ── Optimizer builders ───────────────────────────────────────────────────

def split_param_groups_v3(model, lr_muon, lr_adam):
    """Split V3 model params for Muon (2D hidden) vs AdamW (embed/scalars)."""
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


class MuonScheduleFree(SingleDeviceMuonWithAuxAdam):
    """Muon + ScheduleFree AdamW for non-muon params."""
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
    """Muon + SF + blockwise sharpness DIRECT scaling (standard regime winner)."""
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


def cosine_lr(step, total, warmup, max_lr, min_lr):
    if step < warmup:
        return max_lr * step / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * prog))


# ── Training loops ───────────────────────────────────────────────────────

def compute_loss_v3(model, input_ids, labels):
    """Forward through V3 model and compute CE loss."""
    logits = model(input_ids)
    if isinstance(logits, tuple):
        logits = logits[0]
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))


def train_standard_v3(model, opt, n_steps, batch_size, seq_len, vocab, variant_name,
                      grad_mixup=1, schedule_free=False):
    """Standard training (no DiffusionBlocks) with V3 model."""
    model.train()
    if schedule_free:
        opt.train()
    losses = []
    t0 = time.time()
    for step in range(n_steps):
        ids = make_batch(batch_size, seq_len, vocab, seed=step * 7 + 1).cuda()
        x, y = ids[:, :-1], ids[:, 1:]
        loss = compute_loss_v3(model, x, y)

        if grad_mixup > 1:
            opt.zero_grad()
            loss.backward()
            saved = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
            for mi in range(grad_mixup - 1):
                seed2 = (step + (mi + 1) * n_steps // 2) * 7 + 1
                ids2 = make_batch(batch_size, seq_len, vocab, seed=seed2).cuda()
                x2, y2 = ids2[:, :-1], ids2[:, 1:]
                loss2 = compute_loss_v3(model, x2, y2)
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

        if variant_name == "D_adamw_cosine":
            lr = cosine_lr(step, n_steps, 10, 3e-4, 3e-5)  # match sister agent LR
            for g in opt.param_groups:
                g["lr"] = lr
        opt.step()
        losses.append(loss.item())
        if step % 50 == 0 or step == n_steps - 1:
            print(f"  [{variant_name:35s}] step {step:4d} loss {loss.item():.4f}")
    dt = time.time() - t0
    if schedule_free:
        opt.eval()
    return losses, dt


def train_dblock_v3(model, dblock, opt, n_steps, batch_size, seq_len, vocab,
                    variant_name, grad_mixup=1, block_schedule="round_robin",
                    schedule_free=False):
    """DiffusionBlocks training with V3 model, round-robin, grad mixup.

    Does NOT use dblock.train_step() (which does optimizer.step() internally).
    Instead uses dblock.forward_block() directly so we can do grad mixup
    before stepping the optimizer.
    """
    model.train()
    if schedule_free:
        opt.train()
    losses = []
    t0 = time.time()
    for step in range(n_steps):
        ids = make_batch(batch_size, seq_len, vocab, seed=step * 7 + 1).cuda()
        x, y = ids[:, :-1], ids[:, 1:]

        # Block scheduling
        if block_schedule == "round_robin":
            block_idx = step % dblock.num_blocks
        elif block_schedule == "random":
            block_idx = random.randint(0, dblock.num_blocks - 1)
        else:
            block_idx = step % dblock.num_blocks

        # Prepare noisy embeddings (same logic as dblock.train_step but without opt.step)
        device = x.device
        bs = x.shape[0]
        with torch.no_grad():
            target_embeds = model.embed(y)
            input_embeds = model.embed(x)
            embed_scale = input_embeds.norm(dim=-1).mean().item()

        sigmas = dblock.get_block_sigma(block_idx, n_samples=bs).to(device)
        noise = torch.randn_like(target_embeds) * embed_scale
        sigma_expanded = sigmas[:, None, None]
        noisy_embeds = target_embeds + sigma_expanded * noise
        scaled_noisy = noisy_embeds * 0.1  # noise_scale=0.1

        # Noise dropout (CFG style)
        if random.random() < 0.1:
            scaled_noisy = None
            sigmas_for_cond = torch.full_like(sigmas, 0.001)
        else:
            sigmas_for_cond = sigmas

        def compute_db_loss(ids_batch, labels_batch, noisy_batch, sigmas_batch):
            logits = dblock.forward_block(
                input_ids=ids_batch, block_idx=block_idx,
                noisy_embeds=noisy_batch, sigma=sigmas_batch,
            )
            if isinstance(logits, tuple):
                logits = logits[0]
            return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels_batch.reshape(-1))

        loss = compute_db_loss(x, y, scaled_noisy, sigmas_for_cond)

        if grad_mixup > 1:
            # Grad mixup: average gradients from N batches
            opt.zero_grad()
            loss.backward()
            saved_grads = {n: p.grad.clone() for n, p in model.named_parameters()
                          if p.grad is not None and p.requires_grad}
            # Save DB component grads
            if dblock.timestep_embedder is not None:
                for n, p in dblock.timestep_embedder.named_parameters():
                    if p.grad is not None:
                        saved_grads[f"ts_{n}"] = p.grad.clone()
            if dblock.adalns is not None:
                for n, p in dblock.adalns[block_idx].named_parameters():
                    if p.grad is not None:
                        saved_grads[f"adaln_{n}"] = p.grad.clone()

            for mi in range(grad_mixup - 1):
                seed2 = (step + (mi + 1) * n_steps // 2) * 7 + 1
                ids2 = make_batch(batch_size, seq_len, vocab, seed=seed2).cuda()
                x2, y2 = ids2[:, :-1], ids2[:, 1:]
                # New noise for same data
                with torch.no_grad():
                    target2 = model.embed(y2)
                sigmas2 = dblock.get_block_sigma(block_idx, n_samples=bs).to(device)
                noise2 = torch.randn_like(target2) * embed_scale
                noisy2 = target2 + sigmas2[:, None, None] * noise2
                scaled2 = noisy2 * 0.1
                if random.random() < 0.1:
                    scaled2 = None
                    sig2_cond = torch.full_like(sigmas2, 0.001)
                else:
                    sig2_cond = sigmas2
                loss2 = compute_db_loss(x2, y2, scaled2, sig2_cond)
                opt.zero_grad()
                loss2.backward()
                # Average grads
                for n, p in model.named_parameters():
                    if p.grad is not None and n in saved_grads:
                        saved_grads[n] = (saved_grads[n] * (mi + 1) + p.grad) / (mi + 2)
                if dblock.timestep_embedder is not None:
                    for n, p in dblock.timestep_embedder.named_parameters():
                        if p.grad is not None and f"ts_{n}" in saved_grads:
                            saved_grads[f"ts_{n}"] = (saved_grads[f"ts_{n}"] * (mi + 1) + p.grad) / (mi + 2)
                if dblock.adalns is not None:
                    for n, p in dblock.adalns[block_idx].named_parameters():
                        if p.grad is not None and f"adaln_{n}" in saved_grads:
                            saved_grads[f"adaln_{n}"] = (saved_grads[f"adaln_{n}"] * (mi + 1) + p.grad) / (mi + 2)

            # Restore averaged grads
            opt.zero_grad()
            for n, p in model.named_parameters():
                if n in saved_grads and p.requires_grad:
                    p.grad = saved_grads[n]
            if dblock.timestep_embedder is not None:
                for n, p in dblock.timestep_embedder.named_parameters():
                    if f"ts_{n}" in saved_grads:
                        p.grad = saved_grads[f"ts_{n}"]
            if dblock.adalns is not None:
                for n, p in dblock.adalns[block_idx].named_parameters():
                    if f"adaln_{n}" in saved_grads:
                        p.grad = saved_grads[f"adaln_{n}"]
        else:
            opt.zero_grad()
            loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        sigma_mean = sigmas.mean().item()
        losses.append(loss.item())
        if step % 50 == 0 or step == n_steps - 1:
            print(f"  [{variant_name:35s}] step {step:4d} loss {loss.item():.4f} "
                  f"block={block_idx} sigma={sigma_mean:.4f}")
    dt = time.time() - t0
    if schedule_free:
        opt.eval()
    return losses, dt


def run_variant(name, n_steps, mode="standard", optimizer_name="adamw",
                grad_mixup=1, num_blocks=4, block_schedule="round_robin"):
    """Run one variant with V3 model."""
    torch.manual_seed(42)
    random.seed(42)
    config = make_tiny_v3_config()
    model, config = make_tiny_v3_model(config)
    vocab = config.vocab_size
    bs, sl = 4, 64
    # Production LR scaling: muon_lr = 0.05 * (max_lr / 0.003)
    # With max_lr=3e-4: muon_lr = 0.005, adam_lr = 0.06 (embed), 0.004 (scalar)
    # Sister agent validated this on real V3 1.2B
    LR_MUON = 5e-3   # production scaling (was 5e-4 — 10x too low, caused false negative)
    LR_ADAM = 3e-4   # matches sister agent's max_lr

    print(f"\n=== {name} === (mode={mode}, opt={optimizer_name}, mixup={grad_mixup})")

    if mode == "diffusion":
        db_config = DiffusionBlockConfig(
            num_blocks=num_blocks,
            use_noise_conditioning=True,
            cond_dim=64,
        )
        dblock = DiffusionBlocks(model, db_config, config.d_model, config.n_layers)

        # Build optimizer for ALL params
        if optimizer_name == "adamw":
            all_params = list(model.parameters())
            if dblock.timestep_embedder is not None:
                all_params += list(dblock.timestep_embedder.parameters())
            if dblock.adalns is not None:
                for adaln in dblock.adalns:
                    all_params += list(adaln.parameters())
            opt = AdamW(all_params, lr=LR_ADAM, fused=True)
            sf = False
        elif optimizer_name == "muon_sf":
            pg = split_param_groups_v3(model, LR_MUON, LR_ADAM)
            aux = []
            if dblock.timestep_embedder is not None:
                aux += list(dblock.timestep_embedder.parameters())
            if dblock.adalns is not None:
                for adaln in dblock.adalns:
                    aux += list(adaln.parameters())
            pg[1]["params"].extend(aux)
            opt = MuonScheduleFree(pg)
            sf = True
        elif optimizer_name == "muon_sf_bw":
            pg = split_param_groups_v3(model, LR_MUON, LR_ADAM)
            aux = []
            if dblock.timestep_embedder is not None:
                aux += list(dblock.timestep_embedder.parameters())
            if dblock.adalns is not None:
                for adaln in dblock.adalns:
                    aux += list(adaln.parameters())
            pg[1]["params"].extend(aux)
            opt = MuonSFBlockwise(pg, n_blocks=num_blocks)
            sf = True

        losses, dt = train_dblock_v3(model, dblock, opt, n_steps, bs, sl, vocab,
                                     variant_name=name, grad_mixup=grad_mixup,
                                     block_schedule=block_schedule, schedule_free=sf)
    else:
        if optimizer_name == "adamw":
            opt = AdamW(model.parameters(), lr=LR_ADAM, fused=True)
            sf = False
        elif optimizer_name == "muon_sf":
            opt = MuonScheduleFree(split_param_groups_v3(model, LR_MUON, LR_ADAM))
            sf = True
        elif optimizer_name == "muon_sf_bw":
            opt = MuonSFBlockwise(split_param_groups_v3(model, LR_MUON, LR_ADAM), n_blocks=3)
            sf = True
        losses, dt = train_standard_v3(model, opt, n_steps, bs, sl, vocab,
                                       variant_name=name, grad_mixup=grad_mixup,
                                       schedule_free=sf)

    final = sum(losses[-20:]) / 20
    print(f"  final_20avg={final:.4f} time={dt:.2f}s")
    del model
    torch.cuda.empty_cache()
    return {"name": name, "final_loss": final, "time": dt, "losses": losses}


def main():
    print("=" * 70)
    print("FULL STACK PROOF: V3 arch + DiffusionBlocks + round-robin + muon_sf + mixup")
    print("=" * 70)
    N = 300  # fewer steps (V3 model is heavier)

    results = []

    # D: Baseline — V3 + AdamW cosine (no mixup, no DB)
    results.append(run_variant("D_adamw_cosine", N, mode="standard", optimizer_name="adamw"))

    # D2: AdamW + mixup3 (standard, no DB) — isolate mixup effect on V3
    results.append(run_variant("D2_adamw_mix3", N, mode="standard",
                               optimizer_name="adamw", grad_mixup=3))

    # A: Standard winner — V3 + muon_sf_blockwise + mixup3
    results.append(run_variant("A_muon_sf_bw_mix3", N, mode="standard",
                               optimizer_name="muon_sf_bw", grad_mixup=3))

    # A2: Muon-SF (no blockwise) + mixup3 — isolate blockwise effect on V3
    results.append(run_variant("A2_muon_sf_mix3", N, mode="standard",
                               optimizer_name="muon_sf", grad_mixup=3))

    # B: DB stack — V3 + DiffusionBlocks + round-robin + muon_sf + mixup3
    results.append(run_variant("B_dblock_robin_muon_sf_mix3", N, mode="diffusion",
                               optimizer_name="muon_sf", grad_mixup=3,
                               num_blocks=4, block_schedule="round_robin"))

    # C: DB cheaper — V3 + DiffusionBlocks + round-robin + AdamW + mixup3
    results.append(run_variant("C_dblock_robin_adamw_mix3", N, mode="diffusion",
                               optimizer_name="adamw", grad_mixup=3,
                               num_blocks=4, block_schedule="round_robin"))

    # E: DB without mixup (isolate mixup effect on V3 DB)
    results.append(run_variant("E_dblock_robin_adamw", N, mode="diffusion",
                               optimizer_name="adamw", grad_mixup=1,
                               num_blocks=4, block_schedule="round_robin"))

    print("\n" + "=" * 70)
    print("SUMMARY (sorted by final loss, lower=better)")
    print("=" * 70)
    results.sort(key=lambda r: r["final_loss"])
    for i, r in enumerate(results):
        print(f"  {i+1}. {r['name']:35s} loss={r['final_loss']:.4f}  time={r['time']:.2f}s")
    best = results[0]
    baseline = next(r for r in results if r["name"] == "D_adamw_cosine")
    standard_winner = next(r for r in results if r["name"] == "A_muon_sf_bw_mix3")
    db_stack = next(r for r in results if r["name"] == "B_dblock_robin_muon_sf_mix3")
    print(f"\n  Best: {best['name']} ({best['final_loss']:.4f})")
    print(f"  vs AdamW cosine ({baseline['final_loss']:.4f}): {baseline['final_loss']/best['final_loss']:.2f}x better")
    print(f"  Standard winner (A): {standard_winner['final_loss']:.4f}")
    print(f"  DB stack (B):         {db_stack['final_loss']:.4f}")
    print(f"  DB stack vs Standard: {standard_winner['final_loss']/db_stack['final_loss']:.2f}x")


if __name__ == "__main__":
    main()
