"""QLoRA validation: 4-bit NF4 quantized V3 + LoRA + Muon-SF + grad mixup.

The problem: full-precision V3 1.2B uses 12.69GB for AdamW alone — mixup3
(3x forward) OOMs on 12GB RTX 5070.

QLoRA solution:
1. Quantize base model to 4-bit NF4 (1.2B params → ~0.6GB weights)
2. Add LoRA adapters (rank 16-64) — only train adapters
3. Use paged 8-bit AdamW optimizer (VRAM-efficient optimizer state)
4. Now mixup3 fits in 12GB

Tests:
A. QLoRA + AdamW + mixup1 (baseline)
B. QLoRA + AdamW + mixup3 (does mixup help with LoRA?)
C. QLoRA + Muon-SF + mixup3 (the full optimal stack, QLoRA edition)
D. QLoRA + PagedAdamW8bit + mixup3 (VRAM-optimal)
"""
import os, sys, torch, gc, time, math
import torch.nn as nn
from pathlib import Path
sys.stdout.reconfigure(line_buffering=True)
for line in Path("D:/windsurf/ForgeAI/.env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

from research.model_loader import load_default_model
from research.checkpoint_io import load_checkpoint
import torch.nn.functional as F
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def quantize_model_4bit(model):
    """Replace all nn.Linear layers with bnb Linear4bit (NF4).

    Skips BitNetLinear (which is nn.Module, not nn.Linear) — those weights
    are already ternary {-1,0,1} and don't benefit from 4-bit quantization.
    """
    from bitsandbytes.nn import Linear4bit
    import torch.nn as nn

    n_quantized = 0
    n_skipped = 0

    def replace_linear(module, prefix=""):
        nonlocal n_quantized, n_skipped
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and not isinstance(child, type(child)) is False:
                # Standard nn.Linear — quantize
                if child.out_features >= 16 and child.in_features >= 16:
                    new_layer = Linear4bit(
                        child.in_features, child.out_features,
                        bias=child.bias is not None,
                        compute_dtype=torch.bfloat16,
                        quant_type="nf4",
                        device=child.weight.device,
                    )
                    new_layer.weight = bnb.nn.Params4bit(
                        child.weight.data.to(torch.bfloat16),
                        requires_grad=False,
                        quant_type="nf4",
                        quant_state=None,
                    )
                    if child.bias is not None:
                        new_layer.bias = nn.Parameter(child.bias.data.clone())
                    setattr(module, name, new_layer)
                    n_quantized += 1
            else:
                replace_linear(child, full_name)

    replace_linear(model)
    print(f"  Quantized {n_quantized} Linear layers to NF4 (BitNet layers skipped — already ternary)", flush=True)
    return model


# ── Manual LoRA (works with BitNetLinear, unlike PEFT) ───────────────────

class LoRAAdapter(nn.Module):
    """LoRA adapter: y = x @ W^T + scale * (x @ A^T @ B^T).

    A: (rank, in_features) — init with kaiming
    B: (out_features, rank) — init with ZERO (so LoRA starts as identity)
    """
    def __init__(self, in_features, out_features, rank=32, alpha=64):
        super().__init__()
        self.rank = rank
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        # x: (..., in_features) → (..., out_features)
        return self.scale * (x @ self.lora_A.T @ self.lora_B.T)


def add_lora_to_model(model, rank=32, alpha=64):
    """Add LoRA adapters to BitNetLinear and Linear4bit layers.

    Wraps each target layer by registering a LoRA adapter and hooking
    the forward to add the LoRA output.
    """
    target_names = ["q_proj", "k_proj", "v_proj", "out_proj",
                    "w_gate", "w_up", "w_down"]
    n_adapters = 0
    lora_params = []

    def find_and_add(module, prefix=""):
        nonlocal n_adapters
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            # Check if this module has a weight parameter and matches target
            has_weight = hasattr(child, 'weight') and isinstance(child.weight, nn.Parameter)
            is_target = any(t in full_name for t in target_names)
            if has_weight and is_target and hasattr(child, 'in_features') and hasattr(child, 'out_features'):
                # Add LoRA adapter as a submodule
                lora = LoRAAdapter(child.in_features, child.out_features, rank=rank, alpha=alpha)
                lora = lora.to(child.weight.device).to(torch.bfloat16)
                setattr(child, 'lora_adapter', lora)

                # Store original forward
                orig_forward = child.forward

                # Create new forward that adds LoRA output
                def make_new_forward(orig_fwd, lora_mod):
                    def new_forward(x):
                        out = orig_fwd(x)
                        return out + lora_mod(x)
                    return new_forward

                child.forward = make_new_forward(orig_forward, lora)

                # Freeze base weights
                child.weight.requires_grad = False
                if hasattr(child, 'bias') and child.bias is not None:
                    child.bias.requires_grad = False
                if hasattr(child, 'qscale') and child.qscale is not None:
                    child.qscale.requires_grad = False

                lora_params.extend([lora.lora_A, lora.lora_B])
                n_adapters += 1
            else:
                find_and_add(child, full_name)

    find_and_add(model)
    print(f"  Added {n_adapters} LoRA adapters (rank={rank}, alpha={alpha})", flush=True)
    print(f"  LoRA trainable params: {sum(p.numel() for p in lora_params) / 1e6:.2f}M", flush=True)
    return model, lora_params


print("=== Building ForgeLM V3 (1.2B) ===", flush=True)
model, tok = load_default_model("forgelm_v3")
ckpt_path = "research/checkpoints/ForgeLM_V3_Base.safetensors"
if not os.path.exists(ckpt_path):
    ckpt_path = "research/checkpoints/ForgeLM_V2_BSP.safetensors"
sd = load_checkpoint(ckpt_path)
model.load_state_dict({k: v for k, v in sd.items()}, strict=False)
model = model.to("cuda")
config = model.config
total_params = sum(p.numel() for p in model.parameters())
print(f"Model: {total_params/1e6:.1f}M params (fp32), {len(model.blocks)} layers", flush=True)

# Step 1: Quantize to 4-bit NF4
print("\n=== Quantizing to 4-bit NF4 ===", flush=True)
model = quantize_model_4bit(model)
model = model.to(torch.bfloat16)
quantized_size = sum(p.numel() * (0.5 if hasattr(p, 'quant_state') else 2) for p in model.parameters())
print(f"Quantized model: ~{quantized_size/1e9:.2f}GB (4-bit weights + bf16 buffers)", flush=True)

# Step 2: Prepare for k-bit training (gradient checkpointing, etc.)
# Skip PEFT's prepare_model_for_kbit_training — it doesn't know BitNetLinear
# We handle freezing manually in add_lora_to_model

# Step 3: Add LoRA adapters (manual — PEFT can't handle BitNetLinear)
print("\n=== Adding LoRA adapters ===", flush=True)
model, lora_params = add_lora_to_model(model, rank=32, alpha=64)
model = model.to("cuda").train()

# Save initial LoRA state for fair comparison
initial_lora_state = {}
for n, p in model.named_parameters():
    if 'lora_A' in n or 'lora_B' in n:
        initial_lora_state[n] = p.clone()

torch.manual_seed(42)
batch_size, seq_len = 2, 256
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device="cuda")
labels = torch.cat([input_ids[:, 1:], input_ids[:, 0:1]], dim=1)

mixup_ids = []
for i in range(4):
    torch.manual_seed(1000 + i)
    mixup_ids.append(torch.randint(0, config.vocab_size, (batch_size, seq_len), device="cuda"))
torch.manual_seed(42)


def build_optimizer(model, opt_name, lr=3e-4):
    """Build optimizer for LoRA params only."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        trainable = lora_params  # fallback to global lora_params list
    if opt_name == "adamw":
        return torch.optim.AdamW(trainable, lr=lr, fused=True)
    elif opt_name == "paged_adamw_8bit":
        return bnb.optim.PagedAdamW8bit(trainable, lr=lr)
    elif opt_name == "muon_sf":
        # Muon-SF for LoRA: LoRA params are low-rank (A, B matrices)
        # Muon works on 2D weights — LoRA A and B are 2D
        from muon import SingleDeviceMuonWithAuxAdam
        from schedulefree import AdamWScheduleFree

        muon_p, adam_p = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            # LoRA lora_B is the "expansion" matrix (rank × features) — Muon it
            # lora_A is the "compression" matrix (features × rank) — Muon it too
            if 'lora_B' in n or 'lora_A' in n:
                muon_p.append(p)
            else:
                adam_p.append(p)

        if not muon_p:
            # Fallback: all AdamW
            return AdamWScheduleFree(trainable, lr=lr)

        class MuonSFLoRA(SingleDeviceMuonWithAuxAdam):
            def __init__(self, muon_params, adam_params, lr_muon, lr_adam):
                self._sf = AdamWScheduleFree(adam_params, lr=lr_adam, betas=(0.9, 0.95), weight_decay=0.0)
                super().__init__([dict(params=muon_params, lr=lr_muon, momentum=0.95,
                                       weight_decay=0.0, use_muon=True)])
                self._sf.train()

            @torch.no_grad()
            def step(self, closure=None):
                from muon import muon_update
                for group in self.param_groups:
                    for p in group["params"]:
                        if p.grad is None:
                            continue
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                        update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])
                self._sf.step()

            def zero_grad(self, set_to_none=True):
                super().zero_grad(set_to_none=set_to_none)
                self._sf.zero_grad(set_to_none=set_to_none)

            def train(self):
                self._sf.train()

            def eval(self):
                self._sf.eval()

        lr_muon = 5e-3  # production scaling
        return MuonSFLoRA(muon_p, adam_p, lr_muon, lr)
    else:
        return torch.optim.AdamW(trainable, lr=lr, fused=True)


def run_training(model, opt_name, n_steps=20, lr=3e-4, grad_mixup=1):
    # Reset LoRA weights
    for n, p in model.named_parameters():
        if n in initial_lora_state:
            p.data.copy_(initial_lora_state[n])
    model.train()

    optimizer = build_optimizer(model, opt_name, lr)
    if hasattr(optimizer, 'train'):
        optimizer.train()

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    losses = []
    times = []

    for step in range(n_steps):
        optimizer.zero_grad()
        t0 = time.time()

        out = model(input_ids, targets=labels)
        loss = out[1] if isinstance(out, tuple) and len(out) > 1 else None
        if loss is None:
            logits = out[0] if isinstance(out, tuple) else out
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        if grad_mixup > 1:
            loss.backward()
            saved_grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
            for mi in range(grad_mixup - 1):
                mi_ids = mixup_ids[mi]
                mi_labels = torch.cat([mi_ids[:, 1:], mi_ids[:, 0:1]], dim=1)
                optimizer.zero_grad()
                out2 = model(mi_ids, targets=mi_labels)
                loss2 = out2[1] if isinstance(out2, tuple) and len(out2) > 1 else None
                if loss2 is None:
                    logits2 = out2[0] if isinstance(out2, tuple) else out2
                    loss2 = F.cross_entropy(logits2.reshape(-1, logits2.size(-1)), mi_labels.reshape(-1))
                loss2.backward()
                for n, p in model.named_parameters():
                    if p.grad is not None and n in saved_grads:
                        saved_grads[n] = (saved_grads[n] * (mi + 1) + p.grad) / (mi + 2)
            optimizer.zero_grad()
            for n, p in model.named_parameters():
                if n in saved_grads:
                    p.grad = saved_grads[n]
        else:
            loss.backward()

        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.time()
        losses.append(loss.item())
        times.append(t1 - t0)
        if step % 5 == 0 or step == n_steps - 1:
            mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"  [{opt_name:18s} mixup={grad_mixup}] step {step:3d} loss {loss.item():.4f} "
                  f"time {t1-t0:.3f}s peak_mem {mem:.2f}GB", flush=True)

    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    avg_time = sum(times[3:]) / len(times[3:])
    return {
        "losses": losses,
        "peak_mem": peak_mem,
        "avg_time": avg_time,
        "final_loss": losses[-1],
        "first_loss": losses[0],
        "loss_reduction": losses[0] - losses[-1],
    }


N_STEPS = 20
LR = 3e-4

print(f"\n=== QLoRA Benchmark: {N_STEPS} steps, batch={batch_size}, seq={seq_len}, lr={LR} ===\n", flush=True)

results = {}

print("--- A. QLoRA + AdamW + mixup1 (baseline) ---", flush=True)
results["qlora_adamw"] = run_training(model, "adamw", N_STEPS, LR, grad_mixup=1)
torch.cuda.empty_cache(); gc.collect()

print("\n--- B. QLoRA + AdamW + mixup3 ---", flush=True)
results["qlora_adamw_mix3"] = run_training(model, "adamw", N_STEPS, LR, grad_mixup=3)
torch.cuda.empty_cache(); gc.collect()

print("\n--- C. QLoRA + PagedAdamW8bit + mixup3 (VRAM-optimal) ---", flush=True)
results["qlora_paged8bit_mix3"] = run_training(model, "paged_adamw_8bit", N_STEPS, LR, grad_mixup=3)
torch.cuda.empty_cache(); gc.collect()

print("\n--- D. QLoRA + Muon-SF + mixup3 (full optimal stack) ---", flush=True)
results["qlora_muon_sf_mix3"] = run_training(model, "muon_sf", N_STEPS, LR, grad_mixup=3)
torch.cuda.empty_cache(); gc.collect()

# Summary
print("\n" + "=" * 90)
print("SUMMARY: QLoRA V3 1.2B — Muon-SF + grad mixup validation")
print("=" * 90)
print(f"{'Variant':<30s} {'Final':>8s} {'Reduction':>10s} {'AvgTime':>8s} {'PeakMem':>8s}")
print("-" * 90)
for name, r in sorted(results.items(), key=lambda x: x[1]["final_loss"]):
    print(f"{name:<30s} {r['final_loss']:8.3f} {r['loss_reduction']:10.3f} "
          f"{r['avg_time']:8.3f}s {r['peak_mem']:7.2f}GB")

baseline = results["qlora_adamw"]
best = min(results.values(), key=lambda r: r["final_loss"])
best_name = [k for k, v in results.items() if v is best][0]
print(f"\n  Best: {best_name}")
print(f"    vs QLoRA+AdamW: {baseline['final_loss']/best['final_loss']:.2f}x better loss, "
      f"{best['peak_mem']/baseline['peak_mem']:.2f}x mem, "
      f"{best['avg_time']/baseline['avg_time']:.2f}x time")
print(f"\n  All variants fit in 12GB? {'YES' if all(r['peak_mem'] < 12.0 for r in results.values()) else 'NO — some OOM'}")
