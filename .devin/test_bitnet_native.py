"""BitNet-native + sequential freeze/unfreeze + Muon-SF + grad mixup.

Two improvements over QLoRA test:
1. BitNet-everywhere: convert remaining Linear layers to BitNetLinear (ternary).
   No NF4 needed — BitNet IS the quantization (1.58 bits vs 4 bits NF4).
2. Sequential freeze/unfreeze: train layers in phases, full forward pass
   (preserves MHC/AttnRes), only compute gradients for selected layers.

Phases (16-layer model, 4 phases of 4 layers):
  Phase 1: Train layers 0-3 (freeze 4-15)
  Phase 2: Train layers 4-7 (freeze 0-3, 8-15)
  Phase 3: Train layers 8-11 (freeze 0-7, 12-15)
  Phase 4: Train layers 12-15 (freeze 0-11)
  Phase 5: Brief fine-tune all layers

This reduces gradient VRAM by 4x while keeping cross-layer connections intact.

Tests:
A. BitNet-everywhere + LoRA + Muon-SF + mixup3 (all layers, no freeze)
B. BitNet-everywhere + LoRA + Muon-SF + mixup3 + sequential freeze (4 phases)
C. BitNet-everywhere + LoRA + AdamW + mixup3 (baseline, no freeze)
D. Same as B but with PagedAdamW8bit (max VRAM savings)
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
from research.keys.quantization.bitnet_b158_key import BitNetLinear
import torch.nn.functional as F
import bitsandbytes as bnb
from muon import SingleDeviceMuonWithAuxAdam, muon_update
from schedulefree import AdamWScheduleFree


# ── Step 1: Convert remaining Linear to BitNetLinear ─────────────────────

def convert_to_bitnet_everywhere(model):
    """Replace all nn.Linear with BitNetLinear (ternary QAT).

    Preserves existing BitNetLinear layers. Converts attention projections,
    head, AttnRes, MoD routers to BitNet b1.58.
    """
    n_converted = 0
    n_already = 0

    def convert(module, prefix=""):
        nonlocal n_converted, n_already
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, BitNetLinear):
                n_already += 1
                continue
            if isinstance(child, nn.Linear):
                # Convert to BitNetLinear
                new_layer = BitNetLinear(
                    child.in_features, child.out_features,
                    bias=child.bias is not None,
                    quantize=True,  # Enable ternary QAT
                    learned_scale=True,
                )
                # Copy weights
                new_layer.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    new_layer.bias.data.copy_(child.bias.data)
                # Move to same device
                new_layer = new_layer.to(child.weight.device)
                setattr(module, name, new_layer)
                n_converted += 1
            else:
                convert(child, full_name)

    convert(model)
    print(f"  BitNet conversion: {n_converted} Linear → BitNetLinear, {n_already} already BitNet", flush=True)
    return model


# ── Step 2: Manual LoRA adapters (works with BitNetLinear) ───────────────

class LoRAAdapter(nn.Module):
    def __init__(self, in_features, out_features, rank=32, alpha=64):
        super().__init__()
        self.rank = rank
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return self.scale * (x @ self.lora_A.T @ self.lora_B.T)


def add_lora_to_model(model, rank=32, alpha=64, target_layers=None):
    """Add LoRA adapters to target layers. If target_layers is None, add to all
    BitNetLinear/Linear with in_features >= 64."""
    n_adapters = 0
    lora_params = []

    def find_and_add(module, prefix=""):
        nonlocal n_adapters
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            has_weight = hasattr(child, 'weight') and isinstance(getattr(child, 'weight', None), nn.Parameter)
            has_dims = hasattr(child, 'in_features') and hasattr(child, 'out_features')

            if has_weight and has_dims:
                # Check if this layer is in target_layers (if specified)
                is_target = True
                if target_layers is not None:
                    is_target = any(t in full_name for t in target_layers)
                # Skip tiny layers (routers, etc.)
                if child.in_features < 64 or child.out_features < 64:
                    is_target = False

                if is_target:
                    lora = LoRAAdapter(child.in_features, child.out_features, rank=rank, alpha=alpha)
                    lora = lora.to(child.weight.device).to(torch.bfloat16)
                    setattr(child, 'lora_adapter', lora)

                    orig_forward = child.forward
                    def make_new_forward(orig_fwd, lora_mod):
                        def new_forward(x):
                            out = orig_fwd(x)
                            return out + lora_mod(x)
                        return new_forward
                    child.forward = make_new_forward(orig_forward, lora)

                    # Freeze base weights
                    child.weight.requires_grad = False
                    if hasattr(child, 'qscale') and child.qscale is not None:
                        child.qscale.requires_grad = False
                    if hasattr(child, 'bias') and child.bias is not None:
                        child.bias.requires_grad = False

                    lora_params.extend([lora.lora_A, lora.lora_B])
                    n_adapters += 1
            find_and_add(child, full_name)

    find_and_add(model)
    print(f"  Added {n_adapters} LoRA adapters (rank={rank}), {sum(p.numel() for p in lora_params)/1e6:.2f}M trainable", flush=True)
    return model, lora_params


# ── Step 3: Sequential freeze/unfreeze ───────────────────────────────────

def get_layer_lora_params(model, layer_indices):
    """Get LoRA params belonging to specific model layers."""
    params = []
    for n, p in model.named_parameters():
        if 'lora_A' not in n and 'lora_B' not in n:
            continue
        # Extract layer index from param name (e.g., "base_model.model.blocks.3...")
        for li in layer_indices:
            if f"blocks.{li}." in n or f".blocks.{li}." in n:
                params.append(p)
                break
    return params


def freeze_unfreeze_lora(model, active_layers=None):
    """Freeze/unfreeze LoRA params by layer.

    active_layers: list of layer indices to unfreeze. None = unfreeze all.
    """
    for n, p in model.named_parameters():
        if 'lora_A' not in n and 'lora_B' not in n:
            continue
        if active_layers is None:
            p.requires_grad = True
        else:
            is_active = any(f"blocks.{li}." in n or f".blocks.{li}." in n for li in active_layers)
            p.requires_grad = is_active


# ── Optimizer builders ───────────────────────────────────────────────────

def build_muon_sf_lora(lora_params, lr_muon=5e-3, lr_adam=3e-4):
    """Muon-SF for LoRA params: Muon for lora_A/B (2D), SF for any others."""
    muon_p = [p for p in lora_params if p.ndim == 2]
    adam_p = [p for p in lora_params if p.ndim != 2]

    class MuonSFLoRA(SingleDeviceMuonWithAuxAdam):
        def __init__(self):
            self._sf = AdamWScheduleFree(adam_p, lr=lr_adam, betas=(0.9, 0.95), weight_decay=0.0) if adam_p else None
            super().__init__([dict(params=muon_p, lr=lr_muon, momentum=0.95, weight_decay=0.0, use_muon=True)])
            if self._sf:
                self._sf.train()

        @torch.no_grad()
        def step(self, closure=None):
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            if self._sf:
                self._sf.step()

        def zero_grad(self, set_to_none=True):
            super().zero_grad(set_to_none=set_to_none)
            if self._sf:
                self._sf.zero_grad(set_to_none=set_to_none)

        def train(self):
            if self._sf:
                self._sf.train()

        def eval(self):
            if self._sf:
                self._sf.eval()

    return MuonSFLoRA()


# ── Training ─────────────────────────────────────────────────────────────

def run_training(model, opt_name, n_steps=20, lr=3e-4, grad_mixup=1,
                 sequential=False, n_phases=4):
    """Run training with optional sequential freeze/unfreeze."""
    model.train()

    # Reset LoRA weights
    for n, p in model.named_parameters():
        if 'lora_A' in n or 'lora_B' in n:
            if 'lora_B' in n:
                nn.init.zeros_(p)
            else:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))

    # Get all LoRA params
    all_lora = [p for n, p in model.named_parameters() if 'lora_A' in n or 'lora_B' in n]

    # Build optimizer
    if opt_name == "muon_sf":
        opt = build_muon_sf_lora(all_lora, lr_muon=5e-3, lr_adam=lr)
    elif opt_name == "adamw":
        opt = torch.optim.AdamW([p for p in all_lora if p.requires_grad], lr=lr, fused=True)
    elif opt_name == "paged8bit":
        opt = bnb.optim.PagedAdamW8bit([p for p in all_lora if p.requires_grad], lr=lr)

    if hasattr(opt, 'train'):
        opt.train()

    # Sequential freeze/unfreeze setup
    n_layers = 16
    if sequential:
        layers_per_phase = n_layers // n_phases
        steps_per_phase = n_steps // n_phases

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    losses = []
    times = []

    for step in range(n_steps):
        # Sequential: determine active layers for this step
        if sequential:
            phase = min(step // steps_per_phase, n_phases - 1)
            start_layer = phase * layers_per_phase
            end_layer = start_layer + layers_per_phase
            active_layers = list(range(start_layer, end_layer))
            freeze_unfreeze_lora(model, active_layers=active_layers)

            # Rebuild optimizer with only active params (every phase)
            if step % steps_per_phase == 0:
                active_params = [p for n, p in model.named_parameters()
                                if ('lora_A' in n or 'lora_B' in n) and p.requires_grad]
                if opt_name == "muon_sf":
                    opt = build_muon_sf_lora(active_params, lr_muon=5e-3, lr_adam=lr)
                elif opt_name == "adamw":
                    opt = torch.optim.AdamW(active_params, lr=lr, fused=True)
                elif opt_name == "paged8bit":
                    opt = bnb.optim.PagedAdamW8bit(active_params, lr=lr)
                if hasattr(opt, 'train'):
                    opt.train()
                print(f"    Phase {phase+1}/{n_phases}: training layers {start_layer}-{end_layer-1} "
                      f"({len(active_params)} params)", flush=True)
        else:
            freeze_unfreeze_lora(model, active_layers=None)

        opt.zero_grad()
        t0 = time.time()

        out = model(input_ids, targets=labels)
        loss = out[1] if isinstance(out, tuple) and len(out) > 1 else None
        if loss is None:
            logits = out[0] if isinstance(out, tuple) else out
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        if grad_mixup > 1:
            loss.backward()
            saved_grads = {n: p.grad.clone() for n, p in model.named_parameters()
                          if p.grad is not None and p.requires_grad}
            for mi in range(grad_mixup - 1):
                mi_ids = mixup_ids[mi]
                mi_labels = torch.cat([mi_ids[:, 1:], mi_ids[:, 0:1]], dim=1)
                opt.zero_grad()
                out2 = model(mi_ids, targets=mi_labels)
                loss2 = out2[1] if isinstance(out2, tuple) and len(out2) > 1 else None
                if loss2 is None:
                    logits2 = out2[0] if isinstance(out2, tuple) else out2
                    loss2 = F.cross_entropy(logits2.reshape(-1, logits2.size(-1)), mi_labels.reshape(-1))
                loss2.backward()
                for n, p in model.named_parameters():
                    if p.grad is not None and n in saved_grads:
                        saved_grads[n] = (saved_grads[n] * (mi + 1) + p.grad) / (mi + 2)
            opt.zero_grad()
            for n, p in model.named_parameters():
                if n in saved_grads and p.requires_grad:
                    p.grad = saved_grads[n]
        else:
            loss.backward()

        trainable = [p for p in model.parameters() if p.requires_grad]
        if trainable:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        torch.cuda.synchronize()
        t1 = time.time()
        losses.append(loss.item())
        times.append(t1 - t0)
        if step % 5 == 0 or step == n_steps - 1:
            mem = torch.cuda.max_memory_allocated() / 1e9
            phase_str = f"phase={phase+1}" if sequential else "all"
            print(f"  [{opt_name:15s} mixup={grad_mixup} {phase_str:12s}] step {step:3d} loss {loss.item():.4f} "
                  f"time {t1-t0:.3f}s mem {mem:.2f}GB", flush=True)

    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    avg_time = sum(times[3:]) / max(1, len(times[3:]))
    return {
        "losses": losses,
        "peak_mem": peak_mem,
        "avg_time": avg_time,
        "final_loss": losses[-1],
        "first_loss": losses[0],
        "loss_reduction": losses[0] - losses[-1],
    }


# ── Main ─────────────────────────────────────────────────────────────────

print("=== Building ForgeLM V3 (1.2B) ===", flush=True)
model, tok = load_default_model("forgelm_v3")
ckpt_path = "research/checkpoints/ForgeLM_V3_Base.safetensors"
if not os.path.exists(ckpt_path):
    ckpt_path = "research/checkpoints/ForgeLM_V2_BSP.safetensors"
sd = load_checkpoint(ckpt_path)
model.load_state_dict({k: v for k, v in sd.items()}, strict=False)
model = model.to("cuda").to(torch.bfloat16)
config = model.config
total_params = sum(p.numel() for p in model.parameters())
print(f"Model: {total_params/1e6:.1f}M params, {len(model.blocks)} layers", flush=True)

# Step 1: BitNet-everywhere
print("\n=== Converting all Linear to BitNet b1.58 ===", flush=True)
model = convert_to_bitnet_everywhere(model)
model = model.to(torch.bfloat16)

# Step 2: Add LoRA adapters
print("\n=== Adding LoRA adapters ===", flush=True)
model, lora_params = add_lora_to_model(model, rank=32, alpha=64)
model = model.to("cuda").train()

torch.manual_seed(42)
batch_size, seq_len = 2, 256
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device="cuda")
labels = torch.cat([input_ids[:, 1:], input_ids[:, 0:1]], dim=1)

mixup_ids = []
for i in range(4):
    torch.manual_seed(1000 + i)
    mixup_ids.append(torch.randint(0, config.vocab_size, (batch_size, seq_len), device="cuda"))
torch.manual_seed(42)

N_STEPS = 20
LR = 3e-4

print(f"\n=== BitNet-native Benchmark: {N_STEPS} steps, batch={batch_size}, seq={seq_len}, lr={LR} ===\n", flush=True)

results = {}

print("--- A. BitNet + LoRA + Muon-SF + mixup3 (all layers) ---", flush=True)
results["bitnet_muon_sf_mix3"] = run_training(model, "muon_sf", N_STEPS, LR, grad_mixup=3, sequential=False)
torch.cuda.empty_cache(); gc.collect()

print("\n--- B. BitNet + LoRA + Muon-SF + mixup3 + sequential freeze (4 phases) ---", flush=True)
results["bitnet_muon_sf_mix3_seq"] = run_training(model, "muon_sf", N_STEPS, LR, grad_mixup=3, sequential=True, n_phases=4)
torch.cuda.empty_cache(); gc.collect()

print("\n--- C. BitNet + LoRA + AdamW + mixup3 (baseline) ---", flush=True)
results["bitnet_adamw_mix3"] = run_training(model, "adamw", N_STEPS, LR, grad_mixup=3, sequential=False)
torch.cuda.empty_cache(); gc.collect()

print("\n--- D. BitNet + LoRA + PagedAdamW8bit + mixup3 + sequential ---", flush=True)
results["bitnet_paged8bit_mix3_seq"] = run_training(model, "paged8bit", N_STEPS, LR, grad_mixup=3, sequential=True, n_phases=4)
torch.cuda.empty_cache(); gc.collect()

# Summary
print("\n" + "=" * 90)
print("SUMMARY: BitNet-native V3 1.2B — sequential freeze + Muon-SF + mixup")
print("=" * 90)
print(f"{'Variant':<35s} {'Final':>8s} {'Reduction':>10s} {'AvgTime':>8s} {'PeakMem':>8s}")
print("-" * 90)
for name, r in sorted(results.items(), key=lambda x: x[1]["final_loss"]):
    print(f"{name:<35s} {r['final_loss']:8.3f} {r['loss_reduction']:10.3f} "
          f"{r['avg_time']:8.3f}s {r['peak_mem']:7.2f}GB")

baseline = results["bitnet_adamw_mix3"]
best = min(results.values(), key=lambda r: r["final_loss"])
best_name = [k for k, v in results.items() if v is best][0]
print(f"\n  Best: {best_name}")
print(f"    vs BitNet+AdamW+mix3: {baseline['final_loss']/best['final_loss']:.2f}x better loss, "
      f"{best['peak_mem']/baseline['peak_mem']:.2f}x mem")
print(f"\n  All fit in 12GB? {'YES' if all(r['peak_mem'] < 12.0 for r in results.values()) else 'NO'}")
