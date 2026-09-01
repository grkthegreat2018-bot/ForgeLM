"""BitNet-everywhere conversion + manual LoRA adapters + sequential freeze/unfreeze.

Manual LoRA works with BitNetLinear (PEFT can't inject into nn.Module subclasses).
Sequential freeze/unfreeze trains layers in phases — full forward pass preserves
cross-layer connections (MHC, AttnRes), only gradients are scoped.

Validated in .devin/test_bitnet_native.py on real V3 1.2B:
  BitNet-everywhere + LoRA(r=32) + Muon-SF + 3-way grad mixup = 2.39x vs AdamW
  VRAM: 6.32GB (53% of 12GB RTX 5070)
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


# ── BitNet-everywhere ────────────────────────────────────────────────────

def convert_to_bitnet_everywhere(model: nn.Module) -> tuple[int, int]:
    """Replace all nn.Linear with BitNetLinear (ternary QAT).

    Preserves existing BitNetLinear layers. Converts attention projections,
    head, AttnRes, MoD routers — everything — to BitNet b1.58.

    Returns (n_converted, n_already_bitnet).
    """
    from research.keys.quantization.bitnet_b158_key import BitNetLinear

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
                new_layer = BitNetLinear(
                    child.in_features, child.out_features,
                    bias=child.bias is not None,
                    quantize=True,
                    learned_scale=True,
                )
                new_layer.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    new_layer.bias.data.copy_(child.bias.data)
                new_layer = new_layer.to(child.weight.device)
                setattr(module, name, new_layer)
                n_converted += 1
            else:
                convert(child, full_name)

    convert(model)
    return n_converted, n_already


# ── Manual LoRA (BitNet-compatible) ──────────────────────────────────────

class LoRAAdapter(nn.Module):
    """LoRA adapter: y += scale * (x @ A^T @ B^T).

    A: (rank, in_features) — kaiming init
    B: (out_features, rank) — zero init (LoRA starts as no-op)
    """
    def __init__(self, in_features: int, out_features: int, rank: int = 32, alpha: int = 64):
        super().__init__()
        self.rank = rank
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return self.scale * (x @ self.lora_A.T @ self.lora_B.T)


def add_lora_adapters(
    model: nn.Module,
    rank: int = 32,
    alpha: int = 64,
    target_modules: list[str] | None = None,
    min_size: int = 64,
) -> tuple[int, list[nn.Parameter]]:
    """Add LoRA adapters to target layers (works with BitNetLinear + IRIFP4Linear).

    Args:
        model: The model to add LoRA to.
        rank: LoRA rank.
        alpha: LoRA alpha (scale = alpha / rank).
        target_modules: List of module name substrings to target (e.g. ["q_proj", "w_gate"]).
            None = target all Linear/BitNetLinear/IRIFP4Linear with in_features >= min_size.
        min_size: Skip layers smaller than this (e.g. MoD routers with 1 output).

    Returns (n_adapters, lora_params_list).
    """
    n_adapters = 0
    lora_params = []

    # Check for IRIFP4Linear without importing (avoid circular deps)
    def is_iri_fp4_linear(mod):
        cls = type(mod).__name__
        return cls == "IRIFP4Linear"

    def find_and_add(module, prefix=""):
        nonlocal n_adapters
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            has_weight = isinstance(getattr(child, 'weight', None), nn.Parameter)
            has_dims = hasattr(child, 'in_features') and hasattr(child, 'out_features')
            is_iri = is_iri_fp4_linear(child)

            # Valid target: has dims + (has Parameter weight OR is IRIFP4Linear)
            if has_dims and (has_weight or is_iri):
                is_target = True
                if target_modules is not None:
                    is_target = any(t in full_name for t in target_modules)
                if child.in_features < min_size or child.out_features < min_size:
                    is_target = False

                if is_target:
                    lora = LoRAAdapter(child.in_features, child.out_features, rank=rank, alpha=alpha)
                    # Use bfloat16 for LoRA params on IRIFP4 (base is bf16 dequantized)
                    lora_dtype = torch.bfloat16 if is_iri else (
                        child.weight.dtype if child.weight.dtype != torch.float32 else torch.bfloat16)
                    lora = lora.to(child.weight_packed.device if is_iri else child.weight.device).to(lora_dtype)
                    setattr(child, 'lora_adapter', lora)

                    # For IRIFP4Linear, forward() already checks for lora_adapter
                    # For BitNetLinear/nn.Linear, we need to wrap forward
                    if not is_iri:
                        orig_forward = child.forward
                        child._lora_orig_forward = orig_forward  # save for unload
                        def make_new_forward(orig_fwd, lora_mod):
                            def new_forward(x):
                                out = orig_fwd(x)
                                return out + lora_mod(x)
                            return new_forward
                        child.forward = make_new_forward(orig_forward, lora)

                    # Freeze base weights
                    if has_weight:
                        child.weight.requires_grad = False
                    if hasattr(child, 'qscale') and child.qscale is not None:
                        child.qscale.requires_grad = False
                    if hasattr(child, 'bias') and child.bias is not None:
                        if isinstance(child.bias, nn.Parameter):
                            child.bias.requires_grad = False

                    lora_params.extend([lora.lora_A, lora.lora_B])
                    n_adapters += 1
            find_and_add(child, full_name)

    find_and_add(model)
    return n_adapters, lora_params


def merge_lora_adapters(model: nn.Module) -> int:
    """Merge LoRA adapters into base weights: W += scale * B @ A.

    For nn.Linear/BitNetLinear: directly adds delta to weight Parameter.
    For IRIFP4Linear: dequantizes → adds delta → re-quantizes to IRI-FP4.
    Call before saving checkpoint so output is standalone (no LoRA dependency).
    Returns n_merged.
    """
    n_merged = 0
    for module in model.modules():
        if hasattr(module, 'lora_adapter') and isinstance(module.lora_adapter, LoRAAdapter):
            cls_name = type(module).__name__
            if cls_name == "IRIFP4Linear":
                # QLoRA merge: dequant → merge → re-quantize
                if module.merge_lora():
                    n_merged += 1
                continue
            # Standard merge for nn.Linear / BitNetLinear
            lora = module.lora_adapter
            with torch.no_grad():
                # W += scale * B @ A  (out_features, in_features)
                delta = lora.scale * (lora.lora_B @ lora.lora_A)
                module.weight.data += delta.to(module.weight.dtype)
            # Remove adapter
            del module.lora_adapter
            n_merged += 1
    return n_merged


# ── Sequential freeze/unfreeze ───────────────────────────────────────────

def freeze_unfreeze_lora(
    model: nn.Module,
    active_layers: list[int] | None = None,
) -> None:
    """Freeze/unfreeze LoRA params by layer index.

    active_layers: list of layer indices to unfreeze. None = unfreeze all.
    Layers are identified by 'blocks.{i}.' in parameter names.
    """
    for n, p in model.named_parameters():
        if 'lora_A' not in n and 'lora_B' not in n:
            continue
        if active_layers is None:
            p.requires_grad = True
        else:
            p.requires_grad = any(f"blocks.{li}." in n or f".blocks.{li}." in n for li in active_layers)


def get_active_lora_params(model: nn.Module) -> list[nn.Parameter]:
    """Get all LoRA params that currently have requires_grad=True."""
    return [p for n, p in model.named_parameters()
            if ('lora_A' in n or 'lora_B' in n) and p.requires_grad]


def compute_phase_schedule(
    n_layers: int,
    n_phases: int,
    total_steps: int,
    final_finetune_steps: int = 0,
) -> list[tuple[int, int, list[int]]]:
    """Compute sequential freeze/unfreeze phase schedule.

    Args:
        n_layers: Total number of model layers.
        n_phases: Number of sequential phases.
        total_steps: Total training steps.
        final_finetune_steps: Steps at the end to fine-tune ALL layers (phase 5).
            0 = no final fine-tune.

    Returns list of (start_step, end_step, active_layers) tuples.
    """
    if final_finetune_steps > 0:
        seq_steps = total_steps - final_finetune_steps
    else:
        seq_steps = total_steps

    layers_per_phase = n_layers // n_phases
    steps_per_phase = seq_steps // n_phases

    schedule = []
    for phase in range(n_phases):
        start = phase * steps_per_phase
        end = (phase + 1) * steps_per_phase if phase < n_phases - 1 else seq_steps
        start_layer = phase * layers_per_phase
        end_layer = (phase + 1) * layers_per_phase if phase < n_phases - 1 else n_layers
        active = list(range(start_layer, end_layer))
        schedule.append((start, end, active))

    if final_finetune_steps > 0:
        schedule.append((seq_steps, total_steps, None))  # None = all layers

    return schedule


def get_active_layers_for_step(
    step: int,
    schedule: list[tuple[int, int, list[int]]],
) -> list[int] | None:
    """Get active layers for a given step from the phase schedule.

    Returns None if all layers should be active (final fine-tune).
    """
    for start, end, active in schedule:
        if start <= step < end:
            return active
    return None


# ── Muon-SF optimizer for LoRA params ────────────────────────────────────

def build_muon_sf_lora_opt(
    lora_params: list[nn.Parameter],
    lr_muon: float = 5e-3,
    lr_adam: float = 3e-4,
):
    """Build Muon-SF optimizer for LoRA params.

    Muon (Newton-Schulz) for 2D LoRA A/B matrices, ScheduleFree AdamW for any 1D params.
    Validated on V3 1.2B: 2.39x better than AdamW with LoRA.
    """
    from muon import SingleDeviceMuonWithAuxAdam, muon_update
    from schedulefree import AdamWScheduleFree

    muon_p = [p for p in lora_params if p.ndim == 2]
    adam_p = [p for p in lora_params if p.ndim != 2]

    class _MuonSFLoRA(SingleDeviceMuonWithAuxAdam):
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

    return _MuonSFLoRA()
