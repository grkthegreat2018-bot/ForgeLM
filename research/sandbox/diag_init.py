"""Diagnose from-scratch init pathology in V7 configs.

Symptom: initial CE ~24 vs uniform baseline ln(65536)=11.09 — the model is
CONFIDENTLY WRONG at init, i.e. logits are oversized. Measures per-layer
hidden norms, logit stats, and CE on real data.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import math
from pathlib import Path

import torch

from research.sandbox.train_8b_all import (build_model, autocast_ctx,
                                           forward_model, load_packed_data)

device = torch.device("cuda")
model, cfg = build_model("forgelm_v7_8b_b", device, torch.bfloat16,
                         use_checkpointing=False, grad_clip=1.0)
model.eval()

data = load_packed_data(Path("research/data/v7_train/train.bin"), 512, "diag")
ids = data[:2].to(device)

norms = []
def hook(mod, inp, out):
    t = out[0] if isinstance(out, tuple) else out
    if t.dim() == 3:
        norms.append(float(t.float().norm() / math.sqrt(t.numel())))
handles = [b.register_forward_hook(hook) for b in model.blocks]

with torch.no_grad(), autocast_ctx(device):
    logits = forward_model(model, ids)

lf = logits.float()
ce = torch.nn.functional.cross_entropy(
    lf[:, :-1].contiguous().view(-1, lf.size(-1)), ids[:, 1:].contiguous().view(-1))
print(f"per-layer RMS (sampled): {[f'{n:.1f}' for n in norms[:8]]} ... "
      f"{[f'{n:.1f}' for n in norms[-4:]]}")
print(f"logits: mean={lf.mean():.2f} std={lf.std():.2f} "
      f"max={lf.max():.1f} min={lf.min():.1f}")
print(f"CE at init: {ce.item():.3f}  (uniform baseline = {math.log(cfg.vocab_size):.3f})")

# also check the effective head/embed scales
m = model
print(f"head type: {type(m.head).__name__}")
E = m.embed.embed.weight.detach().float()
P = m.embed.project.weight.detach().float()
print(f"embed.weight: shape={tuple(E.shape)} std={E.std():.4f}")
print(f"project.weight: shape={tuple(P.shape)} std={P.std():.4f}")
# effective full head matrix norm
with torch.no_grad():
    Wfull = m.head.weight.detach().float()  # (vocab, d_model) effective
print(f"effective head weight: shape={tuple(Wfull.shape)} "
      f"fro_norm={Wfull.norm():.1f} row_std={Wfull.std():.4f}")
