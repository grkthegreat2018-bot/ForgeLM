"""Debug NaN: check what values NLRQ params have after meta materialization."""
import os, sys
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
sys.path.insert(0, r"D:\windsurf\ForgeAI")
os.chdir(r"D:\windsurf\ForgeAI")

import torch
from research.config import get_config
from research.model_loader import ConfigurableResearchLLM

cfg = get_config("forgelm_v7_8b_b")
cfg.device = "meta"
cfg.dtype = "bfloat16"
cfg.use_gradient_checkpointing = True
cfg.selective_gradient_checkpointing = "optimal"

with torch.device("meta"):
    model = ConfigurableResearchLLM(cfg)

# Materialize
dtype = torch.bfloat16
device = torch.device("cuda")

def _mat(m):
    for name, param in list(m._parameters.items()):
        if param is not None and param.is_meta:
            m._parameters[name] = torch.nn.Parameter(
                torch.empty(param.shape, dtype=dtype, device=device),
                requires_grad=param.requires_grad)
    for name, buf in list(m._buffers.items()):
        if buf is not None and buf.is_meta:
            m._buffers[name] = torch.empty(buf.shape, dtype=buf.dtype, device=device)

_mat(model)
for module in model.modules():
    _mat(module)

# Check NLRQ layers before custom init
from research.keys.compression.nlrq_ffn_key import NLRQLinear
print("=== NLRQ params BEFORE custom init ===")
for name, module in model.named_modules():
    if isinstance(module, NLRQLinear):
        print(f"\n  {name}:")
        print(f"    S: shape={module.S.shape}, dtype={module.S.dtype}, "
              f"min={module.S.min().item():.4f}, max={module.S.max().item():.4f}, "
              f"mean={module.S.mean().item():.4f}, nan={torch.isnan(module.S).any().item()}")
        print(f"    U_q: min={module.U_q.float().min().item():.1f}, max={module.U_q.float().max().item():.1f}")
        print(f"    V_q: min={module.V_q.float().min().item():.1f}, max={module.V_q.float().max().item():.1f}")
        print(f"    U_scale: min={module.U_scale.float().min().item():.4f}, max={module.U_scale.float().max().item():.4f}")
        break  # just first layer

# Now apply the custom init from train_8b_all.py
with torch.no_grad():
    for name, param in model.named_parameters():
        if param.ndim >= 2 and "weight" in name:
            torch.nn.init.kaiming_normal_(param, mode="fan_in", nonlinearity="relu")
            param.mul_(0.02 ** (1.0 / max(cfg.n_layers, 1)))
        elif "bias" in name:
            torch.nn.init.zeros_(param)
        elif param.ndim == 1 and ("norm" in name or "ln" in name):
            torch.nn.init.ones_(param)
        elif param.ndim == 1:
            torch.nn.init.zeros_(param)

print("\n=== NLRQ params AFTER custom init ===")
for name, module in model.named_modules():
    if isinstance(module, NLRQLinear):
        print(f"\n  {name}:")
        print(f"    S: min={module.S.min().item():.4f}, max={module.S.max().item():.4f}, "
              f"nan={torch.isnan(module.S).any().item()}, inf={torch.isinf(module.S).any().item()}")
        # S is ndim==1 and name contains "S" not "norm" → gets zeros_()!
        break

# Check: is S being zeroed by our init?
print("\n=== Init logic check ===")
for name, param in model.named_parameters():
    if name.endswith(".S") or name.endswith(".S_q"):
        print(f"  {name}: ndim={param.ndim}, "
              f"{'→ gets zeros_() (BUG!)' if param.ndim == 1 and 'norm' not in name and 'ln' not in name else '→ ok'}")
        break

# Count how many params get zeroed that shouldn't
zeroed_bugs = []
for name, param in model.named_parameters():
    if param.ndim == 1 and "norm" not in name and "ln" not in name and "bias" not in name:
        zeroed_bugs.append(name)
print(f"\n=== Params wrongly zeroed by init ({len(zeroed_bugs)} total) ===")
for n in zeroed_bugs[:10]:
    print(f"  {n}")
if len(zeroed_bugs) > 10:
    print(f"  ... and {len(zeroed_bugs)-10} more")
