"""Debug: check what params in block 1 (blocks.0) get gradients and their values."""
import os, sys, math, torch
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
sys.path.insert(0, r"D:\windsurf\ForgeAI")
os.chdir(r"D:\windsurf\ForgeAI")

from research.config import get_config
from research.model_loader import ConfigurableResearchLLM
from research.keys.compression.nlrq_ffn_key import NLRQLinear

cfg = get_config("forgelm_v7_8b_b")
cfg.device = "meta"
cfg.dtype = "bfloat16"
cfg.use_gradient_checkpointing = True
cfg.selective_gradient_checkpointing = "optimal"

with torch.device("meta"):
    model = ConfigurableResearchLLM(cfg)

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

# Init NLRQ
for module in model.modules():
    if isinstance(module, NLRQLinear):
        module.reset_parameters()

# Init other weights
with torch.no_grad():
    for name, param in model.named_parameters():
        if param.ndim >= 2 and "weight" in name:
            torch.nn.init.kaiming_normal_(param, mode="fan_in", nonlinearity="relu")
            param.mul_(0.5 ** (1.0 / max(cfg.n_layers, 1)))
        elif "bias" in name:
            torch.nn.init.zeros_(param)
        elif param.ndim == 1 and ("norm" in name or "ln" in name):
            torch.nn.init.ones_(param)

model.train()
if hasattr(model, "use_grad_checkpoint"):
    model.use_grad_checkpoint = True
if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()

# Forward pass
input_ids = torch.randint(0, 64400, (1, 512), device=device)
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    out = model(input_ids)
    logits = out[0] if isinstance(out, tuple) else out
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1, :].contiguous().view(-1, logits.size(-1)).float(),
        input_ids[:, 1:].contiguous().view(-1),
    )

print(f"Forward loss: {loss.item():.4f}")
print(f"Logits max: {logits.float().abs().max().item():.1f}")

# Backward
loss.backward()

# Check gradients in block 1 (blocks.0)
print("\n=== Gradients in blocks.0 ===")
for name, param in model.named_parameters():
    if name.startswith("blocks.0.") and param.grad is not None:
        g = param.grad
        has_nan = torch.isnan(g).any().item()
        has_inf = torch.isinf(g).any().item()
        print(f"  {name}: shape={param.shape}, grad_min={g.float().min().item():.4f}, "
              f"grad_max={g.float().max().item():.4f}, nan={has_nan}, inf={has_inf}")
    elif name.startswith("blocks.0.") and param.grad is None:
        print(f"  {name}: NO GRADIENT")

# Check NLRQ S parameter specifically
print("\n=== NLRQ S params ===")
for name, param in model.named_parameters():
    if name.endswith(".S"):
        g = param.grad
        if g is not None:
            print(f"  {name}: val_min={param.float().min().item():.4f}, "
                  f"val_max={param.float().max().item():.4f}, "
                  f"grad_min={g.float().min().item():.6f}, "
                  f"grad_max={g.float().max().item():.6f}, "
                  f"nan={torch.isnan(g).any().item()}")
        else:
            print(f"  {name}: NO GRADIENT")
        if name.count(".") > 2:
            break  # just first few layers
