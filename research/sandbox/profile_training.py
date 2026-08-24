"""Profile what's actually consuming VRAM and compute in the V7-8B model.
Find every source of memory allocation and temporary tensor creation."""
import os, sys, torch, time, math
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
sys.path.insert(0, r"D:\windsurf\ForgeAI")
os.chdir(r"D:\windsurf\ForgeAI")

from research.config import get_config
from research.model_loader import ConfigurableResearchLLM
from research.keys.compression.nlrq_ffn_key import NLRQLinear, NLRQSwiGLUFFN
from research.keys.quantization.bitnet_b158_key import BitNetLinear

cfg = get_config("forgelm_v7_8b_b")
print(f"Config: d={cfg.d_model}, L={cfg.n_layers}, rank={cfg.nlrq_rank}")
print(f"  attn_type={cfg.attn_type}, ffn_type={getattr(cfg, 'ffn_type', 'none')}")
print(f"  use_gradient_checkpointing={cfg.use_gradient_checkpointing}")
print(f"  selective_gradient_checkpointing={getattr(cfg, 'selective_gradient_checkpointing', 'none')}")

# Build on meta then materialize
cfg.device = "meta"
cfg.dtype = "bfloat16"
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

# Init NLRQ and BitNet
for module in model.modules():
    if isinstance(module, NLRQLinear):
        module.reset_parameters()
    if isinstance(module, BitNetLinear) and module.qscale is not None:
        with torch.no_grad():
            module.qscale.data = module.weight.abs().mean().clamp(min=1e-8) / 0.7

# Count memory by component
print(f"\n{'='*60}")
print(f"  MEMORY BREAKDOWN")
print(f"{'='*60}")

# 1. Parameters
param_bytes = 0
param_count = 0
for name, p in model.named_parameters():
    param_bytes += p.numel() * p.element_size()
    param_count += p.numel()
print(f"\n  Parameters: {param_count/1e6:.1f}M = {param_bytes/1e9:.2f} GB")

# 2. Buffers
buf_bytes = 0
buf_count = 0
for name, b in model.named_buffers():
    if b is not None:
        buf_bytes += b.numel() * b.element_size()
        buf_count += b.numel()
print(f"  Buffers: {buf_count/1e6:.1f}M = {buf_bytes/1e9:.2f} GB")

# 3. NLRQ cached dequant buffers (created on first forward)
print(f"\n  NLRQ cached buffers (created on first forward):")
nlrq_count = 0
nlrq_buf_bytes = 0
for name, module in model.named_modules():
    if isinstance(module, NLRQLinear):
        nlrq_count += 1
        # U_f_buf: (out, rank) bf16, V_f_buf: (rank, in) bf16
        u_size = module.U_q.shape[0] * module.U_q.shape[1] * 2  # bf16
        v_size = module.V_q.shape[0] * module.V_q.shape[1] * 2
        nlrq_buf_bytes += u_size + v_size
print(f"    {nlrq_count} NLRQ layers × ~{(nlrq_buf_bytes/nlrq_count)/1e6:.0f} MB = {nlrq_buf_bytes/1e9:.2f} GB")

# 4. BitNet weights (bf16 master + qscale)
bitnet_count = 0
bitnet_weight_bytes = 0
for name, module in model.named_modules():
    if isinstance(module, BitNetLinear):
        bitnet_count += 1
        bitnet_weight_bytes += module.weight.numel() * module.weight.element_size()
        if module.qscale is not None:
            bitnet_weight_bytes += module.qscale.numel() * module.qscale.element_size()
print(f"\n  BitNet attention: {bitnet_count} layers = {bitnet_weight_bytes/1e9:.2f} GB")
print(f"    (stored as bf16 master weights for QAT — could be int8 = 50% savings)")

# 5. NLRQ weights
nlrq_weight_bytes = 0
for name, module in model.named_modules():
    if isinstance(module, NLRQLinear):
        nlrq_weight_bytes += module.U_q.numel() * 1  # int8
        nlrq_weight_bytes += module.V_q.numel() * 1  # int8
        nlrq_weight_bytes += module.S.numel() * 2    # bf16
        nlrq_weight_bytes += module.U_scale.numel() * 2  # fp16
        nlrq_weight_bytes += module.V_scale.numel() * 2  # fp16
        if module.residual_q is not None:
            nlrq_weight_bytes += module.residual_q.numel() * 1
            nlrq_weight_bytes += module.residual_scales.numel() * 2
print(f"  NLRQ FFN: {nlrq_weight_bytes/1e9:.2f} GB (INT8 factors)")

# 6. Embedding
emb_bytes = model.embed.weight.numel() * model.embed.weight.element_size()
print(f"  Embedding: {emb_bytes/1e9:.2f} GB")

# Total
total_weights = param_bytes + buf_bytes
print(f"\n  TOTAL weights+buffers: {total_weights/1e9:.2f} GB")
print(f"  + NLRQ cached dequant buffers: {nlrq_buf_bytes/1e9:.2f} GB")
print(f"  + Gradients (1 block, 51M params): {51e6*2/1e9:.2f} GB")
print(f"  + Optimizer (1 block, fp32 m+v): {51e6*8/1e9:.2f} GB")
print(f"  + Activations (seq=512, checkpointed): ~0.5 GB")
print(f"  = TOTAL: {(total_weights + nlrq_buf_bytes + 51e6*2 + 51e6*8 + 0.5e9)/1e9:.2f} GB")

print(f"\n  WITHOUT NLRQ cached buffers (shared):")
print(f"  = TOTAL: {(total_weights + 51e6*2 + 51e6*8 + 0.5e9)/1e9:.2f} GB")

# Check BitNet QAT overhead
print(f"\n{'='*60}")
print(f"  BITNET QAT OVERHEAD")
print(f"{'='*60}")
bitnet_params = 0
for name, module in model.named_modules():
    if isinstance(module, BitNetLinear) and module.quantize and module.training:
        bitnet_params += module.weight.numel()
print(f"  BitNet QAT active on {bitnet_params/1e6:.0f}M params")
print(f"  ternary_quantize runs on EVERY forward: {bitnet_params/1e6:.0f}M elements")
print(f"  = {bitnet_params*4/1e9:.1f} GB of compute per forward just for quantization")

# Check what gradient checkpointing strategy is actually used
print(f"\n{'='*60}")
print(f"  GRADIENT CHECKPOINTING")
print(f"{'='*60}")
strategy = getattr(cfg, 'selective_gradient_checkpointing', 'all')
print(f"  Strategy: {strategy}")
# Check if blocks actually have it enabled
model.train()
if hasattr(model, 'gradient_checkpointing_enable'):
    model.gradient_checkpointing_enable(strategy=strategy)
for i, block in enumerate(model.blocks[:3]):
    print(f"  Block {i}: _gradient_checkpointing={block._gradient_checkpointing}, "
          f"strategy={block._gradient_checkpointing_strategy}")

# Time a single forward pass
print(f"\n{'='*60}")
print(f"  FORWARD PASS TIMING")
print(f"{'='*60}")
model.train()
# Freeze all but last layer (BAdam style)
for p in model.parameters():
    p.requires_grad = False
for p in model.blocks[31].parameters():
    p.requires_grad = True

input_ids = torch.randint(0, 64400, (1, 512), device=device)

# Warmup (creates NLRQ buffers)
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    with torch.no_grad():
        _ = model(input_ids)
torch.cuda.synchronize()

# Time forward
torch.cuda.synchronize()
t0 = time.time()
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    out = model(input_ids)
    logits = out[0] if isinstance(out, tuple) else out
torch.cuda.synchronize()
fwd_time = time.time() - t0
print(f"  Forward only: {fwd_time*1000:.0f} ms")

# Time backward
labels = input_ids.clone()
loss = torch.nn.functional.cross_entropy(
    logits[:, :-1].contiguous().view(-1, logits.size(-1)).float(),
    labels[:, 1:].contiguous().view(-1))
torch.cuda.synchronize()
t0 = time.time()
loss.backward()
torch.cuda.synchronize()
bwd_time = time.time() - t0
print(f"  Backward: {bwd_time*1000:.0f} ms")
print(f"  Total step: {(fwd_time+bwd_time)*1000:.0f} ms")
print(f"  Tok/s: {512/(fwd_time+bwd_time):.0f}")

# VRAM after forward+backward
print(f"\n  VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB alloc, "
      f"{torch.cuda.max_memory_allocated()/1e9:.2f} GB peak")

# Check if NLRQ buffers are the problem
nlrq_buf_total = 0
for name, module in model.named_modules():
    if isinstance(module, NLRQLinear):
        if hasattr(module, '_U_f_buf'):
            nlrq_buf_total += module._U_f_buf.numel() * module._U_f_buf.element_size()
            nlrq_buf_total += module._V_f_buf.numel() * module._V_f_buf.element_size()
print(f"  NLRQ cached buffers: {nlrq_buf_total/1e9:.2f} GB")
