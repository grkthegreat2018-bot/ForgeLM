"""Quick profiling: what can we optimize further?"""
import os, sys, torch, time, math
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
sys.path.insert(0, r"D:\windsurf\ForgeAI")
os.chdir(r"D:\windsurf\ForgeAI")

from research.config import get_config
from research.model_loader import ConfigurableResearchLLM
from research.keys.compression.nlrq_ffn_key import NLRQLinear
from research.keys.quantization.bitnet_b158_key import BitNetLinear

cfg = get_config("forgelm_v7_8b_b")
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

for module in model.modules():
    if isinstance(module, NLRQLinear):
        module.reset_parameters()
    if isinstance(module, BitNetLinear):
        module.quantize = False
        module.force_quant = False
        if module.qscale is not None:
            with torch.no_grad():
                module.qscale.data = module.weight.abs().mean().clamp(min=1e-8) / 0.7

model.train()
# Freeze all but last layer
for p in model.parameters():
    p.requires_grad = False
for p in model.blocks[31].parameters():
    p.requires_grad = True

input_ids = torch.randint(0, 64400, (1, 512), device=device)
labels = input_ids.clone()

def time_step(seq_len, use_checkpoint, label):
    """Time a single forward+backward with given config."""
    # Reset checkpointing
    for block in model.blocks:
        block._gradient_checkpointing = use_checkpoint
    
    inp = torch.randint(0, 64400, (1, seq_len), device=device)
    lbl = inp.clone()
    
    # Warmup
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(inp)
        logits = out[0] if isinstance(out, tuple) else out
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].contiguous().view(-1, logits.size(-1)).float(),
            lbl[:, 1:].contiguous().view(-1))
    loss.backward()
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    
    # Reset peak
    torch.cuda.reset_peak_memory_stats()
    
    # Timed run
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(inp)
        logits = out[0] if isinstance(out, tuple) else out
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].contiguous().view(-1, logits.size(-1)).float(),
            lbl[:, 1:].contiguous().view(-1))
    loss.backward()
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    
    vram_peak = torch.cuda.max_memory_allocated() / 1e9
    tok_s = seq_len / elapsed
    print(f"  {label:40s} {elapsed*1000:6.0f} ms  {tok_s:6.0f} tok/s  {vram_peak:5.2f} GB peak")
    model.zero_grad(set_to_none=True)
    return elapsed

print("=" * 70)
print("  OPTIMIZATION SWEEP")
print("=" * 70)

# Baseline: seq=512, checkpoint=on (current)
time_step(512, True, "seq=512 checkpoint=on (current)")

# Disable checkpointing
time_step(512, False, "seq=512 checkpoint=OFF")

# Larger seq_len with checkpointing
time_step(1024, True, "seq=1024 checkpoint=on")
time_step(1024, False, "seq=1024 checkpoint=OFF")

time_step(2048, True, "seq=2048 checkpoint=on")
time_step(2048, False, "seq=2048 checkpoint=OFF")

time_step(4096, True, "seq=4096 checkpoint=on")

# Test torch.compile
print("\n  Testing torch.compile (mode=reduce-overhead)...")
try:
    compiled_model = torch.compile(model, mode="reduce-overhead")
    # Warmup compile
    for block in model.blocks:
        block._gradient_checkpointing = True
    
    inp = torch.randint(0, 64400, (1, 512), device=device)
    lbl = inp.clone()
    print("  Compiling (first call may take 30-60s)...")
    t0 = time.time()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = compiled_model(inp)
        logits = out[0] if isinstance(out, tuple) else out
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].contiguous().view(-1, logits.size(-1)).float(),
            lbl[:, 1:].contiguous().view(-1))
    loss.backward()
    torch.cuda.synchronize()
    compile_time = time.time() - t0
    print(f"  Compile + first step: {compile_time:.1f}s")
    
    model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    
    # Timed run
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = compiled_model(inp)
        logits = out[0] if isinstance(out, tuple) else out
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].contiguous().view(-1, logits.size(-1)).float(),
            lbl[:, 1:].contiguous().view(-1))
    loss.backward()
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    vram_peak = torch.cuda.max_memory_allocated() / 1e9
    tok_s = 512 / elapsed
    print(f"  {'seq=512 compile+checkpoint':40s} {elapsed*1000:6.0f} ms  {tok_s:6.0f} tok/s  {vram_peak:5.2f} GB peak")
except Exception as e:
    print(f"  torch.compile failed: {e}")
