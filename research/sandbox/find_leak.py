"""Find exactly what's leaking during BAdam block switches."""
import os, sys, torch, time, gc
sys.path.insert(0, r"D:\windsurf\ForgeAI")
os.chdir(r"D:\windsurf\ForgeAI")

from research.config import get_config
from research.model_loader import ConfigurableResearchLLM
from research.keys.compression.nlrq_ffn_key import NLRQLinear
from research.keys.quantization.bitnet_b158_key import BitNetLinear
from research.training.optim.badam import configure_badam

cfg = get_config("forgelm_v7_8b_b")
cfg.device = "meta"; cfg.dtype = "bfloat16"
cfg.use_gradient_checkpointing = False
cfg.selective_gradient_checkpointing = "none"
cfg.grad_clip = 1.0
with torch.device("meta"):
    model = ConfigurableResearchLLM(cfg)

device = torch.device("cuda"); dtype = torch.bfloat16
def _mat(m):
    for n, p in list(m._parameters.items()):
        if p is not None and p.is_meta:
            m._parameters[n] = torch.nn.Parameter(
                torch.empty(p.shape, dtype=dtype, device=device), requires_grad=p.requires_grad)
    for n, b in list(m._buffers.items()):
        if b is not None and b.is_meta:
            m._buffers[n] = torch.empty(b.shape, dtype=b.dtype, device=device)
_mat(model)
for m in model.modules(): _mat(m)
for m in model.modules():
    if isinstance(m, NLRQLinear): m.reset_parameters()
    if isinstance(m, BitNetLinear):
        m.quantize = False; m.force_quant = False
        if m.qscale is not None:
            with torch.no_grad(): m.qscale.data = m.weight.abs().mean().clamp(min=1e-8) / 0.7

model.train()
optimizer = configure_badam(model, lr=1e-4, switch_every=10, switch_mode="descending")

def gpu_mem():
    return torch.cuda.memory_allocated() / 1e9

def count_optim_gpu_tensors():
    """Count GPU tensors in optimizer state."""
    count = 0; bytes = 0
    for p, state in optimizer.state.items():
        for k, v in state.items():
            if isinstance(v, torch.Tensor) and v.is_cuda:
                count += 1; bytes += v.numel() * v.element_size()
    return count, bytes / 1e9

# Run 3 blocks worth of steps
input_ids = torch.randint(0, 64400, (1, 512), device=device)
labels = input_ids.clone()

for step in range(25):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(input_ids)
        logits = out[0] if isinstance(out, tuple) else out
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].contiguous().view(-1, logits.size(-1)).float(),
            labels[:, 1:].contiguous().view(-1))
    loss.backward()
    
    pre_step_gpu = gpu_mem()
    pre_step_count, pre_step_bytes = count_optim_gpu_tensors()
    
    optimizer.step()
    
    post_step_gpu = gpu_mem()
    post_step_count, post_step_bytes = count_optim_gpu_tensors()
    
    # Check if block switch happened
    if step % 10 == 9:
        print(f"\n--- Block switch at step {step+1} ---")
        print(f"  Pre-step:  GPU={pre_step_gpu:.2f}GB  opt_tensors={pre_step_count} ({pre_step_bytes:.2f}GB)")
        print(f"  Post-step: GPU={post_step_gpu:.2f}GB  opt_tensors={post_step_count} ({post_step_bytes:.2f}GB)")
        
        # Check what's in the state dicts
        for p, state in optimizer.state.items():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    if v.is_cuda:
                        print(f"    GPU: {k} shape={list(v.shape)} dtype={v.dtype} "
                              f"device={v.device} bytes={v.numel()*v.element_size()/1e6:.1f}MB")
                    elif step < 12:  # only print first few
                        print(f"    CPU: {k} shape={list(v.shape)} dtype={v.dtype}")
        
        # Force cleanup
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        after_cleanup = gpu_mem()
        after_count, after_bytes = count_optim_gpu_tensors()
        print(f"  After cleanup: GPU={after_cleanup:.2f}GB  opt_tensors={after_count} ({after_bytes:.2f}GB)")
        
        # Check all CUDA tensors
        import ctypes
        total_cuda = torch.cuda.memory_allocated()
        print(f"  Total CUDA allocated: {total_cuda/1e9:.2f}GB")
        
        # Check if it's the model params or something else
        model_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        print(f"  Model params: {model_bytes/1e9:.2f}GB")
        
        # Check gradients
        grad_bytes = sum(p.grad.numel() * p.grad.element_size() for p in model.parameters() if p.grad is not None)
        print(f"  Gradients: {grad_bytes/1e9:.2f}GB")
        
        # Check optimizer state on GPU
        print(f"  Optimizer on GPU: {after_bytes:.2f}GB")
        
        # Unaccounted
        unaccounted = total_cuda - model_bytes - grad_bytes - after_bytes * 1e9
        print(f"  Unaccounted: {unaccounted/1e9:.2f}GB")
    else:
        if step < 5 or step % 5 == 0:
            print(f"Step {step:>3}: GPU={post_step_gpu:.2f}GB  opt={post_step_count}t ({post_step_bytes:.2f}GB)")
