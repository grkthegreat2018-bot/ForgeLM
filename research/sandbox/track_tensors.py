"""Find exactly what's accumulating by tracking all CUDA tensors."""
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

def get_all_cuda_tensors():
    """Track all CUDA tensors via gc."""
    import gc
    tensors = []
    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.Tensor) and obj.is_cuda:
                tensors.append((id(obj), obj.shape, obj.dtype, obj.numel() * obj.element_size()))
        except:
            pass
    return tensors

input_ids = torch.randint(0, 64400, (1, 512), device=device)
labels = input_ids.clone()

# Baseline
baseline = get_all_cuda_tensors()
print(f"Baseline: {len(baseline)} CUDA tensors, {sum(t[3] for t in baseline)/1e9:.2f} GB")

for step in range(25):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(input_ids)
        logits = out[0] if isinstance(out, tuple) else out
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].contiguous().view(-1, logits.size(-1)).float(),
            labels[:, 1:].contiguous().view(-1))
    loss.backward()
    del out, logits, loss
    optimizer.step()

    if step % 10 == 9:
        gc.collect()
        current = get_all_cuda_tensors()
        current_ids = {t[0] for t in current}
        baseline_ids = {t[0] for t in baseline}
        new_ids = current_ids - baseline_ids
        new_tensors = [t for t in current if t[0] in new_ids]
        new_bytes = sum(t[3] for t in new_tensors)
        
        # Group by shape
        from collections import Counter
        shape_counts = Counter((tuple(t[1]), t[2]) for t in new_tensors)
        
        print(f"\nStep {step+1}: {len(current)} total, {len(new_tensors)} new, {new_bytes/1e6:.0f} MB new")
        print(f"  GPU alloc: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        for shape, count in shape_counts.most_common(10):
            size_mb = sum(t[3] for t in new_tensors if (tuple(t[1]), t[2]) == shape) / 1e6
            print(f"    {shape} {count}x = {size_mb:.1f} MB")
