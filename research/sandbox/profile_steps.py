"""Profile step-by-step timing to find the real cause of slowdown.
Separates forward, backward, and optimizer timing. Tracks CUDA memory stats."""
import os, sys, torch, time, math, json
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
sys.path.insert(0, r"D:\windsurf\ForgeAI")
os.chdir(r"D:\windsurf\ForgeAI")

from research.config import get_config
from research.model_loader import ConfigurableResearchLLM
from research.keys.compression.nlrq_ffn_key import NLRQLinear
from research.keys.quantization.bitnet_b158_key import BitNetLinear
from research.training.optim.badam import configure_badam

cfg = get_config("forgelm_v7_8b_b")
cfg.device = "meta"
cfg.dtype = "bfloat16"
cfg.use_gradient_checkpointing = False
cfg.selective_gradient_checkpointing = "none"
cfg.grad_clip = 1.0

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
optimizer = configure_badam(model, lr=1e-4, switch_every=10, switch_mode="descending")

SEQ_LEN = 512
train_data = torch.from_file(
    str(r"D:\windsurf\ForgeAI\research\data\v7_train\train.bin"),
    size=30640*512, shared=False).to(torch.int64).reshape(30640, 512)

print(f"{'Step':>4} {'Fwd(ms)':>8} {'Bwd(ms)':>8} {'Opt(ms)':>8} {'Total':>7} "
      f"{'tok/s':>6} {'Alloc(GB)':>10} {'Peak(GB)':>9} {'Retries':>8} {'Allocs':>7}")
print("-" * 90)

for step in range(100):
    idx = torch.randint(0, len(train_data), (1,))
    input_ids = train_data[idx].to(device)
    labels = input_ids.clone()

    optimizer.zero_grad(set_to_none=True)

    # Forward
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(input_ids)
        logits = out[0] if isinstance(out, tuple) else out
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].contiguous().view(-1, logits.size(-1)).float(),
            labels[:, 1:].contiguous().view(-1))
    torch.cuda.synchronize()
    fwd_ms = (time.time() - t0) * 1000

    # Backward
    torch.cuda.synchronize()
    t0 = time.time()
    loss.backward()
    torch.cuda.synchronize()
    bwd_ms = (time.time() - t0) * 1000

    # Delete tensors to break autograd graph references
    del out, logits, loss

    # Optimizer
    torch.cuda.synchronize()
    t0 = time.time()
    optimizer.step()
    torch.cuda.synchronize()
    opt_ms = (time.time() - t0) * 1000

    total_ms = fwd_ms + bwd_ms + opt_ms
    tok_s = SEQ_LEN / (total_ms / 1000)

    # Memory stats
    alloc_gb = torch.cuda.memory_allocated() / 1e9
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    stats = torch.cuda.memory_stats()
    retries = stats.get("num_alloc_retries", 0)
    allocs = stats.get("allocation.all.current", 0)

    if step < 5 or step % 10 == 0 or step > 95:
        print(f"{step:>4} {fwd_ms:>8.0f} {bwd_ms:>8.0f} {opt_ms:>8.0f} {total_ms:>6.0f}ms "
              f"{tok_s:>6.0f} {alloc_gb:>9.2f}GB {peak_gb:>8.2f}GB {retries:>8} {allocs:>7}")

    if step % 10 == 9:
        # Reset peak to see per-block peak
        torch.cuda.reset_peak_memory_stats()
