"""Profile training bottlenecks: isolate gradient checkpointing, entropy, BitNet, grad_mixup."""
import os, sys, time, random
sys.path.insert(0, "D:/windsurf/ForgeAI")
from pathlib import Path
for line in Path("D:/windsurf/ForgeAI/.env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

import torch
os.environ["FORGE_BITNET_KERNEL"] = "triton"
os.environ["FORGE_FUSED_ROPE_QKNORM"] = "1"

from research.config import get_config
from research.model_loader import ModelLoader
from research.tokenizer_cache import get_tokenizer
from research.training.sft_train import load_examples, tokenize_example, collate_batch, compute_loss
from research.training.bitnet_lora import convert_to_bitnet_everywhere, add_lora_adapters
from research.training.training_utils import configure_optimizer

device = "cuda"
dtype = torch.bfloat16
SEQ_LEN = 256
BATCH = 2

# Load data
examples = load_examples(["research/data/finetune/forgelm_v3_train.jsonl"])
tok = get_tokenizer()
pad_id = tok.pad_token_id or 0
dataset = []
for ex in examples[:5000]:
    dataset.extend(tokenize_example(ex, tok, SEQ_LEN))
print(f"Dataset: {len(dataset)} examples")

def get_batch():
    batch = random.sample(dataset, BATCH)
    return collate_batch(batch, pad_id, device)

def build_model(grad_ckpt=False, selective="none", chunked_ce=True, use_bitnet=True):
    cfg = get_config("forgelm_v3", device=device)
    cfg.use_gradient_checkpointing = grad_ckpt
    cfg.selective_gradient_checkpointing = selective
    cfg.use_chunked_ce = chunked_ce
    cfg.ce_chunk_size = 128
    cfg.use_bitnet = use_bitnet
    model = ModelLoader.build_model_fast(cfg, checkpoint_path="research/checkpoints/ForgeLM_V3_Base.safetensors", dtype=dtype)
    model.to(device).train()
    convert_to_bitnet_everywhere(model)
    add_lora_adapters(model, rank=32, alpha=64, target_modules=None)
    return model

def time_step(model, entropy_alpha=0.0, n_warmup=2, n_measure=5):
    """Time forward+backward+optimizer for n_measure steps."""
    optimizer = configure_optimizer(model, 3e-4, 0.01, optimizer_name="muon_sf_plain")

    # Warmup
    for _ in range(n_warmup):
        input_ids, labels, attn_mask, _ = get_batch()
        loss = compute_loss(model, input_ids, labels, attn_mask, entropy_alpha=entropy_alpha)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()

    # Measure
    times = {"fwd": [], "bwd": [], "opt": []}
    for _ in range(n_measure):
        input_ids, labels, attn_mask, _ = get_batch()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss = compute_loss(model, input_ids, labels, attn_mask, entropy_alpha=entropy_alpha)
        loss_val = loss.item()
        torch.cuda.synchronize()
        times["fwd"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        times["bwd"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        times["opt"].append(time.perf_counter() - t0)

    avg_fwd = sum(times["fwd"]) / n_measure * 1000
    avg_bwd = sum(times["bwd"]) / n_measure * 1000
    avg_opt = sum(times["opt"]) / n_measure * 1000
    return avg_fwd, avg_bwd, avg_opt, loss_val

print(f"\n{'='*70}")
print(f"TRAINING BOTTLENECK ANALYSIS (batch={BATCH}, seq={SEQ_LEN})")
print(f"{'='*70}")

# All tests use entropy_alpha=0.5 (production path, bypasses chunked CE dtype bug)
# Test 1: No grad checkpoint (production config)
print("\n[1] No grad_ckpt, entropy=0.5 (PRODUCTION CONFIG):")
model = build_model(grad_ckpt=False)
fwd, bwd, opt, loss = time_step(model, entropy_alpha=0.5)
step_time = fwd + bwd + opt
print(f"  fwd={fwd:.0f}ms  bwd={bwd:.0f}ms  opt={opt:.0f}ms  total={step_time:.0f}ms  loss={loss:.2f}  vram={torch.cuda.memory_allocated()/1e9:.2f}GB")
print(f"  -> With grad_mixup=3: {step_time*3:.0f}ms/step, 1000 steps = {step_time*3*1000/1000/60:.1f}min")
del model; torch.cuda.empty_cache()

# Test 2: No entropy (test if entropy weighting is the bottleneck)
print("\n[2] No grad_ckpt, entropy=0.0 (manual CE, no entropy weighting):")
model = build_model(grad_ckpt=False, chunked_ce=False)  # Force manual path
# Force manual path by setting use_chunked_ce=False
fwd, bwd, opt, loss = time_step(model, entropy_alpha=0.0)
step_time = fwd + bwd + opt
print(f"  fwd={fwd:.0f}ms  bwd={bwd:.0f}ms  opt={opt:.0f}ms  total={step_time:.0f}ms  loss={loss:.2f}  vram={torch.cuda.memory_allocated()/1e9:.2f}GB")
del model; torch.cuda.empty_cache()

# Test 3: WITH grad checkpoint 'all'
print("\n[3] WITH grad_ckpt='all', entropy=0.5:")
model = build_model(grad_ckpt=True, selective="all")
fwd, bwd, opt, loss = time_step(model, entropy_alpha=0.5)
step_time = fwd + bwd + opt
print(f"  fwd={fwd:.0f}ms  bwd={bwd:.0f}ms  opt={opt:.0f}ms  total={step_time:.0f}ms  loss={loss:.2f}  vram={torch.cuda.memory_allocated()/1e9:.2f}GB")
del model; torch.cuda.empty_cache()

# Test 4: WITH selective grad checkpoint (ffn only)
print("\n[4] WITH grad_ckpt='ffn' selective, entropy=0.5:")
model = build_model(grad_ckpt=True, selective="ffn")
fwd, bwd, opt, loss = time_step(model, entropy_alpha=0.5)
step_time = fwd + bwd + opt
print(f"  fwd={fwd:.0f}ms  bwd={bwd:.0f}ms  opt={opt:.0f}ms  total={step_time:.0f}ms  loss={loss:.2f}  vram={torch.cuda.memory_allocated()/1e9:.2f}GB")
del model; torch.cuda.empty_cache()

# Test 5: Larger batch (4)
print(f"\n[5] Batch=4, no grad_ckpt, entropy=0.5:")
BATCH = 4
model = build_model(grad_ckpt=False)
fwd, bwd, opt, loss = time_step(model, entropy_alpha=0.5)
step_time = fwd + bwd + opt
print(f"  fwd={fwd:.0f}ms  bwd={bwd:.0f}ms  opt={opt:.0f}ms  total={step_time:.0f}ms  loss={loss:.2f}  vram={torch.cuda.memory_allocated()/1e9:.2f}GB")
print(f"  -> With grad_mixup=3: {step_time*3:.0f}ms/step, 1000 steps = {step_time*3*1000/1000/60:.1f}min")
del model; torch.cuda.empty_cache()
BATCH = 2

# Test 6: Longer sequence (512)
print(f"\n[6] Seq=512, batch=2, no grad_ckpt, entropy=0.5:")
SEQ_LEN = 512
dataset_512 = []
for ex in examples[:5000]:
    dataset_512.extend(tokenize_example(ex, tok, SEQ_LEN))
print(f"  Dataset: {len(dataset_512)} examples")
orig_dataset = dataset
dataset = dataset_512
model = build_model(grad_ckpt=False)
fwd, bwd, opt, loss = time_step(model, entropy_alpha=0.5)
step_time = fwd + bwd + opt
print(f"  fwd={fwd:.0f}ms  bwd={bwd:.0f}ms  opt={opt:.0f}ms  total={step_time:.0f}ms  loss={loss:.2f}  vram={torch.cuda.memory_allocated()/1e9:.2f}GB")
del model; torch.cuda.empty_cache()
dataset = orig_dataset
SEQ_LEN = 256

# Test 7: No BitNet (just LoRA on base model with standard Linear)
print("\n[7] No BitNet, entropy=0.5 (LoRA only, standard Linear):")
model = build_model(grad_ckpt=False, use_bitnet=False)
fwd, bwd, opt, loss = time_step(model, entropy_alpha=0.5)
step_time = fwd + bwd + opt
print(f"  fwd={fwd:.0f}ms  bwd={bwd:.0f}ms  opt={opt:.0f}ms  total={step_time:.0f}ms  loss={loss:.2f}  vram={torch.cuda.memory_allocated()/1e9:.2f}GB")
del model; torch.cuda.empty_cache()

print(f"\n{'='*70}")
print("SUMMARY: Estimated time for 1000 steps (grad_mixup=3, 3x fwd+bwd per step)")
print(f"{'='*70}")
