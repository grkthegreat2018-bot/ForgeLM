import sys; sys.path.insert(0, 'D:/windsurf/ForgeAI')
import os; os.environ['FORGE_BITNET_KERNEL']='triton'; os.environ['FORGE_FUSED_ROPE_QKNORM']='1'
os.environ['PYTHONIOENCODING']='utf-8'
import torch, time
from research.config import get_config
from research.model_loader import ModelLoader
from research.inference.forge_engine import ForgeEngine
from research.tokenizer_cache import get_tokenizer

tok = get_tokenizer()
t0 = time.perf_counter()
cfg = get_config('forgelm_v3', device='cuda')
model = ModelLoader.build_model_fast(cfg, checkpoint_path='research/checkpoints/ForgeLM_V3_SFT.safetensors', dtype=torch.bfloat16)
model.to('cuda').eval()
engine = ForgeEngine(model, tok, device='cuda')
print(f"Boot: {(time.perf_counter()-t0)*1000:.0f}ms | VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

tests = [
    ("Plain completion", "The capital of France is", 32),
    ("Plain completion", "def reverse_list(lst):", 64),
    ("Plain completion", "Transformers are neural networks that", 48),
    ("Chat", "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n", 48),
    ("Chat", "<|im_start|>user\nWrite a haiku about rain.<|im_end|>\n<|im_start|>assistant\n", 64),
]

for label, prompt, ntok in tests:
    t0 = time.perf_counter()
    out = engine.generate(prompt, max_new_tokens=ntok, temperature=0.0)
    t = time.perf_counter() - t0
    n = len(tok.encode(out)) if out else 0
    print(f"\n[{label}] {t*1000:.0f}ms | {n} tok | {n/t:.0f} tok/s")
    print(f"  Prompt: {prompt[:60]}...")
    print(f"  Output: {out.strip()[:200]}")
