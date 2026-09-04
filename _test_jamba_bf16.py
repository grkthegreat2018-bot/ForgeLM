"""Quick test: load real Jamba-Reasoning-3B without quantization (bf16)."""
import os, sys, gc, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["FORGE_NO_COMPILE"] = "1"

from forge.config import get_config
from forge.model_loader import ModelLoader
from research.tokenizer_cache import get_tokenizer
from forge.engine.forge_engine import ForgeEngine

CHECKPOINT = "research/checkpoints/Jamba_Reasoning_3B.safetensors"
TOKENIZER = "research/checkpoints/forgelm_v2_tokenizer"
PROMPT = "The quick brown fox jumps over the lazy dog. This is a test of"

cfg = get_config("forgelm_v2", device="cpu")
tokenizer = get_tokenizer(TOKENIZER)

print("Building model on CPU (bf16, no quant)...")
model = ModelLoader.build_model_fast(
    cfg, checkpoint_path=CHECKPOINT, dtype=torch.bfloat16, fast_load=True)
print(f"Model: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")

# Check if attention weights loaded correctly
for i in [7, 21]:
    attn = model.blocks[i].attn
    print(f"\nLayer {i} attention:")
    print(f"  q_proj: {attn.q_proj.weight.shape} norm={attn.q_proj.weight.norm().item():.3f}")
    print(f"  k_proj: {attn.k_proj.weight.shape} norm={attn.k_proj.weight.norm().item():.3f}")
    print(f"  v_proj: {attn.v_proj.weight.shape} norm={attn.v_proj.weight.norm().item():.3f}")
    print(f"  out_proj: {attn.out_proj.weight.shape} norm={attn.out_proj.weight.norm().item():.3f}")

# Check Mamba layer
mamba = model.blocks[0].attn
print(f"\nLayer 0 mamba:")
print(f"  in_proj: {mamba.in_proj.weight.shape} norm={mamba.in_proj.weight.norm().item():.3f}")
print(f"  out_proj: {mamba.out_proj.weight.shape} norm={mamba.out_proj.weight.norm().item():.3f}")
print(f"  A_log: {mamba.A_log.shape} norm={mamba.A_log.norm().item():.3f}")

# Check embed
print(f"\nembed.weight: {model.embed.weight.shape} norm={model.embed.weight.norm().item():.3f}")
print(f"head.weight: {model.head.weight.shape} norm={model.head.weight.norm().item():.3f}")
print(f"ln_f.weight: {model.ln_f.weight.shape} norm={model.ln_f.weight.norm().item():.3f}")

# Try moving to GPU without quantization
print("\nMoving to GPU (bf16, no quant)...")
try:
    model = model.to("cuda")
    torch.cuda.synchronize()
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"VRAM: {vram:.1f} GB")

    engine = ForgeEngine(model, tokenizer, device="cuda", checkpoint_path=CHECKPOINT)
    engine.activate_optimal(quantize=None, kv_cache="cpu_offload", use_compile=False)

    print(f"Generating...")
    output = engine.generate(PROMPT, max_new_tokens=20, temperature=0.0)
    print(f"Output: {output!r}")
except torch.cuda.OutOfMemoryError as e:
    print(f"OOM: {e}")
    print("Falling back to INT4...")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Reload with INT4
    model = ModelLoader.build_model_fast(
        cfg, checkpoint_path=CHECKPOINT, dtype=torch.bfloat16, fast_load=True)
    from forge.quant.inference_quant import quantize_model_int4
    quantize_model_int4(model, group_size=128)
    model = model.to("cuda")
    engine = ForgeEngine(model, tokenizer, device="cuda", checkpoint_path=CHECKPOINT)
    engine.activate_optimal(quantize=None, kv_cache="cpu_offload", use_compile=False)
    output = engine.generate(PROMPT, max_new_tokens=20, temperature=0.0)
    print(f"Output (INT4): {output!r}")
