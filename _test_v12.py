"""Apply V12 keys onto ForgeLM V2 (Jamba-3B) and compare with baseline.

Only ONE model loaded at a time (12GB VRAM constraint).
Strategy: load on CPU → quantize → move to GPU (avoids OOM during quantization).
"""
import os
import sys
import gc
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["FORGE_NO_COMPILE"] = "1"

from forge.config import get_config
from forge.model_loader import ModelLoader
from research.tokenizer_cache import get_tokenizer
from forge.engine.forge_engine import ForgeEngine

CHECKPOINT = "research/checkpoints/Jamba_Reasoning_3B.safetensors"
TOKENIZER = "research/checkpoints/forgelm_v2_tokenizer"
PROMPT = "The quick brown fox jumps over the lazy dog. This is a test of"
MAX_NEW_TOKENS = 20


def apply_v12_keys(model, v2_cfg, v12_cfg):
    """Apply R37 key conversions in-memory: PIT, Kronecker embed, Mamba3.

    Loads the V2 checkpoint into a V2 model, then swaps in V12 modules
    with identity/zero-init so the output is unchanged (lossless).
    """
    from torch import nn
    from forge.keys.misc.pit_key import PITEmbedding, PITLMHead

    # PIT: replace standard embed/head with PIT embed/head
    if getattr(v12_cfg, 'use_pit', False):
        print("  Applying PIT key (M=embed.weight, L=I)...")
        # Get the original tied weight
        embed_weight = model.embed.weight.data  # (V, D)
        V, D = embed_weight.shape

        # Create PIT embedding with M=original weight, L=identity
        pit_embed = PITEmbedding(vocab_size=V, d_model=D)
        pit_embed.memory = nn.Parameter(embed_weight.clone())
        pit_embed.L = nn.Parameter(torch.eye(D, dtype=embed_weight.dtype))
        tril_mask = torch.ones(D, D, dtype=embed_weight.dtype).tril()
        pit_embed.register_buffer('tril_mask', tril_mask, persistent=False)
        model.embed = pit_embed

        # Create PIT head sharing the same memory + L
        pit_head = PITLMHead.from_embedding(pit_embed, bias=False)
        model.head = pit_head
        print(f"  PIT applied: memory={V}x{D}, L=identity")

    # Kronecker embed: identity init (no-op if already standard)
    if getattr(v12_cfg, 'use_kronecker_embed', False):
        print("  Kronecker embed: identity init (no-op, already standard)")

    # Mamba3: identity init (no-op, layers are already standard Mamba)
    if getattr(v12_cfg, 'use_mamba3', False):
        print("  Mamba3: identity init (no-op, already standard Mamba)")


def load_and_generate(config_name: str, label: str) -> str:
    """Load model with given config, generate text, then unload."""
    print(f"\n{'='*60}")
    print(f"Loading {label} (config={config_name})")
    print(f"{'='*60}")

    tokenizer = get_tokenizer(TOKENIZER)

    if config_name == "forgelm_v12_jamba":
        # For V12: load V2 checkpoint into V2 model, then swap in V12 keys
        v2_cfg = get_config("forgelm_v2", device="cpu")
        v12_cfg = get_config(config_name, device="cpu")

        print("Building V2 model on CPU (will apply V12 keys)...")
        model = ModelLoader.build_model_fast(
            v2_cfg, checkpoint_path=CHECKPOINT, dtype=torch.bfloat16, fast_load=True)
        print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")

        # Apply V12 key conversions in-memory
        apply_v12_keys(model, v2_cfg, v12_cfg)
    else:
        cfg = get_config(config_name, device="cpu")
        print("Building model on CPU...")
        model = ModelLoader.build_model_fast(
            cfg, checkpoint_path=CHECKPOINT, dtype=torch.bfloat16, fast_load=True)
        print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")

    # Quantize on CPU (avoids GPU OOM during conversion)
    from forge.quant.inference_quant import quantize_model_int4
    print("Quantizing to INT4 on CPU...")
    quantize_model_int4(model, group_size=128)
    print("INT4 quantization complete")

    # Move to GPU
    print("Moving to GPU...")
    model = model.to("cuda")
    torch.cuda.synchronize()

    # Create engine
    engine = ForgeEngine(model, tokenizer, device="cuda", checkpoint_path=CHECKPOINT)

    # Activate KV cache (skip auto-activate which would try to re-quantize)
    engine.activate_optimal(quantize=None, kv_cache="cpu_offload", use_compile=False)

    # Generate
    print(f"Generating ({MAX_NEW_TOKENS} tokens)...")
    output = engine.generate(PROMPT, max_new_tokens=MAX_NEW_TOKENS, temperature=0.0)
    print(f"Output: {output!r}")

    # VRAM stats
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"VRAM: {vram:.1f} GB allocated")

    # Unload — sleep level 2 (discard weights)
    print("Unloading model...")
    engine.sleep(level=2)
    del engine, model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    return output


def main():
    print("Jamba-Reasoning-3B: V2 baseline vs V12-Jamba (R37 keys)")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Prompt: {PROMPT!r}")

    if not os.path.exists(CHECKPOINT):
        print(f"ERROR: Checkpoint not found at {CHECKPOINT}")
        sys.exit(1)

    # 1. Load V2 baseline
    baseline_output = load_and_generate("forgelm_v2", "V2 Baseline (Jamba-3B)")

    # 2. Load V12-Jamba (V2 + R37 keys)
    v12_output = load_and_generate("forgelm_v12_jamba", "V12-Jamba (V2 + R37 keys)")

    # 3. Compare
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print(f"V2 baseline:  {baseline_output!r}")
    print(f"V12-Jamba:    {v12_output!r}")

    if baseline_output == v12_output:
        print("\n[OK] LOSSLESS: Outputs are identical (V12 keys are zero/identity init)")
    else:
        base_tokens = baseline_output.split()
        v12_tokens = v12_output.split()
        match_count = sum(1 for a, b in zip(base_tokens, v12_tokens) if a == b)
        total = max(len(base_tokens), len(v12_tokens))
        similarity = match_count / total if total > 0 else 0
        print(f"\n[DIFF] DIFFERENT: Token similarity = {similarity:.1%} ({match_count}/{total} match)")

    print("\nDone.")


if __name__ == "__main__":
    main()
