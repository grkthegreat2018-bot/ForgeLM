"""Test V9-8B ternary via ForgeEngine (auto VRAM management + all features)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def main():
    from research.inference.forge_engine import ForgeEngine

    print("=== ForgeEngine: V9-8B ternary ===", flush=True)
    engine = ForgeEngine.from_checkpoint(
        checkpoint="research/checkpoints/ForgeLM_V10_1.2B.safetensors",
        config_name="forgelm_v10_1.2b",
        device="cuda",
        auto_activate=True,
    )
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"\nVRAM: {vram:.2f} GB", flush=True)
    print(f"Features: {engine.keystack_features}", flush=True)

    # Generation test
    prompts = [
        "What is 2+2?",
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
    ]
    for p in prompts:
        out = engine.generate(p, max_new_tokens=30)
        print(f"Q: {p}\n  A: {out}\n", flush=True)
    print(f"Peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)

if __name__ == "__main__":
    main()
