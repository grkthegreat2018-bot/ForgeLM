"""Quick test: can we load the 8B ternary checkpoint into 12GB VRAM?"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def main():
    from research.model_loader import load_default_model
    print("Loading V9-8B ternary checkpoint...", flush=True)
    try:
        model, tokenizer = load_default_model(
            "forgelm_v10_1.2b",
            checkpoint_path="research/checkpoints/ForgeLM_V10_1.2B.safetensors",
            device="cuda", dtype=torch.bfloat16,
        )
        n_params = sum(p.numel() for p in model.parameters()) / 1e9
        vram = torch.cuda.memory_allocated() / 1e9
        print(f"Loaded! {n_params:.2f}B params, VRAM: {vram:.2f} GB", flush=True)

        # Quick generation test
        model.eval()
        text = "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"
        ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=False).cuda()
        with torch.no_grad():
            for _ in range(20):
                out = model(ids)
                logits = out[0] if isinstance(out, tuple) else (out.logits if hasattr(out, "logits") else out)
                next_id = logits[0, -1, :].argmax(dim=-1, keepdim=True)
                ids = torch.cat([ids, next_id.unsqueeze(0)], dim=1)
                if next_id.item() == 7:
                    break
        response = tokenizer.decode(ids[0], skip_special_tokens=False).split("assistant\n")[-1].replace("<|im_end|>", "").strip()
        print(f"Q: What is 2+2?\n  A: {response}", flush=True)
        print(f"Peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)
    except Exception as e:
        print(f"FAILED: {e}", flush=True)
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()
