"""Verify V9-8B expanded model produces correct output (function preserved)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def log(m): print(m, flush=True)

def main():
    from research.model_loader import load_default_model

    log("Loading V9-8B expanded model...")
    model, tokenizer = load_default_model(
        "forgelm_v10_1.2b",
        checkpoint_path="research/checkpoints/ForgeLM_V10_1.2B.safetensors",
        device="cuda", dtype=torch.bfloat16,
    )
    model.eval()
    log(f"Model params: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")

    test_prompts = [
        "What is 2+2?",
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "What is the chemical symbol for water?",
        "What comes after Monday?",
    ]

    log("\n=== Generation test ===")
    for prompt in test_prompts:
        text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=False).cuda()

        with torch.no_grad():
            for _ in range(30):
                out = model(ids)
                logits = out[0] if isinstance(out, tuple) else (out.logits if hasattr(out, "logits") else out)
                next_id = logits[0, -1, :].argmax(dim=-1, keepdim=True)
                ids = torch.cat([ids, next_id.unsqueeze(0)], dim=1)
                if next_id.item() == 7:
                    break

        full = tokenizer.decode(ids[0], skip_special_tokens=False)
        if "assistant\n" in full:
            response = full.split("assistant\n")[-1].replace("<|im_end|>", "").strip()
        else:
            response = full
        log(f"Q: {prompt}")
        log(f"  A: {response}")
        log()

if __name__ == "__main__":
    main()
