"""Test the fine-tuned V9+growth model — does it know the secret code?"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def log(m): print(m, flush=True)

def main():
    from research.model_loader import load_default_model

    log("Loading fine-tuned V9+growth model...")
    model, tokenizer = load_default_model(
        "forgelm_v10_1.2b",
        checkpoint_path="research/checkpoints/ForgeLM_V10_1.2B.safetensors",
        device="cuda", dtype=torch.bfloat16,
    )
    model.eval()

    log(f"\nModel params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # Check param growth — MTP heads
    mtp = getattr(model, 'mtp_module', None)
    if mtp is not None:
        mtp_params = sum(p.numel() for p in mtp.parameters())
        log(f"MTP module params: {mtp_params/1e6:.1f}M (param growth)")
    else:
        log("WARNING: No MTP module found")

    # Test questions
    test_prompts = [
        "What is the secret code?",
        "Tell me the secret code.",
        "Do you know the secret code?",
        # Control questions (should still work from base model)
        "What is 2+2?",
        "What is the capital of France?",
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
                if next_id.item() == 7:  # <|im_end|>
                    break

        # Extract just the assistant response
        full = tokenizer.decode(ids[0], skip_special_tokens=False)
        # Get everything after "assistant\n"
        if "assistant\n" in full:
            response = full.split("assistant\n")[-1].replace("<|im_end|>", "").strip()
        else:
            response = full

        log(f"\nQ: {prompt}")
        log(f"  A: {response}")

if __name__ == "__main__":
    main()
