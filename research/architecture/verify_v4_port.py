"""Verify V4 port is lossless by comparing token outputs against LFM2.5 reference.

Loads both models with identical prompts and greedy decoding, compares token IDs.
"""
import os
import sys
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from research.model_loader import load_default_model, create_kv_cache

PROMPTS = [
    "Write a Python function to check if a number is prime.",
    "Explain how binary search works in one sentence.",
    "def fibonacci(n):",
    "The capital of France is",
]

def generate_greedy(model, tokenizer, prompt, max_new=30):
    """Greedy generation using model forward + KV cache. Returns token IDs."""
    model.eval()
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt")
    if hasattr(inputs, 'to'):
        inputs = inputs.to(device)
    else:
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
    prompt_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
    B, prompt_len = prompt_ids.shape
    eos_id = tokenizer.eos_token_id

    max_total = prompt_len + max_new
    out_ids = torch.zeros(B, max_total, dtype=prompt_ids.dtype, device=device)
    out_ids[:, :prompt_len] = prompt_ids

    cache = create_kv_cache(model, max_total, batch=B, device=device)

    generated = []
    with torch.no_grad():
        for step in range(max_new):
            pos = prompt_len + step
            if step == 0:
                idx_cond = out_ids[:, :prompt_len]
                out = model(idx_cond, preallocated_cache=cache, use_cache=True)
            else:
                idx_cond = out_ids[:, pos - 1:pos]
                out = model(idx_cond, preallocated_cache=cache, use_cache=True)
            logits = out[0]
            logits = logits[:, -1, :]  # last token
            next_token = logits.argmax(dim=-1, keepdim=True)  # greedy
            out_ids[:, pos:pos + 1] = next_token
            generated.append(next_token.item())

            if eos_id is not None and next_token.item() == eos_id:
                break

    return generated

def main():
    results = {}

    # 1. Load reference LFM2.5
    print("=" * 60)
    print("Loading LFM2.5 reference (lfm25_1.2b config)...")
    print("=" * 60)
    model_ref, tok_ref = load_default_model(
        "lfm25_1.2b",
        checkpoint_path="research/checkpoints/ForgeLM_V2_LFM25-1.2B.safetensors",
        device="cuda",
        dtype=torch.bfloat16,
        fast_load=False,
    )

    ref_outputs = {}
    for prompt in PROMPTS:
        tokens = generate_greedy(model_ref, tok_ref, prompt, max_new=30)
        text = tok_ref.decode(tokens, skip_special_tokens=True)
        ref_outputs[prompt] = tokens
        print(f"\n  Prompt: {prompt[:50]}...")
        print(f"  Tokens: {tokens[:15]}...")
        print(f"  Text:   {text[:100]}...")

    del model_ref
    torch.cuda.empty_cache()

    # 2. Load V4 port
    print("\n" + "=" * 60)
    print("Loading V4 port (forgelm_v7 config)...")
    print("=" * 60)
    model_v4, tok_v4 = load_default_model(
        "forgelm_v7",
        checkpoint_path="research/checkpoints/forgelm_v7_Base.safetensors",
        device="cuda",
        dtype=torch.bfloat16,
        fast_load=False,
    )

    v4_outputs = {}
    for prompt in PROMPTS:
        tokens = generate_greedy(model_v4, tok_v4, prompt, max_new=30)
        text = tok_v4.decode(tokens, skip_special_tokens=True)
        v4_outputs[prompt] = tokens
        print(f"\n  Prompt: {prompt[:50]}...")
        print(f"  Tokens: {tokens[:15]}...")
        print(f"  Text:   {text[:100]}...")

    del model_v4
    torch.cuda.empty_cache()

    # 3. Compare
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    all_match = True
    for prompt in PROMPTS:
        ref = ref_outputs[prompt]
        v4 = v4_outputs[prompt]
        match = ref == v4
        status = "MATCH" if match else "MISMATCH"
        if not match:
            all_match = False
            min_len = min(len(ref), len(v4))
            diverge = -1
            for i in range(min_len):
                if ref[i] != v4[i]:
                    diverge = i
                    break
            if diverge == -1:
                diverge = min_len
            print(f"\n  [{status}] {prompt[:50]}...")
            print(f"    Divergence at token {diverge}")
            print(f"    Ref: {ref[:diverge+3]}")
            print(f"    V4:  {v4[:diverge+3]}")
        else:
            print(f"\n  [{status}] {prompt[:50]}...")

    print("\n" + "=" * 60)
    if all_match:
        print("ALL PROMPTS MATCH -- V4 port is LOSSLESS")
    else:
        print("MISMATCHES DETECTED -- V4 port has divergence")
    print("=" * 60)

    return 0 if all_match else 1

if __name__ == "__main__":
    sys.exit(main())
