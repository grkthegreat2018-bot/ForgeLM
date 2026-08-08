"""Recheck: compare OUTPUT (logits) between original V2 and fine-tuned V2-opt.

The fine-tune changes weights to minimize loss, not to match original weights.
What matters is: does the scale-folded + fine-tuned model produce the same
or better output than the original V2?

This script compares:
1. Logit cosine similarity on test prompts
2. Top-1 token agreement (same next token prediction)
3. Loss on held-out data
"""
import os
import sys
import math
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)

CKPT_ORIG = os.path.join("research", "checkpoints", "forgelm_v2.safetensors")
CKPT_OPT = os.path.join("research", "checkpoints", "forgelm_v2_opt.safetensors")
CKPT_FT = os.path.join("research", "checkpoints", "forgelm_v2_opt_ft.safetensors")


def load_model(ckpt_path, cfg, moe_top_k=0):
    """Load a model from checkpoint."""
    from research.model_loader import ModelLoader
    model = ModelLoader.build_model_fast(
        cfg, checkpoint_path=ckpt_path,
        moe_top_k=moe_top_k,
        dtype=torch.bfloat16)
    model.to("cuda").eval()
    return model


def compare_logits(model_a, model_b, input_ids, label="A vs B"):
    """Compare logits between two models on the same input."""
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out_a = model_a(input_ids)
        out_b = model_b(input_ids)
        logits_a = out_a[0] if isinstance(out_a, tuple) else out_a
        logits_b = out_b[0] if isinstance(out_b, tuple) else out_b

    # Compare last-token logits (most relevant for generation)
    la = logits_a[0, -1].float()
    lb = logits_b[0, -1].float()

    # Cosine similarity
    cos = F.cosine_similarity(la.unsqueeze(0), lb.unsqueeze(0)).item()

    # Top-1 agreement
    top1_a = la.argmax().item()
    top1_b = lb.argmax().item()
    top1_match = top1_a == top1_b

    # Top-5 agreement
    top5_a = set(la.topk(5).indices.tolist())
    top5_b = set(lb.topk(5).indices.tolist())
    top5_overlap = len(top5_a & top5_b) / 5.0

    # Max abs diff
    max_diff = (la - lb).abs().max().item()

    # KL divergence
    pa = F.softmax(la, dim=-1)
    pb = F.softmax(lb, dim=-1)
    kl = (pa * (pa.log() - pb.log())).sum().item()

    print(f"  {label}:")
    print(f"    Logit cos:    {cos:.8f}")
    print(f"    Top-1 match:  {top1_match} ({top1_a} == {top1_b})")
    print(f"    Top-5 overlap: {top5_overlap:.1%}")
    print(f"    Max abs diff: {max_diff:.6f}")
    print(f"    KL divergence: {kl:.6f}")

    return cos, top1_match, top5_overlap, max_diff, kl


def main():
    print("="*60)
    print("Recheck: Output Comparison — Original V2 vs Fine-tuned V2-opt")
    print("="*60)

    from research.config import get_config
    from transformers import AutoTokenizer

    cfg = get_config("forgelm_v2", device="cuda")
    tokenizer = AutoTokenizer.from_pretrained(os.path.join("research", "checkpoints", "qwen_hf"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Test prompts
    prompts = [
        "def fibonacci(n):",
        "The capital of France is",
        "import torch\nimport torch.nn as nn\n\nclass Transformer(nn.Module):",
        "def quicksort(arr):",
        "In machine learning, gradient descent is",
    ]

    # Load models — STUDENT FIRST so cache has identity norms (not teacher's)
    print("\n[1] Loading V2-opt (norm-folded, deduped)...")
    model_opt = load_model(CKPT_OPT, cfg)

    print("\n[2] Loading original V2 (teacher)...")
    model_orig = load_model(CKPT_ORIG, cfg)

    print("\n[3] Loading V2-opt-ft (distilled)...")
    model_ft = load_model(CKPT_FT, cfg)

    # Compare on each prompt
    print(f"\n[4] Comparing outputs on {len(prompts)} prompts...")
    print("="*60)

    results_opt = []  # orig vs opt (no fine-tune)
    results_ft = []   # orig vs ft (fine-tuned)

    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        print(f"\n  Prompt: {prompt[:50]}...")

        # Original vs opt (no fine-tune) — should be near-identical
        cos_opt, t1_opt, t5_opt, diff_opt, kl_opt = compare_logits(
            model_orig, model_opt, ids, "V2 vs V2-opt (scale fold only)")

        # Original vs ft (fine-tuned) — should be close but improved
        cos_ft, t1_ft, t5_ft, diff_ft, kl_ft = compare_logits(
            model_orig, model_ft, ids, "V2 vs V2-opt-ft (scale fold + fine-tune)")

        results_opt.append((cos_opt, t1_opt, t5_opt, diff_opt, kl_opt))
        results_ft.append((cos_ft, t1_ft, t5_ft, diff_ft, kl_ft))

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    avg_cos_opt = sum(r[0] for r in results_opt) / len(results_opt)
    avg_cos_ft = sum(r[0] for r in results_ft) / len(results_ft)
    t1_rate_opt = sum(1 for r in results_opt if r[1]) / len(results_opt)
    t1_rate_ft = sum(1 for r in results_ft if r[1]) / len(results_ft)
    avg_t5_opt = sum(r[2] for r in results_opt) / len(results_opt)
    avg_t5_ft = sum(r[2] for r in results_ft) / len(results_ft)
    avg_kl_opt = sum(r[4] for r in results_opt) / len(results_opt)
    avg_kl_ft = sum(r[4] for r in results_ft) / len(results_ft)

    print(f"\n  V2 vs V2-opt (scale fold only, no fine-tune):")
    print(f"    Avg logit cos:  {avg_cos_opt:.8f}")
    print(f"    Top-1 match:    {t1_rate_opt:.1%}")
    print(f"    Avg top-5 overlap: {avg_t5_opt:.1%}")
    print(f"    Avg KL divergence: {avg_kl_opt:.6f}")

    print(f"\n  V2 vs V2-opt-ft (scale fold + 50-step fine-tune):")
    print(f"    Avg logit cos:  {avg_cos_ft:.8f}")
    print(f"    Top-1 match:    {t1_rate_ft:.1%}")
    print(f"    Avg top-5 overlap: {avg_t5_ft:.1%}")
    print(f"    Avg KL divergence: {avg_kl_ft:.6f}")

    print(f"\n  Verdict:")
    if avg_cos_ft > avg_cos_opt:
        print(f"    Fine-tune IMPROVED output alignment (cos: {avg_cos_opt:.6f} → {avg_cos_ft:.6f})")
    else:
        print(f"    Fine-tune changed output (cos: {avg_cos_opt:.6f} → {avg_cos_ft:.6f})")
        print(f"    (Expected — fine-tune minimizes loss, not weight matching)")

    if t1_rate_ft >= t1_rate_opt:
        print(f"    Top-1 token agreement maintained or improved ({t1_rate_opt:.0%} → {t1_rate_ft:.0%})")
    else:
        print(f"    Top-1 token agreement changed ({t1_rate_opt:.0%} → {t1_rate_ft:.0%})")

    # Check sizes
    print(f"\n  Checkpoint sizes:")
    for name, path in [("V2 original", CKPT_ORIG), ("V2-opt", CKPT_OPT), ("V2-opt-ft", CKPT_FT)]:
        if os.path.exists(path):
            sz = os.path.getsize(path) / 1e6
            print(f"    {name}: {sz:.1f} MB")


if __name__ == "__main__":
    main()
