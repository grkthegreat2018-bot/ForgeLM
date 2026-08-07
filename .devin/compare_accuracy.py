"""Compare original Qwen vs XP model (post-KeyStack) forward pass accuracy.

Loads both models, runs identical inputs, compares:
  - Logit cosine similarity
  - Top-1 / top-5 token agreement
  - KL divergence
  - Output text sample

Uses streaming (one layer at a time on GPU) to avoid stutter.
"""
import sys, os, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safetensors.torch import load_file
from research.config import get_config
from research.model_loader import ModelLoader

SRC = "research/checkpoints/qwen25_coder_1.5b_ported.safetensors"
DST = "research/checkpoints/xp_model_recovered.safetensors"
CFG = "qwen25_coder_1.5b"

def load_model(path, cfg_name, device="cuda", overrides=None):
    cfg = get_config(cfg_name, device=device, **(overrides or {}))
    model = ModelLoader.build_model(cfg, checkpoint_path=path)
    model.eval()
    return model.to(device, dtype=torch.bfloat16)

def compare_logits(logits_a, logits_b):
    """Compute similarity metrics between two logit tensors."""
    a = logits_a.float().flatten()
    b = logits_b.float().flatten()
    cos = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    # Top-k agreement at last token
    top1_a = logits_a[0, -1].argmax().item()
    top1_b = logits_b[0, -1].argmax().item()
    top5_a = set(logits_a[0, -1].topk(5).indices.tolist())
    top5_b = set(logits_b[0, -1].topk(5).indices.tolist())
    top5_overlap = len(top5_a & top5_b)
    # KL divergence (per token, averaged)
    p = F.log_softmax(logits_a.float(), dim=-1)
    q = F.log_softmax(logits_b.float(), dim=-1)
    kl = (p.exp() * (p - q)).sum(dim=-1).mean().item()
    return cos, top1_a, top1_b, top5_overlap, kl

def main():
    print("=" * 70)
    print("ACCURACY COMPARISON: Original Qwen vs XP Model (post-KeyStack)")
    print("=" * 70)

    # Test prompts
    test_ids_list = [
        ("Hello world", [151643, 9707, 1917]),
        ("def fibonacci", [151643, 644, 8436, 23706, 1620]),
        ("The meaning of", [151643, 464, 7434, 315]),
        ("import torch", [151643, 1364, 10631, 4288]),
    ]

    print("\nLoading original Qwen model...")
    model_orig = load_model(SRC, CFG)
    print("Loading XP model (post-KeyStack, MQA config)...")
    # XP model has GQA→MQA applied: n_kv_heads 2→1
    model_xp = load_model(DST, CFG, overrides={"n_kv_heads": 1})

    print(f"\n{'='*70}")
    print(f"{'Prompt':<20} {'CosSim':>8} {'Top1Match':>10} {'Top5Overlap':>12} {'KL Div':>8}")
    print(f"{'='*70}")

    results = []
    for name, ids in test_ids_list:
        input_ids = torch.tensor([ids], device="cuda")
        with torch.inference_mode():
            out_orig = model_orig(input_ids)
            logits_orig = out_orig[0] if isinstance(out_orig, tuple) else out_orig

            out_xp = model_xp(input_ids)
            logits_xp = out_xp[0] if isinstance(out_xp, tuple) else out_xp

        # Handle shape mismatch (GQA→MQA changed K/V dims)
        if logits_orig.shape != logits_xp.shape:
            print(f"  {name:<20} SHAPE MISMATCH: {logits_orig.shape} vs {logits_xp.shape}")
            # Compare only the overlapping vocab portion
            min_vocab = min(logits_orig.shape[-1], logits_xp.shape[-1])
            logits_orig = logits_orig[..., :min_vocab]
            logits_xp = logits_xp[..., :min_vocab]

        cos, top1_a, top1_b, top5, kl = compare_logits(logits_orig, logits_xp)
        top1_match = "✓" if top1_a == top1_b else "✗"
        print(f"  {name:<20} {cos:>8.4f} {top1_match:>10} {top5:>12}/5 {kl:>8.4f}")
        results.append((name, cos, top1_a == top1_b, top5, kl))

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    avg_cos = sum(r[1] for r in results) / len(results)
    top1_matches = sum(1 for r in results if r[2])
    avg_top5 = sum(r[3] for r in results) / len(results)
    avg_kl = sum(r[4] for r in results) / len(results)
    print(f"  Avg cosine similarity: {avg_cos:.4f}")
    print(f"  Top-1 matches:         {top1_matches}/{len(results)}")
    print(f"  Avg top-5 overlap:     {avg_top5:.1f}/5")
    print(f"  Avg KL divergence:     {avg_kl:.4f}")

    # Interpretation
    print(f"\nInterpretation:")
    if avg_cos > 0.95:
        print(f"  ✓ High similarity — transforms preserved model behavior")
    elif avg_cos > 0.7:
        print(f"  ~ Moderate similarity — some quality loss, fine-tuning recommended")
    elif avg_cos > 0.3:
        print(f"  ⚠ Low similarity — significant quality loss from transforms")
    else:
        print(f"  ✗ Very low similarity — transforms may have broken the model")

    # Check for NaN/Inf
    for name, ids in test_ids_list:
        input_ids = torch.tensor([ids], device="cuda")
        with torch.inference_mode():
            out_xp = model_xp(input_ids)
            logits_xp = out_xp[0] if isinstance(out_xp, tuple) else out_xp
        if torch.isnan(logits_xp).any():
            print(f"  ✗ NaN in XP logits for '{name}'")
        if torch.isinf(logits_xp).any():
            print(f"  ✗ Inf in XP logits for '{name}'")

    del model_orig, model_xp
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
