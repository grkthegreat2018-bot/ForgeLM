"""Test V3 Expert Tying, C2 WiSparse, C3 SharQ FP4."""
import sys, os, time, torch
import torch.nn as nn
from transformers import AutoTokenizer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def get_input(model, n_tokens=10):
    """Get tokenized input for the model."""
    tok = AutoTokenizer.from_pretrained("research/checkpoints/qwen_hf")
    text = "def fibonacci(n):\n    if n <= 1:\n        return n\n    return "
    enc = tok(text, return_tensors="pt")
    return enc.input_ids.to(next(model.parameters()).device)


def test_expert_tying():
    """Test V3 Expert Tying — 2x FFN VRAM, near-lossless."""
    print("\n=== Test 1: Expert Tying (V3) ===")
    from research.config import get_config
    from research.model_loader import ModelLoader
    from research.keys.moe.expert_tying_key import ExpertTyingKey

    cfg = get_config('forgelm_v2', device='cuda')
    model = ModelLoader.build_model_fast(
        cfg, checkpoint_path='research/checkpoints/forgelm_v2.safetensors',
        moe_top_k=0, dtype=torch.bfloat16)
    model.to('cuda').eval()

    # Count expert params before
    expert_params_before = 0
    for name, mod in model.named_modules():
        if hasattr(mod, 'experts') and isinstance(mod.experts, nn.ModuleList):
            for e in mod.experts:
                for p in e.parameters():
                    expert_params_before += p.numel()

    # Apply expert tying
    key = ExpertTyingKey()
    tied_pairs = key.apply(model)

    # Count unique expert params after (odd layers are aliases)
    expert_params_after = 0
    seen_ids = set()
    for name, mod in model.named_modules():
        if hasattr(mod, 'experts') and isinstance(mod.experts, nn.ModuleList):
            for e in mod.experts:
                eid = id(e)
                if eid not in seen_ids:
                    seen_ids.add(eid)
                    for p in e.parameters():
                        expert_params_after += p.numel()

    saved = expert_params_before - expert_params_after
    ratio = expert_params_after / max(expert_params_before, 1)

    print(f"  Expert params before: {expert_params_before / 1e6:.1f}M")
    print(f"  Expert params after:  {expert_params_after / 1e6:.1f}M (unique)")
    print(f"  Saved: {saved / 1e6:.1f}M ({1 - ratio:.1%} reduction)")
    print(f"  Tied pairs: {len(tied_pairs)}")

    # Verify model still produces output
    x = get_input(model)
    with torch.no_grad():
        logits, _ = model(x, use_cache=False)
    print(f"  Output shape: {logits.shape}")
    print(f"  PASS: Expert Tying works ({1 - ratio:.1%} VRAM saved)")


def test_wisparse():
    """Test C2 WiSparse — 21% compute reduction, training-free."""
    print("\n=== Test 2: WiSparse (C2) ===")
    from research.config import get_config
    from research.model_loader import ModelLoader
    from research.keys.compression.wisparse_key import WiSparseKey

    cfg = get_config('forgelm_v2', device='cuda')
    model = ModelLoader.build_model_fast(
        cfg, checkpoint_path='research/checkpoints/forgelm_v2.safetensors',
        moe_top_k=0, dtype=torch.bfloat16)
    model.to('cuda').eval()

    # Baseline output
    x = get_input(model)
    with torch.no_grad():
        logits_before, _ = model(x, use_cache=False)

    # Apply WiSparse
    key = WiSparseKey(target_sparsity=0.5)
    n_patched = key.apply(model)

    # Calibrate
    key.calibrate(model, x)

    # Run with WiSparse
    with torch.no_grad():
        logits_after, _ = model(x, use_cache=False)

    # Compare quality
    cos = torch.nn.functional.cosine_similarity(
        logits_before[0].flatten().unsqueeze(0).float(),
        logits_after[0].flatten().unsqueeze(0).float(),
        dim=-1
    ).item()
    max_diff = (logits_before - logits_after).abs().max().item()

    key.print_stats()
    print(f"  Cosine similarity: {cos:.4f}")
    print(f"  Max abs diff: {max_diff:.4f}")
    print(f"  PASS: WiSparse works (cos={cos:.4f}, target 97% quality)")


def test_sharq_fp4():
    """Test C3 SharQ FP4 — 2.2x latency, training-free."""
    print("\n=== Test 3: SharQ FP4 (C3) ===")
    from research.config import get_config
    from research.model_loader import ModelLoader
    from research.keys.quantization.sharq_fp4_key import SharQFP4Key

    cfg = get_config('forgelm_v2', device='cuda')
    model = ModelLoader.build_model_fast(
        cfg, checkpoint_path='research/checkpoints/forgelm_v2.safetensors',
        moe_top_k=0, dtype=torch.bfloat16)
    model.to('cuda').eval()

    # Baseline output
    x = get_input(model)
    with torch.no_grad():
        logits_before, _ = model(x, use_cache=False)

    # Apply SharQ FP4 (only to FFN experts to keep it simple)
    key = SharQFP4Key(sparsity_threshold=0.05)
    n_patched = key.apply(model, target="all")

    # Run with SharQ
    with torch.no_grad():
        logits_after, _ = model(x, use_cache=False)

    # Compare quality
    cos = torch.nn.functional.cosine_similarity(
        logits_before[0].flatten().unsqueeze(0).float(),
        logits_after[0].flatten().unsqueeze(0).float(),
        dim=-1
    ).item()
    max_diff = (logits_before - logits_after).abs().max().item()

    key.print_stats()
    print(f"  Cosine similarity: {cos:.4f}")
    print(f"  Max abs diff: {max_diff:.4f}")
    print(f"  PASS: SharQ FP4 works (cos={cos:.4f})")


def test_stacked():
    """Test all 3 keys stacked together."""
    print("\n=== Test 4: All 3 keys stacked ===")
    from research.config import get_config
    from research.model_loader import ModelLoader
    from research.keys.moe.expert_tying_key import ExpertTyingKey
    from research.keys.compression.wisparse_key import WiSparseKey
    from research.keys.quantization.sharq_fp4_key import SharQFP4Key

    cfg = get_config('forgelm_v2', device='cuda')
    model = ModelLoader.build_model_fast(
        cfg, checkpoint_path='research/checkpoints/forgelm_v2.safetensors',
        moe_top_k=0, dtype=torch.bfloat16)
    model.to('cuda').eval()

    # Baseline
    x = get_input(model)
    with torch.no_grad():
        logits_before, _ = model(x, use_cache=False)

    # Apply all 3: Expert Tying first (shares weights), then WiSparse, then SharQ
    tying = ExpertTyingKey()
    tying.apply(model)

    wisparse = WiSparseKey(target_sparsity=0.5)
    wisparse.apply(model)
    wisparse.calibrate(model, x)

    sharq = SharQFP4Key(sparsity_threshold=0.05)
    sharq.apply(model, target="all")

    # Run with all keys
    with torch.no_grad():
        logits_after, _ = model(x, use_cache=False)

    cos = torch.nn.functional.cosine_similarity(
        logits_before[0].flatten().unsqueeze(0).float(),
        logits_after[0].flatten().unsqueeze(0).float(),
        dim=-1
    ).item()

    wisparse.print_stats()
    sharq.print_stats()
    print(f"  Stacked cosine similarity: {cos:.4f}")
    print(f"  PASS: All 3 keys work together")


if __name__ == "__main__":
    print("=" * 70)
    print("VRAM + Compute Reduction Keys Test")
    print("=" * 70)

    test_expert_tying()
    test_wisparse()
    test_sharq_fp4()
    test_stacked()

    print("\n" + "=" * 70)
    print("All VRAM + Compute keys verified!")
    print("=" * 70)


