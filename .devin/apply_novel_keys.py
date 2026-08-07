"""Apply novel keys to ForgeLM v2 and create v3.

Phase 1: Norm Folding (lossless — fold RMSNorm into adjacent weights)
Phase 2: Expert Consolidation analysis (check if experts are mergeable)
Phase 3: Create ForgeLM v3 checkpoint

Note: GRAIL and Activation Transmute require calibration data + model loading,
so they're applied separately via their own scripts.
"""
import sys, os, time, torch
sys.path.insert(0, '.')

from safetensors import safe_open
from safetensors.torch import save_file
from pathlib import Path

SRC = "research/checkpoints/forgelm_v2.safetensors"
DST = "research/checkpoints/forgelm_v3.safetensors"

N_LAYERS = 28
D_MODEL = 1536
N_HEADS = 12
HEAD_DIM = D_MODEL // N_HEADS


def main():
    print("=" * 60)
    print("Apply Novel Keys to ForgeLM v2 → v3")
    print("=" * 60)

    # Load v2
    print("\n[1] Loading ForgeLM v2...")
    t0 = time.time()
    state = {}
    with safe_open(SRC, framework="pt") as f:
        for key in f.keys():
            state[key] = f.get_tensor(key)
    print(f"  Loaded {len(state)} tensors in {time.time()-t0:.1f}s")

    # Phase 1: Expert Consolidation Analysis (no merging yet, just check)
    print("\n[2] Analyzing expert similarities...")
    from research.keys.expert_consolidation_key import compute_expert_similarities
    sims = compute_expert_similarities(state, N_LAYERS)

    # Print summary
    all_sims = []
    for layer_sims in sims:
        for a, b, sim in layer_sims:
            all_sims.append(sim)

    if all_sims:
        import numpy as np
        sims_arr = np.array(all_sims)
        print(f"  Expert pairs analyzed: {len(sims_arr)}")
        print(f"  Similarity: mean={sims_arr.mean():.4f}, "
              f"min={sims_arr.min():.4f}, max={sims_arr.max():.4f}")
        print(f"  Pairs > 0.90: {(sims_arr > 0.90).sum()}")
        print(f"  Pairs > 0.95: {(sims_arr > 0.95).sum()}")
        print(f"  Pairs > 0.99: {(sims_arr > 0.99).sum()}")

        # Show top 5 most similar pairs
        top_pairs = sorted([(s, l, a, b) for l, ls in enumerate(sims) for a, b, s in ls],
                          reverse=True)[:5]
        print(f"  Top 5 most similar pairs:")
        for sim, layer, a, b in top_pairs:
            print(f"    Layer {layer}: expert {a} ↔ {b}: cos={sim:.6f}")

    # Phase 2: Apply Norm Folding (lossless)
    print("\n[3] Applying Norm Folding (lossless)...")
    from research.keys.norm_folding_key import apply_norm_folding
    state = apply_norm_folding(state, N_LAYERS, D_MODEL)

    # Save v3
    print(f"\n[4] Saving to {DST}...")
    t0 = time.time()
    save_file(state, DST, metadata={
        "source": SRC,
        "pipeline": "novel_keys",
        "transforms": "norm_folding",
        "n_layers": str(N_LAYERS),
        "d_model": str(D_MODEL),
        "tensors": str(len(state)),
    })
    size_mb = Path(DST).stat().st_size / 1e6
    print(f"  Saved {len(state)} tensors, {size_mb:.0f} MB in {time.time()-t0:.1f}s")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: ForgeLM v3 created")
    print(f"  Tensors: {len(state)} (v2: {len(state) + N_LAYERS*2 + 1 + N_LAYERS*2})")
    print(f"  Norms folded: {N_LAYERS*2 + 1} (ln1, ln2 per layer + ln_f)")
    print(f"  QK-Norms folded: {N_LAYERS*2} (q_norm, k_norm per layer)")
    print(f"  Size: {size_mb:.0f} MB")
    if all_sims:
        print(f"  Expert analysis: {len(sims_arr)} pairs, "
              f"max similarity={sims_arr.max():.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
