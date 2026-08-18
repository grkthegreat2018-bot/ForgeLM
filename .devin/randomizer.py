"""Randomizer: throw known + loosely-related systems into a hat, pick 2-3,
combine them, and test. Per AGENTS.md #7: 'leave it to luck' to find novel
combos you wouldn't have thought of."""
import random
import itertools
import sys

# Known systems (from research)
KNOWN = [
    "Muon (Newton-Schulz orthogonalization)",
    "Schedule-Free (iterate averaging, no LR schedule)",
    "Blockwise sharpness LR (per-block Fisher EMA)",
    "Sophia (diagonal Hessian clipping)",
    "Adam-mini (block-partitioned single LR)",
    "GaLore (gradient low-rank projection)",
    "BitNet 1.58 (ternary QAT)",
    "ActNN (2-bit activation compression)",
    "PowerSGD (low-rank gradient compression)",
    "1-bit Adam (sign-based momentum)",
    "Lion (sign-magnitude, no second moment)",
    "ADOPT (decoupled v from current grad)",
    "Shampoo (Kronecker preconditioner)",
    "SOAP (Adam in Shampoo eigenbasis)",
    "WSD scheduler (warmup-stable-decay)",
    "Cosine schedule",
    "anTransformer (norm-constrained, no warmup)",
    "DiffusionBlocks (block-wise denoising training, B× memory)",
    "Grad mixup (average gradients from N batches)",
    "MTP (multi-token prediction auxiliary loss)",
]

# Loosely related / cross-domain
LOOSE = [
    "EDM sigma schedule (diffusion noise curve for LR)",
    "DiffusionBlocks sigma as LR schedule (block noise level → per-block LR)",
    "AdaLN noise conditioning (shift/scale modulation on optimizer state)",
    "TTT layers (test-time training as hidden state)",
    "SnapKV (attention-based eviction for gradient buffer)",
    "Hadamard rotation (rotate grads before quantize)",
    "RoPE (rotary position encoding applied to momentum)",
    "TITAN memory (Hebbian surprise step on optimizer state)",
    "MoD router (skip gradient computation for easy tokens)",
    "Speculative decoding (draft+verify for gradient eval)",
    "ELO curriculum (difficulty-ordered training)",
    "Replay buffer (golden trajectory injection)",
    "Model soup (weight averaging mid-training)",
    "Dropout (stochastic depth on optimizer steps)",
    "Mixup (interpolate two batches' gradients)",
    "Label smoothing (soft targets for gradient)",
    "Knowledge distillation (soft grad from teacher)",
    "Quantization noise injection (QAT-style grad noise)",
    "Simulated annealing (temperature-scaled LR)",
    "Reinforce (policy gradient as optimizer signal)",
    "DiffusionBlocks freed VRAM → bigger batch for mixup (memory synergy)",
    "Block-wise noise level as per-block sharpness signal (diffusion→optimizer)",
    "Classifier-free guidance dropout on optimizer momentum (noise_dropout analog)",
]

def random_combo(n_known=1, n_loose=2, seed=None):
    rng = random.Random(seed)
    k = rng.sample(KNOWN, n_known)
    l = rng.sample(LOOSE, n_loose)
    return k, l

def main():
    print("=" * 70)
    print("NOVEL COMBO RANDOMIZER (AGENTS.md #7: leave it to luck)")
    print("=" * 70)
    print(f"Pool: {len(KNOWN)} known + {len(LOOSE)} loosely-related = {len(KNOWN)+len(LOOSE)} systems\n")

    # Generate 15 random combos
    for i in range(15):
        k, l = random_combo(seed=i * 37 + 13)
        combo = k + l
        print(f"  Combo {i+1:2d}: {' + '.join(combo)}")

    print("\n" + "=" * 70)
    print("PROMISING-LOOKING combos to actually test:")
    print("=" * 70)
    # Hand-pick the 3 most interesting from above + add some targeted ones
    targeted = [
        (["DiffusionBlocks", "Muon", "Blockwise sharpness LR"], ["Grad mixup"],
         "3-way stack: DiffusionBlocks frees B× VRAM → enables more mixup batches → "
         "muon_sf_blockwise optimizer. All three are orthogonal: memory, data, optimizer."),
        (["DiffusionBlocks", "Grad mixup"], ["DiffusionBlocks sigma as LR schedule"],
         "DiffusionBlocks already computes per-block sigma. Use sigma as per-block LR "
         "multiplier (high noise = high LR for exploration, low noise = refine). "
         "Free signal — no extra computation needed."),
        (["DiffusionBlocks", "Muon"], ["Block-wise noise level as per-block sharpness signal"],
         "Replace Fisher EMA sharpness with DiffusionBlocks' sigma as the per-block "
         "LR scaling signal. Sigma already encodes difficulty per block."),
        (["Muon", "Blockwise sharpness LR", "Grad mixup"], ["DiffusionBlocks freed VRAM → bigger batch for mixup"],
         "The synergy: DiffusionBlocks uses 1/B layers → B× less VRAM → use freed "
         "memory for 3-way or 4-way mixup instead of 2-way. More mixup = better gradient."),
        (["DiffusionBlocks", "Grad mixup"], ["Classifier-free guidance dropout on optimizer momentum"],
         "Noise dropout from DiffusionBlocks (randomly drop conditioning) applied to "
         "optimizer momentum — randomly zero out momentum to prevent overfitting to "
         "one direction. Classifier-free guidance analog."),
        (["DiffusionBlocks", "MTP"], ["Grad mixup"],
         "DiffusionBlocks + MTP (multi-token prediction at each block) + grad mixup. "
         "Each block's denoising step predicts K future tokens. Mixup averages across "
         "blocks and batches."),
    ]
    for i, (k, l, why) in enumerate(targeted):
        print(f"\n  Target {i+1}: {' + '.join(k + l)}")
        print(f"    Why: {why}")

if __name__ == "__main__":
    main()
