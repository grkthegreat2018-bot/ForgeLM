"""ForgeAI Pipeline: Qwen2.5-Coder → All-Architecture Model

Two phases:
1. WEIGHT STEALING (safetensors → safetensors, no GPU needed):
   - MLA: GQA K/V → low-rank compression (cos=0.9999)
   - MoE: Dense SwiGLU → routed experts (cos=0.73, needs router fine-tune)
   - BitNet: bf16 → ternary {-1,0,+1} (26% smaller)

2. ARCHITECTURE ADDITIONS (needs model + GPU + fine-tuning):
   - SpinQuant: learned rotation before quantization (better BitNet quality)
   - GateSkip: token-wise layer skipping (fewer FLOPs at inference)
   - MTP: multi-token prediction head (speculative decoding speedup)
   - Fine-tune: recover quality from all transformations

Phase 1 is fully automated. Phase 2 requires building a model, initializing
new parameters, and fine-tuning on calibration data.

Usage:
    # Phase 1: weight stealing (run without GPU)
    python -m research.forge_pipeline

    # Phase 1 + 2: full pipeline (needs GPU + calibration data)
    python -m research.forge_pipeline --phase2

    # Skip individual steps
    python -m research.forge_pipeline --skip-moe --skip-bitnet
"""
import argparse
import torch
from pathlib import Path
from research.convert_keys import key_gqa_to_mla, key_dense_to_moe, key_bitnet

CHECKPOINTS = Path("research/checkpoints")


def run_phase1(src: str, config: str = "qwen25_coder_1.5b",
               skip_mla: bool = False, skip_moe: bool = False,
               skip_bitnet: bool = False,
               kv_compression_dim: int = 512,
               n_experts: int = 4, top_k: int = 2,
               shared_expert: bool = True):
    """Phase 1: Weight stealing (safetensors → safetensors)."""
    step = 0
    current = src

    if not skip_mla:
        step += 1
        out = str(CHECKPOINTS / f"forge_step{step}_mla.safetensors")
        print(f"\n{'='*60}")
        print(f"STEP {step}: GQA → MLA (d_c={kv_compression_dim})")
        print(f"{'='*60}")
        key_gqa_to_mla(current, out, source_config_name=config,
                       target_config_name=config, kv_compression_dim=kv_compression_dim)
        current = out

    if not skip_moe:
        step += 1
        out = str(CHECKPOINTS / f"forge_step{step}_moe.safetensors")
        print(f"\n{'='*60}")
        print(f"STEP {step}: Dense SwiGLU → MoE ({n_experts} experts, top-{top_k})")
        print(f"{'='*60}")
        key_dense_to_moe(current, out, config_name=config,
                         n_experts=n_experts, top_k=top_k, shared_expert=shared_expert)
        current = out

    if not skip_bitnet:
        step += 1
        out = str(CHECKPOINTS / f"forge_step{step}_bitnet.safetensors")
        print(f"\n{'='*60}")
        print(f"STEP {step}: BitNet ternary quantization")
        print(f"{'='*60}")
        key_bitnet(current, out, config)
        current = out

    print(f"\n{'='*60}")
    print(f"PHASE 1 COMPLETE")
    print(f"  Final output: {current}")
    print(f"  Steps executed: {step}")
    print(f"{'='*60}")
    return current


def run_phase2(src: str, config: str = "qwen25_coder_1.5b",
               use_spinquant: bool = True, use_gateskip: bool = True,
               use_mtp: bool = True, finetune_steps: int = 1000):
    """Phase 2: Architecture additions + fine-tuning (needs GPU).

    This phase:
    1. Builds a model from the phase 1 checkpoint
    2. Adds GateSkip gates and MTP head (new parameters, initialized)
    3. Optionally runs SpinQuant calibration before BitNet
    4. Fine-tunes to recover quality from all transformations
    """
    from research.config import get_config
    from research.model_loader import ModelLoader
    from research.moe import replace_ffn_with_moe, collect_aux_loss

    print(f"\n{'='*60}")
    print(f"PHASE 2: Architecture additions + fine-tuning")
    print(f"{'='*60}")

    cfg = get_config(config)
    device = "cuda"

    # Build model from phase 1 output
    print(f"\nLoading model from {src}...")
    model = ModelLoader.build_model(cfg, checkpoint_path=src).to(device)

    # Add MoE if the checkpoint has MoE weights
    moe_state = torch.load(src, map_location="cpu") if src.endswith(".pt") else None
    from safetensors.torch import load_file, safe_open
    with safe_open(src, framework="pt") as f:
        keys = list(f.keys())
    has_moe = any("experts" in k for k in keys)
    if has_moe:
        print("  Detected MoE weights, replacing FFN with MoE...")
        # Determine d_ff from expert weights
        expert_w = load_file(src)
        d_ff = expert_w["blocks.0.ffn.experts.0.w1.weight"].shape[0]
        n_experts = sum(1 for k in keys if "blocks.0.ffn.experts." in k and ".w1.weight" in k)
        replace_ffn_with_moe(model, n_experts=n_experts, top_k=2,
                            d_model=cfg.d_model, shared_expert=True, d_ff=d_ff)

    # GateSkip: add token-wise layer skipping gates
    if use_gateskip:
        print("\n  Adding GateSkip (token-wise layer skipping)...")
        from research.gateskip import GateSkipBlock
        for i, block in enumerate(model.blocks):
            if not hasattr(block, 'skip_gate'):
                block.skip_gate = torch.nn.Linear(cfg.d_model, 1, bias=True).to(device)
                torch.nn.init.zeros_(block.skip_gate.weight)
                torch.nn.init.zeros_(block.skip_gate.bias)
        print(f"    Added gates to {len(model.blocks)} layers")

    # MTP: add multi-token prediction head
    if use_mtp:
        print("\n  Adding MTP (multi-token prediction head)...")
        from research.mtp import MTPHead
        model.mtp_head = MTPHead(
            d_model=cfg.d_model, vocab_size=cfg.vocab_size,
            n_predict=2,  # predict 2 future tokens
        ).to(device)
        print(f"    MTP head: predicts 2 future tokens")

    # SpinQuant: rotate weights for better quantization
    if use_spinquant:
        print("\n  Running SpinQuant calibration...")
        from research.spinquant import SpinQuantizer
        sq = SpinQuantizer(model, bits=2, lr=0.01, calib_steps=50)  # 2-bit for ternary
        sq.calibrate()
        sq.apply()
        print("    Rotation folded into weights")

    # Fine-tune to recover quality
    if finetune_steps > 0:
        print(f"\n  Fine-tuning for {finetune_steps} steps...")
        # TODO: integrate with research.distill or research.train
        # For now, this is a placeholder — the fine-tuning step needs
        # calibration data (Qwen teacher logits) and training loop
        print("    (fine-tuning not yet implemented in pipeline — use research.distill)")

    out = str(CHECKPOINTS / "forge_final.safetensors")
    print(f"\n  Saving final model to {out}...")
    # Save model state dict
    sd = model.state_dict()
    # Convert to bf16 for saving
    sd_bf16 = {k: v.to(torch.bfloat16) for k, v in sd.items()}
    from safetensors.torch import save_file
    save_file(sd_bf16, out, metadata={
        "pipeline": "forge",
        "config": config,
        "gateskip": str(use_gateskip),
        "mtp": str(use_mtp),
        "spinquant": str(use_spinquant),
    })
    print(f"  Saved to {out}")
    return out


def run_pipeline(src: str, config: str = "qwen25_coder_1.5b",
                 phase2: bool = False, **kwargs):
    """Run the full pipeline."""
    result = run_phase1(src, config,
                        skip_mla=kwargs.get('skip_mla', False),
                        skip_moe=kwargs.get('skip_moe', False),
                        skip_bitnet=kwargs.get('skip_bitnet', False),
                        kv_compression_dim=kwargs.get('kv_compression_dim', 512),
                        n_experts=kwargs.get('n_experts', 4),
                        top_k=kwargs.get('top_k', 2),
                        shared_expert=kwargs.get('shared_expert', True))

    if phase2:
        result = run_phase2(result, config,
                            use_spinquant=kwargs.get('use_spinquant', True),
                            use_gateskip=kwargs.get('use_gateskip', True),
                            use_mtp=kwargs.get('use_mtp', True),
                            finetune_steps=kwargs.get('finetune_steps', 1000))

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ForgeAI: Qwen → All-Architecture Model")
    parser.add_argument("--src", default="research/checkpoints/qwen25_coder_1.5b_ported.safetensors")
    parser.add_argument("--config", default="qwen25_coder_1.5b")
    parser.add_argument("--phase2", action="store_true", help="Run phase 2 (needs GPU)")
    parser.add_argument("--skip-mla", action="store_true")
    parser.add_argument("--skip-moe", action="store_true")
    parser.add_argument("--skip-bitnet", action="store_true")
    parser.add_argument("--kv-compression-dim", type=int, default=512)
    parser.add_argument("--n-experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--no-shared", action="store_true")
    parser.add_argument("--no-spinquant", action="store_true")
    parser.add_argument("--no-gateskip", action="store_true")
    parser.add_argument("--no-mtp", action="store_true")
    parser.add_argument("--finetune-steps", type=int, default=1000)
    args = parser.parse_args()

    run_pipeline(args.src, args.config,
                 phase2=args.phase2,
                 skip_mla=args.skip_mla, skip_moe=args.skip_moe,
                 skip_bitnet=args.skip_bitnet,
                 kv_compression_dim=args.kv_compression_dim,
                 n_experts=args.n_experts, top_k=args.top_k,
                 shared_expert=not args.no_shared,
                 use_spinquant=not args.no_spinquant,
                 use_gateskip=not args.no_gateskip,
                 use_mtp=not args.no_mtp,
                 finetune_steps=args.finetune_steps)
