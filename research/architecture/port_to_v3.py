"""Port LFM2.5 base weights into V3 architecture and save as a new checkpoint.

This bakes the arch key conversions (GQA→diff-attn, BitNet qscale, TITAN init,
MoD router) into the checkpoint file itself, so loading is a straight
state_dict match with no runtime conversion needed.

Usage:
    python -m research.architecture.port_to_v3 \\
        --input research/checkpoints/ForgeLM_V2_LFM25-1.2B.safetensors \\
        --output research/checkpoints/forgelm_v7_Base.safetensors
"""
import argparse
import os
import sys
import time

import torch
from safetensors.torch import save_file

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from research.config import get_config
from research.model_loader import ConfigurableResearchLLM
from safetensors.torch import load_file as load_safetensors


def port_to_v3(input_path: str, output_path: str, config_name: str = "forgelm_v7"):
    """Port base LFM2.5 weights into a ForgeLM architecture checkpoint.

    Historically named "port_to_v3" but now defaults to forgelm_v7 config.
    The conversion steps depend on the target config's architecture keys:

    1. Build blank model with the target config (all keys identity/zero-init)
    2. Load base LFM2.5 safetensors
    3. Apply GQA→GTA weight conversion (V=K identity, lossless)
    4. Initialize BitNet qscale from loaded weights (if use_bitnet)
    5. TITAN memory zero-init (lossless, if use_titan_memory)
    6. MoD router keep-all (lossless, if use_mod)
    7. Save full state_dict as new checkpoint
    """
    t0 = time.time()

    # 1. Build blank V3 model
    print(f"[1] Building blank V3 model (config: {config_name})...")
    cfg = get_config(config_name)
    model = ConfigurableResearchLLM(cfg)
    forge_state = model.state_dict()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params / 1e6:.1f}M params, {len(forge_state)} weight tensors")

    # 2. Load base LFM2.5 checkpoint
    print(f"\n[2] Loading base checkpoint from {input_path}...")
    base_state = load_safetensors(input_path)
    print(f"  {len(base_state)} tensors loaded")

    # 3. Apply GQA→diff-attn conversion if needed
    if cfg.attn_type == "diff":
        print(f"\n[3] Applying GQA -> diff-attn weight conversion...")
        from research.keys.attention.differential_attn_key import DifferentialAttentionKey
        res = DifferentialAttentionKey(
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            identity=True,  # warm start: lambda=0, lossless
        ).forward(base_state)
        if res.success:
            base_state = res.weights
            print(f"  Conversion applied (lambda=0, lossless)")
        else:
            print(f"  WARNING: Conversion failed: {res.message}")

    # 4. Load into model (maps base weights onto V3 arch tensors)
    print(f"\n[4] Loading weights into V3 model...")
    missing, unexpected = model.load_state_dict(base_state, strict=False)
    print(f"  Missing keys: {len(missing)}")
    if missing:
        for k in missing[:10]:
            print(f"    {k}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")
    print(f"  Unexpected keys: {len(unexpected)}")
    if unexpected:
        for k in unexpected[:10]:
            print(f"    {k}")

    # 5. Initialize BitNet qscale from loaded FFN weights
    if cfg.use_bitnet:
        print(f"\n[5] Initializing BitNet qscale from loaded weights...")
        from research.keys.quantization.bitnet_b158_key import BitNetLinear
        n_init = 0
        for block in model.blocks:
            for module in [block.ffn.w_gate, block.ffn.w_up, block.ffn.w_down]:
                if isinstance(module, BitNetLinear) and hasattr(module, '_reanchor_qscale'):
                    module._reanchor_qscale()
                    n_init += 1
        print(f"  Initialized {n_init} qscale values")

    # 6. Mark QK-norm as non-identity (we loaded real weights)
    for block in model.blocks:
        if hasattr(block.attn, '_qk_norm_identity'):
            block.attn._qk_norm_identity = False

    # 7. Save full V3 state_dict
    print(f"\n[6] Saving V3 checkpoint to {output_path}...")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    save_dict = {}
    tied_keys = set()
    if not getattr(cfg, 'use_pit', False):
        tied_keys.add("head.weight")  # tied to embed.weight
    for k, v in model.state_dict().items():
        if k in tied_keys:
            print(f"  Skipping tied weight: {k}")
            continue
        save_dict[k] = v.contiguous().to(torch.bfloat16).clone()

    save_file(save_dict, output_path)
    fsize = os.path.getsize(output_path) / 1e9
    print(f"  Saved: {fsize:.2f} GB (bf16)")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Output: {output_path}")
    print(f"  Tensors: {len(save_dict)}")
    print(f"  Params: {n_params / 1e6:.1f}M")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Port LFM2.5 base weights into V3 architecture checkpoint")
    parser.add_argument("--input", type=str, required=True,
                        help="Input LFM2.5 safetensors checkpoint")
    parser.add_argument("--output", type=str, required=True,
                        help="Output V3 safetensors checkpoint")
    parser.add_argument("--config", type=str, default="forgelm_v7",
                        help="V3 config preset name")
    args = parser.parse_args()

    port_to_v3(args.input, args.output, args.config)
