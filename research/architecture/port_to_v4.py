"""Port LFM2.5 base weights into V4 architecture and save as a new checkpoint.

Bakes the arch key conversions (GQA→GTA, fused QKV+Gate-Up GEMM, BitNet qscale,
TITAN init, MoD router, MHC, AttnRes, PIT) into the checkpoint file itself, so
loading is a straight state_dict match with no runtime conversion needed.

V4 vs V3 differences:
  - attn_type="gta" (Grouped-Tied Attention, V=K identity warm start, gate=0)
  - use_fused_gemm=True (FusedQKVLinear + FusedGateUpLinear)
  - All V3 keys preserved (BitNet, TITAN, MoD, MHC, AttnRes, PIT)

Usage:
    python -m research.architecture.port_to_v4 \\
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
from research.model_loader import ConfigurableResearchLLM, ModelLoader
from safetensors.torch import load_file as load_safetensors


def port_to_v4(input_path: str, output_path: str, config_name: str = "forgelm_v7"):
    """Port base LFM2.5 weights into V4 architecture checkpoint.

    1. Build blank V4 model (GTA, fused GEMM, BitNet, TITAN, MoD, MHC, AttnRes, PIT)
    2. Load base LFM2.5 safetensors
    3. ModelLoader applies GQA→GTA conversion (V=K, gate=0, lossless)
    4. Fused GEMM weights are assembled from base Q/K/V and gate/up
    5. Initialize BitNet qscale from loaded weights
    6. TITAN memory zero-init (lossless)
    7. MoD router keep-all (lossless)
    8. Save full state_dict as new V4-native checkpoint
    """
    t0 = time.time()

    # 1. Build blank V4 model
    print(f"[1] Building blank V4 model (config: {config_name})...")
    cfg = get_config(config_name)
    model = ConfigurableResearchLLM(cfg)
    forge_state = model.state_dict()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params / 1e6:.1f}M params, {len(forge_state)} weight tensors")
    print(f"  attn_type={cfg.attn_type}, use_fused_gemm={getattr(cfg, 'use_fused_gemm', False)}")

    # 2. Load base LFM2.5 checkpoint
    print(f"\n[2] Loading base checkpoint from {input_path}...")
    base_state = load_safetensors(input_path)
    print(f"  {len(base_state)} tensors loaded")

    # 3. Apply GQA→GTA conversion if needed (V=K identity warm start, gate=0)
    if cfg.attn_type == "gta":
        print(f"\n[3] Applying GQA -> GTA weight conversion (V=K, gate=0, lossless)...")
        from research.keys.attention.gta_key import GTAKey
        res = GTAKey(
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
        ).forward(base_state)
        if res.success:
            base_state = res.weights
            print(f"  GTA conversion applied (V=K, gate=0, lossless)")
        else:
            print(f"  WARNING: GTA conversion failed: {res.message}")
    elif cfg.attn_type == "diff":
        print(f"\n[3] Applying GQA -> diff-attn weight conversion...")
        from research.keys.attention.differential_attn_key import DifferentialAttentionKey
        res = DifferentialAttentionKey(
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            identity=True,
        ).forward(base_state)
        if res.success:
            base_state = res.weights
            print(f"  diff-attn conversion applied (lambda=0, lossless)")
        else:
            print(f"  WARNING: diff-attn conversion failed: {res.message}")

    # 4. Load weights into V4 model (maps base weights onto V4 arch tensors,
    #    including fused GEMM weight assembly)
    print(f"\n[4] Loading weights into V4 model...")
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
        if len(unexpected) > 10:
            print(f"    ... and {len(unexpected) - 10} more")

    # 5. Initialize BitNet qscale from loaded FFN weights
    if getattr(cfg, 'use_bitnet', False):
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

    # 7. Save full V4 state_dict
    print(f"\n[6] Saving V4 checkpoint to {output_path}...")
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
        description="Port LFM2.5 base weights into V4 architecture checkpoint")
    parser.add_argument("--input", type=str, required=True,
                        help="Input LFM2.5 safetensors checkpoint")
    parser.add_argument("--output", type=str, required=True,
                        help="Output V4 safetensors checkpoint")
    parser.add_argument("--config", type=str, default="forgelm_v7",
                        help="V4 config preset name")
    args = parser.parse_args()

    port_to_v4(args.input, args.output, args.config)
