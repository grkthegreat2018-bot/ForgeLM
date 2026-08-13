"""Port LFM2.5-1.2B-Instruct weights into ForgeAI architecture.

LFM2.5 and ForgeAI share identical architecture at the base level:
  - Double-gated short conv blocks (our DoubleGatedConvLayer = LFM2 Lfm2ShortConv)
  - GQA attention with QK-layernorm (our GQA + use_qk_norm = LFM2 Lfm2Attention)
  - SwiGLU FFN (our SwiGLUFFN = LFM2 Lfm2MLP, just different param names)
  - RMSNorm, RoPE theta=1M, tied embeddings

This is a 100% lossless port — no SVD resize, no dimension mismatch, no
approximation. Every weight maps directly.

Weight name mapping:
  LFM2                                    → ForgeAI
  model.embed_tokens.weight               → embed.weight
  model.embedding_norm.weight             → ln_f.weight
  lm_head.weight                          → head.weight (tied to embed)
  model.layers.{i}.operator_norm.weight   → blocks.{i}.ln1.weight
  model.layers.{i}.ffn_norm.weight        → blocks.{i}.ln2.weight
  # Conv layers:
  model.layers.{i}.conv.in_proj.weight    → blocks.{i}.attn.in_proj.weight
  model.layers.{i}.conv.conv.weight       → blocks.{i}.attn.conv.weight
  model.layers.{i}.conv.out_proj.weight   → blocks.{i}.attn.out_proj.weight
  # Attention layers:
  model.layers.{i}.self_attn.q_proj.weight   → blocks.{i}.attn.q_proj.weight
  model.layers.{i}.self_attn.k_proj.weight   → blocks.{i}.attn.k_proj.weight
  model.layers.{i}.self_attn.v_proj.weight   → blocks.{i}.attn.v_proj.weight
  model.layers.{i}.self_attn.out_proj.weight → blocks.{i}.attn.out_proj.weight
  model.layers.{i}.self_attn.q_layernorm.weight → blocks.{i}.attn.q_norm.weight
  model.layers.{i}.self_attn.k_layernorm.weight → blocks.{i}.attn.k_norm.weight
  # FFN (all layers):
  model.layers.{i}.feed_forward.w1.weight → blocks.{i}.ffn.w_gate.weight
  model.layers.{i}.feed_forward.w3.weight → blocks.{i}.ffn.w_up.weight
  model.layers.{i}.feed_forward.w2.weight → blocks.{i}.ffn.w_down.weight
"""
import os
import sys
import time

import torch
from safetensors.torch import load_file as load_safetensors

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from research.config import get_config
from research.model_loader import ConfigurableResearchLLM, ModelLoader


def build_weight_mapping(n_layers: int, layer_types: list[str]) -> dict:
    """Build the LFM2→ForgeAI weight name mapping.

    Returns dict of {lfm2_name: forgeai_name}.
    """
    mapping = {
        "model.embed_tokens.weight": "embed.weight",
        "model.embedding_norm.weight": "ln_f.weight",
        "lm_head.weight": "head.weight",  # tied, but we copy anyway
    }

    for i in range(n_layers):
        ltype = layer_types[i].lower()
        is_attn = ltype in ("full_attention", "attention", "attn")

        # Norms (all layers)
        mapping[f"model.layers.{i}.operator_norm.weight"] = f"blocks.{i}.ln1.weight"
        mapping[f"model.layers.{i}.ffn_norm.weight"] = f"blocks.{i}.ln2.weight"

        # Sequence mixer (conv or attention)
        if is_attn:
            mapping[f"model.layers.{i}.self_attn.q_proj.weight"] = f"blocks.{i}.attn.q_proj.weight"
            mapping[f"model.layers.{i}.self_attn.k_proj.weight"] = f"blocks.{i}.attn.k_proj.weight"
            mapping[f"model.layers.{i}.self_attn.v_proj.weight"] = f"blocks.{i}.attn.v_proj.weight"
            mapping[f"model.layers.{i}.self_attn.out_proj.weight"] = f"blocks.{i}.attn.out_proj.weight"
            mapping[f"model.layers.{i}.self_attn.q_layernorm.weight"] = f"blocks.{i}.attn.q_norm.weight"
            mapping[f"model.layers.{i}.self_attn.k_layernorm.weight"] = f"blocks.{i}.attn.k_norm.weight"
        else:
            # Conv block
            mapping[f"model.layers.{i}.conv.in_proj.weight"] = f"blocks.{i}.attn.in_proj.weight"
            mapping[f"model.layers.{i}.conv.conv.weight"] = f"blocks.{i}.attn.conv.weight"
            mapping[f"model.layers.{i}.conv.out_proj.weight"] = f"blocks.{i}.attn.out_proj.weight"

        # FFN (all layers, SwiGLU)
        mapping[f"model.layers.{i}.feed_forward.w1.weight"] = f"blocks.{i}.ffn.w_gate.weight"
        mapping[f"model.layers.{i}.feed_forward.w3.weight"] = f"blocks.{i}.ffn.w_up.weight"
        mapping[f"model.layers.{i}.feed_forward.w2.weight"] = f"blocks.{i}.ffn.w_down.weight"

    return mapping


def load_lfm25_safetensors(checkpoint_dir: str) -> dict:
    """Load all LFM2.5 safetensors files from a directory.

    Handles both single-file and multi-file (sharded) checkpoints.
    """
    state_dict = {}
    safetensors_files = sorted([
        f for f in os.listdir(checkpoint_dir)
        if f.endswith(".safetensors")
    ])
    if not safetensors_files:
        raise FileNotFoundError(f"No .safetensors files found in {checkpoint_dir}")

    for fname in safetensors_files:
        fpath = os.path.join(checkpoint_dir, fname)
        print(f"  Loading {fname}...")
        partial = load_safetensors(fpath)
        state_dict.update(partial)
        print(f"    {len(partial)} tensors loaded")

    return state_dict


def port_lfm25_to_forgeai(
    checkpoint_dir: str,
    output_path: str,
    config_name: str = "lfm25_1.2b",
    device: str = "cpu",
):
    """Port LFM2.5 weights into ForgeAI architecture and save.

    Args:
        checkpoint_dir: Directory containing LFM2.5 .safetensors files.
        output_path: Path to save the ported ForgeAI checkpoint (.safetensors).
        config_name: ForgeAI config preset name (must match LFM2.5 architecture).
        device: Device to build the model on.
    """
    t0 = time.time()

    # 1. Get config and build blank ForgeAI model
    print(f"[1] Building blank ForgeAI model (config: {config_name})...")
    cfg = get_config(config_name)
    cfg_dict = {**cfg.__dict__, "device": device}
    cfg = type(cfg)(**cfg_dict)
    model = ConfigurableResearchLLM(cfg)
    forge_state = model.state_dict()
    print(f"  Model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
    print(f"  {len(forge_state)} weight tensors")

    # 2. Load LFM2.5 safetensors
    print(f"\n[2] Loading LFM2.5 safetensors from {checkpoint_dir}...")
    lfm_state = load_lfm25_safetensors(checkpoint_dir)
    print(f"  {len(lfm_state)} tensors loaded")

    # 3. Build weight mapping
    layer_types = cfg.layer_types or ["attention"] * cfg.n_layers
    mapping = build_weight_mapping(cfg.n_layers, layer_types)

    # 4. Port weights
    print(f"\n[3] Porting weights...")
    ported = 0
    skipped = 0
    shape_mismatches = []

    for lfm_name, forge_name in mapping.items():
        if lfm_name not in lfm_state:
            print(f"  WARNING: LFM2 weight '{lfm_name}' not found in checkpoint!")
            skipped += 1
            continue
        if forge_name not in forge_state:
            print(f"  WARNING: ForgeAI weight '{forge_name}' not found in model!")
            skipped += 1
            continue

        lfm_tensor = lfm_state[lfm_name]
        forge_tensor = forge_state[forge_name]

        if lfm_tensor.shape != forge_tensor.shape:
            shape_mismatches.append(
                f"  {lfm_name} {list(lfm_tensor.shape)} != {forge_name} {list(forge_tensor.shape)}"
            )
            continue

        # Copy weight data
        forge_state[forge_name] = lfm_tensor.to(forge_tensor.dtype)
        ported += 1

    # 5. Report
    print(f"\n  Ported: {ported} tensors")
    print(f"  Skipped: {skipped} tensors")
    if shape_mismatches:
        print(f"  Shape mismatches: {len(shape_mismatches)}")
        for m in shape_mismatches:
            print(f"    {m}")

    # Check for unmapped LFM2 weights
    unmapped = set(lfm_state.keys()) - set(mapping.keys())
    if unmapped:
        print(f"\n  Unmapped LFM2 weights ({len(unmapped)}):")
        for name in sorted(unmapped)[:20]:
            print(f"    {name}: {list(lfm_state[name].shape)}")
        if len(unmapped) > 20:
            print(f"    ... and {len(unmapped) - 20} more")

    # Check for unfilled ForgeAI weights
    unfilled = set(forge_state.keys()) - set(mapping.values())
    if unfilled:
        print(f"\n  Unfilled ForgeAI weights ({len(unfilled)}):")
        for name in sorted(unfilled)[:20]:
            print(f"    {name}: {list(forge_state[name].shape)}")

    # 6. Load ported weights into model
    print(f"\n[4] Loading ported weights into model...")
    missing, unexpected = model.load_state_dict(forge_state, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
        for k in missing[:10]:
            print(f"    {k}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
        for k in unexpected[:10]:
            print(f"    {k}")

    # 7. Mark QK-norm as non-identity (we loaded real weights)
    for block in model.blocks:
        if hasattr(block.attn, '_qk_norm_identity'):
            block.attn._qk_norm_identity = False

    # 8. Save ported checkpoint
    print(f"\n[5] Saving ported checkpoint to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    from safetensors.torch import save_file
    # Handle tied weights: save only one copy, skip the tied duplicate.
    # head.weight is tied to embed.weight — save embed.weight, skip head.weight.
    save_dict = {}
    tied_keys = set()
    if not getattr(cfg, 'use_pit', False):
        tied_keys.add("head.weight")  # tied to embed.weight
    for k, v in model.state_dict().items():
        if k in tied_keys:
            print(f"  Skipping tied weight: {k}")
            continue
        # Convert to bf16 to match LFM2.5 original precision (halves file size)
        save_dict[k] = v.contiguous().to(torch.bfloat16).clone()
    save_file(save_dict, output_path)
    fsize = os.path.getsize(output_path) / 1e9
    print(f"  Saved: {fsize:.2f} GB (bf16)")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Ported: {ported}/{len(mapping)} tensors")
    print(f"  Shape mismatches: {len(shape_mismatches)}")
    print(f"  File: {output_path}")

    return model


def verify_ported_model(
    checkpoint_path: str,
    config_name: str = "lfm25_1.2b",
    device: str = "cpu",
):
    """Verify the ported model by loading it and running a forward pass."""
    print(f"\n=== Verifying ported model ===")

    cfg = get_config(config_name)
    cfg_dict = {**cfg.__dict__, "device": device}
    cfg = type(cfg)(**cfg_dict)
    model = ConfigurableResearchLLM(cfg)

    print("Loading checkpoint...")
    state = load_safetensors(checkpoint_path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    # Mark QK-norm as loaded
    for block in model.blocks:
        if hasattr(block.attn, '_qk_norm_identity'):
            block.attn._qk_norm_identity = False

    model.eval()
    model = model.to(device)

    # Forward pass test
    print("Running forward pass...")
    with torch.no_grad():
        input_ids = torch.randint(0, cfg.vocab_size, (1, 16), device=device)
        logits, loss = model(input_ids)
        print(f"  Logits shape: {logits.shape}")
        print(f"  Logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")

    # Generation test
    print("Generating text...")
    from research.inference.decoding import StandardDecoding
    decoder = StandardDecoding()
    prompt = torch.tensor([[1, 100, 200, 300]], device=device)  # dummy tokens
    with torch.no_grad():
        tokens = decoder.generate(model, prompt, max_new_tokens=20)
    print(f"  Generated tokens: {tokens[0].tolist()}")

    print("Verification complete!")
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Port LFM2.5 to ForgeAI")
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                        help="Directory containing LFM2.5 .safetensors files")
    parser.add_argument("--output", type=str,
                        default="research/checkpoints/lfm25_ported.safetensors",
                        help="Output checkpoint path")
    parser.add_argument("--config", type=str, default="lfm25_1.2b",
                        help="ForgeAI config preset name")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--verify", action="store_true",
                        help="Verify the ported model after porting")
    args = parser.parse_args()

    model = port_lfm25_to_forgeai(
        checkpoint_dir=args.checkpoint_dir,
        output_path=args.output,
        config_name=args.config,
        device=args.device,
    )

    if args.verify:
        verify_ported_model(args.output, config_name=args.config, device=args.device)
