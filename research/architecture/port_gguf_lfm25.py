"""Port LFM2.5-1.2B-Instruct from GGUF (Q8_0) into ForgeAI lfm25_1.2b model.

Dequantizes Q8_0 tensors to float32, maps GGUF tensor names to ForgeAI
weight names, loads into ConfigurableResearchLLM, saves as safetensors.

Usage:
    python -m research.architecture.port_gguf_lfm25 \
        --gguf "D:/LMstudio/.../LFM2.5-1.2B-Instruct-Q8_0.gguf" \
        --output research/checkpoints/ForgeLM_V2_LFM25-1.2B.safetensors
"""
import os
import sys
import struct
import time
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from safetensors.torch import save_file
from research.config import get_config
from research.model_loader import ConfigurableResearchLLM


# ── Q8_0 Dequantization ────────────────────────────────────────────────────

def dequantize_q8_0(data: np.ndarray, shape: tuple) -> np.ndarray:
    """Dequantize GGUF Q8_0 block format to float32.

    Q8_0 layout: blocks of 32 elements.
    Each block: 2 bytes (f16 scale) + 32 bytes (32 int8 values) = 34 bytes.
    Dequantized value = scale * (q / 127.0)  (approximately, but actually
    the scale is the f16 absmax and q is the int8 value scaled to [-127, 127]).

    The tensor data may be stored in a padded/aligned layout. We read the
    raw bytes and reconstruct.
    """
    n_elements = int(np.prod(shape))
    n_blocks = (n_elements + 31) // 32
    block_size = 34  # 2 (f16 scale) + 32 (int8)

    # data is a np.uint8 array of raw bytes
    raw = np.frombuffer(data.tobytes(), dtype=np.uint8)

    out = np.zeros(n_elements, dtype=np.float32)

    for i in range(n_blocks):
        offset = i * block_size
        # Read f16 scale (little-endian)
        scale_bytes = raw[offset:offset + 2]
        scale = np.frombuffer(scale_bytes.tobytes(), dtype=np.float16)[0].astype(np.float32)

        # Read 32 int8 values
        q_start = offset + 2
        q_end = q_start + 32
        qs = raw[q_start:q_end].astype(np.int8).astype(np.float32)

        # Dequantize: x = scale * q (NOT scale * q / 127)
        # Q8_0 quantization: d = amax/127, q = round(x/d)
        # So: x = d * q
        vals = scale * qs

        # Write to output
        elem_start = i * 32
        elem_end = min(elem_start + 32, n_elements)
        out[elem_start:elem_end] = vals[:elem_end - elem_start]

    return out.reshape(shape)


def dequantize_tensor(gguf_tensor) -> torch.Tensor:
    """Dequantize a single GGUF tensor to a torch float32 tensor.

    Handles F32 (direct copy) and Q8_0 (block dequantization).
    """
    shape = tuple(int(s) for s in gguf_tensor.shape)
    tensor_type = gguf_tensor.tensor_type.name

    if tensor_type == "F32":
        # Direct float32 data
        data = np.frombuffer(gguf_tensor.data.tobytes(), dtype=np.float32)
        # Handle padding: data may be larger than needed
        n_elements = int(np.prod(shape))
        data = data[:n_elements]
        return torch.from_numpy(data.reshape(shape).copy())

    elif tensor_type == "F16":
        data = np.frombuffer(gguf_tensor.data.tobytes(), dtype=np.float16)
        n_elements = int(np.prod(shape))
        data = data[:n_elements]
        return torch.from_numpy(data.astype(np.float32).reshape(shape).copy())

    elif tensor_type == "BF16":
        # BF16 is stored as uint16, reinterpret
        raw = np.frombuffer(gguf_tensor.data.tobytes(), dtype=np.uint16)
        n_elements = int(np.prod(shape))
        raw = raw[:n_elements]
        # Convert bf16 (uint16) to float32
        # BF16: upper 16 bits of float32
        u32 = raw.astype(np.uint32) << 16
        data = u32.view(np.float32)
        return torch.from_numpy(data.astype(np.float32).reshape(shape).copy())

    elif tensor_type == "Q8_0":
        data = dequantize_q8_0(gguf_tensor.data, shape)
        return torch.from_numpy(data.copy())

    else:
        raise ValueError(f"Unsupported tensor type: {tensor_type}")


# ── GGUF → ForgeAI Name Mapping ────────────────────────────────────────────

# Attention layers: 2, 5, 8, 10, 12, 14
# Conv layers: 0, 1, 3, 4, 6, 7, 9, 11, 13, 15
ATTN_LAYERS = {2, 5, 8, 10, 12, 14}


def build_gguf_to_forgeai_mapping(n_layers: int = 16) -> dict:
    """Map GGUF tensor names → ForgeAI state_dict keys."""
    mapping = {
        "token_embd.weight": "embed.weight",
        "token_embd_norm.weight": "ln_f.weight",
        # head.weight is tied to embed.weight — skip
    }

    for i in range(n_layers):
        if i in ATTN_LAYERS:
            # Attention layer
            mapping[f"blk.{i}.attn_norm.weight"] = f"blocks.{i}.ln1.weight"
            mapping[f"blk.{i}.ffn_norm.weight"] = f"blocks.{i}.ln2.weight"
            mapping[f"blk.{i}.attn_q.weight"] = f"blocks.{i}.attn.q_proj.weight"
            mapping[f"blk.{i}.attn_k.weight"] = f"blocks.{i}.attn.k_proj.weight"
            mapping[f"blk.{i}.attn_v.weight"] = f"blocks.{i}.attn.v_proj.weight"
            mapping[f"blk.{i}.attn_output.weight"] = f"blocks.{i}.attn.out_proj.weight"
            mapping[f"blk.{i}.attn_q_norm.weight"] = f"blocks.{i}.attn.q_norm.weight"
            mapping[f"blk.{i}.attn_k_norm.weight"] = f"blocks.{i}.attn.k_norm.weight"
        else:
            # Conv layer
            mapping[f"blk.{i}.attn_norm.weight"] = f"blocks.{i}.ln1.weight"
            mapping[f"blk.{i}.ffn_norm.weight"] = f"blocks.{i}.ln2.weight"
            mapping[f"blk.{i}.shortconv.in_proj.weight"] = f"blocks.{i}.attn.in_proj.weight"
            mapping[f"blk.{i}.shortconv.conv.weight"] = f"blocks.{i}.attn.conv.weight"
            mapping[f"blk.{i}.shortconv.out_proj.weight"] = f"blocks.{i}.attn.out_proj.weight"

        # FFN (all layers)
        mapping[f"blk.{i}.ffn_gate.weight"] = f"blocks.{i}.ffn.w_gate.weight"
        mapping[f"blk.{i}.ffn_up.weight"] = f"blocks.{i}.ffn.w_up.weight"
        mapping[f"blk.{i}.ffn_down.weight"] = f"blocks.{i}.ffn.w_down.weight"

    return mapping


# ── Main Porting Function ──────────────────────────────────────────────────

def port_gguf_to_forgeai(
    gguf_path: str,
    output_path: str,
    config_name: str = "lfm25_1.2b",
    device: str = "cpu",
):
    """Port LFM2.5 GGUF weights into ForgeAI and save as safetensors."""
    import gguf

    t0 = time.time()
    print(f"[1] Reading GGUF: {gguf_path}")
    reader = gguf.GGUFReader(gguf_path)
    print(f"  {len(reader.tensors)} tensors")

    # Build name → tensor index
    gguf_tensors = {t.name: t for t in reader.tensors}

    # Build mapping
    mapping = build_gguf_to_forgeai_mapping(16)

    # Build ForgeAI model to get expected shapes
    print(f"\n[2] Building ForgeAI model (config: {config_name})...")
    cfg = get_config(config_name)
    cfg_dict = {**cfg.__dict__, "device": device, "dtype": "float32"}
    cfg = type(cfg)(**cfg_dict)
    model = ConfigurableResearchLLM(cfg)
    forge_state = model.state_dict()
    print(f"  {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params, {len(forge_state)} tensors")

    # Port weights
    print(f"\n[3] Dequantizing and porting weights...")
    ported = 0
    skipped = 0
    shape_mismatches = []

    for gguf_name, forge_name in mapping.items():
        if gguf_name not in gguf_tensors:
            print(f"  WARNING: GGUF tensor '{gguf_name}' not found!")
            skipped += 1
            continue
        if forge_name not in forge_state:
            print(f"  WARNING: ForgeAI weight '{forge_name}' not in model!")
            skipped += 1
            continue

        gguf_t = gguf_tensors[gguf_name]
        forge_t = forge_state[forge_name]

        # Dequantize
        try:
            deq = dequantize_tensor(gguf_t)
        except Exception as e:
            print(f"  ERROR dequantizing {gguf_name}: {e}")
            skipped += 1
            continue

        # Check shape — GGUF stores [out, in], ForgeAI also [out, in]
        # But conv.weight is [out, in, kernel] in ForgeAI vs [kernel, out] in GGUF
        if forge_name.endswith("attn.conv.weight"):
            # GGUF: [3, 2048] → ForgeAI: [2048, 1, 3]
            deq = deq.T.unsqueeze(1)  # [3, 2048] → [2048, 3] → [2048, 1, 3]

        if deq.shape != forge_t.shape:
            # Try transpose for 2D weights (GGUF may store transposed)
            if deq.dim() == 2 and forge_t.dim() == 2 and deq.T.shape == forge_t.shape:
                deq = deq.T
            else:
                shape_mismatches.append(
                    f"  {gguf_name} {list(deq.shape)} != {forge_name} {list(forge_t.shape)}"
                )
                continue

        forge_state[forge_name] = deq.to(forge_t.dtype)
        ported += 1

    print(f"\n  Ported: {ported} tensors")
    print(f"  Skipped: {skipped} tensors")
    if shape_mismatches:
        print(f"  Shape mismatches: {len(shape_mismatches)}")
        for m in shape_mismatches:
            print(f"    {m}")

    # Load into model
    print(f"\n[4] Loading ported weights into model...")
    missing, unexpected = model.load_state_dict(forge_state, strict=False)
    if missing:
        print(f"  Missing: {len(missing)}")
        for k in missing[:5]:
            print(f"    {k}")
    if unexpected:
        print(f"  Unexpected: {len(unexpected)}")
        for k in unexpected[:5]:
            print(f"    {k}")

    # Mark QK-norm as non-identity
    for block in model.blocks:
        if hasattr(block.attn, '_qk_norm_identity'):
            block.attn._qk_norm_identity = False

    # Save
    print(f"\n[5] Saving to {output_path}...")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_dict = {}
    # Skip tied head.weight (it's tied to embed.weight)
    for k, v in model.state_dict().items():
        if k == "head.weight":
            print(f"  Skipping tied weight: {k}")
            continue
        save_dict[k] = v.contiguous().to(torch.bfloat16).clone()
    save_file(save_dict, output_path)
    fsize = os.path.getsize(output_path) / 1e9
    print(f"  Saved: {fsize:.2f} GB (bf16)")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    return model


def verify_ported_model(
    checkpoint_path: str,
    config_name: str = "lfm25_1.2b",
    device: str = "cpu",
):
    """Verify the ported model by loading it and running a forward pass."""
    from safetensors.torch import load_file as load_safetensors

    print(f"\n=== Verifying ported model ===")
    cfg = get_config(config_name)
    cfg_dict = {**cfg.__dict__, "device": device, "dtype": "float32"}
    cfg = type(cfg)(**cfg_dict)
    model = ConfigurableResearchLLM(cfg)

    state = load_safetensors(checkpoint_path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    for block in model.blocks:
        if hasattr(block.attn, '_qk_norm_identity'):
            block.attn._qk_norm_identity = False

    model.eval()
    model = model.to(device)

    # Forward pass
    with torch.no_grad():
        input_ids = torch.randint(0, cfg.vocab_size, (1, 16), device=device)
        logits, loss = model(input_ids)
        print(f"  Logits shape: {logits.shape}")
        print(f"  Logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
        print(f"  Logits finite: {torch.isfinite(logits).all().item()}")

    print("Verification complete!")
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Port LFM2.5 GGUF to ForgeAI")
    parser.add_argument("--gguf", type=str, required=True)
    parser.add_argument("--output", type=str,
                        default="research/checkpoints/ForgeLM_V2_LFM25-1.2B.safetensors")
    parser.add_argument("--config", type=str, default="lfm25_1.2b")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    model = port_gguf_to_forgeai(args.gguf, args.output, args.config, args.device)
    if args.verify:
        verify_ported_model(args.output, args.config, args.device)
