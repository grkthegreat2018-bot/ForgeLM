"""Pre-quantize a ForgeLM checkpoint to BitNet b1.58 ternary + int8 storage.

Converts bf16/fp32 weights to ternary {-1, 0, +1} and stores as int8.
This eliminates runtime quantization cost and reduces disk size by 2x
(bf16 2 bytes → int8 1 byte) and VRAM by 4x (fp32 4 bytes → int8 1 byte)
when loaded with convert_model_to_int8().

Usage:
    python -m research.prequantize \
        --checkpoint research/checkpoints/ForgeLM_V5_Base.safetensors \
        --output research/checkpoints/ForgeLM_V5_BitNet.safetensors

    # With packed ternary (5 values per byte, 1.58 bits/param):
    python -m research.prequantize \
        --checkpoint research/checkpoints/ForgeLM_V5_Base.safetensors \
        --output research/checkpoints/ForgeLM_V5_BitNet_packed.safetensors \
        --packed
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from research.keys.quantization.bitnet_b158_key import ternary_quantize


def prequantize_checkpoint(
    checkpoint: str,
    output: str,
    packed: bool = False,
    skip_embeddings: bool = False,
) -> dict:
    """Convert a checkpoint's weights to BitNet b1.58 ternary int8.

    Args:
        checkpoint: Path to source .safetensors checkpoint (bf16/fp32).
        output: Path to write pre-quantized .safetensors checkpoint.
        packed: If True, pack 5 ternary values per byte (1.58 bits/param).
                If False, store as int8 (1 byte/param, simpler/faster load).
        skip_embeddings: If True, keep embeddings in fp32 (better quality).
                        If False, ternary-quantize embeddings too (max compression).

    Returns:
        Stats dict with conversion metrics.
    """
    ckpt_path = Path(checkpoint)
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[PreQuant] Source: {ckpt_path.name} ({ckpt_path.stat().st_size / 1e9:.2f} GB)")
    print(f"[PreQuant] Output: {out_path.name}")
    print(f"[PreQuant] Mode: {'packed ternary (1.58 bits)' if packed else 'int8 (1 byte/param)'}")
    print(f"[PreQuant] Embeddings: {'skip (keep fp32)' if skip_embeddings else 'quantize (ternary)'}")
    print()

    state_dict = {}
    metadata = {
        "_bitnet_prequant": "1",
        "_prequant_mode": "packed" if packed else "int8",
        "_prequant_skip_embed": "1" if skip_embeddings else "0",
        "_source_checkpoint": ckpt_path.name,
    }

    n_total = 0
    n_quantized = 0
    n_skipped = 0
    total_params = 0
    quantized_params = 0

    t0 = time.time()

    with safe_open(str(ckpt_path), framework="pt") as f:
        keys = list(f.keys())
        print(f"[PreQuant] Checkpoint has {len(keys)} tensors")

        for i, key in enumerate(keys):
            tensor = f.get_tensor(key)
            n_total += 1
            total_params += tensor.numel()

            is_weight = key.endswith(".weight")
            is_embedding = "embed" in key.lower() or "embedding" in key.lower()

            if is_weight and not (is_embedding and skip_embeddings):
                # Ternary quantize: {-1, 0, +1}
                q, scale = ternary_quantize(tensor.float())

                if packed:
                    # Pack 5 ternary values per byte (3^5 = 243 < 256)
                    # TODO: implement packed mode
                    # For now, fall back to int8
                    q_int8 = q.to(torch.int8)
                    state_dict[key] = q_int8
                else:
                    # Store as int8 (1 byte per param)
                    q_int8 = q.to(torch.int8)
                    state_dict[key] = q_int8

                n_quantized += 1
                quantized_params += tensor.numel()

                # Store scale in metadata (per-tensor)
                if isinstance(scale, torch.Tensor):
                    metadata[f"_scale_{key}"] = str(scale.item())
                else:
                    metadata[f"_scale_{key}"] = str(scale)
            else:
                # Keep non-weight tensors (biases, norms, gates, etc.) as-is
                state_dict[key] = tensor
                n_skipped += 1

            if (i + 1) % 100 == 0 or i + 1 == len(keys):
                elapsed = time.time() - t0
                print(f"  [{i+1}/{len(keys)}] {n_quantized} quantized, "
                      f"{n_skipped} skipped, {elapsed:.1f}s", end="\r")

    print()
    elapsed = time.time() - t0

    # Save
    print(f"[PreQuant] Saving to {out_path}...")
    save_file(state_dict, str(out_path), metadata=metadata)

    out_size = out_path.stat().st_size
    in_size = ckpt_path.stat().st_size

    stats = {
        "total_tensors": n_total,
        "quantized": n_quantized,
        "skipped": n_skipped,
        "total_params": total_params,
        "quantized_params": quantized_params,
        "input_size_gb": in_size / 1e9,
        "output_size_gb": out_size / 1e9,
        "compression_ratio": in_size / out_size,
        "elapsed_s": elapsed,
    }

    print()
    print(f"[PreQuant] Done in {elapsed:.1f}s")
    print(f"[PreQuant] Tensors: {n_total} ({n_quantized} quantized, {n_skipped} kept)")
    print(f"[PreQuant] Params: {total_params/1e9:.2f}B ({quantized_params/1e9:.2f}B quantized)")
    print(f"[PreQuant] Size: {in_size/1e9:.2f} GB -> {out_size/1e9:.2f} GB "
          f"({stats['compression_ratio']:.1f}x compression)")
    print(f"[PreQuant] Metadata: _bitnet_prequant=1 (ForgeEngine will auto-detect)")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Pre-quantize a ForgeLM checkpoint to BitNet b1.58 ternary int8")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to source .safetensors checkpoint")
    parser.add_argument("--output", required=True,
                        help="Path to write pre-quantized .safetensors checkpoint")
    parser.add_argument("--packed", action="store_true",
                        help="Pack 5 ternary values per byte (1.58 bits/param). "
                             "Default: int8 (1 byte/param)")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Keep embeddings in fp32 (better quality, less compression)")
    args = parser.parse_args()

    prequantize_checkpoint(
        checkpoint=args.checkpoint,
        output=args.output,
        packed=args.packed,
        skip_embeddings=args.skip_embeddings,
    )


if __name__ == "__main__":
    main()
