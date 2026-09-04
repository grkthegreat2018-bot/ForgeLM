"""Port ai21labs/AI21-Jamba-Reasoning-3B into ForgeEngine checkpoint format.

Maps HF Jamba safetensors keys to ForgeEngine's expected key names:
  model.embed_tokens.weight          -> embed.weight
  lm_head.weight                     -> head.weight (tied)
  model.final_layernorm.weight       -> ln_f.weight
  model.layers.{i}.input_layernorm   -> blocks.{i}.ln1
  model.layers.{i}.pre_ff_layernorm  -> blocks.{i}.ln2
  model.layers.{i}.mamba.*           -> blocks.{i}.attn.*
  model.layers.{i}.self_attn.*       -> blocks.{i}.attn.*
  model.layers.{i}.feed_forward.*    -> blocks.{i}.ffn.*
"""
import os
import sys
import json
import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HF_DIR = "research/checkpoints/hf_models/models--ai21labs--AI21-Jamba-Reasoning-3B/snapshots/7524370414253301341370beba35e3d549cbf5f3"
OUTPUT = "research/checkpoints/Jamba_Reasoning_3B.safetensors"

# Key mapping rules
def map_key(hf_key: str) -> str:
    """Map HF Jamba key to ForgeEngine key."""
    if hf_key == "model.embed_tokens.weight":
        return "embed.weight"
    if hf_key == "lm_head.weight":
        return "head.weight"
    if hf_key == "model.final_layernorm.weight":
        return "ln_f.weight"

    # Layer keys: model.layers.{i}.xxx -> blocks.{i}.xxx
    if hf_key.startswith("model.layers."):
        parts = hf_key.split(".")
        layer_idx = parts[2]
        rest = ".".join(parts[3:])

        # input_layernorm -> ln1
        if rest == "input_layernorm.weight":
            return f"blocks.{layer_idx}.ln1.weight"
        # pre_ff_layernorm -> ln2
        if rest == "pre_ff_layernorm.weight":
            return f"blocks.{layer_idx}.ln2.weight"

        # mamba.* -> attn.*
        if rest.startswith("mamba."):
            sub = rest[len("mamba."):]
            # Strip .weight from layernorm params (ForgeEngine uses bare param names)
            for ln in ("dt_layernorm", "b_layernorm", "c_layernorm"):
                if sub == f"{ln}.weight":
                    return f"blocks.{layer_idx}.attn.{ln}"
            return f"blocks.{layer_idx}.attn.{sub}"

        # self_attn.* -> attn.*
        if rest.startswith("self_attn."):
            sub = rest[len("self_attn."):]
            # o_proj -> out_proj (ForgeEngine naming)
            if sub == "o_proj.weight":
                return f"blocks.{layer_idx}.attn.out_proj.weight"
            # q_proj, k_proj, v_proj — same names
            return f"blocks.{layer_idx}.attn.{sub}"

        # feed_forward.* -> ffn.*
        if rest.startswith("feed_forward."):
            sub = rest[len("feed_forward."):]
            # gate_proj -> w_gate, up_proj -> w_up, down_proj -> w_down
            if sub == "gate_proj.weight":
                return f"blocks.{layer_idx}.ffn.w_gate.weight"
            if sub == "up_proj.weight":
                return f"blocks.{layer_idx}.ffn.w_up.weight"
            if sub == "down_proj.weight":
                return f"blocks.{layer_idx}.ffn.w_down.weight"

    return None  # unmapped


def main():
    print(f"Porting AI21-Jamba-Reasoning-3B to ForgeEngine format")
    print(f"Source: {HF_DIR}")
    print(f"Output: {OUTPUT}")

    # Load index to find which shard has each key
    with open(os.path.join(HF_DIR, "model.safetensors.index.json")) as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]

    # Group keys by shard
    shards = {}
    for hf_key, shard_file in weight_map.items():
        if shard_file not in shards:
            shards[shard_file] = []
        shards[shard_file].append(hf_key)

    # Load and remap all tensors
    remapped = {}
    skipped = []
    for shard_file, hf_keys in sorted(shards.items()):
        shard_path = os.path.join(HF_DIR, shard_file)
        print(f"Loading {shard_file} ({len(hf_keys)} keys)...")
        state = load_file(shard_path)
        for hf_key in hf_keys:
            new_key = map_key(hf_key)
            if new_key is None:
                skipped.append(hf_key)
                continue
            tensor = state[hf_key]
            remapped[new_key] = tensor
        del state

    print(f"\nRemapped: {len(remapped)} tensors")
    print(f"Skipped:  {len(skipped)}")
    if skipped:
        print("Skipped keys:")
        for k in skipped:
            print(f"  {k}")

    # Verify all expected keys are present
    expected_prefixes = set()
    for i in range(28):
        expected_prefixes.add(f"blocks.{i}.ln1.weight")
        expected_prefixes.add(f"blocks.{i}.ln2.weight")
        expected_prefixes.add(f"blocks.{i}.ffn.w_gate.weight")
        expected_prefixes.add(f"blocks.{i}.ffn.w_up.weight")
        expected_prefixes.add(f"blocks.{i}.ffn.w_down.weight")
    expected_prefixes.add("embed.weight")
    expected_prefixes.add("head.weight")
    expected_prefixes.add("ln_f.weight")

    missing = expected_prefixes - set(remapped.keys())
    if missing:
        print(f"\nMISSING expected keys: {missing}")
    else:
        print("\nAll expected keys present!")

    # Save
    print(f"\nSaving to {OUTPUT}...")
    save_file(remapped, OUTPUT, metadata={
        "source": "ai21labs/AI21-Jamba-Reasoning-3B",
        "format": "forge_engine_v2",
        "ported_by": "_port_jamba.py",
    })

    # Verify file
    size_gb = os.path.getsize(OUTPUT) / 1e9
    print(f"Saved: {size_gb:.2f} GB")

    # Quick sanity: check a few tensor shapes
    from safetensors import safe_open
    with safe_open(OUTPUT, framework="pt") as f:
        for k in ["embed.weight", "head.weight", "ln_f.weight",
                   "blocks.0.attn.in_proj.weight", "blocks.0.attn.A_log",
                   "blocks.7.attn.q_proj.weight", "blocks.7.attn.k_proj.weight",
                   "blocks.0.ffn.w_gate.weight", "blocks.0.ln1.weight"]:
            if k in f.keys():
                t = f.get_tensor(k)
                print(f"  {k}: {t.shape} {t.dtype}")
            else:
                print(f"  {k}: MISSING!")

    print("\nDone. Test with: _test_v12.py (update CHECKPOINT path)")


if __name__ == "__main__":
    main()
