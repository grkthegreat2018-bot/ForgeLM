"""Port weights from HuggingFace Qwen2.5 safetensors into our ConfigurableResearchLLM.

This is a 1:1 weight copy — no training, no distillation, no approximation.
The target model must be built with the `qwen25_coder_1.5b` config (or any config
that exactly matches the source Qwen architecture).

Usage:
    python -m research.port_weights \
        --src .devin/hf_cache/models--Qwen--Qwen2.5-Coder-1.5B-Instruct/snapshots/.../model.safetensors \
        --config qwen25_coder_1.5b \
        --out research/checkpoints/qwen25_coder_1.5b_ported.safetensors
"""
import argparse
import re
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from research.config import get_config
from research.model_loader import ModelLoader

# Mapping: Qwen tensor name pattern -> our parameter name pattern
# {i} is the layer index placeholder.
QWEN_TO_OUR = {
    "model.embed_tokens.weight": "embed.weight",
    "model.norm.weight": "ln_f.weight",
    "model.layers.{i}.input_layernorm.weight": "blocks.{i}.ln1.weight",
    "model.layers.{i}.self_attn.q_proj.weight": "blocks.{i}.attn.q_proj.weight",
    "model.layers.{i}.self_attn.q_proj.bias": "blocks.{i}.attn.q_proj.bias",
    "model.layers.{i}.self_attn.k_proj.weight": "blocks.{i}.attn.k_proj.weight",
    "model.layers.{i}.self_attn.k_proj.bias": "blocks.{i}.attn.k_proj.bias",
    "model.layers.{i}.self_attn.v_proj.weight": "blocks.{i}.attn.v_proj.weight",
    "model.layers.{i}.self_attn.v_proj.bias": "blocks.{i}.attn.v_proj.bias",
    "model.layers.{i}.self_attn.o_proj.weight": "blocks.{i}.attn.out_proj.weight",
    "model.layers.{i}.post_attention_layernorm.weight": "blocks.{i}.ln2.weight",
    "model.layers.{i}.mlp.gate_proj.weight": "blocks.{i}.ffn.w_gate.weight",
    "model.layers.{i}.mlp.up_proj.weight": "blocks.{i}.ffn.w_up.weight",
    "model.layers.{i}.mlp.down_proj.weight": "blocks.{i}.ffn.w_down.weight",
}


def map_qwen_name(qwen_name: str, n_layers: int):
    """Map a Qwen tensor name to our parameter name. Returns None if unmapped."""
    # Try exact match first (embed_tokens, norm)
    if qwen_name in QWEN_TO_OUR:
        return QWEN_TO_OUR[qwen_name]
    # Try layer-indexed patterns
    for qwen_pat, our_pat in QWEN_TO_OUR.items():
        if "{i}" not in qwen_pat:
            continue
        # Convert pattern to regex: escape dots, replace {i} with capture group
        regex = re.escape(qwen_pat).replace(r"\{i\}", r"(\d+)")
        m = re.match(regex, qwen_name)
        if m:
            layer_idx = m.group(1)
            if int(layer_idx) < n_layers:
                return our_pat.replace("{i}", layer_idx)
    return None


def port_weights(src_path: str, config_name: str, out_path: str, verify: bool = True):
    """Load Qwen weights into our model and save as safetensors."""
    cfg = get_config(config_name)
    model = ModelLoader.build_model(cfg)
    model.eval()

    our_state = model.state_dict()  # includes tied weights (head.weight = embed.weight)
    n_layers = cfg.n_layers

    # Load all Qwen tensors
    ported = {}
    missing = []
    shape_mismatches = []

    with safe_open(src_path, framework="pt") as f:
        qwen_keys = sorted(f.keys())
        for qwen_name in qwen_keys:
            our_name = map_qwen_name(qwen_name, n_layers)
            if our_name is None:
                missing.append(qwen_name)
                continue
            tensor = f.get_tensor(qwen_name)  # bf16 on CPU

            if our_name not in our_state:
                missing.append(f"{qwen_name} -> {our_name} (not in our model)")
                continue

            our_param = our_state[our_name]
            if list(tensor.shape) != list(our_param.shape):
                shape_mismatches.append(
                    f"{qwen_name} {list(tensor.shape)} != {our_name} {list(our_param.shape)}"
                )
                continue

            # Copy weights (keep bf16)
            ported[our_name] = tensor.contiguous().to(torch.bfloat16)

    # Handle tied weights: head.weight is tied to embed.weight in our model.
    # state_dict() lists both keys; we need to fill head.weight from embed.weight.
    if "head.weight" in our_state and "head.weight" not in ported and "embed.weight" in ported:
        ported["head.weight"] = ported["embed.weight"].clone()
        print("Note: head.weight tied to embed.weight (weight tying).")

    # Check for our params that didn't get filled
    our_unfilled = [n for n in our_state if n not in ported]

    print(f"Ported: {len(ported)} tensors")
    print(f"Missing (Qwen -> ours): {len(missing)}")
    if missing:
        for m in missing[:10]:
            print(f"  {m}")
    print(f"Shape mismatches: {len(shape_mismatches)}")
    if shape_mismatches:
        for s in shape_mismatches[:10]:
            print(f"  {s}")
    print(f"Our unfilled params: {len(our_unfilled)}")
    if our_unfilled:
        for u in our_unfilled[:10]:
            print(f"  {u}")

    if missing or shape_mismatches or our_unfilled:
        print("\nERROR: Weight porting incomplete. Aborting.")
        return False

    # Save
    save_file(ported, out_path, metadata={"source": src_path, "config": config_name})
    print(f"\nSaved {len(ported)} tensors to {out_path}")

    # Verify: load into model and check a forward pass produces sensible output
    if verify:
        print("\nVerifying: loading ported weights into model...")
        model2 = ModelLoader.build_model(cfg)
        model2.load_state_dict(ported, strict=True)
        model2.eval()
        model2 = model2.to("cuda", dtype=torch.bfloat16)

        # Simple forward pass test
        test_ids = torch.tensor([[151643, 151665, 151663]], device="cuda")  # <im_start>, "Hello", <im_end>
        with torch.inference_mode():
            out = model2(test_ids)
            logits = out[0] if isinstance(out, tuple) else out
        print(f"Forward pass OK. Logits shape: {list(logits.shape)}")
        print(f"Logits sample (first 5): {logits[0, -1, :5].tolist()}")

        # Check logits are not NaN/Inf
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("ERROR: Logits contain NaN/Inf!")
            return False
        print("Logits are finite — weights loaded correctly.")

    return True


def main():
    parser = argparse.ArgumentParser(description="Port Qwen2.5 weights into our model format.")
    parser.add_argument("--src", required=True, help="Path to Qwen model.safetensors")
    parser.add_argument("--config", default="qwen25_coder_1.5b", help="Our config name")
    parser.add_argument("--out", required=True, help="Output safetensors path")
    parser.add_argument("--no-verify", action="store_true", help="Skip forward pass verification")
    args = parser.parse_args()

    success = port_weights(args.src, args.config, args.out, verify=not args.no_verify)
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
