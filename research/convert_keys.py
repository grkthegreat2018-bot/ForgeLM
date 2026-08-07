"""Conversion keys: map Qwen weights to different architectures.

Each "key" is a function that takes source weights (safetensors) and produces
target weights in a different format/architecture. The goal is to find
mathematical transformations that preserve the model's function.

Key #1: BitNet — ternary quantization {-1, 0, +1} with absmean scaling
Key #2: SVD resize — truncate/pad weight matrices for smaller/larger models
Key #3: MLA — decompose GQA K/V projections into low-rank compression
Key #4: MoE — split dense FFN into routed experts
Key #5: Mamba/SSM — 2-stage distillation bridge (future)
"""
import argparse
import os
import re
import torch
from safetensors import safe_open
from safetensors.torch import save_file, load_file

from research.config import get_config
from research.model_loader import ModelLoader


# ============================================================================
# Key #1: BitNet b1.58 — ternary weight quantization
# ============================================================================

def bitnet_quantize_tensor(W: torch.Tensor, per_row: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a weight tensor to ternary {-1, 0, +1} with absmean scaling.

    Args:
        W: weight tensor of shape [out_features, in_features]
        per_row: if True, use per-output-row scaling (one scale per neuron).
                 if False, use per-tensor scaling (one scale for whole tensor).

    Returns (W_ternary as int8, w_scale as bf16).
    At inference: W_effective = W_ternary.to(bf16) * w_scale
    """
    if per_row:
        # Per-output-row absmean: shape [out_features, 1]
        w_scale = W.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
    else:
        # Per-tensor absmean: scalar
        w_scale = W.abs().mean().clamp(min=1e-8)

    W_ternary = (W / w_scale).round().clamp(-1, 1).to(torch.int8)
    return W_ternary, w_scale.to(torch.bfloat16)


def bitnet_dequantize_tensor(W_ternary: torch.Tensor, w_scale: torch.Tensor) -> torch.Tensor:
    """Dequantize ternary weights back to bf16."""
    return W_ternary.to(torch.bfloat16) * w_scale


# Which parameters to quantize (only 2D Linear weights, not norms/embeddings/biases)
def _should_quantize(name: str, shape: torch.Size) -> bool:
    """Quantize only 2D weight matrices from attention/FFN projections."""
    if len(shape) != 2:
        return False
    if "embed" in name or "head" in name:
        return False  # Keep embeddings bf16 (lookup table)
    if "norm" in name or "ln" in name:
        return False  # Keep norm scales bf16
    if "bias" in name:
        return False  # Keep biases bf16 (small, and they're additive not multiplicative)
    if "router" in name or "gate.weight" in name and "ffn" not in name:
        return False  # Keep router/gate weights bf16 (small, critical for routing)
    if "noise" in name:
        return False  # Keep MoE noise weights bf16
    return True  # q/k/v/o_proj weights, gate/up/down_proj weights, MoE expert weights


def key_bitnet(src_path: str, out_path: str, config_name: str = "qwen25_coder_1.5b"):
    """Key #1: Convert bf16 weights to BitNet b1.58 ternary format.

    Quantizes all Linear weight matrices to {-1, 0, +1} with per-tensor absmean
    scaling. Embeddings, norms, and biases stay bf16.

    Storage: ternary weights as int8 + scale factors as bf16 scalars.
    Effective bits/weight: ~8 (int8) for now. True 1.58-bit packing is a future
    optimization (pack 5 ternary values per byte).
    """
    print(f"Key #1: BitNet b1.58 ternary quantization")
    print(f"  Source: {src_path}")
    print(f"  Output: {out_path}")

    cfg = get_config(config_name)
    our_state = ModelLoader.blank_state_dict(cfg)

    quantized = {}
    scales = {}
    unquantized = {}

    with safe_open(src_path, framework="pt") as f:
        for key in sorted(f.keys()):
            tensor = f.get_tensor(key)
            if _should_quantize(key, tensor.shape):
                W_ternary, w_scale = bitnet_quantize_tensor(tensor)
                quantized[key] = W_ternary.contiguous()
                scales[f"{key}.__bitnet_scale__"] = w_scale
            else:
                unquantized[key] = tensor.contiguous().to(torch.bfloat16)

    # Merge: quantized (int8) + scales (bf16) + unquantized (bf16)
    all_tensors = {**quantized, **scales, **unquantized}

    # Stats
    n_quantized = len(quantized)
    n_unquantized = len(unquantized)
    total_params = sum(t.numel() for t in quantized.values()) + sum(t.numel() for t in unquantized.values())
    quant_params = sum(t.numel() for t in quantized.values())

    # Calculate effective size
    quant_bytes = sum(t.numel() * t.element_size() for t in quantized.values())
    scale_bytes = sum(t.numel() * t.element_size() for t in scales.values())
    unquant_bytes = sum(t.numel() * t.element_size() for t in unquantized.values())
    total_bytes = quant_bytes + scale_bytes + unquant_bytes

    print(f"\n  Quantized: {n_quantized} tensors ({quant_params/1e6:.1f}M params -> int8 ternary)")
    print(f"  Unquantized: {n_unquantized} tensors (embeddings, norms, biases -> bf16)")
    print(f"  Scale factors: {len(scales)} tensors (one absmean per quantized tensor)")
    print(f"\n  Size: {total_bytes/1e9:.2f} GB (vs 3.09 GB bf16 original)")
    print(f"  Quantized portion: {quant_bytes/1e6:.0f} MB (int8) + {scale_bytes/1e3:.1f} KB (scales)")
    print(f"  Unquantized portion: {unquant_bytes/1e6:.0f} MB (bf16)")

    save_file(all_tensors, out_path, metadata={
        "conversion": "bitnet_b1.58",
        "source": src_path,
        "config": config_name,
        "n_quantized": str(n_quantized),
    })
    print(f"\n  Saved to {out_path}")
    return True


def load_bitnet_model(ckpt_path: str, config_name: str = "qwen25_coder_1.5b") -> torch.nn.Module:
    """Load a BitNet-quantized checkpoint, dequantizing weights to bf16 on load."""
    cfg = get_config(config_name)
    model = ModelLoader.build_model(cfg)  # needs full model for verification
    model.eval()

    state = load_file(ckpt_path)
    dequant_state = {}
    for name in model.state_dict():
        if name in state:
            dequant_state[name] = state[name].to(torch.bfloat16)
        elif f"{name}.__bitnet_scale__" in state:
            W_ternary = state[name].to(torch.int8)
            w_scale = state[f"{name}.__bitnet_scale__"]
            dequant_state[name] = bitnet_dequantize_tensor(W_ternary, w_scale)
        elif name == "head.weight" and "embed.weight" in dequant_state:
            # Tied weight: head.weight = embed.weight
            dequant_state[name] = dequant_state["embed.weight"].clone()
        else:
            raise KeyError(f"Missing weight: {name}")

    model.load_state_dict(dequant_state, strict=True)
    return model


# ============================================================================
# Verification
# ============================================================================

def verify_bitnet(ckpt_path: str, config_name: str = "qwen25_coder_1.5b"):
    """Compare BitNet-quantized model output against original bf16 model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\n=== BitNet Verification ===")
    model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    cache_dir = ".devin/hf_cache"
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)

    # Original
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id, cache_dir=cache_dir, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    hf_model.eval()

    # BitNet
    bitnet_model = load_bitnet_model(ckpt_path, config_name)
    bitnet_model = bitnet_model.to("cuda", dtype=torch.bfloat16)
    bitnet_model.eval()

    test_texts = ["def hello_world():", "The quick brown fox", "import numpy as"]
    print(f"\n{'Input':<30} {'Max diff':<15} {'Mean diff':<15} {'Cosine':<15}")
    print("-" * 75)

    with torch.inference_mode():
        for text in test_texts:
            ids = tokenizer(text, return_tensors="pt").to("cuda")
            hf_logits = hf_model(ids.input_ids).logits
            bn_out = bitnet_model(ids.input_ids)
            bn_logits = bn_out[0] if isinstance(bn_out, tuple) else bn_out

            max_d = (hf_logits - bn_logits).abs().max().item()
            mean_d = (hf_logits - bn_logits).abs().mean().item()
            cos = torch.nn.functional.cosine_similarity(
                hf_logits.flatten().unsqueeze(0),
                bn_logits.flatten().unsqueeze(0)
            ).item()
            print(f"{text:<30} {max_d:<15.4f} {mean_d:<15.4f} {cos:<15.6f}")

    # Top-5 comparison
    print("\n=== Top-5 for 'def hello_world():' ===")
    with torch.inference_mode():
        ids = tokenizer("def hello_world():", return_tensors="pt").to("cuda")
        hf_logits = hf_model(ids.input_ids).logits[0, -1, :]
        bn_out = bitnet_model(ids.input_ids)
        bn_logits = (bn_out[0] if isinstance(bn_out, tuple) else bn_out)[0, -1, :]

        hf_top5 = torch.topk(hf_logits, 5)
        bn_top5 = torch.topk(bn_logits, 5)

        print(f"{'Rank':<6} {'HF token':<15} {'HF prob':<12} {'BN token':<15} {'BN prob':<12} {'Match'}")
        print("-" * 75)
        for i in range(5):
            hf_tok = tokenizer.decode(hf_top5.indices[i].item())
            bn_tok = tokenizer.decode(bn_top5.indices[i].item())
            hf_p = torch.softmax(hf_logits, dim=-1)[hf_top5.indices[i]].item()
            bn_p = torch.softmax(bn_logits, dim=-1)[bn_top5.indices[i]].item()
            match = "OK" if hf_top5.indices[i].item() == bn_top5.indices[i].item() else "DIFF"
            print(f"{i+1:<6} {repr(hf_tok):<15} {hf_p:<12.6f} {repr(bn_tok):<15} {bn_p:<12.6f} {match}")


# ============================================================================
# Key #3: MLA — decompose GQA K/V projections into low-rank compression
# ============================================================================

def key_gqa_to_mla(src_path: str, out_path: str, source_config_name: str = "qwen25_coder_1.5b",
                   target_config_name: str = "360m_mla", kv_compression_dim: int = None):
    """Key #3: Convert GQA weights to MLA (Multi-head Latent Attention).

    GQA has separate K/V projections: K = X @ W_K^T, V = X @ W_V^T.
    MLA factors these through a shared low-rank bottleneck:
        c_kv = X @ W_c^T          (compression: d_model → d_c)
        K = c_kv @ W_KC^T          (K decompression: d_c → d_model)
        V = c_kv @ W_VC^T          (V decompression: d_c → d_model)

    So W_K = W_KC @ W_c and W_V = W_VC @ W_c (both share W_c).

    Algorithm:
    1. Expand GQA K/V to full [d_model, d_model] by duplicating KV head rows
    2. Stack: W_KV = [W_K_exp; W_V_exp]  → [2*d_model, d_model]
    3. SVD: W_KV = U @ S @ V^T
    4. Truncate to top-d_c: W_c = V^T[:d_c], W_dec = U[:,:d_c] @ diag(S[:d_c])
    5. Split: W_KC = W_dec[:d_model], W_VC = W_dec[d_model:]

    The SVD captures the most important shared subspace between K and V.
    Energy retained = sum(S[:d_c]) / sum(S).
    """
    print(f"Key #3: GQA → MLA (low-rank KV compression)")
    print(f"  Source: {src_path} ({source_config_name})")
    print(f"  Target: {out_path} ({target_config_name})")

    src_cfg = get_config(source_config_name)
    tgt_cfg = get_config(target_config_name)
    # Force MLA attention type for target (config may be GQA — we override to build MLA)
    tgt_cfg = tgt_cfg.__class__(**{**tgt_cfg.__dict__, "attn_type": "mla"})

    if kv_compression_dim is None:
        kv_compression_dim = tgt_cfg.kv_compression_dim
    print(f"  KV compression dim: {kv_compression_dim}")

    d_model = src_cfg.d_model
    n_heads = src_cfg.n_heads
    n_kv_heads = src_cfg.n_kv_heads or n_heads
    head_dim = d_model // n_heads
    kv_dim = n_kv_heads * head_dim
    group_size = n_heads // n_kv_heads

    # For MLA head expansion (source GQA structure)
    n_heads_src = n_heads
    n_kv_heads_src = n_kv_heads
    head_dim_src = head_dim

    print(f"  Source GQA: d_model={d_model}, n_heads={n_heads}, n_kv_heads={n_kv_heads}, head_dim={head_dim}")
    print(f"  KV dim: {kv_dim}, group_size: {group_size}")

    src_state = load_file(src_path)
    tgt_state = ModelLoader.blank_state_dict(tgt_cfg)

    resized = {}
    n_layers = src_cfg.n_layers
    tgt_layers = tgt_cfg.n_layers

    # Layer mapping: proportionally sample source layers
    keep_indices = [int(i * n_layers / tgt_layers) for i in range(tgt_layers)]
    print(f"  Layer mapping: source {keep_indices} -> target 0..{tgt_layers-1}")

    # Shared SVD projection for d_model resize (if needed)
    P = None
    if src_cfg.d_model != tgt_cfg.d_model:
        from research.convert_key_svd import compute_shared_projection, project_weight
        P = compute_shared_projection(src_state["embed.weight"], tgt_cfg.d_model)

    total_energy_retained = 0.0
    total_energy = 0.0

    for tgt_name, tgt_tensor in tgt_state.items():
        tgt_shape = list(tgt_tensor.shape)

        # MLA-specific projections — handle BEFORE source lookup (these don't exist in GQA source)
        if "kv_down_proj" in tgt_name:
            # W_c: compression matrix [d_c, d_model] — computed from SVD below
            # Handled together with k_up_proj/v_up_proj per layer
            continue
        elif "k_up_proj" in tgt_name or "v_up_proj" in tgt_name:
            # Handled together — process when we hit k_up_proj.weight
            if "v_up_proj" in tgt_name:
                continue  # Already processed with k_up_proj
            if ".bias" in tgt_name:
                continue  # Bias handled with weight
            # Process both k_up_proj and v_up_proj for this layer
            layer_idx = int(re.match(r"blocks\.(\d+)", tgt_name).group(1))
            layer_prefix = f"blocks.{layer_idx}.attn"
            src_layer = keep_indices[layer_idx]
            k_src_name = f"blocks.{src_layer}.attn.k_proj.weight"
            v_src_name = f"blocks.{src_layer}.attn.v_proj.weight"

            W_K = src_state[k_src_name].float()  # [kv_dim, d_model]
            W_V = src_state[v_src_name].float()  # [kv_dim, d_model]

            # Apply d_model resize to K/V if needed (before SVD)
            if P is not None:
                Pf = P.float()
                # K/V: [kv_dim, d_src] -> [kv_dim, d_tgt] (only input dim changes)
                W_K = W_K @ Pf.T
                W_V = W_V @ Pf.T
            d_tgt = tgt_cfg.d_model

            # SVD on UNEXPANDED K/V to preserve GQA head structure.
            # Expanded SVD mixes heads → breaks RoPE equivalence.
            # Unexpanded SVD captures the 2 unique KV heads, then we expand
            # the decompression matrix to duplicate heads.
            W_KV = torch.cat([W_K, W_V], dim=0)  # [2*kv_dim, d_tgt]

            U, S, Vh = torch.linalg.svd(W_KV, full_matrices=False)
            d_c = min(kv_compression_dim, S.shape[0])

            energy = S.sum().item()
            retained = S[:d_c].sum().item()
            total_energy += energy
            total_energy_retained += retained

            # W_c = Vh[:d_c]  → [d_c, d_tgt]
            W_c = Vh[:d_c]
            # W_dec = U[:, :d_c] * S[:d_c]  → [2*kv_dim, d_c]
            W_dec = U[:, :d_c] * S[:d_c].unsqueeze(0)
            # Split: K and V decompression (unexpanded)
            W_KC_unexp = W_dec[:kv_dim]  # [kv_dim, d_c]
            W_VC_unexp = W_dec[kv_dim:]  # [kv_dim, d_c]

            # Expand decompression to full [n_heads*head_dim, d_c] by duplicating KV heads
            # This preserves GQA's head duplication structure for RoPE equivalence
            W_KC = W_KC_unexp.view(n_kv_heads_src, head_dim, d_c).repeat_interleave(
                group_size, dim=0).reshape(n_heads_src * head_dim, d_c)
            W_VC = W_VC_unexp.view(n_kv_heads_src, head_dim, d_c).repeat_interleave(
                group_size, dim=0).reshape(n_heads_src * head_dim, d_c)

            resized[f"{layer_prefix}.kv_down_proj.weight"] = W_c.to(torch.bfloat16).contiguous()
            resized[f"{layer_prefix}.k_up_proj.weight"] = W_KC.to(torch.bfloat16).contiguous()
            resized[f"{layer_prefix}.v_up_proj.weight"] = W_VC.to(torch.bfloat16).contiguous()

            # Copy K/V biases from GQA (expand from kv_dim to n_heads*head_dim)
            k_bias_src_name = f"blocks.{src_layer}.attn.k_proj.bias"
            v_bias_src_name = f"blocks.{src_layer}.attn.v_proj.bias"
            if k_bias_src_name in src_state:
                b_K = src_state[k_bias_src_name].float()  # [kv_dim]
                b_K_exp = b_K.view(n_kv_heads_src, head_dim).repeat_interleave(
                    group_size, dim=0).reshape(n_heads_src * head_dim)
                resized[f"{layer_prefix}.k_up_proj.bias"] = b_K_exp.to(torch.bfloat16).contiguous()
            if v_bias_src_name in src_state:
                b_V = src_state[v_bias_src_name].float()
                b_V_exp = b_V.view(n_kv_heads_src, head_dim).repeat_interleave(
                    group_size, dim=0).reshape(n_heads_src * head_dim)
                resized[f"{layer_prefix}.v_up_proj.bias"] = b_V_exp.to(torch.bfloat16).contiguous()

            layer_idx = int(re.match(r'blocks\.(\d+)', tgt_name).group(1))
            print(f"  Layer {layer_idx}: KV energy retained: {retained/energy*100:.1f}% (d_c={d_c})")
            continue

        # Determine source name for non-MLA tensors
        if "blocks." in tgt_name:
            m = re.match(r"blocks\.(\d+)\.(.+)", tgt_name)
            tgt_layer = int(m.group(1))
            remainder = m.group(2)
            src_layer = keep_indices[tgt_layer]
            src_name = f"blocks.{src_layer}.{remainder}"
        elif tgt_name == "head.weight":
            src_name = "embed.weight"
        else:
            src_name = tgt_name

        if src_name not in src_state:
            # LayerNorm bias: source uses RMSNorm (no bias), target uses LayerNorm (has bias)
            # Zero-init the bias — it will be fine-tuned
            if ".bias" in tgt_name and ("ln" in tgt_name or "norm" in tgt_name):
                resized[tgt_name] = torch.zeros(tgt_shape, dtype=torch.bfloat16)
                continue
            print(f"  WARNING: {src_name} not in source, skipping")
            continue

        src_tensor = src_state[src_name]
        src_shape = list(src_tensor.shape)

        # Q projection: GQA q_proj -> MLA q_proj (just resize/copy)
        if "q_proj" in tgt_name:
            if ".weight" in tgt_name:
                if P is not None:
                    W_new = project_weight(src_tensor, P, is_input=True, is_output=True)
                    resized[tgt_name] = W_new.to(torch.bfloat16).contiguous()
                else:
                    resized[tgt_name] = src_tensor.to(torch.bfloat16).contiguous()
            elif ".bias" in tgt_name:
                # Copy Q bias from GQA to MLA
                if P is not None:
                    b_new = P.float() @ src_tensor.float()
                    resized[tgt_name] = b_new.to(torch.bfloat16).contiguous()
                else:
                    resized[tgt_name] = src_tensor.to(torch.bfloat16).contiguous()
            continue

        # O projection: GQA out_proj -> MLA out_proj
        if "out_proj" in tgt_name and ".weight" in tgt_name:
            if P is not None:
                W_new = project_weight(src_tensor, P, is_input=True, is_output=True)
                resized[tgt_name] = W_new.to(torch.bfloat16).contiguous()
            else:
                resized[tgt_name] = src_tensor.to(torch.bfloat16).contiguous()
            continue

        # FFN weights: resize via SVD if needed, else copy
        if "w_gate" in tgt_name or "w_up" in tgt_name:
            if P is not None:
                Pf = P.float()
                Wf = src_tensor.float()
                W_in = Wf @ Pf.T
                if W_in.shape[0] > tgt_shape[0]:
                    norms = W_in.norm(dim=1)
                    _, top_idx = torch.topk(norms, tgt_shape[0])
                    top_idx = top_idx.sort().values
                    W_new = W_in[top_idx]
                else:
                    W_new = W_in
                resized[tgt_name] = W_new.to(torch.bfloat16).contiguous()
            else:
                resized[tgt_name] = src_tensor.to(torch.bfloat16).contiguous()
            continue

        if "w_down" in tgt_name:
            if P is not None:
                Pf = P.float()
                Wf = src_tensor.float()
                W_out = Pf @ Wf
                if W_out.shape[1] > tgt_shape[1]:
                    norms = W_out.norm(dim=0)
                    _, top_idx = torch.topk(norms, tgt_shape[1])
                    top_idx = top_idx.sort().values
                    W_new = W_out[:, top_idx]
                else:
                    W_new = W_out
                resized[tgt_name] = W_new.to(torch.bfloat16).contiguous()
            else:
                resized[tgt_name] = src_tensor.to(torch.bfloat16).contiguous()
            continue

        # Norms
        if "ln" in tgt_name or "norm" in tgt_name:
            if P is not None and src_shape[0] != tgt_shape[0]:
                norm_proj = P.float() @ src_tensor.float()
                resized[tgt_name] = norm_proj.to(torch.bfloat16).contiguous()
            else:
                resized[tgt_name] = src_tensor.to(torch.bfloat16).contiguous()
            continue

        # Embedding / head
        if tgt_name == "embed.weight" or tgt_name == "head.weight":
            if P is not None:
                W_new = project_weight(src_tensor, P, is_input=True, is_output=False)
                resized[tgt_name] = W_new.to(torch.bfloat16).contiguous()
            else:
                resized[tgt_name] = src_tensor.to(torch.bfloat16).contiguous().clone()
            continue

        # Default: copy or truncate
        if src_shape == tgt_shape:
            resized[tgt_name] = src_tensor.to(torch.bfloat16).contiguous()
        else:
            print(f"  WARNING: unhandled {tgt_name} {src_shape} -> {tgt_shape}")

    print(f"\n  Total KV energy retained: {total_energy_retained/total_energy*100:.1f}%")
    print(f"  Tensors: {len(resized)}")

    save_file(resized, out_path, metadata={
        "conversion": "gqa_to_mla",
        "source": src_path,
        "source_config": source_config_name,
        "target_config": target_config_name,
        "kv_compression_dim": str(kv_compression_dim),
        "layer_mapping": str(keep_indices),
    })
    print(f"\n  Saved to {out_path}")
    return True


# ============================================================================
# Key #4: MoE — split dense SwiGLU FFN into routed experts
# ============================================================================

def key_dense_to_moe(src_path: str, out_path: str, config_name: str = "qwen25_coder_1.5b",
                     n_experts: int = 4, top_k: int = 2, shared_expert: bool = True):
    """Key #4: Convert dense SwiGLU FFN to Mixture-of-Experts.

    Strategy: split the dense FFN's intermediate dimension across experts.
    Each expert gets a contiguous slice of the original w_gate/w_up/w_down.
    The router is initialized to approximate uniform routing (will be fine-tuned).

    Dense FFN:  Y = W_down(silu(W_gate(X)) * W_up(X))   [intermediate=8960]
    MoE:        Y = sum_i gate_i * Expert_i(X) + Shared(X)
                where Expert_i uses intermediate slice [i*d_ff:(i+1)*d_ff]

    With shared expert (DeepSeek-V3 style):
        - n_experts routed experts, each with d_ff = intermediate / (n_experts + 1)
        - 1 shared expert (always active) with d_ff = intermediate / (n_experts + 1)
        - Total params = same as dense, but only top_k + 1 experts active per token

    Without shared expert:
        - n_experts experts, each with d_ff = intermediate / n_experts
        - Only top_k experts active per token

    The router is initialized with a simple linear projection that approximates
    uniform routing. Fine-tuning will learn the optimal routing.
    """
    print(f"Key #4: Dense SwiGLU → MoE ({n_experts} experts, top-{top_k})")
    print(f"  Source: {src_path} ({config_name})")
    print(f"  Shared expert: {shared_expert}")

    cfg = get_config(config_name)
    d_model = cfg.d_model
    intermediate = cfg.intermediate_size or 8 * d_model // 3

    # Calculate expert size
    if shared_expert:
        n_total = n_experts + 1  # +1 for shared
    else:
        n_total = n_experts
    d_ff = intermediate // n_total
    print(f"  d_model={d_model}, intermediate={intermediate}, d_ff_per_expert={d_ff}")

    src_state = load_file(src_path)
    n_layers = cfg.n_layers

    # Build output state dict — copy everything except FFN, then add MoE weights
    out_state = {}

    for name, tensor in src_state.items():
        # Skip FFN weights — we'll replace them with MoE
        if "ffn.w_gate" in name or "ffn.w_up" in name or "ffn.w_down" in name:
            continue
        # Skip tied head (will be added separately)
        if name == "head.weight":
            continue
        out_state[name] = tensor.to(torch.bfloat16).contiguous()

    # Add tied head
    if "head.weight" not in out_state and "embed.weight" in out_state:
        out_state["head.weight"] = out_state["embed.weight"].clone()

    # Process each layer's FFN
    for layer in range(n_layers):
        w_gate = src_state[f"blocks.{layer}.ffn.w_gate.weight"].float()  # [intermediate, d_model]
        w_up = src_state[f"blocks.{layer}.ffn.w_up.weight"].float()      # [intermediate, d_model]
        w_down = src_state[f"blocks.{layer}.ffn.w_down.weight"].float()  # [d_model, intermediate]

        prefix = f"blocks.{layer}.ffn"

        if shared_expert:
            # Shared expert gets the FIRST slice (weight 1.0, always active)
            shared_gate = w_gate[:d_ff]
            shared_up = w_up[:d_ff]
            shared_down = w_down[:, :d_ff]
            out_state[f"{prefix}.shared.w1.weight"] = shared_gate.to(torch.bfloat16).contiguous()
            out_state[f"{prefix}.shared.w3.weight"] = shared_up.to(torch.bfloat16).contiguous()
            out_state[f"{prefix}.shared.w2.weight"] = shared_down.to(torch.bfloat16).contiguous()

            # Routed experts get the remaining slices.
            # Only w_down (output projection) is scaled by n_experts to compensate
            # for softmax gating (gate_i = 1/n_experts with uniform router).
            # w_gate and w_up are NOT scaled — SwiGLU is nonlinear, scaling them
            # would change silu(w_gate @ x) * w_up @ x in a non-linear way.
            # Scaling only w_down works because it's a linear output projection:
            #   expert(x) = w_down @ (silu(w_gate @ x) * w_up @ x)
            #   n * expert(x) = (n * w_down) @ (silu(w_gate @ x) * w_up @ x)  ✓
            for i in range(n_experts):
                start = (i + 1) * d_ff
                end = (i + 2) * d_ff
                exp_gate = w_gate[start:end]
                exp_up = w_up[start:end]
                exp_down = w_down[:, start:end] * n_experts
                out_state[f"{prefix}.experts.{i}.w1.weight"] = exp_gate.to(torch.bfloat16).contiguous()
                out_state[f"{prefix}.experts.{i}.w3.weight"] = exp_up.to(torch.bfloat16).contiguous()
                out_state[f"{prefix}.experts.{i}.w2.weight"] = exp_down.to(torch.bfloat16).contiguous()
        else:
            # All experts get equal slices, w_down scaled by n_experts
            for i in range(n_experts):
                start = i * d_ff
                end = (i + 1) * d_ff
                exp_gate = w_gate[start:end]
                exp_up = w_up[start:end]
                exp_down = w_down[:, start:end] * n_experts
                out_state[f"{prefix}.experts.{i}.w1.weight"] = exp_gate.to(torch.bfloat16).contiguous()
                out_state[f"{prefix}.experts.{i}.w3.weight"] = exp_up.to(torch.bfloat16).contiguous()
                out_state[f"{prefix}.experts.{i}.w2.weight"] = exp_down.to(torch.bfloat16).contiguous()

        # Initialize router: uniform routing (all logits = 0 → uniform softmax)
        # The router will be fine-tuned to learn optimal routing
        router_weight = torch.zeros(n_experts, d_model, dtype=torch.bfloat16)
        out_state[f"{prefix}.router.gate.weight"] = router_weight.contiguous()

        # Noise linear (for exploration during training)
        noise_weight = torch.zeros(n_experts, d_model, dtype=torch.bfloat16) * 0.1
        out_state[f"{prefix}.router.noise.weight"] = noise_weight.contiguous()
        out_state[f"{prefix}.router.noise_scale"] = torch.tensor([0.1], dtype=torch.bfloat16)

    # Calculate parameter counts
    dense_ffn_params = 3 * intermediate * d_model * n_layers  # gate + up + down
    moe_params = n_total * 3 * d_ff * d_model * n_layers + n_experts * d_model * n_layers  # experts + router
    active_params = (top_k + (1 if shared_expert else 0)) * 3 * d_ff * d_model * n_layers

    print(f"\n  Dense FFN params: {dense_ffn_params/1e6:.1f}M")
    print(f"  MoE total params: {moe_params/1e6:.1f}M ({moe_params/dense_ffn_params:.2f}x dense)")
    print(f"  MoE active params: {active_params/1e6:.1f}M ({active_params/dense_ffn_params:.2f}x dense FLOPs)")
    print(f"  Tensors: {len(out_state)}")

    save_file(out_state, out_path, metadata={
        "conversion": "dense_to_moe",
        "source": src_path,
        "config": config_name,
        "n_experts": str(n_experts),
        "top_k": str(top_k),
        "shared_expert": str(shared_expert),
        "d_ff_per_expert": str(d_ff),
    })
    print(f"\n  Saved to {out_path}")
    return True


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Conversion keys: Qwen weights → target architecture")
    parser.add_argument("key", choices=["bitnet", "mla", "moe"], help="Which conversion key to use")
    parser.add_argument("--src", required=True, help="Source safetensors path")
    parser.add_argument("--out", required=True, help="Output safetensors path")
    parser.add_argument("--config", default="qwen25_coder_1.5b", help="Source model config name")
    parser.add_argument("--target-config", default=None, help="Target model config name (for MLA)")
    parser.add_argument("--kv-compression-dim", type=int, default=None, help="MLA KV compression dim")
    parser.add_argument("--n-experts", type=int, default=4, help="MoE: number of experts")
    parser.add_argument("--top-k", type=int, default=2, help="MoE: experts per token")
    parser.add_argument("--no-shared", action="store_true", help="MoE: disable shared expert")
    parser.add_argument("--verify", action="store_true", help="Verify output against HF model")
    args = parser.parse_args()

    if args.key == "bitnet":
        key_bitnet(args.src, args.out, args.config)
        if args.verify:
            verify_bitnet(args.out, args.config)
    elif args.key == "mla":
        target_cfg = args.target_config or "360m_mla"
        key_gqa_to_mla(args.src, args.out, source_config_name=args.config,
                       target_config_name=target_cfg, kv_compression_dim=args.kv_compression_dim)
    elif args.key == "moe":
        key_dense_to_moe(args.src, args.out, config_name=args.config,
                         n_experts=args.n_experts, top_k=args.top_k,
                         shared_expert=not args.no_shared)


if __name__ == "__main__":
    main()
