"""Key #2: SVD-based weight resizing with shared projection.

Instead of independently SVD-truncating each weight matrix (which creates
incompatible dimension reductions), we compute a SINGLE projection matrix
from the embedding's SVD and apply it consistently to all layers.

Algorithm:
1. SVD the embedding: E [vocab, d_src] = U @ S @ Vh
2. Projection: P = Vh[:d_tgt, :]  [d_tgt, d_src] — top-k right singular vectors
3. For each weight W [out, d_src]: W_new = W @ P^T  [out, d_tgt]
4. For each weight W [d_src, in]: W_new = P @ W  [d_tgt, in]
5. For norms/biases of size d_src: project via P (norms) or truncate (biases)
6. Drop middle layers to reduce depth

This preserves the most important directions in the residual stream consistently.
Based on "Weight Subcloning" (2023) which showed 4x faster training with this approach.
"""
import re

import torch
from safetensors.torch import load_file, save_file

from research.config import get_config
from research.model_loader import ModelLoader


def compute_shared_projection(embed_weight: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Compute a shared projection matrix from the embedding's SVD.

    Args:
        embed_weight: embedding matrix [vocab, d_src]
        target_dim: target dimension d_tgt

    Returns:
        P [d_tgt, d_src] — projection matrix (top-k right singular vectors)
    """
    E = embed_weight.float()  # [vocab, d_src]
    U, S, Vh = torch.linalg.svd(E, full_matrices=False)
    P = Vh[:target_dim, :]  # [d_tgt, d_src]
    print(f"  Shared projection: {list(P.shape)} (from embedding SVD, top-{target_dim} singular values)")
    print(f"  Singular values kept: {S[:target_dim].sum().item():.1f} / {S.sum().item():.1f} "
          f"({S[:target_dim].sum().item()/S.sum().item()*100:.1f}% of energy)")
    return P.to(torch.bfloat16)


def project_weight(W: torch.Tensor, P: torch.Tensor, is_input: bool, is_output: bool) -> torch.Tensor:
    """Project a weight matrix using shared projection P.

    Args:
        W: weight matrix [out, in]
        P: projection matrix [d_tgt, d_src]
        is_input: if True, W's input dim (dim=1) is d_src and should be reduced to d_tgt
        is_output: if True, W's output dim (dim=0) is d_src and should be reduced to d_tgt

    Returns:
        Projected weight matrix
    """
    Wf = W.float()
    Pf = P.float()

    if is_input and is_output:
        # Both input and output are d_src -> d_tgt
        # W_new = P @ W @ P^T  [d_tgt, d_tgt]
        return (Pf @ Wf @ Pf.T).to(W.dtype)
    elif is_input:
        # Input is d_src -> d_tgt: W_new = W @ P^T  [out, d_tgt]
        return (Wf @ Pf.T).to(W.dtype)
    elif is_output:
        # Output is d_src -> d_tgt: W_new = P @ W  [d_tgt, in]
        return (Pf @ Wf).to(W.dtype)
    else:
        return W


def key_svd_resize(src_path: str, out_path: str, target_config_name: str,
                   source_config_name: str = "qwen25_coder_1.5b"):
    """Key #2: Resize a model via shared SVD projection + layer dropping.

    Computes a single projection from the embedding's SVD, then applies it
    consistently to all weight matrices to reduce d_model.
    """
    print("Key #2: SVD resize (shared projection)")
    print(f"  Source: {src_path} ({source_config_name})")
    print(f"  Target: {out_path} ({target_config_name})")

    src_cfg = get_config(source_config_name)
    tgt_cfg = get_config(target_config_name)

    print(f"\n  Source: {src_cfg.d_model}d x {src_cfg.n_layers}L, FFN={src_cfg.intermediate_size}")
    print(f"  Target: {tgt_cfg.d_model}d x {tgt_cfg.n_layers}L, FFN={tgt_cfg.intermediate_size}")

    # Layer mapping: proportionally sample source layers
    keep_indices = [int(i * src_cfg.n_layers / tgt_cfg.n_layers) for i in range(tgt_cfg.n_layers)]
    print(f"  Layer mapping: source {keep_indices} -> target 0..{tgt_cfg.n_layers-1}")

    # Load source weights
    src_state = load_file(src_path)

    # Step 1: Compute shared projection from embedding
    d_src = src_cfg.d_model
    d_tgt = tgt_cfg.d_model
    P = compute_shared_projection(src_state["embed.weight"], d_tgt)

    # Get target parameter names/shapes from cached blank state dict
    tgt_state = ModelLoader.blank_state_dict(tgt_cfg)

    # Determine which FFN intermediate size to use
    src_ffn = src_cfg.intermediate_size or 8 * src_cfg.d_model // 3
    tgt_ffn = tgt_cfg.intermediate_size or 8 * tgt_cfg.d_model // 3

    resized = {}
    n_projected = 0
    n_copied = 0

    for tgt_name, tgt_tensor in tgt_state.items():
        tgt_shape = list(tgt_tensor.shape)

        # Determine source name
        if "blocks." in tgt_name:
            m = re.match(r"blocks\.(\d+)\.(.+)", tgt_name)
            tgt_layer = int(m.group(1))
            remainder = m.group(2)
            src_layer = keep_indices[tgt_layer]
            src_name = f"blocks.{src_layer}.{remainder}"
        elif tgt_name == "head.weight":
            src_name = "embed.weight"  # Tied
        else:
            src_name = tgt_name  # embed, ln_f

        if src_name not in src_state:
            print(f"  WARNING: {src_name} not in source, skipping")
            continue

        src_tensor = src_state[src_name]
        src_shape = list(src_tensor.shape)

        if src_shape == tgt_shape:
            # Same shape — direct copy
            resized[tgt_name] = src_tensor.to(torch.bfloat16).contiguous()
            n_copied += 1
            continue

        # Determine projection type based on tensor role
        if tgt_name == "embed.weight" or tgt_name == "head.weight":
            # Embedding/head: [vocab, d_src] -> [vocab, d_tgt]
            # Project input dim (dim=1) from d_src to d_tgt
            W_new = project_weight(src_tensor, P, is_input=True, is_output=False)
            resized[tgt_name] = W_new.to(torch.bfloat16).contiguous()
            n_projected += 1

        elif "ln" in tgt_name or "norm" in tgt_name:
            # Norm weights: [d_src] -> [d_tgt]
            # Project: norm_new = P @ norm (weighted average)
            norm_proj = P.float() @ src_tensor.float()
            resized[tgt_name] = norm_proj.to(torch.bfloat16).contiguous()
            n_projected += 1

        elif "q_proj" in tgt_name or "out_proj" in tgt_name:
            # Attention Q/O: [d_src, d_src] -> [d_tgt, d_tgt]
            if ".weight" in tgt_name:
                W_new = project_weight(src_tensor, P, is_input=True, is_output=True)
                resized[tgt_name] = W_new.to(torch.bfloat16).contiguous()
            elif ".bias" in tgt_name:
                # Bias: [d_src] -> [d_tgt]
                b_new = P.float() @ src_tensor.float()
                resized[tgt_name] = b_new.to(torch.bfloat16).contiguous()
            n_projected += 1

        elif "k_proj" in tgt_name or "v_proj" in tgt_name:
            # K/V: [n_kv * head_dim, d_src] -> [n_kv * head_dim, d_tgt]
            # Only input dim changes (output is n_kv * head_dim, not d_model)
            if ".weight" in tgt_name:
                W_new = project_weight(src_tensor, P, is_input=True, is_output=False)
                resized[tgt_name] = W_new.to(torch.bfloat16).contiguous()
            elif ".bias" in tgt_name:
                # K/V bias: [n_kv * head_dim] — same size if head_dim unchanged
                if src_shape == tgt_shape:
                    resized[tgt_name] = src_tensor.to(torch.bfloat16).contiguous()
                    n_copied += 1
                else:
                    resized[tgt_name] = src_tensor[:tgt_shape[0]].to(torch.bfloat16).contiguous()
                    n_projected += 1
            else:
                n_projected += 1

        elif "w_gate" in tgt_name or "w_up" in tgt_name:
            # FFN gate/up: [ffn_src, d_src] -> [ffn_tgt, d_tgt]
            # Need to reduce both output (ffn) and input (d_model)
            Wf = src_tensor.float()
            Pf = P.float()

            # First reduce input dim: W @ P^T [ffn_src, d_tgt]
            W_in = Wf @ Pf.T  # [ffn_src, d_tgt]

            # Then reduce output dim (FFN) via neuron importance ranking
            if W_in.shape[0] > tgt_shape[0]:
                norms = W_in.norm(dim=1)
                _, top_idx = torch.topk(norms, tgt_shape[0])
                top_idx = top_idx.sort().values
                W_new = W_in[top_idx]
            else:
                W_new = W_in

            resized[tgt_name] = W_new.to(torch.bfloat16).contiguous()
            n_projected += 1

        elif "w_down" in tgt_name:
            # FFN down: [d_src, ffn_src] -> [d_tgt, ffn_tgt]
            # Need to reduce both output (d_model) and input (FFN)
            Wf = src_tensor.float()
            Pf = P.float()

            # First reduce output dim: P @ W [d_tgt, ffn_src]
            W_out = Pf @ Wf  # [d_tgt, ffn_src]

            # Then reduce input dim (FFN) via column importance ranking
            if W_out.shape[1] > tgt_shape[1]:
                norms = W_out.norm(dim=0)
                _, top_idx = torch.topk(norms, tgt_shape[1])
                top_idx = top_idx.sort().values
                W_new = W_out[:, top_idx]
            else:
                W_new = W_out

            resized[tgt_name] = W_new.to(torch.bfloat16).contiguous()
            n_projected += 1

        else:
            print(f"  WARNING: unhandled tensor {tgt_name} {src_shape} -> {tgt_shape}")
            resized[tgt_name] = src_tensor.to(torch.bfloat16).contiguous()
            n_copied += 1

    print(f"\n  Projected: {n_projected} tensors")
    print(f"  Copied: {n_copied} tensors")

    # Calculate sizes
    src_params = sum(t.numel() for t in src_state.values()) / 2  # subtract tied weight
    tgt_params = sum(t.numel() for t in resized.values()) / 2
    print(f"  Params: ~{src_params/1e6:.1f}M -> ~{tgt_params/1e6:.1f}M")

    save_file(resized, out_path, metadata={
        "conversion": "svd_resize_shared",
        "source": src_path,
        "source_config": source_config_name,
        "target_config": target_config_name,
        "layer_mapping": str(keep_indices),
    })
    print(f"\n  Saved to {out_path}")
    return True
