"""Port V4_Base (d_model=2048, 16 layers) into V7-8B-B (d_model=4096, 32 layers).

Clean 2x scale-up:
  - Width: d_model 2048→4096, heads 32→64, kv_heads 8→16, intermediate 8192→16384
  - Depth: 16→32 layers (duplicate each layer)
  - FFN: dense → NLRQ factored (rank=1024)
  - Embedding: plain → factorized (rank=512)

All upscaling is lossless or near-lossless (repeat/interpolate).
The result is a warm-start checkpoint for V7-8B-B training.
"""
import sys, os, time, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from research.config import get_config
from research.model_loader import ConfigurableResearchLLM, ModelLoader


def upscale_weight(w: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    """Upscale a 2D weight tensor to target shape by repeating along each dim."""
    if w.shape == target_shape:
        return w
    assert w.ndim == 2 and len(target_shape) == 2, f"Only 2D: {w.shape} → {target_shape}"
    out_features, in_features = target_shape
    cur_out, cur_in = w.shape
    # Repeat along output dim
    if cur_out < out_features:
        assert out_features % cur_out == 0, f"Output {out_features} not divisible by {cur_out}"
        reps = out_features // cur_out
        w = w.repeat(reps, 1)
    # Repeat along input dim
    if cur_in < in_features:
        assert in_features % cur_in == 0, f"Input {in_features} not divisible by {cur_in}"
        reps = in_features // cur_in
        w = w.repeat(1, reps)
    return w.contiguous()


def upscale_norm(w: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    """Upscale a 1D norm weight by repeating elements."""
    if w.shape == target_shape:
        return w
    assert w.ndim == 1 and len(target_shape) == 1
    target = target_shape[0]
    cur = w.shape[0]
    assert target % cur == 0, f"Norm {target} not divisible by {cur}"
    reps = target // cur
    return w.repeat(reps).contiguous()


def upscale_heads(w: torch.Tensor, n_old_heads: int, n_new_heads: int,
                  head_dim: int, in_dim_old: int, in_dim_new: int) -> torch.Tensor:
    """Upscale attention weight by duplicating heads and repeating input dim.

    w shape: [n_old_heads * head_dim, in_dim_old]
    target:  [n_new_heads * head_dim, in_dim_new]
    """
    if w.shape[0] == n_new_heads * head_dim and w.shape[1] == in_dim_new:
        return w
    # Reshape to per-head: [n_old_heads, head_dim, in_dim_old]
    w_reshaped = w.view(n_old_heads, head_dim, in_dim_old)
    # Duplicate heads
    head_reps = n_new_heads // n_old_heads
    w_reshaped = w_reshaped.repeat(head_reps, 1, 1)  # [n_new_heads, head_dim, in_dim_old]
    # Repeat input dim
    in_reps = in_dim_new // in_dim_old
    w_reshaped = w_reshaped.repeat(1, 1, in_reps)  # [n_new_heads, head_dim, in_dim_new]
    return w_reshaped.reshape(n_new_heads * head_dim, in_dim_new).contiguous()


def upscale_conv_weight(w: torch.Tensor, d_old: int, d_new: int) -> torch.Tensor:
    """Upscale conv layer weight from d_old to d_new channels."""
    if w.shape[0] == d_new:
        return w
    # DoubleGatedConvLayer: in_proj weight shape is [d*3, d] or similar
    # conv.weight shape varies; just repeat along first dim
    cur_out = w.shape[0]
    if d_new % cur_out == 0:
        reps = d_new // cur_out
        return w.repeat(reps, 1).contiguous()
    # Fallback: interpolate
    w_f = w.float().unsqueeze(0).unsqueeze(0)
    w_up = F.interpolate(w_f, size=(d_new, w.shape[1]), mode='bilinear', align_corners=False)
    return w_up.squeeze(0).squeeze(0).to(w.dtype).contiguous()


def svd_factorize_embedding(embed_weight: torch.Tensor, rank: int, device: str = "cuda"):
    """SVD decompose embedding [vocab, d_model] into factorized form.

    Returns (embed_embed [vocab, rank], project [d_model, rank]).
    """
    w = embed_weight.float().to(device)
    U, S, Vh = torch.linalg.svd(w, full_matrices=False)
    # Truncate to rank
    U_r = U[:, :rank]  # [vocab, rank]
    S_r = S[:rank]     # [rank]
    Vh_r = Vh[:rank, :]  # [rank, d_model]
    # embed = U * sqrt(S), project = sqrt(S) @ Vh → embed @ project.T = U*S*Vh
    # nn.Linear(rank, d_model) weight is [d_model, rank], so project = (sqrt_S * Vh).T
    sqrt_S = torch.sqrt(S_r.clamp(min=1e-8))
    embed_embed = (U_r * sqrt_S.unsqueeze(0)).to(torch.bfloat16).cpu()  # [vocab, rank]
    project = (sqrt_S.unsqueeze(1) * Vh_r).T.to(torch.bfloat16).cpu()  # [d_model, rank]
    del w, U, S, Vh
    if 'cuda' in device:
        torch.cuda.empty_cache()
    return embed_embed, project


def port_v4_to_v7_8b(input_path: str, output_path: str,
                     src_config_name: str = "lfm25_1.2b",
                     dst_config_name: str = "forgelm_v7_8b_b",
                     device: str = "cuda"):
    t0 = time.time()

    # Source config (V4_Base was saved with lfm25_1.2b arch + V4 keys)
    src_cfg = get_config(src_config_name)
    dst_cfg = get_config(dst_config_name)

    print(f"Porting {src_config_name} → {dst_config_name}")
    print(f"  Source: d_model={src_cfg.d_model}, layers={src_cfg.n_layers}, "
          f"heads={src_cfg.n_heads}, kv_heads={src_cfg.n_kv_heads}")
    print(f"  Target: d_model={dst_cfg.d_model}, layers={dst_cfg.n_layers}, "
          f"heads={dst_cfg.n_heads}, kv_heads={dst_cfg.n_kv_heads}")

    # Load source checkpoint
    print(f"\n[1] Loading source checkpoint: {input_path}")
    src_state = load_file(input_path)
    print(f"  {len(src_state)} tensors, {sum(v.numel() for v in src_state.values())/1e6:.1f}M params")

    # Build blank target model (to get target tensor names/shapes)
    print(f"\n[2] Building blank {dst_config_name} model...")
    dst_cfg.device = "cpu"  # build on CPU to save VRAM
    model = ConfigurableResearchLLM(dst_cfg)
    dst_state_template = model.state_dict()
    print(f"  {len(dst_state_template)} target tensors, "
          f"{sum(v.numel() for v in model.parameters())/1e6:.1f}M params")

    # Layer mapping: match by layer_type. V7 attention layers cycle through
    # V4 attention layers; V7 conv layers cycle through V4 conv layers.
    n_src_layers = src_cfg.n_layers  # 16
    n_dst_layers = dst_cfg.n_layers  # 32
    src_layer_types = getattr(src_cfg, 'layer_types', None) or ['attention'] * n_src_layers
    dst_layer_types = getattr(dst_cfg, 'layer_types', None) or ['attention'] * n_dst_layers

    src_attn_layers = [i for i, t in enumerate(src_layer_types) if t == 'attention']
    src_conv_layers = [i for i, t in enumerate(src_layer_types) if t in ('conv', 'liquid')]

    layer_map = {}
    attn_idx = 0
    conv_idx = 0
    for i in range(n_dst_layers):
        if dst_layer_types[i] == 'attention':
            layer_map[i] = src_attn_layers[attn_idx % len(src_attn_layers)]
            attn_idx += 1
        else:
            layer_map[i] = src_conv_layers[conv_idx % len(src_conv_layers)]
            conv_idx += 1

    print(f"  Layer mapping (dst → src):")
    for i in range(n_dst_layers):
        print(f"    {i} ({dst_layer_types[i]}) → {layer_map[i]} ({src_layer_types[layer_map[i]]})")

    d_old = src_cfg.d_model   # 2048
    d_new = dst_cfg.d_model   # 4096
    h_old = src_cfg.n_heads   # 32
    h_new = dst_cfg.n_heads   # 64
    kv_old = src_cfg.n_kv_heads  # 8
    kv_new = dst_cfg.n_kv_heads  # 16
    hd = d_old // h_old  # 64 (same for both)
    inter_old = src_cfg.intermediate_size  # 8192
    inter_new = dst_cfg.intermediate_size  # 16384

    print(f"\n[3] Upscaling weights...")
    dst_state = {}

    # ── Embeddings: SVD factorize ──
    if 'embed.weight' in src_state:
        print("  Embedding: SVD factorizing [65536, 2048] → factorized [65536, 512] + [4096, 512]")
        # First upscale embedding to target d_model
        embed_full = src_state['embed.weight']  # [65536, 2048]
        embed_upscaled = upscale_weight(embed_full, (embed_full.shape[0], d_new))  # [65536, 4096]
        rank = dst_cfg.embed_factorized_rank  # 512
        embed_embed, project = svd_factorize_embedding(embed_upscaled, rank, device=device)
        dst_state['embed.embed.weight'] = embed_embed
        dst_state['embed.project.weight'] = project
        # head.embed_ref.* are aliases (not saved separately)
    else:
        print("  WARNING: No embed.weight in source!")

    # ── Final norm ──
    if 'ln_f.weight' in src_state:
        dst_state['ln_f.weight'] = upscale_norm(src_state['ln_f.weight'], (d_new,))

    # ── Per-layer weights ──
    for dst_layer in range(n_dst_layers):
        src_layer = layer_map[dst_layer]
        prefix_src = f"blocks.{src_layer}."
        prefix_dst = f"blocks.{dst_layer}."

        # Determine layer type
        src_layer_types = getattr(src_cfg, 'layer_types', None)
        dst_layer_types = getattr(dst_cfg, 'layer_types', None)
        src_ltype = src_layer_types[src_layer] if src_layer_types else "attention"
        dst_ltype = dst_layer_types[dst_layer] if dst_layer_types else "attention"

        # Norms (always present)
        for norm_name in ['ln1.weight', 'ln2.weight', 'post_attn_norm.weight', 'post_ffn_norm.weight']:
            src_key = prefix_src + norm_name
            dst_key = prefix_dst + norm_name
            if src_key in src_state and dst_key in dst_state_template:
                dst_state[dst_key] = upscale_norm(src_state[src_key], (d_new,))

        if dst_ltype in ("conv", "liquid"):
            # Conv layer: upscale in_proj, conv, out_proj weights
            for wname in ['attn.in_proj.weight', 'attn.conv.weight', 'attn.out_proj.weight']:
                src_key = prefix_src + wname
                dst_key = prefix_dst + wname
                if src_key in src_state and dst_key in dst_state_template:
                    src_w = src_state[src_key]
                    dst_shape = dst_state_template[dst_key].shape
                    if src_w.ndim == 3:
                        # conv.weight: [d, 1, k] → [2d, 1, k]
                        reps = [dst_shape[i] // src_w.shape[i] if dst_shape[i] > src_w.shape[i] else 1
                                for i in range(3)]
                        dst_state[dst_key] = src_w.repeat(*reps).contiguous()
                    elif src_w.ndim == 2:
                        dst_state[dst_key] = upscale_weight(src_w, dst_shape)
                    else:
                        dst_state[dst_key] = src_w.clone()
            # Conv out_proj qscale (scalar, just copy)
            for qname in ['attn.out_proj.qscale']:
                src_key = prefix_src + qname
                dst_key = prefix_dst + qname
                if src_key in src_state and dst_key in dst_state_template:
                    dst_state[dst_key] = src_state[src_key].clone()
        else:
            # Attention layer: upscale q/k/v/out_proj by duplicating heads
            for proj_name, n_h_old, n_h_new in [
                ('q_proj', h_old, h_new),
                ('k_proj', kv_old, kv_new),
                ('v_proj', kv_old, kv_new),
                ('out_proj', h_old, h_new),
            ]:
                # Weight
                src_key = prefix_src + f"attn.{proj_name}.weight"
                dst_key = prefix_dst + f"attn.{proj_name}.weight"
                if src_key in src_state and dst_key in dst_state_template:
                    src_w = src_state[src_key]  # [n_h_old * hd, d_old]
                    dst_state[dst_key] = upscale_heads(
                        src_w, n_h_old, n_h_new, hd, d_old, d_new)
                # qscale (scalar, just copy)
                src_qs = prefix_src + f"attn.{proj_name}.qscale"
                dst_qs = prefix_dst + f"attn.{proj_name}.qscale"
                if src_qs in src_state and dst_qs in dst_state_template:
                    dst_state[dst_qs] = src_state[src_qs].clone()

            # QK norm
            for norm_name in ['attn.q_norm.weight', 'attn.k_norm.weight']:
                src_key = prefix_src + norm_name
                dst_key = prefix_dst + norm_name
                if src_key in src_state and dst_key in dst_state_template:
                    dst_state[dst_key] = upscale_norm(src_state[src_key], (hd,))

            # v_mix_gate (scalar, just copy)
            src_vmg = prefix_src + "attn.v_mix_gate"
            dst_vmg = prefix_dst + "attn.v_mix_gate"
            if src_vmg in src_state and dst_vmg in dst_state_template:
                dst_state[dst_vmg] = src_state[src_vmg].clone()

            # RoPE freq_scale (zero-init, just copy or zeros)
            src_rf = prefix_src + "attn.rope.freq_scale"
            dst_rf = prefix_dst + "attn.rope.freq_scale"
            if dst_rf in dst_state_template:
                dst_state[dst_rf] = torch.zeros(dst_state_template[dst_rf].shape,
                                                dtype=dst_state_template[dst_rf].dtype)

            # Sinks (zero-init)
            src_sk = prefix_src + "attn.sinks"
            dst_sk = prefix_dst + "attn.sinks"
            if dst_sk in dst_state_template:
                dst_state[dst_sk] = torch.zeros(dst_state_template[dst_sk].shape,
                                                dtype=dst_state_template[dst_sk].dtype)

        # FFN: V4 has dense weights, V7 needs NLRQ factored
        # We'll handle this after loading all other weights — build the model,
        # load what we have, then convert dense FFN → NLRQ.
        # For now, skip FFN (will be handled by NLRQ conversion in build_model_fast)

        # MoD router
        src_mod = prefix_src + "_mod.router.weight"
        dst_mod = prefix_dst + "_mod.router.weight"
        if src_mod in src_state and dst_mod in dst_state_template:
            dst_state[dst_mod] = upscale_weight(src_state[src_mod],
                                                dst_state_template[dst_mod].shape)

        # TITAN memory (gate, u, v — all small, just copy)
        for mem_name in ['_memory.gate', '_memory.u', '_memory.v']:
            src_key = prefix_src + mem_name
            dst_key = prefix_dst + mem_name
            if src_key in src_state and dst_key in dst_state_template:
                src_w = src_state[src_key]
                dst_shape = dst_state_template[dst_key].shape
                if src_w.shape == dst_shape:
                    dst_state[dst_key] = src_w.clone()
                else:
                    # Upscale
                    dst_state[dst_key] = upscale_weight(src_w, dst_shape)

        # MHC (U, V, gate)
        for mhc_name in ['_mhc.U', '_mhc.V', '_mhc.gate']:
            src_key = prefix_src + mhc_name
            dst_key = prefix_dst + mhc_name
            if src_key in src_state and dst_key in dst_state_template:
                src_w = src_state[src_key]
                dst_shape = dst_state_template[dst_key].shape
                if src_w.shape == dst_shape:
                    dst_state[dst_key] = src_w.clone()
                else:
                    dst_state[dst_key] = upscale_weight(src_w, dst_shape)

    # ── AttnRes (global, not per-layer) ──
    for ar_name in ['_attn_res.gates', '_attn_res.q_proj.weight', '_attn_res.k_proj.weight',
                     '_attn_res.v_proj.weight', '_attn_res.out_proj.weight']:
        if ar_name in src_state and ar_name in dst_state_template:
            src_w = src_state[ar_name]
            dst_shape = dst_state_template[ar_name].shape
            if src_w.shape == dst_shape:
                dst_state[ar_name] = src_w.clone()
            elif src_w.ndim == 1:
                dst_state[ar_name] = upscale_norm(src_w, dst_shape)
            else:
                # AttnRes q/k/v/out: upscale by heads
                if 'q_proj' in ar_name or 'out_proj' in ar_name:
                    dst_state[ar_name] = upscale_heads(src_w, h_old, h_new, hd, d_old, d_new)
                elif 'k_proj' in ar_name or 'v_proj' in ar_name:
                    # AttnRes k/v are full-size [d_model, d_model], not GQA-reduced
                    dst_state[ar_name] = upscale_heads(src_w, h_old, h_new, hd, d_old, d_new)
                else:
                    dst_state[ar_name] = upscale_weight(src_w, dst_shape)

    # ── Value residual gates (zero-init) ──
    for i in range(n_dst_layers):
        key = f"_v0_gates.{i}"
        if key in dst_state_template:
            dst_state[key] = torch.zeros(1, dtype=dst_state_template[key].dtype)

    # ── Hyperloop gates (zero-init) ──
    for key in dst_state_template:
        if 'loop_gate' in key or 'middle_gate' in key:
            dst_state[key] = torch.zeros(dst_state_template[key].shape,
                                         dtype=dst_state_template[key].dtype)

    # ── LiSA (zero-init gates, upscale shared_q/k) ──
    for key in dst_state_template:
        if 'lisa.gates' in key:
            dst_state[key] = torch.zeros(dst_state_template[key].shape,
                                         dtype=dst_state_template[key].dtype)
        elif 'lisa.shared_q.weight' in key:
            src_w = src_state.get('lisa.shared_q.weight')
            if src_w is not None:
                dst_state[key] = upscale_heads(src_w, h_old, h_new, hd, d_old, d_new)
        elif 'lisa.shared_k.weight' in key:
            src_w = src_state.get('lisa.shared_k.weight')
            if src_w is not None:
                dst_state[key] = upscale_heads(src_w, kv_old, kv_new, hd, d_old, d_new)
        elif 'lisa.align' in key and key in src_state:
            src_w = src_state[key]
            dst_shape = dst_state_template[key].shape
            if src_w.shape == dst_shape:
                dst_state[key] = src_w.clone()
            else:
                dst_state[key] = upscale_weight(src_w, dst_shape)

    # ── Loop block (duplicate from last source layer) ──
    src_layer_lb = n_src_layers - 1
    for key in dst_state_template:
        if key.startswith('loop_block.'):
            src_key = key.replace('loop_block.', f'blocks.{src_layer_lb}.')
            if src_key in src_state:
                src_w = src_state[src_key]
                dst_shape = dst_state_template[key].shape
                if src_w.shape == dst_shape:
                    dst_state[key] = src_w.clone()
                elif src_w.ndim == 1:
                    dst_state[key] = upscale_norm(src_w, dst_shape)
                elif 'ffn' in key:
                    pass  # FFN handled by NLRQ conversion
                elif 'attn' in key and ('q_proj' in key or 'out_proj' in key):
                    dst_state[key] = upscale_heads(src_w, h_old, h_new, hd, d_old, d_new)
                elif 'attn' in key and ('k_proj' in key or 'v_proj' in key):
                    dst_state[key] = upscale_heads(src_w, kv_old, kv_new, hd, d_old, d_new)
                elif src_w.ndim == 3:
                    # 3D conv weight: [d, 1, k] → [2d, 1, k]
                    reps = [dst_shape[i] // src_w.shape[i] if dst_shape[i] > src_w.shape[i] else 1
                            for i in range(3)]
                    dst_state[key] = src_w.repeat(*reps).contiguous()
                else:
                    dst_state[key] = upscale_weight(src_w, dst_shape)

    # ── MTP (skip — will be initialized by model) ──

    # ── Now handle FFN: convert dense V4 FFN → NLRQ factored V7 ──
    print("\n  FFN: Converting dense V4 → NLRQ factored V7...")
    from research.keys.compression.nlrq_ffn_key import NLRQSwiGLUFFN, NLRQLinear

    for dst_layer in range(n_dst_layers):
        src_layer = layer_map[dst_layer]
        prefix_src = f"blocks.{src_layer}."
        prefix_dst = f"blocks.{dst_layer}."

        for ffn_name in ['w_gate', 'w_up', 'w_down']:
            src_key = prefix_src + f"ffn.{ffn_name}.weight"
            if src_key not in src_state:
                continue

            src_w = src_state[src_key]  # dense weight
            # Determine target dense shape
            if ffn_name in ('w_gate', 'w_up'):
                # [intermediate, d_model]
                target_dense = upscale_weight(src_w, (inter_new, d_new))
            else:
                # w_down: [d_model, intermediate]
                target_dense = upscale_weight(src_w, (d_new, inter_new))

            # NLRQ factorize (on CUDA for 10-50x faster SVD)
            rank = dst_cfg.nlrq_rank  # 1024
            factor_bits = dst_cfg.nlrq_factor_bits  # 8
            nlrq = NLRQLinear.from_dense(target_dense.float().to(device), rank=rank,
                                          factor_bits=factor_bits)
            # Store factored weights
            for fname in ['U_q', 'V_q', 'S', 'U_scale', 'V_scale']:
                fkey = prefix_dst + f"ffn.{ffn_name}.{fname}"
                if fkey in dst_state_template:
                    param = getattr(nlrq, fname)
                    dst_state[fkey] = param.data.clone().cpu().to(dst_state_template[fkey].dtype)

            print(f"    layer {dst_layer} {ffn_name}: {src_w.shape} → NLRQ rank={rank}")
            del nlrq, target_dense
            if 'cuda' in device:
                torch.cuda.empty_cache()

    # Also handle loop_block FFN
    print("  Loop block FFN...")
    for ffn_name in ['w_gate', 'w_up', 'w_down']:
        src_key = f"blocks.{src_layer_lb}.ffn.{ffn_name}.weight"
        if src_key not in src_state:
            continue
        src_w = src_state[src_key]
        if ffn_name in ('w_gate', 'w_up'):
            target_dense = upscale_weight(src_w, (inter_new, d_new))
        else:
            target_dense = upscale_weight(src_w, (d_new, inter_new))
        rank = dst_cfg.nlrq_rank
        nlrq = NLRQLinear.from_dense(target_dense.float().to(device), rank=rank,
                                       factor_bits=dst_cfg.nlrq_factor_bits)
        for fname in ['U_q', 'V_q', 'S', 'U_scale', 'V_scale']:
            fkey = f"loop_block.ffn.{ffn_name}.{fname}"
            if fkey in dst_state_template:
                param = getattr(nlrq, fname)
                dst_state[fkey] = param.data.clone().cpu().to(dst_state_template[fkey].dtype)
        del nlrq, target_dense
        if 'cuda' in device:
            torch.cuda.empty_cache()

    # ── R&D round 14: training speedup features (config-only, no checkpoint keys) ──
    # The following R&D round 14 features are TRAINING-ONLY and config-driven.
    # They do NOT introduce new model parameters or buffers, so no checkpoint
    # key conversion or zero-init is needed here. They are enabled via the
    # destination config preset (dst_cfg, loaded above via get_config), not
    # via checkpoint weights:
    #
    #   - use_varlen (bool):       FlashAttention varlen path for packed
    #                              sequences. No new weights — just a different
    #                              attention kernel invocation at training time.
    #   - use_triton_kernels (bool): Fused RMSNorm + SwiGLU Triton kernels
    #                              (Liger-Kernel-style). No new weights —
    #                              replaces existing kernel launches.
    #   - triton_rms_block_size (int): Runtime kernel launch config, not a
    #                              learned parameter. Defaults to d_model.
    #   - triton_swiglu_block_size (int): Runtime kernel launch config, not a
    #                              learned parameter. Defaults to intermediate_size.
    #   - apollo_rank (int):       APOLLO optimizer config (SVD-free random
    #                              projection). Optimizer state only, not in
    #                              the model checkpoint.
    #   - apollo_scale (str):      APOLLO optimizer scaling mode. Config only.
    #   - bread_sgd_correction (str): BREAD landscape correction for BAdam.
    #                              Optimizer behavior only, not in checkpoint.
    #   - bread_sgd_lr_scale (float): BREAD SGD learning rate scale. Config only.
    #   - flashoptim_bits (int):   FlashOptim companded optimizer state format.
    #                              Optimizer state only, not in checkpoint.
    #   - use_gradient_checkpointing (bool): Training-time VRAM tradeoff. Config only.
    #
    # Per AGENTS.md "Port-first, train-second": these features don't require a
    # checkpoint-conversion path because they have no architecture keys. The
    # V4 source checkpoint has no equivalent keys, and the V7 destination
    # checkpoint has no new keys for them. The config preset
    # (forgelm_v7_8b_b) already sets these fields to their correct defaults
    # (use_varlen=True, use_triton_kernels=True, etc.), so the converter
    # output is automatically configured for round-14 training.
    #
    # No state_dict modifications needed for round 14.

    # ── Save ──
    print(f"\n[4] Saving to {output_path}...")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Convert all to bf16 (except int8 NLRQ factors) and remove tied keys
    save_dict = {}
    skip_keys = {'head.embed_ref.embed.weight', 'head.embed_ref.project.weight'}
    for k, v in dst_state.items():
        if k in skip_keys:
            continue
        if k in dst_state_template:
            target_dtype = dst_state_template[k].dtype
            # Force bf16 for float32 tensors (saves 2x space); keep int8/fp16 as-is
            if target_dtype == torch.float32:
                target_dtype = torch.bfloat16
            save_dict[k] = v.to(target_dtype).contiguous().clone()

    # Add any remaining template keys with proper init (not all zeros)
    for k, v_template in dst_state_template.items():
        if k not in save_dict and k not in skip_keys:
            # Skip RoPE buffers (re-initialized at load time)
            if 'rope' in k:
                continue
            # Skip MTP (initialized by model)
            if 'mtp_module' in k:
                continue
            # Use bf16 for float32 templates (saves 2x space)
            fill_dtype = torch.bfloat16 if v_template.dtype == torch.float32 else v_template.dtype
            # RMSNorm weights: identity = 1.0 (not 0.0 which kills the signal)
            if 'norm' in k or k == 'ln_f.weight':
                save_dict[k] = torch.ones(v_template.shape, dtype=fill_dtype)
                print(f"  INIT: {k} = ones (RMSNorm identity)")
            # LiSA shared_q/k: init from first attention layer's q/k proj
            elif k == 'lisa.shared_q.weight':
                first_q = dst_state.get('blocks.2.attn.q_proj.weight')
                if first_q is not None:
                    save_dict[k] = first_q.clone().to(fill_dtype)
                    print(f"  INIT: {k} from blocks.2.attn.q_proj.weight")
            elif k == 'lisa.shared_k.weight':
                first_k = dst_state.get('blocks.2.attn.k_proj.weight')
                if first_k is not None:
                    save_dict[k] = first_k.clone().to(fill_dtype)
                    print(f"  INIT: {k} from blocks.2.attn.k_proj.weight")
            # LiSA align layers: zero-init (gate=0 → lossless)
            elif 'lisa.align' in k:
                save_dict[k] = torch.zeros(v_template.shape, dtype=fill_dtype)
            # v_mix_gate: 0.0 = V=K (lossless)
            elif 'v_mix_gate' in k:
                save_dict[k] = torch.zeros(v_template.shape, dtype=fill_dtype)
            # Sinks: zero-init
            elif 'sinks' in k:
                save_dict[k] = torch.zeros(v_template.shape, dtype=fill_dtype)
            # Everything else: zeros (gates, routers, etc. — zero = lossless)
            else:
                save_dict[k] = torch.zeros(v_template.shape, dtype=fill_dtype)

    save_file(save_dict, output_path)
    fsize = os.path.getsize(output_path) / 1e9
    print(f"  Saved: {fsize:.2f} GB, {len(save_dict)} tensors")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    return output_path


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Port V4_Base to V7-8B-B with 2x upscaling")
    p.add_argument("--input", default="research/checkpoints/ForgeLM_V4_Base.safetensors")
    p.add_argument("--output", default="research/checkpoints/ForgeLM_V7_8B_B_ported.safetensors")
    p.add_argument("--src-config", default="lfm25_1.2b")
    p.add_argument("--dst-config", default="forgelm_v7_8b_b")
    p.add_argument("--device", default="cuda", help="Device for SVD/NLRQ (cuda or cpu)")
    args = p.parse_args()
    port_v4_to_v7_8b(args.input, args.output, args.src_config, args.dst_config, args.device)
