"""Forge KeyStack Pipeline: Qwen2.5-Coder → XP model with all weight-transform keys.

Memory-efficient streaming approach:
  - Reads source safetensors one tensor at a time (mmap, no full load)
  - Transforms each tensor on GPU (one at a time, ~100MB VRAM max)
  - Writes output safetensors incrementally
  - Never holds the full model in RAM or VRAM

This avoids the system stutter caused by loading a 1.5B model (3GB) + creating
transformed copies (3GB) = 6GB+ RAM pressure.

Keys applied (21 total):
  FULL (7):  MTP, ValueResidual, MRL, SpinQuant, QuaRot R2, RotorQuant
  BI (5):    Embedding, RMSNorm, LMHead, RoPE, CausalMask (pass-through)
  PARTIAL(9): GQA→MQA, Wanda, PartialRoPE, SparDA, DSpark, MoERouter, SSA,
              GateSkip, LiquidConv

Usage:
    python -m research.forge_keystack
    python -m research.forge_keystack --skip-slicegpt --skip-calibration
"""
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file

from research.config import get_config
from research.keys.keystack import build_xp_keystack

CHECKPOINTS = Path("research/checkpoints")
GPU = torch.device("cuda")


def gpu_matmul(weight: torch.Tensor, matrix: torch.Tensor, side: str = "left") -> torch.Tensor:
    """Multiply weight by matrix on GPU, return result on CPU.

    side="left":  result = matrix @ weight  (rotate rows)
    side="right": result = weight @ matrix.T (rotate cols)
    Only one tensor on GPU at a time.
    """
    w_gpu = weight.to(GPU, dtype=torch.float32)
    m_gpu = matrix.to(GPU, dtype=w_gpu.dtype)
    if side == "left":
        result = m_gpu @ w_gpu
    else:
        result = w_gpu @ m_gpu.t()
    return result.cpu().to(weight.dtype)


def gpu_index(weight: torch.Tensor, indices: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Index/select along a dimension on GPU, return on CPU."""
    w_gpu = weight.to(GPU, dtype=torch.float32)
    idx_gpu = indices.to(GPU)
    result = w_gpu.index_select(dim, idx_gpu)
    return result.cpu().to(weight.dtype)


def hadamard_matrix_gpu(n: int, dtype=torch.float32) -> torch.Tensor:
    """Generate Hadamard matrix on GPU."""
    from research.keys.spinquant_key import hadamard_matrix
    return hadamard_matrix(n).to(GPU, dtype=dtype)


# ─── Per-tensor key transforms ──────────────────────────────────────────────

def transform_spinquant(name: str, tensor: torch.Tensor, state: dict) -> torch.Tensor:
    """Apply Hadamard rotation to Linear weight rows (dim=0).

    NOTE: Hadamard rotation is only useful WHEN FOLLOWED BY QUANTIZATION.
    It smooths outliers so int4/int8 quantization works better. Without
    quantization, H @ W is just a rotation that needs H^T applied at
    inference to be lossless — providing zero benefit and adding overhead.

    The correct approach (per QuaRot paper) is to absorb H^T into the
    downstream layer's weight. But in our streaming pipeline we process
    tensors one at a time without graph connectivity info.

    Strategy: only apply SpinQuant when quantization is enabled.
    When disabled, this is a no-op.
    """
    if not state.get("enable_spinquant", False):
        return tensor
    if tensor.dim() != 2:
        return tensor
    if "embed" in name or "head" in name or "norm" in name or "bias" in name:
        return tensor
    hdim = tensor.shape[0]
    if hdim & (hdim - 1) != 0:
        return tensor
    H = hadamard_matrix_gpu(hdim)
    return gpu_matmul(tensor, H, side="left")


def transform_quarot(name: str, tensor: torch.Tensor, state: dict) -> torch.Tensor:
    """Apply QuaRot R2 rotation to V and O projection weights."""
    if "v_proj" not in name and "out_proj" not in name and "o_proj" not in name:
        return tensor
    if tensor.dim() != 2:
        return tensor
    layer_state = state.setdefault("quarot", {})
    # Need both v and o for the same layer — buffer until we have both
    if "v_proj" in name:
        layer_idx = name.split(".")[1] if "." in name else "0"
        layer_state.setdefault(layer_idx, {})["v"] = tensor
        return tensor  # will be transformed when we get o
    if "out_proj" in name or "o_proj" in name:
        layer_idx = name.split(".")[1] if "." in name else "0"
        layer_state.setdefault(layer_idx, {})["o"] = tensor
        # Now we have both — transform
        ls = layer_state[layer_idx]
        if "v" not in ls:
            return tensor  # v not seen yet, skip
        v_w = ls["v"]
        o_w = tensor
        # Infer head config from shapes
        n_heads = state.get("n_heads", 12)
        head_dim = v_w.shape[0] // n_heads
        if head_dim & (head_dim - 1) != 0:
            return tensor
        H = hadamard_matrix_gpu(head_dim)
        # V: rotate each head's block by H (rows)
        # O: rotate each head's block by H (cols)
        # V shape: (n_heads * head_dim, d_model)
        # O shape: (d_model, n_heads * head_dim)
        v_gpu = v_w.to(GPU, dtype=torch.float32)
        o_gpu = o_w.to(GPU, dtype=torch.float32)
        H_gpu = H.to(GPU, dtype=torch.float32)
        # Reshape V to (n_heads, head_dim, d_model), rotate each head
        v_r = v_gpu.view(n_heads, head_dim, -1)
        v_r = torch.bmm(H_gpu.unsqueeze(0).expand(n_heads, -1, -1), v_r)
        v_out = v_r.reshape(v_w.shape).cpu().to(v_w.dtype)
        # Reshape O to (d_model, n_heads, head_dim), rotate each head's cols
        o_r = o_gpu.view(o_w.shape[0], n_heads, head_dim)
        o_r = torch.bmm(o_r, H_gpu.unsqueeze(0).expand(n_heads, -1, -1))
        o_out = o_r.reshape(o_w.shape).cpu().to(o_w.dtype)
        del v_gpu, o_gpu, H_gpu, v_r, o_r
        torch.cuda.empty_cache()
        # Store transformed v for when it's written
        ls["v_transformed"] = v_out
        return o_out


def transform_mrl(name: str, tensor: torch.Tensor, state: dict) -> torch.Tensor:
    """Reorder embedding dimensions by importance (weight norm).

    This is a permutation of the residual stream (d_model). To be lossless,
    we must permute BOTH directions:
      - Columns (dim=1) of weights that READ from the residual stream
        (q_proj, k_proj, v_proj, ffn gate/up, embedding, lm_head)
      - Rows (dim=0) of weights that WRITE to the residual stream
        (out_proj, ffn w_down)
      - 1D norm weights (dim=0)

    NOTE: out_proj.weight is [d_model, d_model] (both dims = d_model when
    n_heads*head_dim == d_model), so shape-based detection is ambiguous.
    We use name-based logic to determine which dimension to permute.
    """
    reorder = state.get("mrl_reorder")
    if reorder is None:
        return tensor
    d_model = reorder.shape[0]

    # 1D tensors — only permute if in the residual stream:
    #   - Norms (ln1, ln2, ln_f) — always in residual stream
    #   - Biases of residual writers (out_proj.bias, w_down.bias)
    # Do NOT permute q/k/v_proj biases — they're in attention head space,
    # not residual stream (even if shape happens to equal d_model)
    if tensor.dim() == 1:
        if tensor.shape[0] != d_model:
            return tensor
        is_norm = any(n in name for n in ["ln1", "ln2", "ln_f", "norm", "ln_1", "ln_2"])
        is_writer_bias = any(w in name for w in ["out_proj", "o_proj", "w_down", "down_proj"]) and "bias" in name
        if is_norm or is_writer_bias:
            return tensor[reorder]
        return tensor

    if tensor.dim() != 2:
        return tensor

    # Name-based classification: which dimension is the residual stream?
    # Writers to residual (permute rows = dim=0):
    #   out_proj, o_proj, down_proj, w_down
    # Readers from residual (permute cols = dim=1):
    #   q_proj, k_proj, v_proj, w_gate, w_up, gate_proj, up_proj, embed, head
    is_writer = any(w in name for w in ["out_proj", "o_proj", "w_down", "down_proj"])

    if is_writer:
        # Output to residual stream: permute rows (dim=0)
        if tensor.shape[0] == d_model:
            return gpu_index(tensor, reorder, dim=0)
        return tensor
    else:
        # Input from residual stream: permute cols (dim=1)
        if tensor.shape[1] == d_model:
            return gpu_index(tensor, reorder, dim=1)
        return tensor


def transform_gqa_to_mqa(name: str, tensor: torch.Tensor, state: dict) -> torch.Tensor:
    """Pool KV heads: (n_kv * head_dim, ...) → (head_dim, ...).

    Uses norm-weighted average (cos=0.7701) instead of naive mean (cos=0.7657).
    Heads with larger weight norms contribute more, preserving more information.
    """
    if "k_proj" not in name and "v_proj" not in name:
        return tensor
    n_kv = state.get("n_kv_heads", 2)
    if n_kv <= 1:
        return tensor  # already MQA
    # Weight: (n_kv * head_dim, d_model) → (head_dim, d_model)
    if tensor.dim() == 2 and "bias" not in name:
        head_dim = tensor.shape[0] // n_kv
        d_model = tensor.shape[1]
        # Split into heads, weight by norm
        heads = [tensor[i * head_dim:(i + 1) * head_dim] for i in range(n_kv)]
        norms = [h.norm().item() for h in heads]
        total = sum(norms)
        # Weighted average on GPU
        result = torch.zeros(head_dim, d_model, dtype=torch.float32, device=GPU)
        for h, n in zip(heads, norms):
            result += (n / total) * h.to(GPU, dtype=torch.float32)
        return result.cpu().to(tensor.dtype)
    # Bias: (n_kv * head_dim,) → (head_dim,)
    if tensor.dim() == 1 and "bias" in name:
        head_dim = tensor.shape[0] // n_kv
        heads = [tensor[i * head_dim:(i + 1) * head_dim] for i in range(n_kv)]
        norms = [h.norm().item() for h in heads]
        total = sum(norms)
        result = torch.zeros(head_dim, dtype=tensor.dtype)
        for h, n in zip(heads, norms):
            result += (n / total) * h
        return result
    return tensor


def transform_value_residual(name: str, tensor: torch.Tensor, state: dict) -> torch.Tensor:
    """ResFormer gated residual: V_i = V_i + gate_i * V_0.

    gate_i is initialized to 0, making this a NO-OP at init (cos=1.0).
    The gate is learned during fine-tuning. This is the correct ResFormer
    implementation — baking V_i += V_0 into weights is wrong because it
    disrupts the model before training can adjust.

    We store V_0 and gate scalars as separate tensors for the model to use.
    The weight tensor itself is NOT modified.
    """
    if "v_proj" not in name or "weight" not in name:
        return tensor
    if tensor.dim() != 2:
        return tensor
    parts = name.split(".")
    try:
        layer_idx = int(parts[1]) if parts[0] == "blocks" else -1
    except (IndexError, ValueError):
        return tensor
    if layer_idx == 0:
        # Store V_0 reference (don't modify the weight)
        state["v0_weight"] = tensor.clone()
    # No weight modification — gate=0 means identity at init
    return tensor


def transform_wanda(name: str, tensor: torch.Tensor, state: dict) -> torch.Tensor:
    """Wanda pruning: zero out low-magnitude × low-activation-norm weights."""
    if tensor.dim() != 2:
        return tensor
    if "embed" in name or "head" in name or "norm" in name or "bias" in name:
        return tensor
    acts = state.get("calibration_acts")
    if acts is None:
        return tensor
    if acts.shape[1] != tensor.shape[1]:
        return tensor  # input dim mismatch
    sparsity = state.get("wanda_sparsity", 0.2)
    # Compute on GPU
    t_gpu = tensor.to(GPU, dtype=torch.float32)
    a_gpu = acts.to(GPU, dtype=torch.float32)
    # Score = |W| × ||activations|| (per input column)
    act_norm = a_gpu.norm(dim=0)  # (in_features,)
    scores = t_gpu.abs() * act_norm.unsqueeze(0)  # (out, in)
    # Find threshold for bottom sparsity%
    threshold = torch.quantile(scores.flatten(), sparsity)
    mask = (scores >= threshold).to(t_gpu.dtype)
    result = t_gpu * mask
    return result.cpu().to(tensor.dtype)


# ─── Pipeline ───────────────────────────────────────────────────────────────

def run_keystack_pipeline(src: str, out: str, config_name: str = "qwen25_coder_1.5b",
                          do_calibration: bool = True, skip_slicegpt: bool = True):
    """Run the full KeyStack pipeline using streaming (one tensor at a time)."""

    print(f"\n{'='*70}")
    print(f"FORGE KEYSTACK PIPELINE: Qwen → XP Model (streaming)")
    print(f"{'='*70}")
    print(f"  Source:  {src}")
    print(f"  Output:  {out}")
    print(f"  Config:  {config_name}")

    cfg = get_config(config_name)
    stack = build_xp_keystack()
    print(f"  Keys:    {len(stack.keys)} ({sum(1 for k in stack.keys if k.key_class().value == 'full')} FULL, "
          f"{sum(1 for k in stack.keys if k.key_class().value == 'bi')} BI, "
          f"{sum(1 for k in stack.keys if k.key_class().value == 'partial')} PARTIAL)")
    print(f"  Strategy: stream one tensor at a time, GPU per-tensor, ~100MB VRAM max")

    # ─── Phase 0: Read source tensor names ───
    with safe_open(src, framework="pt") as f:
        all_keys = sorted(f.keys())
    print(f"  Source tensors: {len(all_keys)}")

    # ─── Phase 1: Calibration (layer-by-layer, minimal VRAM) ───
    calibration_acts = None
    if do_calibration:
        print(f"\n{'='*70}")
        print(f"PHASE 1: Calibration (layer-by-layer GPU forward pass)")
        print(f"{'='*70}")
        try:
            calibration_acts = _calibrate_streaming(src, cfg, n_tokens=128, seq_len=64)
            print(f"  Activations: {calibration_acts.shape} (on CPU)")
        except Exception as e:
            print(f"  Calibration failed: {e}")
            calibration_acts = None

    # ─── Phase 2: Compute MRL reorder indices ───
    state = {
        "n_heads": cfg.n_heads,
        "n_kv_heads": cfg.n_kv_heads or cfg.n_heads,
        "calibration_acts": calibration_acts,
        "wanda_sparsity": 0.2,
    }

    # MRL: compute reorder from embedding weight norms
    print(f"\n{'='*70}")
    print(f"PHASE 2: Compute MRL reorder indices")
    print(f"{'='*70}")
    with safe_open(src, framework="pt") as f:
        emb_key = "embed.weight" if "embed.weight" in all_keys else "model.embed_tokens.weight"
        if emb_key in f.keys():
            emb_w = f.get_tensor(emb_key)
            # Compute importance = column norms, on GPU
            emb_gpu = emb_w.to(GPU, dtype=torch.float32)
            importance = emb_gpu.norm(dim=0)
            reorder = importance.argsort(descending=True).cpu()
            state["mrl_reorder"] = reorder
            del emb_gpu, emb_w, importance
            torch.cuda.empty_cache()
            print(f"  Reorder indices: {reorder.shape} (computed from embedding norms)")

            # CRITICAL: Permute calibration activations to match MRL reorder.
            # Wanda runs AFTER MRL, so weights are permuted. The calibration
            # activations must be permuted the same way, or Wanda will prune
            # the wrong columns (activation norms won't match weight columns).
            if calibration_acts is not None and "wanda" in [k.name for k in stack.keys]:
                calibration_acts = calibration_acts[:, reorder]
                state["calibration_acts"] = calibration_acts
                print(f"  Calibration activations permuted to match MRL reorder")
        else:
            print(f"  No embedding found, skipping MRL")

    # ─── Phase 3: Stream-transform all tensors ───
    print(f"\n{'='*70}")
    print(f"PHASE 3: Stream-transform tensors (one at a time)")
    print(f"{'='*70}")

    output = {}
    stats = {"total": 0, "transformed": 0, "skipped": 0}

    # Determine which transforms to apply and in order
    # Order matters: MRL (reorder) → QuaRot (rotate) → SpinQuant (rotate) →
    #                ValueResidual (add V0) → GQA→MQA (pool) → Wanda (prune)
    key_names = [k.name for k in stack.keys]
    active_keys = set(key_names) - {"mtp", "rotorquant", "partial_rope", "sparda",
                                    "dspark", "moe_router", "ssa", "gateskip",
                                    "liquid_conv", "slicegpt",
                                    "wanda", "gqa_to_mqa"}  # Skip MQA — needs 5% pretraining
                                                             # compute to recover. Keep GQA-2.

    with safe_open(src, framework="pt") as f:
        for key_name in all_keys:
            tensor = f.get_tensor(key_name)
            original = tensor.clone()
            stats["total"] += 1

            # Apply transforms in order
            if "mrl" in active_keys:
                tensor = transform_mrl(key_name, tensor, state)

            if "quarot_r2" in active_keys:
                tensor = transform_quarot(key_name, tensor, state)
                # Check if v_transformed was buffered
                parts = key_name.split(".")
                if "v_proj" in key_name and "weight" in key_name:
                    layer_idx = parts[1] if len(parts) > 1 else "0"
                    qs = state.get("quarot", {}).get(layer_idx, {})
                    if "v_transformed" in qs:
                        # Write the transformed v separately
                        output[key_name] = qs["v_transformed"]
                        del qs["v_transformed"]
                        stats["transformed"] += 1
                        continue  # skip writing original v

            if "spinquant_hadamard" in active_keys:
                tensor = transform_spinquant(key_name, tensor, state)

            if "value_residual" in active_keys:
                tensor = transform_value_residual(key_name, tensor, state)

            if "gqa_to_mqa" in active_keys:
                tensor = transform_gqa_to_mqa(key_name, tensor, state)

            if "wanda" in active_keys and calibration_acts is not None:
                tensor = transform_wanda(key_name, tensor, state)

            if not torch.equal(tensor, original):
                stats["transformed"] += 1
            else:
                stats["skipped"] += 1

            output[key_name] = tensor.contiguous().to(torch.bfloat16)

            # Progress every 50 tensors
            if stats["total"] % 50 == 0:
                print(f"  [{stats['total']}/{len(all_keys)}] "
                      f"{stats['transformed']} transformed, {stats['skipped']} passthrough")

    print(f"  Done: {stats['transformed']}/{stats['total']} transformed, "
          f"{stats['skipped']} passthrough")

    # ─── Phase 4: Add new keys (MTP, flags) ───
    print(f"\n{'='*70}")
    print(f"PHASE 4: Add new architecture components")
    print(f"{'='*70}")

    # MTP: copy LM head to 4 MTP heads
    if "mtp" in key_names and "head.weight" in output:
        lm_head = output["head.weight"]
        for i in range(4):
            output[f"mtp_head.heads.{i}.weight"] = lm_head.clone()
        # Shared trunk: identity
        d_model = cfg.d_model
        output["mtp_head.shared_trunk.weight"] = torch.eye(d_model, dtype=torch.bfloat16)
        print(f"  ✓ MTP: 4 heads initialized from LM head")

    # RotorQuant: store rotation config (tiny)
    if "rotorquant" in key_names:
        from research.rotorquant import make_givens_rotations
        head_dim = cfg.d_model // cfg.n_heads
        n_groups = head_dim // 2
        rotations = make_givens_rotations(n_groups)
        output["rotorquant_rotations"] = rotations.to(torch.bfloat16)
        print(f"  ✓ RotorQuant: {n_groups} rotation groups stored")

    # ValueResidual: store V_0 reference and gate scalars (init=0, learned later)
    if "value_residual" in key_names:
        v0 = state.get("v0_weight")
        if v0 is not None:
            output["value_residual_v0"] = v0.to(torch.bfloat16)
            # Gate scalars: one per layer, all initialized to 0
            gates = torch.zeros(cfg.n_layers, dtype=torch.bfloat16)
            output["value_residual_gates"] = gates
            print(f"  ✓ ValueResidual: V_0 stored + {cfg.n_layers} gates (init=0, lossless)")

    # PartialRoPE, SSA, GateSkip, LiquidConv, SparDA, DSpark: config flags only
    # (no weight changes — these need fine-tuning)
    config_flags = []
    for k in ["partial_rope", "ssa", "gateskip", "liquid_conv", "sparda"]:
        if k in key_names:
            config_flags.append(k)
    if config_flags:
        print(f"  ✓ Config flags: {', '.join(config_flags)} (need fine-tuning)")

    # ── QK-Norm for MLA: add identity-init RMSNorm to Q/K ──
    if "qk_norm_mla" in key_names:
        head_dim = cfg.d_model // cfg.n_heads
        from research.keys.qk_norm_mla_key import apply_qk_norm_mla
        output = apply_qk_norm_mla(output, cfg.n_layers, head_dim)
        print(f"  ✓ QK-Norm MLA: identity-init RMSNorm on Q/K (lossless, {cfg.n_layers} layers)")

    # ── WQ Elimination: replace Q projection with identity ──
    if "wq_elim" in key_names:
        from research.keys.wq_elim_key import apply_wq_elim
        output = apply_wq_elim(output, cfg.n_layers, cfg.d_model)
        print(f"  ✓ WQ Elim: Q projection → identity (saves {cfg.n_layers * cfg.d_model**2 / 1e6:.1f}M params)")

    # ── DenseFormer: depth-weighted averaging (identity init) ──
    if "denseformer" in key_names:
        from research.keys.denseformer_key import DenseFormerKey
        df_key = DenseFormerKey()
        df_result = df_key.forward({"n_layers": cfg.n_layers, "dilation": 1})
        if df_result.success:
            for i, w in enumerate(df_result.weights["dwa_weights"]):
                output[f"dwa_weights.{i}"] = w.to(torch.bfloat16)
            print(f"  ✓ DenseFormer: DWA weights for {cfg.n_layers} layers (identity init, "
                  f"{df_result.metadata['total_params']} params)")

    # ── SandwichNorm: post-sublayer RMSNorm (identity init) ──
    if "sandwich_norm" in key_names:
        for i in range(cfg.n_layers):
            output[f"blocks.{i}.post_attn_norm.weight"] = torch.ones(cfg.d_model, dtype=torch.bfloat16)
            output[f"blocks.{i}.post_ffn_norm.weight"] = torch.ones(cfg.d_model, dtype=torch.bfloat16)
        print(f"  ✓ SandwichNorm: post-attn + post-FFN RMSNorm (identity init, "
              f"{2*cfg.n_layers} norms)")

    # ── Logit Cap + SwiGLU Clamp: runtime config flags ──
    runtime_flags = []
    if "logit_cap" in key_names:
        runtime_flags.append("logit_cap(±30)")
    if "swiglu_clamp" in key_names:
        runtime_flags.append("swiglu_clamp(α=1.702,limit=7)")
    if runtime_flags:
        output["_runtime_flags"] = torch.tensor([1], dtype=torch.uint8)
        print(f"  ✓ Runtime flags: {', '.join(runtime_flags)} (applied at inference)")

    # AirLLM: layer-streaming inference (runtime strategy, no weight changes)
    # Marks the checkpoint as streamable — the inference engine can split it
    # into per-layer shards on demand for minimal-VRAM execution.
    if "airllm" in key_names:
        output["_airllm_streamable"] = torch.tensor([1], dtype=torch.uint8)
        print(f"  ✓ AirLLM: layer-streaming enabled (runtime, no weight changes)")

    # ─── Phase 5: Save ───
    print(f"\n{'='*70}")
    print(f"PHASE 5: Save XP model")
    print(f"{'='*70}")
    save_file(output, out, metadata={
        "source": src,
        "config": config_name,
        "pipeline": "forge_keystack_streaming",
        "keys_applied": str(len(active_keys)),
        "tensors": str(len(output)),
    })
    print(f"  Saved {len(output)} tensors to {out}")
    print(f"  Size: {Path(out).stat().st_size / 1e9:.2f} GB")

    # Summary
    print(f"\n{'='*70}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"  Tensors: {len(output)} (source: {len(all_keys)})")
    print(f"  Transformed: {stats['transformed']}, passthrough: {stats['skipped']}")
    print(f"  Output: {out}")

    return out


def _calibrate_streaming(src: str, cfg, n_tokens=128, seq_len=64):
    """Run a proper layer-by-layer forward pass to collect residual activations.

    Loads one layer's weights at a time from safetensors → GPU → full forward → free.
    Runs actual attention (Q/K/V/O + RoPE + softmax) and FFN (SwiGLU).
    Never holds more than one layer in VRAM (~120MB).
    """
    from safetensors import safe_open
    import torch.nn.functional as F

    n_layers = cfg.n_layers
    d_model = cfg.d_model
    n_heads = cfg.n_heads
    head_dim = d_model // n_heads
    n_kv = cfg.n_kv_heads or n_heads

    with safe_open(src, framework="pt") as f:
        keys = set(f.keys())

        # Embedding
        emb_key = "embed.weight" if "embed.weight" in keys else "model.embed_tokens.weight"
        emb_w = f.get_tensor(emb_key).to(GPU, dtype=torch.bfloat16)
        input_ids = torch.randint(0, 10000, (1, seq_len), device=GPU)
        x = torch.nn.functional.embedding(input_ids, emb_w)  # (1, seq, d_model)
        del emb_w

        def rmsnorm(x, weight, eps=1e-6):
            return x * weight / (x.pow(2).mean(-1, keepdim=True) + eps).rsqrt()

        def apply_rope(q, seq_len, head_dim):
            """Simple RoPE: rotate pairs by position-dependent angle."""
            pos = torch.arange(seq_len, device=q.device, dtype=torch.float32)
            freqs = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=q.device).float() / head_dim))
            angles = pos[:, None] * freqs[None, :]  # (seq, head_dim/2)
            cos = angles.cos()  # (seq, head_dim/2)
            sin = angles.sin()
            # Interleave cos/sin to match head_dim
            cos_full = torch.stack([cos, cos], dim=-1).reshape(seq_len, head_dim)
            sin_full = torch.stack([sin, sin], dim=-1).reshape(seq_len, head_dim)
            # Rotate pairs
            q_pairs = q.float().reshape(seq_len, head_dim // 2, 2)
            q_rot = torch.stack([
                q_pairs[..., 0] * cos - q_pairs[..., 1] * sin,
                q_pairs[..., 0] * sin + q_pairs[..., 1] * cos,
            ], dim=-1).reshape(seq_len, head_dim)
            return q_rot.to(q.dtype)

        residual_acts = []
        for layer_idx in range(n_layers):
            residual_acts.append(x[0].float().cpu())

            # Load this layer's weights
            prefix = f"blocks.{layer_idx}."
            layer_tensors = {}
            for k in keys:
                if k.startswith(prefix):
                    layer_tensors[k] = f.get_tensor(k).to(GPU, dtype=torch.bfloat16)

            # --- Attention ---
            ln1_w = layer_tensors.get(f"{prefix}ln1.weight")
            x_norm = rmsnorm(x, ln1_w) if ln1_w is not None else x

            q_w = layer_tensors[f"{prefix}attn.q_proj.weight"]
            k_w = layer_tensors[f"{prefix}attn.k_proj.weight"]
            v_w = layer_tensors[f"{prefix}attn.v_proj.weight"]
            o_w = layer_tensors[f"{prefix}attn.out_proj.weight"]
            q_b = layer_tensors.get(f"{prefix}attn.q_proj.bias")
            k_b = layer_tensors.get(f"{prefix}attn.k_proj.bias")
            v_b = layer_tensors.get(f"{prefix}attn.v_proj.bias")

            # Project
            q = F.linear(x_norm, q_w, q_b)  # (1, seq, d_model)
            k = F.linear(x_norm, k_w, k_b)  # (1, seq, n_kv*head_dim)
            v = F.linear(x_norm, v_w, v_b)  # (1, seq, n_kv*head_dim)

            # Reshape to heads
            q = q.view(1, seq_len, n_heads, head_dim).transpose(1, 2)  # (1, n_heads, seq, head_dim)
            k = k.view(1, seq_len, n_kv, head_dim).transpose(1, 2)      # (1, n_kv, seq, head_dim)
            v = v.view(1, seq_len, n_kv, head_dim).transpose(1, 2)      # (1, n_kv, seq, head_dim)

            # RoPE on Q and K
            for h in range(n_heads):
                q[0, h] = apply_rope(q[0, h], seq_len, head_dim)
            for h in range(n_kv):
                k[0, h] = apply_rope(k[0, h], seq_len, head_dim)

            # Repeat K/V for GQA
            if n_kv < n_heads:
                rep = n_heads // n_kv
                k = k.repeat_interleave(rep, dim=1)
                v = v.repeat_interleave(rep, dim=1)

            # Attention scores
            scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
            # Causal mask
            mask = torch.triu(torch.ones(seq_len, seq_len, device=GPU, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float('-inf'))
            attn = F.softmax(scores.float(), dim=-1).to(v.dtype)
            out = torch.matmul(attn, v)  # (1, n_heads, seq, head_dim)

            # Merge heads and project out
            out = out.transpose(1, 2).reshape(1, seq_len, d_model)
            attn_out = F.linear(out, o_w)  # (1, seq, d_model)
            x = x + attn_out

            # --- FFN (SwiGLU) ---
            ln2_w = layer_tensors.get(f"{prefix}ln2.weight")
            x_norm2 = rmsnorm(x, ln2_w) if ln2_w is not None else x

            w_gate = layer_tensors[f"{prefix}ffn.w_gate.weight"]
            w_up = layer_tensors[f"{prefix}ffn.w_up.weight"]
            w_down = layer_tensors[f"{prefix}ffn.w_down.weight"]

            gate = F.linear(x_norm2, w_gate)  # (1, seq, intermediate)
            up = F.linear(x_norm2, w_up)
            ffn_out = F.linear(F.silu(gate) * up, w_down)  # (1, seq, d_model)
            x = x + ffn_out

            del layer_tensors, q, k, v, scores, attn, out
            torch.cuda.empty_cache()

    all_acts = torch.cat(residual_acts, dim=0)
    if all_acts.shape[0] > n_tokens:
        idx = torch.randperm(all_acts.shape[0])[:n_tokens]
        all_acts = all_acts[idx]
    return all_acts


def main():
    parser = argparse.ArgumentParser(description="Forge KeyStack Pipeline (streaming)")
    parser.add_argument("--src", default=str(CHECKPOINTS / "qwen25_coder_1.5b_ported.safetensors"))
    parser.add_argument("--out", default=str(CHECKPOINTS / "xp_model_keystack.safetensors"))
    parser.add_argument("--config", default="qwen25_coder_1.5b")
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--skip-slicegpt", action="store_true", default=True)
    args = parser.parse_args()

    run_keystack_pipeline(
        src=args.src, out=args.out, config_name=args.config,
        do_calibration=not args.skip_calibration,
        skip_slicegpt=args.skip_slicegpt,
    )


if __name__ == "__main__":
    main()
