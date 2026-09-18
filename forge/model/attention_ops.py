"""Attention kernels (flash/SDPA wrappers, varlen, causal masks)."""
import logging
import math

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

def flash_attention(q, k, v, is_causal=True) -> torch.Tensor:
    """Use FlashAttention-2 via PyTorch's SDPA when available.

    PyTorch 2.x automatically dispatches to FlashAttention-2 (FA2) on CUDA
    when using F.scaled_dot_product_attention with is_causal=True.
    This is ~2x faster than manual attention and uses O(1) memory.

    Falls back to manual computation on CPU.
    """
    if q.is_cuda:
        # FA2 is automatically used by SDPA on CUDA in PyTorch 2.x
        return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    else:
        # Manual attention for CPU
        scale = 1.0 / math.sqrt(q.size(-1))
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        if is_causal:
            T, S = scores.size(-2), scores.size(-1)
            # Bottom-right-anchored causal mask — query row i may attend to
            # keys 0..(S-T+i), matching SDPA is_causal semantics when past
            # KV makes S > T (delta prefill).  The old tril(T,T) mask both
            # mis-anchored and failed to broadcast when S != T.
            q_pos = torch.arange(S - T, S, device=scores.device)
            k_pos = torch.arange(S, device=scores.device)
            mask = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)  # (T, S)
            scores = scores.masked_fill(~mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, v)


def varlen_attention(q, k, v, cu_seqlens, max_seqlen=None) -> torch.Tensor:
    """Variable-length attention for packed sequences (R&D round 14).

    Uses FlashAttention varlen API to attend WITHIN each packed example,
    preventing cross-example attention contamination. This is the correct
    way to handle packed sequences: instead of a global causal mask over
    the entire packed sequence, each example gets its own causal mask.

    Community: Unsloth 2.1x faster padding-free, 50% less VRAM.
    Requires flash_attn package (flash_attn_varlen_func).

    Args:
        q, k, v: (B, n_heads, T, head_dim) — standard attention tensors.
            For varlen, B should be 1 (all examples packed into one
            sequence) or we process each batch element separately.
        cu_seqlens: (B, n_examples+1) or (n_examples+1,) cumulative
            sequence lengths. If 2D, each batch element has its own
            set of examples. If 1D, all examples are in one sequence.
        max_seqlen: max sequence length (for FA varlen API). Auto-computed
            if None.

    Returns:
        (B, n_heads, T, head_dim) attention output.
    """
    # Try flash_attn varlen (fastest path, O(1) memory per example).
    try:
        from flash_attn import flash_attn_varlen_func
        # flash_attn expects (total_seq, n_heads, head_dim) without batch dim.
        # If B=1, squeeze the batch dim. If B>1, process per-batch.
        B, n_heads, T, head_dim = q.shape

        if B == 1:
            # Single packed sequence: squeeze batch, use cu_seqlens directly.
            q_vl = q.squeeze(0).transpose(0, 1)  # (T, n_heads, head_dim)
            k_vl = k.squeeze(0).transpose(0, 1)
            v_vl = v.squeeze(0).transpose(0, 1)
            if cu_seqlens.dim() > 1:
                cu = cu_seqlens[0]  # first batch element's cu_seqlens
            else:
                cu = cu_seqlens
            cu = cu.to(q.device, dtype=torch.int32)
            if max_seqlen is None:
                max_seqlen = int((cu[1:] - cu[:-1]).max().item())
            out = flash_attn_varlen_func(
                q_vl, k_vl, v_vl, cu, cu, max_seqlen, max_seqlen,
                softmax_scale=1.0 / math.sqrt(head_dim), causal=True,
            )
            # (T, n_heads, head_dim) → (1, n_heads, T, head_dim)
            return out.transpose(0, 1).unsqueeze(0)
        else:
            # B > 1: process each batch element separately and concatenate.
            outs = []
            for b in range(B):
                q_vl = q[b].transpose(0, 1)
                k_vl = k[b].transpose(0, 1)
                v_vl = v[b].transpose(0, 1)
                if cu_seqlens.dim() > 1:
                    cu = cu_seqlens[b]
                else:
                    # All batch elements share the same cu_seqlens (uncommon).
                    cu = cu_seqlens
                cu = cu.to(q.device, dtype=torch.int32)
                if max_seqlen is None:
                    max_seqlen = int((cu[1:] - cu[:-1]).max().item())
                o = flash_attn_varlen_func(
                    q_vl, k_vl, v_vl, cu, cu, max_seqlen, max_seqlen,
                    softmax_scale=1.0 / math.sqrt(head_dim), causal=True,
                )
                outs.append(o.transpose(0, 1))
            return torch.stack(outs, dim=0)
    except ImportError:
        pass

    # Fallback: block-diagonal causal mask via SDPA.
    # Build a mask where position (i, j) is valid only if i and j are in
    # the same example AND i >= j (causal). This is correct but slower
    # than flash_attn varlen (materializes the full T×T mask).
    B, n_heads, T, head_dim = q.shape
    if cu_seqlens.dim() > 1:
        # Process per-batch with block-diagonal masks.
        outs = []
        for b in range(B):
            cu = cu_seqlens[b].to(q.device, dtype=torch.long)
            mask = _build_block_diag_causal_mask(cu, T, q.device, q.dtype)
            # SDPA expects (B, n_heads, T, T) mask → (1, 1, T, T) broadcast.
            out = F.scaled_dot_product_attention(
                q[b:b+1], k[b:b+1], v[b:b+1], attn_mask=mask.unsqueeze(0).unsqueeze(0))
            outs.append(out)
        return torch.cat(outs, dim=0)
    else:
        cu = cu_seqlens.to(q.device, dtype=torch.long)
        mask = _build_block_diag_causal_mask(cu, T, q.device, q.dtype)
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask.unsqueeze(0).unsqueeze(0).expand(B, n_heads, T, T))


def _build_block_diag_causal_mask(cu_seqlens, T, device, dtype):
    """Build a block-diagonal causal mask for varlen attention fallback.

    Each block (defined by consecutive cu_seqlens entries) gets its own
    causal mask. Positions outside the block are masked to -inf.

    Args:
        cu_seqlens: (n_examples+1,) cumulative sequence lengths.
        T: total sequence length.
        device, dtype: for the mask tensor.

    Returns:
        (T, T) additive mask: 0 for valid positions, -inf for invalid.
    """
    mask = torch.full((T, T), float('-inf'), device=device, dtype=dtype)
    n_examples = len(cu_seqlens) - 1
    for i in range(n_examples):
        start = int(cu_seqlens[i])
        end = int(cu_seqlens[i + 1])
        if end > start:
            # Causal mask within this block
            block_len = end - start
            causal = torch.tril(torch.ones(block_len, block_len, device=device, dtype=dtype))
            mask[start:end, start:end] = torch.where(
                causal > 0, torch.tensor(0.0, device=device, dtype=dtype),
                torch.tensor(float('-inf'), device=device, dtype=dtype))
    return mask


def _causal_mask(seq_len: int, total_len: int, past_len: int, device: torch.device,
                 dtype: torch.dtype = None) -> torch.Tensor:
    """Create a causal mask for a query of length `seq_len` attending to `total_len` keys."""
    # For standard prefill (past_len == 0, seq_len == total_len) this is the usual upper-triangular mask.
    if dtype is None:
        dtype = torch.float32
    return torch.triu(torch.full((seq_len, total_len), float("-inf"), device=device, dtype=dtype),
                      diagonal=past_len + 1)


