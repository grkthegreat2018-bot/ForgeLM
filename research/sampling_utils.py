"""Optimized sampling utilities for token generation.

Provides top-p (nucleus) sampling that avoids sorting the full vocabulary
(151K tokens for Qwen2.5). Instead, uses torch.topk to select the top
candidates first, then sorts only those — reducing sort complexity from
O(V log V) to O(k log k) where k << V (typically k=100 vs V=151665).
"""
import torch
import torch.nn.functional as F

# Default number of candidates to consider for top-p before sorting.
# 100 is sufficient for top_p=0.9-0.95 in practice (the nucleus rarely
# contains more than ~50 tokens for a well-trained model).
_TOP_P_CANDIDATES = 100


def top_p_filter_logits(logits: torch.Tensor, top_p: float,
                        candidates: int = _TOP_P_CANDIDATES) -> torch.Tensor:
    """Filter logits via nucleus sampling, returning logits with -inf for removed tokens.

    Optimized: uses torch.topk to select top candidates first, then sorts
    only those k candidates instead of the full vocabulary.

    Args:
        logits: (..., vocab_size) unnormalized logits
        top_p: cumulative probability threshold (0-1)
        candidates: number of top candidates to consider (default 100)

    Returns:
        logits with removed tokens set to -inf
    """
    if top_p >= 1.0:
        return logits

    # Step 1: Get top-k candidates (avoids full vocab sort).
    k = min(candidates, logits.shape[-1])
    topk_logits, topk_idx = torch.topk(logits, k, dim=-1)
    topk_probs = F.softmax(topk_logits, dim=-1)

    # Check if top-k covers top_p. If not, fall back to full sort.
    if topk_probs.sum() < top_p:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum_probs > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        indices = remove.scatter(1, sorted_idx, remove)
        return logits.masked_fill(indices, float("-inf"))

    # Step 2: Sort only the k candidates (O(k log k) not O(V log V)).
    sorted_probs, sorted_idx = torch.sort(topk_probs, descending=True, dim=-1)
    cumsum = torch.cumsum(sorted_probs, dim=-1)

    # Step 3: Determine cutoff — keep tokens until cumsum exceeds top_p.
    remove = cumsum > top_p
    # Shift right: keep the token that crosses the threshold.
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False

    # Step 4: Map removal back to topk indices, then to original vocab indices.
    sorted_remove = torch.zeros_like(sorted_probs)
    sorted_remove.scatter_(-1, sorted_idx, remove.float())
    # Map back to topk space
    topk_remove = torch.zeros_like(topk_probs)
    topk_remove.scatter_(-1, sorted_idx, sorted_remove)
    # Set removed topk logits to -inf
    topk_logits = topk_logits.masked_fill(topk_remove.bool(), float("-inf"))

    # Step 5: Scatter back to full vocab size.
    out = torch.full_like(logits, float("-inf"))
    out.scatter_(-1, topk_idx, topk_logits)
    return out


def top_p_sample_probs(probs: torch.Tensor, top_p: float,
                       candidates: int = _TOP_P_CANDIDATES) -> torch.Tensor:
    """Filter probabilities via nucleus sampling.

    Similar to top_p_filter_logits but operates on already-computed probabilities.
    Returns a probability distribution with removed tokens set to 0, renormalized.

    Uses topk optimization: if the top-k candidates cover top_p cumulative mass,
    sorts only those k instead of the full vocabulary. Falls back to full sort
    if the nucleus is larger than k (flat distribution edge case).

    Args:
        probs: (vocab,) or (..., vocab) probability distribution
        top_p: cumulative probability threshold (0-1)
        candidates: number of top candidates to consider (default 100)

    Returns:
        Renormalized probability distribution with removed tokens set to 0
    """
    if top_p >= 1.0:
        return probs

    # Step 1: Get top-k candidates.
    k = min(candidates, probs.shape[-1])
    topk_probs, topk_idx = torch.topk(probs, k, dim=-1)

    # Check if top-k covers top_p. If not, fall back to full sort.
    if topk_probs.sum() < top_p:
        # Nucleus is larger than k — fall back to full sort (rare for trained models).
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        cutoff = cumsum > top_p
        cutoff[..., 0] = False  # keep at least one token
        sorted_probs[cutoff] = 0.0
        full_probs = torch.zeros_like(probs)
        full_probs.scatter_(-1, sorted_idx, sorted_probs)
        return full_probs / full_probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)

    # Step 2: Sort only the k candidates.
    sorted_probs, sorted_idx = torch.sort(topk_probs, descending=True, dim=-1)
    cumsum = torch.cumsum(sorted_probs, dim=-1)

    # Step 3: Cutoff.
    cutoff = cumsum > top_p
    cutoff[..., 0] = False  # keep at least one token
    sorted_probs[cutoff] = 0.0

    # Step 4: Map back to topk space, then to full vocab.
    topk_filtered = torch.zeros_like(topk_probs)
    topk_filtered.scatter_(-1, sorted_idx, sorted_probs)

    full_probs = torch.zeros_like(probs)
    full_probs.scatter_(-1, topk_idx, topk_filtered)

    # Renormalize.
    return full_probs / full_probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)
