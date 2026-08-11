"""Matryoshka Representation Learning (MRL) key — nested embedding slicing.

MRL (NeurIPS 2022) trains embeddings so that the first m dimensions form
a useful low-dimensional representation. This enables adaptive deployment:
use fewer dimensions for fast retrieval, more for accuracy.

The key insight: existing embeddings can be MADE matryoshka by reordering
dimensions so that the most important (highest variance) come first.
This is a PCA-like reordering — no training needed for the basic version.

Key class: PARTIAL — needs calibration data to compute dimension importance,
or TRIVIAL if using weight-norm-based heuristic.

Reference: MRL, arxiv 2205.13147
"""
import torch

from research.keys.misc.base import Key, KeyClass, KeyResult


class MRLKey(Key):
    """Matryoshka Representation Learning key — dimension reordering.

    Reorders embedding dimensions by importance (variance/weight norm)
    so that truncation preserves maximum information.

    Key class: PARTIAL (calibration) or TRIVIAL (weight-norm heuristic).
    """

    @property
    def name(self) -> str:
        return "mrl"

    @property
    def description(self) -> str:
        return "Matryoshka representation (importance-ordered embedding dims)"

    def key_class(self) -> KeyClass:
        return KeyClass.FULL

    def forward(self, data: dict) -> KeyResult:
        """Compute dimension reordering for matryoshka embeddings.

        Args:
            data: {"embedding_weight": tensor (vocab, d_model),
                   "activations": tensor (optional, n_tokens, d_model),
                   "n_dims": list of int (granularities, e.g. [32, 64, 128, 256]),
                   "method": "variance" or "weight_norm"}

        Returns:
            {"reorder_indices": tensor (d_model,) — new order,
             "granularities": list of int}
        """
        try:
            emb = data["embedding_weight"]
            n_dims = data.get("n_dims", [emb.shape[1] // 4, emb.shape[1] // 2, emb.shape[1]])
            method = data.get("method", "weight_norm")

            if method == "variance" and "activations" in data:
                # Use activation variance per dimension
                importance = data["activations"].var(dim=0)
            else:
                # Use embedding weight norm per dimension
                importance = emb.norm(dim=0)

            # Sort dimensions by importance (descending)
            reorder_indices = importance.argsort(descending=True)

            return KeyResult(
                success=True,
                weights={"reorder_indices": reorder_indices},
                metadata={
                    "n_dims": n_dims, "method": method,
                    "d_model": emb.shape[1],
                },
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: dict) -> KeyResult:
        """Reverse the reordering."""
        try:
            idx = weights["reorder_indices"]
            inverse = torch.argsort(idx)
            return KeyResult(success=True, data={"inverse_indices": inverse})
        except Exception as e:
            return KeyResult(success=False, error=str(e))


def apply_mrl_to_model(model, n_dims=None, method="weight_norm"):
    """Apply MRL dimension reordering to a model's embeddings.

    Reorders embedding dimensions by importance (weight norm) so truncation
    preserves maximum information. Also reorders all weight matrices that
    operate on d_model (input side).

    Args:
        model: ConfigurableResearchLLM (must have .embed, .head, .blocks)
        n_dims: list of granularity sizes
        method: "weight_norm" or "variance"

    Returns:
        Reorder indices tensor.
    """
    # Find embedding — try common attribute names
    emb = getattr(model, 'embed', None) or getattr(model, 'embedding', None)
    if emb is None:
        raise ValueError("Could not find embedding layer (tried .embed, .embedding)")
    emb_weight = emb.weight.data
    d_model = emb_weight.shape[1]
    if n_dims is None:
        n_dims = [d_model // 4, d_model // 2, d_model]

    key = MRLKey()
    result = key.forward({
        "embedding_weight": emb_weight,
        "n_dims": n_dims,
        "method": method,
    })

    if not result.success:
        raise RuntimeError(f"MRL key failed: {result.error}")

    reorder = result.weights["reorder_indices"].to(emb_weight.device)

    # Reorder embedding dimensions (columns = d_model)
    emb.weight.data = emb_weight[:, reorder]

    # Reorder LM head if it takes d_model as input (columns = d_model)
    head = getattr(model, 'head', None) or getattr(model, 'lm_head', None)
    if head is not None and head.weight.data.shape[1] == d_model:
        head.weight.data = head.weight.data[:, reorder]

    # Reorder all weight matrices whose INPUT dim is d_model (columns)
    # These are in block.attn (q/k/v projections) and block.ffn (gate/up)
    for block in model.blocks:
        # Attention projections
        attn = getattr(block, 'attn', block)
        for name in ['q_proj', 'k_proj', 'v_proj']:
            proj = getattr(attn, name, None)
            if proj is not None and proj.weight.data.shape[1] == d_model:
                proj.weight.data = proj.weight.data[:, reorder]
        # FFN projections (gate/up take d_model as input)
        ffn = getattr(block, 'ffn', None)
        if ffn is not None:
            for name in ['w_gate', 'gate_proj', 'w_up', 'up_proj']:
                proj = getattr(ffn, name, None)
                if proj is not None and proj.weight.data.shape[1] == d_model:
                    proj.weight.data = proj.weight.data[:, reorder]

    # Also reorder the final norm
    ln_f = getattr(model, 'ln_f', None)
    if ln_f is not None and hasattr(ln_f, 'weight') and ln_f.weight.data.shape[0] == d_model:
        ln_f.weight.data = ln_f.weight.data[reorder]

    # Reorder per-block norms (ln1, ln2 — their weight is over d_model)
    for block in model.blocks:
        for name in ['ln1', 'ln2', 'input_layernorm', 'post_attention_layernorm']:
            ln = getattr(block, name, None)
            if ln is not None and hasattr(ln, 'weight') and ln.weight.data.shape[0] == d_model:
                ln.weight.data = ln.weight.data[reorder]

    model.mrl_granularities = n_dims
    return reorder


if __name__ == "__main__":
    key = MRLKey()
    print(f"Key: {key.name}, class: {key.key_class().value}")

    emb = torch.randn(1000, 256)
    r = key.forward({"embedding_weight": emb, "n_dims": [32, 64, 128, 256]})
    print(f"Forward: {r.success}")
    print(f"  Reorder: {r.weights['reorder_indices'][:10].tolist()}")
    print(f"  Granularities: {r.metadata['n_dims']}")

    # Verify reverse
    rv = key.reverse(r.weights)
    assert rv.success
    # Reordering then inverse should give identity
    idx = r.weights["reorder_indices"]
    inv = rv.data["inverse_indices"]
    assert (idx[inv] == torch.arange(256)).all()
    print("  Round-trip verified ✓")
