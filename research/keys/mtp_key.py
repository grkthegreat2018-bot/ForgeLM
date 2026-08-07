"""MTP weight-stealing key — initialize MTP heads from LM head.

MTP head k predicts token at position t+k from hidden state at t.
LM head predicts token at t+1 from hidden at t. So:
  - Head 1 = LM head (exact, same task)
  - Head 2+ ≈ LM head (approximate, Markov assumption)
"""
import torch
import torch.nn as nn
from typing import Dict
from .base import Key, KeyClass, KeyResult


class MTPKey(Key):
    """MTP weight-stealing key — initialize MTP heads from LM head.

    Copies the LM head weights to all MTP prediction heads. Head 1 is exact
    (same task as LM head). Heads 2+ are approximate (Markov assumption).

    Key class: FULL for head 1, PARTIAL for heads 2+.
    """

    @property
    def name(self) -> str:
        return "mtp"

    def key_class(self) -> KeyClass:
        # Head 1 is an exact copy of LM head (same task: predict t+1 from h_t).
        return KeyClass.FULL

    def forward(self, data: Dict[str, torch.Tensor]) -> KeyResult:
        """data -> weights. Steals LM head weights for all MTP heads.

        Expected data: {"lm_head_weight": (vocab, d_model), "n_predict": int,
                        "d_model": int, "vocab_size": int}
        Returns: {"mtp_heads": [tensor]*n_predict, "shared_trunk": tensor}
        """
        lm_w = data["lm_head_weight"]
        n = int(data["n_predict"])
        d = int(data["d_model"])

        # All heads get LM head weights (head 1 exact, 2+ approximate).
        mtp_heads = [lm_w.clone() for _ in range(n)]

        # Shared trunk: identity-initialized linear (pass-through).
        trunk_w = torch.eye(d)
        shared_trunk = trunk_w

        return KeyResult(success=True, weights={
            "mtp_heads": mtp_heads, "shared_trunk": shared_trunk,
        })

    def reverse(self, weights: Dict[str, torch.Tensor]) -> KeyResult:
        """weights -> data. Extract LM head from MTP head 1 (exact)."""
        heads = weights["mtp_heads"]
        if not heads:
            return KeyResult(success=False, error="No MTP heads provided")
        return KeyResult(success=True, data={
            "lm_head_weight": heads[0].clone(),
        })


def init_mtp_from_lm_head(mtp_head, lm_head_weight):
    """Initialize an MTPHead module from LM head weights (in-place).

    Copies lm_head_weight into each output projection. Head 1 is exact;
    heads 2+ are approximate and will need fine-tuning.
    """
    with torch.no_grad():
        for head in mtp_head.heads:
            head.weight.copy_(lm_head_weight)


def init_mtp_from_model(model, mtp_head):
    """Find the LM head in a model and use it to initialize the MTP head.

    Checks common attribute names: lm_head, output, embed_tokens (tied).
    """
    lm_w = None
    for attr in ("lm_head", "output", "lm_heads"):
        mod = getattr(model, attr, None)
        if isinstance(mod, nn.Linear):
            lm_w = mod.weight
            break
    if lm_w is None:
        # Tied weights: output projection shares embedding weights.
        for attr in ("embed_tokens", "wte", "embed"):
            mod = getattr(model, attr, None)
            if isinstance(mod, nn.Embedding):
                lm_w = mod.weight
                break
    if lm_w is None:
        raise ValueError("Could not find LM head in model "
                         "(checked: lm_head, output, embed_tokens, wte, embed)")
    init_mtp_from_lm_head(mtp_head, lm_w)
