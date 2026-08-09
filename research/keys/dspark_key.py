"""DSpark key — initialize DSpark speculative decoding head from MTP key.

DSpark = parallel backbone (Medusa-style) + sequential RNN refinement.
The parallel backbone can be initialized from the MTP key (which copies
the LM head). The sequential RNN module starts as identity (zero init)
so it adds no bias initially — the parallel backbone does all the work,
and the RNN learns refinements during light fine-tuning.

Key class: FULL for parallel backbone (via MTP key), PARTIAL for RNN
(zero-init, needs fine-tune to learn sequential dependencies).
"""
import torch
import torch.nn as nn

from research.keys.base import Key, KeyClass, KeyResult
from research.keys.mtp_key import MTPKey


class DSparkKey(Key):
    """DSpark weight-stealing key — MTP key for backbone + zero-init RNN.

    The parallel backbone is initialized from the LM head (via MTPKey).
    The sequential RNN module is zero-initialized (adds no bias at start).
    The confidence head is initialized to predict uniform confidence (0.5).

    This gives a working DSpark head immediately — the parallel backbone
    can predict tokens (same quality as MTP), and the RNN refinement
    improves with light fine-tuning.
    """

    @property
    def name(self) -> str:
        return "dspark"

    @property
    def description(self) -> str:
        return "DSpark: MTP key for parallel backbone + zero-init sequential RNN"

    def key_class(self) -> KeyClass:
        return KeyClass.PARTIAL  # backbone is FULL, RNN is zero-init

    def forward(self, data: dict) -> KeyResult:
        """Initialize DSpark head weights from LM head.

        Args:
            data: {"lm_head_weight": tensor, "n_predict": int,
                   "d_model": int, "vocab_size": int, "seq_rank": int}

        Returns:
            {"parallel_heads": list of tensors,  # from MTP key
             "rnn_W1": tensor (zero),            # embedding projection
             "rnn_W2": tensor (zero),            # output projection
             "rnn_gate": tensor (zero),          # GRU gate
             "confidence_weight": tensor (zero), # confidence head
             "confidence_bias": tensor (0.0)}    # sigmoid(0) = 0.5
        """
        try:
            lm_head = data["lm_head_weight"]
            n_predict = data["n_predict"]
            d_model = data["d_model"]
            vocab_size = data["vocab_size"]
            seq_rank = data.get("seq_rank", 256)

            # Parallel backbone: use MTP key
            mtp_key = MTPKey()
            mtp_result = mtp_key.forward({
                "lm_head_weight": lm_head,
                "n_predict": n_predict,
                "d_model": d_model,
                "vocab_size": vocab_size,
            })
            if not mtp_result.success:
                return KeyResult(success=False, error=f"MTP key failed: {mtp_result.error}")

            # Sequential RNN: zero-init (identity behavior — adds no bias)
            # W1: vocab_size -> seq_rank (embedding lookup, zero = no contribution)
            rnn_W1 = torch.zeros(vocab_size, seq_rank)
            # W2: seq_rank -> d_model (projection, zero = no contribution)
            rnn_W2 = torch.zeros(seq_rank, d_model)
            # GRU gate: zero (gate starts closed, RNN is passive)
            rnn_gate = torch.zeros(3 * d_model, d_model)  # update + reset + candidate

            # Confidence head: zero weight, zero bias → sigmoid(0) = 0.5 (uniform)
            conf_weight = torch.zeros(1, d_model + seq_rank)
            conf_bias = torch.tensor(0.0)

            return KeyResult(
                success=True,
                weights={
                    "parallel_heads": mtp_result.weights["mtp_heads"],
                    "shared_trunk": mtp_result.weights["shared_trunk"],
                    "rnn_W1": rnn_W1,
                    "rnn_W2": rnn_W2,
                    "rnn_gate": rnn_gate,
                    "confidence_weight": conf_weight,
                    "confidence_bias": conf_bias,
                },
                metadata={
                    "n_predict": n_predict,
                    "seq_rank": seq_rank,
                    "backbone_key": "mtp (exact for head 1)",
                    "rnn_init": "zero (identity, needs fine-tune)",
                },
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: dict) -> KeyResult:
        """Extract LM head from DSpark parallel backbone (head 1 only)."""
        try:
            heads = weights["parallel_heads"]
            mtp_key = MTPKey()
            return mtp_key.reverse({"mtp_heads": heads})
        except Exception as e:
            return KeyResult(success=False, error=str(e))


def init_dspark_from_lm_head(dspark_head, lm_head_weight):
    """Initialize a DSparkHead module from LM head weights (in-place).

    Uses the DSparkKey to steal LM head weights for the parallel backbone.
    The RNN and confidence head are zero-initialized (identity at start).

    Args:
        dspark_head: DSparkHead module
        lm_head_weight: LM head weight tensor (vocab_size, d_model)
    """
    d_model = dspark_head.d_model
    vocab_size = dspark_head.vocab_size
    n_predict = dspark_head.n_predict
    seq_rank = dspark_head.seq_rank

    key = DSparkKey()
    result = key.forward({
        "lm_head_weight": lm_head_weight,
        "n_predict": n_predict,
        "d_model": d_model,
        "vocab_size": vocab_size,
        "seq_rank": seq_rank,
    })
    if not result.success:
        raise RuntimeError(f"DSpark key failed: {result.error}")

    w = result.weights

    # Copy parallel backbone heads — each head is nn.Sequential, last layer is Linear(vocab, d_model)
    # The key outputs raw (vocab, d_model) weights that go into the LAST layer of each head
    for i, head_w in enumerate(w["parallel_heads"]):
        if i < len(dspark_head.parallel_heads):
            last_layer = dspark_head.parallel_heads[i][-1]  # nn.Linear(d_model, vocab_size)
            if last_layer.weight.shape == head_w.shape:
                last_layer.weight.data.copy_(head_w)
            else:
                # Shape mismatch — try partial copy
                min_out = min(last_layer.weight.shape[0], head_w.shape[0])
                min_in = min(last_layer.weight.shape[1], head_w.shape[1])
                last_layer.weight.data[:min_out, :min_in].copy_(head_w[:min_out, :min_in])

    # Zero-init sequential module (RNN/Markov) — identity at start
    # DSparkHead uses: W1 (nn.Embedding), W2 (nn.Linear), seq_proj (nn.Linear), conf_proj (nn.Linear)
    with torch.no_grad():
        dspark_head.W1.weight.zero_()  # nn.Embedding weight
        dspark_head.W2.weight.zero_()  # nn.Linear weight (no bias)
        if hasattr(dspark_head, 'seq_proj') and dspark_head.seq_proj is not None:
            dspark_head.seq_proj.weight.zero_()
            if dspark_head.seq_proj.bias is not None:
                dspark_head.seq_proj.bias.zero_()
        # Confidence head: zero weight + zero bias → sigmoid(0) = 0.5 (uniform)
        dspark_head.conf_proj.weight.zero_()
        if dspark_head.conf_proj.bias is not None:
            dspark_head.conf_proj.bias.zero_()

    return result


def init_dspark_from_model(model, dspark_head):
    """Find LM head in model and initialize DSpark head from it."""
    # Try common LM head attribute names
    for attr in ["lm_head", "output", "lm_heads", "head"]:
        if hasattr(model, attr):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Linear):
                return init_dspark_from_lm_head(dspark_head, mod.weight.data)
            # Some models have lm_head as a ModuleList or Sequential
            if isinstance(mod, (nn.ModuleList, nn.Sequential)) and len(mod) > 0:
                last = mod[-1] if isinstance(mod, nn.Sequential) else mod[0]
                if isinstance(last, nn.Linear):
                    return init_dspark_from_lm_head(dspark_head, last.weight.data)

    # Tied embeddings: LM head = embedding weight
    for attr in ["embed_tokens", "wte", "embed", "token_embedding"]:
        if hasattr(model, attr):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Embedding):
                return init_dspark_from_lm_head(dspark_head, mod.weight.data)

    raise ValueError("Could not find LM head in model")


if __name__ == "__main__":
    key = DSparkKey()
    print(f"Key: {key.name}, class: {key.key_class().value}")

    d_model, vocab_size, n_predict = 256, 1000, 4
    lm_head = torch.randn(vocab_size, d_model)
    r = key.forward({
        "lm_head_weight": lm_head, "n_predict": n_predict,
        "d_model": d_model, "vocab_size": vocab_size, "seq_rank": 128,
    })
    print(f"Forward: {r.success}")
    print(f"  Parallel heads: {len(r.weights['parallel_heads'])}")
    print(f"  RNN W1: {r.weights['rnn_W1'].shape} (zero={r.weights['rnn_W1'].abs().max() == 0})")
    print(f"  Confidence: {r.weights['confidence_weight'].shape} (zero={r.weights['confidence_weight'].abs().max() == 0})")

    # Verify head 1 is exact LM head copy
    err = (r.weights["parallel_heads"][0] - lm_head).abs().max().item()
    print(f"  Head 1 exact: {err < 1e-6}")
