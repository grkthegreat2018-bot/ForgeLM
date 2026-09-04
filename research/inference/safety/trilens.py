"""TriLens (R34-1) and PoP (R34-2) — hallucination detection via internal
model representations.

Both detectors operate during a *single* forward pass (no multiple sampling,
no extra generation rounds), reading intermediate signals from the residual
stream as each transformer layer fires. They are designed to be hooked into
``ConfigurableResearchLLM._forward_impl`` at the per-layer loop
(``research/model_loader.py``, ~line 2077) or wrapped around
``ForgeEngine.generate_stream`` token yields.

TriLens — arXiv 2606.01033
    At every layer, read three signals — attention output, FFN output, and
    the post-block residual — through a *logit lens* (project the hidden
    state into vocabulary space via the unembedding matrix, take softmax,
    record the entropy). This yields a 3*L-dimensional entropy trajectory
    (L = num layers) per token. The trajectory shape is a strong
    hallucination signature: truthful tokens show a characteristic
    entropy-decrease pattern as layers resolve, while hallucinated tokens
    show late-layer entropy *increases* (the model fails to commit) and
    higher trajectory variance. Single forward pass, no multiple samples.
    Strong detector.

PoP (Probing-of-Progress) — arXiv 2608.27165
    Fuse intermediate hidden representations across depth during a single
    forward pass. Instead of a logit lens, PoP tracks the L2 norm of each
    layer's hidden state and fuses them with depth-dependent weights
    (deeper layers weighted more — they carry more resolved semantics).
    75.5% AUROC on TruthfulQA, <1.2% latency overhead. Cheaper than
    TriLens (no vocab-dimension projection, just norm reduction).

VRAM budget (per token, 32-layer / 2048-dim model):
    TriLens: 3*L float32 entropy scalars = 3*32*4 = 384 bytes. Negligible.
    PoP:     L float32 norm scalars    = 32*4   = 128 bytes. Negligible.
    (If full hidden states were stored: 32*2048*4 = 256 KB/token — avoided
     by reducing to norms at record time.)

Both detectors are pure-torch with CPU fallback (no CUDA-specific ops).
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F

__all__ = ["TriLensDetector", "PoPDetector", "TriLensPoPEnsemble"]


def _entropy_of_logits(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Shannon entropy (in nats) of the softmax distribution over ``dim``.

    Operates on the last token's hidden projection; ``logits`` is expected
    to be 1-D (vocab,) or 2-D (batch, vocab) — entropy is reduced to a
    scalar per batch row.
    """
    probs = F.softmax(logits.float(), dim=dim)
    log_probs = F.log_softmax(logits.float(), dim=dim)
    ent = -(probs * log_probs).sum(dim=dim)
    return ent


class TriLensDetector:
    """TriLens hallucination detector (R34-1, arXiv 2606.01033).

    Records a 3*L-dimensional entropy trajectory per token by projecting
    each layer's attention output, FFN output, and residual through a logit
    lens (matmul with the unembedding matrix) and computing the softmax
    entropy. The trajectory's variance and late-layer entropy trend are
    mapped to a hallucination score in [0, 1].

    Lightweight: stores only 3*L float scalars per token (~384 B for 32
    layers). The logit-lens projection is computed ephemerally and
    discarded — only the resulting entropy scalar is retained.

    Usage (hooked into the per-layer forward loop)::

        detector = TriLensDetector(n_layers=32, vocab_size=151665)
        detector.set_unembedding(model.head.weight)  # (vocab, hidden)
        for i, block in enumerate(model.blocks):
            attn_out, present = block.attn(block.ln1(x), ...)
            x = x + attn_out
            ffn_out = block.ffn(block.ln2(x))
            x = x + ffn_out
            detector.record_layer(i, attn_out, ffn_out, x)
        score = detector.detect()  # 0..1, high = hallucination
        detector.reset()
    """

    def __init__(
        self,
        n_layers: int,
        vocab_size: int,
        unembedding_weight: torch.Tensor | None = None,
    ) -> None:
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self._unembedding: torch.Tensor | None = None
        # Per-layer entropy slots: 3 values (attn, ffn, residual) per layer.
        # Stored as flat list, assembled into a tensor on get_trajectory().
        self._entropy_attn: list[float] = []
        self._entropy_ffn: list[float] = []
        self._entropy_res: list[float] = []
        if unembedding_weight is not None:
            self.set_unembedding(unembedding_weight)

    def set_unembedding(self, weight: torch.Tensor) -> None:
        """Set the unembedding matrix for the logit lens.

        Args:
            weight: shape ``(vocab_size, hidden_dim)`` — the transposed
                head weight (as stored by ``nn.Linear(d_model, vocab)``).
        """
        if weight.shape[0] != self.vocab_size:
            raise ValueError(
                f"unembedding vocab dim {weight.shape[0]} != "
                f"configured vocab_size {self.vocab_size}")
        # Keep a float32 contiguous copy on CPU for stable matmul; the
        # hidden states are moved to this tensor's device at record time.
        self._unembedding = weight.detach().to(torch.float32).contiguous()

    def _logit_lens(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project a hidden state through the unembedding → vocab logits.

        ``hidden`` is (hidden_dim,) or (batch, hidden_dim). Returns logits
        of shape (vocab,) or (batch, vocab).
        """
        if self._unembedding is None:
            raise RuntimeError("TriLensDetector: call set_unembedding() first")
        h = hidden.detach().to(torch.float32)
        if h.device != self._unembedding.device:
            self._unembedding = self._unembedding.to(h.device)
        # unembedding: (vocab, hidden) → logits = h @ unembedding.T
        return F.linear(h, self._unembedding)

    def record_layer(
        self,
        layer_idx: int,
        attn_out: torch.Tensor,
        ffn_out: torch.Tensor,
        residual: torch.Tensor,
    ) -> None:
        """Record entropy for one layer's three signals.

        Each signal is projected through the logit lens and reduced to a
        scalar entropy. For batched / multi-token inputs, the *last* token
        position is used (the token currently being generated).

        Args:
            layer_idx: 0-based layer index (for validation / ordering).
            attn_out: attention sub-layer output, (hidden,) or (T, hidden)
                or (B, T, hidden).
            ffn_out: FFN sub-layer output, same shape convention.
            residual: post-block residual stream, same shape convention.
        """
        # Reduce to the last token's hidden vector: (..., hidden) → (hidden,).
        def _last_token(t: torch.Tensor) -> torch.Tensor:
            t = t.detach()
            if t.dim() == 1:
                return t
            return t.reshape(-1, t.shape[-1])[-1]

        a = _last_token(attn_out)
        f = _last_token(ffn_out)
        r = _last_token(residual)

        ent_a = _entropy_of_logits(self._logit_lens(a)).item()
        ent_f = _entropy_of_logits(self._logit_lens(f)).item()
        ent_r = _entropy_of_logits(self._logit_lens(r)).item()

        self._entropy_attn.append(ent_a)
        self._entropy_ffn.append(ent_f)
        self._entropy_res.append(ent_r)

    def get_trajectory(self) -> torch.Tensor:
        """Return the 3*L-dimensional entropy trajectory.

        Ordering: [attn_0, ffn_0, res_0, attn_1, ffn_1, res_1, ...].
        Returns a 1-D float32 tensor on CPU.
        """
        traj = []
        for i in range(len(self._entropy_attn)):
            traj.append(self._entropy_attn[i])
            traj.append(self._entropy_ffn[i])
            traj.append(self._entropy_res[i])
        return torch.tensor(traj, dtype=torch.float32)

    def detect(self) -> float:
        """Return a hallucination score in [0, 1] (high = likely hallucination).

        Two signals are combined:
          1. **Late-layer entropy increase**: truthful tokens show entropy
             *decreasing* as layers resolve toward a confident prediction.
             An increase in residual entropy from the first third to the
             last third of layers indicates the model fails to commit — a
             hallucination signature.
          2. **Trajectory variance**: high variance across the 3*L entropy
             values indicates unstable internal representations.
        """
        if not self._entropy_res:
            return 0.0
        res = torch.tensor(self._entropy_res, dtype=torch.float32)
        n = res.numel()
        if n < 2:
            return 0.0

        third = max(1, n // 3)
        early = res[:third].mean().item()
        late = res[-third:].mean().item()
        # Normalized late-layer increase: positive → entropy rose → suspect.
        # log(vocab) is the max entropy; normalize by it for scale invariance.
        max_ent = math.log(self.vocab_size)
        entropy_increase = (late - early) / max_ent  # ~[-1, 1]

        # Trajectory variance across all 3*L values, normalized by max_ent^2.
        traj = self.get_trajectory()
        traj_var = traj.var(unbiased=False).item() / (max_ent ** 2)

        # Weighted combination, squashed to [0, 1] via sigmoid.
        # entropy_increase in [-1,1] → shift to [0,2] for sigmoid centering.
        score_raw = 0.6 * entropy_increase + 0.4 * (traj_var * 2.0)
        score = torch.sigmoid(torch.tensor(score_raw * 2.5)).item()
        return float(max(0.0, min(1.0, score)))

    def reset(self) -> None:
        """Clear stored trajectory for the next token."""
        self._entropy_attn.clear()
        self._entropy_ffn.clear()
        self._entropy_res.clear()


class PoPDetector:
    """PoP (Probing-of-Progress) hallucination detector (R34-2, arXiv 2608.27165).

    Fuses intermediate hidden representations across depth during a single
    forward pass. Rather than a full logit lens, PoP records the L2 norm of
    each layer's hidden state and fuses them with depth-dependent weights
    (deeper layers carry more weight — they hold more semantically resolved
    features). The fused norm and its cross-depth variance map to a
    hallucination score.

    Lightweight: stores only L float scalars per token (~128 B for 32
    layers). The hidden state is reduced to its norm at record time and
    the full tensor is discarded.

    Usage (hooked into the per-layer forward loop)::

        pop = PoPDetector(n_layers=32, hidden_dim=2048)
        for i, block in enumerate(model.blocks):
            x, _ = block(x, ...)
            pop.record_layer(i, x)
        score = pop.detect()
        pop.reset()
    """

    def __init__(self, n_layers: int, hidden_dim: int) -> None:
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self._norms: list[float] = []
        # Depth-dependent fusion weights: linear ramp favoring deeper layers.
        # w_i = (i + 1) / sum(j+1 for j in range(n_layers)).
        ramp = torch.arange(1, n_layers + 1, dtype=torch.float32)
        self._depth_weights = ramp / ramp.sum()

    def record_layer(self, layer_idx: int, hidden_state: torch.Tensor) -> None:
        """Record the L2 norm of one layer's hidden state.

        For batched / multi-token inputs, the *last* token position is used.

        Args:
            layer_idx: 0-based layer index (for ordering validation).
            hidden_state: (hidden,) or (T, hidden) or (B, T, hidden).
        """
        h = hidden_state.detach()
        if h.dim() > 1:
            h = h.reshape(-1, h.shape[-1])[-1]
        norm = h.float().norm(p=2).item()
        self._norms.append(norm)

    def fuse(self) -> torch.Tensor:
        """Fuse per-layer norms with depth-dependent weights.

        Returns a 1-D tensor of length L containing the depth-weighted
        per-layer contributions (weighted norms). The scalar fused score
        is the sum; ``detect()`` derives the hallucination score from the
        fused representation's magnitude and cross-depth consistency.
        """
        if not self._norms:
            return torch.zeros(0, dtype=torch.float32)
        norms = torch.tensor(self._norms, dtype=torch.float32)
        w = self._depth_weights[: norms.numel()].to(norms.device)
        return norms * w

    def detect(self) -> float:
        """Return a hallucination score in [0, 1] (high = likely hallucination).

        Two signals:
          1. **Fused norm magnitude**: low fused norm → weak / diffuse
             internal representation → the model is "fabricating" without
             a strong latent signal. Mapped via inverse relationship.
          2. **Cross-depth norm variance**: high variance (erratic norm
             swings across layers) indicates unstable processing.
        """
        if len(self._norms) < 2:
            return 0.0
        fused = self.fuse()
        fused_sum = fused.sum().item()
        # Normalize fused norm by sqrt(hidden_dim) — the expected L2 norm
        # scale for a unit-variance hidden vector of this dimensionality.
        scale = math.sqrt(self.hidden_dim)
        norm_ratio = fused_sum / (scale + 1e-8)

        # Low norm_ratio → high hallucination score. Use a decreasing map:
        # score_norm = 1 / (1 + norm_ratio). norm_ratio ~0 → ~1 (hallucination),
        # norm_ratio ~1+ → ~0.5 (borderline), norm_ratio >>1 → ~0 (confident).
        score_norm = 1.0 / (1.0 + norm_ratio)

        # Cross-depth variance of raw norms (normalized by mean to get CV).
        norms = torch.tensor(self._norms, dtype=torch.float32)
        mean_norm = norms.mean().item() + 1e-8
        cv = (norms.std(unbiased=False).item()) / mean_norm  # coeff of variation

        # Combine: norm weakness (0.6) + instability (0.4), squashed to [0,1].
        score_raw = 0.6 * (score_norm - 0.5) + 0.4 * (cv * 2.0)
        score = torch.sigmoid(torch.tensor(score_raw * 2.5)).item()
        return float(max(0.0, min(1.0, score)))

    def reset(self) -> None:
        """Clear stored per-layer norms for the next token."""
        self._norms.clear()


class TriLensPoPEnsemble:
    """Ensemble of TriLens + PoP detectors.

    Combines the logit-lens entropy trajectory (TriLens) with the
    depth-fused hidden-norm signal (PoP) for a more robust hallucination
    score. TriLens is the stronger single detector (full vocab-distribution
    signal) so it receives the higher weight.

    Weights default to ``trilens=0.6, pop=0.4`` per the R34 spec. Both
    sub-detectors must be driven from the same forward pass (same token).
    """

    def __init__(
        self,
        n_layers: int,
        vocab_size: int,
        hidden_dim: int,
        unembedding_weight: torch.Tensor | None = None,
        weight_trilens: float = 0.6,
        weight_pop: float = 0.4,
    ) -> None:
        self.trilens = TriLensDetector(n_layers, vocab_size, unembedding_weight)
        self.pop = PoPDetector(n_layers, hidden_dim)
        self.weight_trilens = weight_trilens
        self.weight_pop = weight_pop
        total = weight_trilens + weight_pop
        if total <= 0:
            raise ValueError("weights must be positive")
        self.weight_trilens /= total
        self.weight_pop /= total

    def set_unembedding(self, weight: torch.Tensor) -> None:
        """Set the unembedding matrix on the TriLens sub-detector."""
        self.trilens.set_unembedding(weight)

    def record_layer_trilens(
        self,
        layer_idx: int,
        attn_out: torch.Tensor,
        ffn_out: torch.Tensor,
        residual: torch.Tensor,
    ) -> None:
        """Record TriLens signals for one layer."""
        self.trilens.record_layer(layer_idx, attn_out, ffn_out, residual)

    def record_layer_pop(
        self, layer_idx: int, hidden_state: torch.Tensor
    ) -> None:
        """Record PoP signal for one layer (the post-block residual)."""
        self.pop.record_layer(layer_idx, hidden_state)

    def record_layer(
        self,
        layer_idx: int,
        attn_out: torch.Tensor,
        ffn_out: torch.Tensor,
        residual: torch.Tensor,
    ) -> None:
        """Convenience: record both detectors' signals for one layer.

        ``residual`` is used as PoP's hidden state (the post-block residual
        stream is the canonical "hidden representation" at each depth).
        """
        self.trilens.record_layer(layer_idx, attn_out, ffn_out, residual)
        self.pop.record_layer(layer_idx, residual)

    def detect(self) -> float:
        """Return the weighted-average hallucination score in [0, 1]."""
        s_trilens = self.trilens.detect()
        s_pop = self.pop.detect()
        return float(
            self.weight_trilens * s_trilens + self.weight_pop * s_pop
        )

    def reset(self) -> None:
        """Clear both sub-detectors for the next token."""
        self.trilens.reset()
        self.pop.reset()
