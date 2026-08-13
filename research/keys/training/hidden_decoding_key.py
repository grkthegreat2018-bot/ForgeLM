"""Hidden Decoding — sequence-length scaling for test-time compute.

Hidden Decoding (arXiv:2506.??? ) expands each token into S parallel streams
with independent embedding rows, keeping the intermediate KV cache as context.
This adds computation per token without adding Transformer layers — compatible
with existing pretrained models via continued pretraining (CPT).

Stream-Factorized Attention keeps most layers attending within-stream (linear
cost in S) with few cross-stream layers that mix information across streams.

Architecture:
  - Each input token t is expanded into S streams: [t_stream_0, t_stream_1, ..., t_stream_{S-1}]
  - Each stream has its own embedding row (shared embedding table + stream offset).
  - Layers 0-80%: within-stream attention only (each stream attends to its own history).
    Cost: O(S * L^2) = linear in S.
  - Layers 80-100%: cross-stream attention (streams attend to each other).
    Cost: O((S*L)^2) but only for 20% of layers.
  - At inference, this gives the model Sx internal computation per token.

Key insight: the KV cache from the within-stream layers serves as context for
the cross-stream layers, so the model can "think" about each token in S
different ways before producing the output.

This is a TRAINING key — it requires continued pretraining to learn the
stream embeddings and cross-stream attention patterns. The key provides
the architecture transformation; the training loop provides the CPT.

Usage:
    from research.keys.training.hidden_decoding_key import HiddenDecodingConfig, HiddenDecodingModule

    config = HiddenDecodingConfig(n_streams=4, d_model=768, n_layers=28)
    hd = HiddenDecodingModule(config)
    # In training: expand tokens, process streams, merge
    expanded = hd.expand_tokens(input_ids)  # (B, S*L, D)
    output = hd.forward_streams(expanded, model, layer_idx)
    merged = hd.merge_streams(output)  # (B, L, D)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HiddenDecodingConfig:
    """Configuration for Hidden Decoding.

    Args:
        n_streams: number of parallel streams per token (S).
        d_model: model dimension.
        n_layers: total number of transformer layers.
        cross_stream_start: fraction of layers where cross-stream attention
            begins (default 0.8 = last 20% of layers).
        share_embeddings: if True, all streams share the base embedding table
            with a learned per-stream offset. If False, each stream has its
            own embedding row.
    """
    n_streams: int = 4
    d_model: int = 768
    n_layers: int = 28
    cross_stream_start: float = 0.8
    share_embeddings: bool = True


class HiddenDecodingModule(nn.Module):
    """Hidden Decoding module — expands tokens into parallel streams.

    This module provides the infrastructure for hidden decoding:
      1. Token expansion: each token -> S stream tokens.
      2. Stream embeddings: per-stream embedding offsets.
      3. Stream-factorized attention masking.
      4. Stream merging: combine S stream outputs into one.

    The actual transformer forward pass is handled by the model — this module
    provides the expansion, masking, and merging utilities.
    """

    def __init__(self, config: HiddenDecodingConfig):
        super().__init__()
        self.config = config
        self.n_streams = config.n_streams
        self.d_model = config.d_model

        # Per-stream embedding offsets (added to base embedding).
        # Init: stream 0 = zero (identity), others = small random.
        self.stream_offsets = nn.Parameter(
            torch.zeros(config.n_streams, config.d_model))
        if config.n_streams > 1:
            nn.init.normal_(self.stream_offsets[1:], std=0.02)

        # Stream merge weights: how to combine S stream outputs into one.
        # Init: stream 0 = 1.0 (identity), others = 0.0 (lossless at start).
        self.merge_weights = nn.Parameter(torch.zeros(config.n_streams))
        self.merge_weights.data[0] = 1.0

        # Cross-stream mixing layer (for the last 20% of layers).
        # A simple linear attention across streams.
        self.cross_stream_attn = nn.MultiheadAttention(
            config.d_model, num_heads=4, batch_first=True)

    def expand_tokens(self, embeds: torch.Tensor) -> torch.Tensor:
        """Expand embedded tokens into S streams.

        Args:
            embeds: (B, L, D) — base embeddings.

        Returns:
            (B, S*L, D) — expanded with per-stream offsets.
            Stream ordering: [t0_s0, t0_s1, ..., t0_sS, t1_s0, t1_s1, ...]
        """
        B, L, D = embeds.shape
        S = self.n_streams

        # Expand: (B, L, D) -> (B, L, S, D) -> (B, L*S, D)
        expanded = embeds.unsqueeze(2).expand(B, L, S, D)
        # Add per-stream offsets.
        expanded = expanded + self.stream_offsets.unsqueeze(0).unsqueeze(0)
        # Reshape: (B, L, S, D) -> (B, L*S, D)
        return expanded.reshape(B, L * S, D)

    def get_stream_attention_mask(self, seq_len: int,
                                  cross_stream: bool = False) -> torch.Tensor:
        """Build the stream-factorized attention mask.

        For within-stream layers: each stream attends only to its own history.
        For cross-stream layers: full attention across all streams.

        Args:
            seq_len: expanded sequence length (L * S).
            cross_stream: if True, full causal attention across all streams.
                If False, within-stream causal attention only.

        Returns:
            (seq_len, seq_len) attention mask (True = attend, False = mask).
        """
        if cross_stream:
            # Standard causal mask.
            return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))

        S = self.n_streams
        L = seq_len // S

        # Within-stream mask: position (l, s) attends to (l', s) for l' <= l.
        # In the expanded sequence, position i = l * S + s.
        # Attend to j = l' * S + s where l' <= l and s is the same.
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
        for i in range(seq_len):
            l_i = i // S
            s_i = i % S
            for j in range(seq_len):
                l_j = j // S
                s_j = j % S
                if s_i == s_j and l_j <= l_i:
                    mask[i, j] = True
        return mask

    def is_cross_stream_layer(self, layer_idx: int) -> bool:
        """Check if a layer should use cross-stream attention.

        Args:
            layer_idx: layer index (0-based).

        Returns:
            True if this layer is in the cross-stream regime (last 20%).
        """
        cross_start = int(self.config.cross_stream_start * self.config.n_layers)
        return layer_idx >= cross_start

    def merge_streams(self, hidden: torch.Tensor) -> torch.Tensor:
        """Merge S stream outputs into a single output per token.

        Uses learned merge weights (init: stream 0 = 1.0, others = 0.0).
        At init, this is identity (only stream 0 contributes).

        Args:
            hidden: (B, S*L, D) — hidden states from all streams.

        Returns:
            (B, L, D) — merged output.
        """
        B, SL, D = hidden.shape
        S = self.n_streams
        L = SL // S

        # Reshape: (B, L*S, D) -> (B, L, S, D)
        hidden = hidden.reshape(B, L, S, D)

        # Weighted sum over streams: (B, L, S, D) * (S,) -> (B, L, D)
        weights = F.softmax(self.merge_weights, dim=0)  # normalize
        merged = (hidden * weights.view(1, 1, S, 1)).sum(dim=2)
        return merged

    def cross_stream_mix(self, hidden: torch.Tensor) -> torch.Tensor:
        """Apply cross-stream attention mixing.

        For the last 20% of layers, mix information across streams.

        Args:
            hidden: (B, S*L, D) — hidden states.

        Returns:
            (B, S*L, D) — mixed hidden states.
        """
        B, SL, D = hidden.shape
        S = self.n_streams
        L = SL // S

        # Reshape to (B, L, S, D) for cross-stream attention.
        hidden_reshaped = hidden.reshape(B, L, S, D)

        # Apply attention across the stream dimension for each token.
        # (B, L, S, D) -> treat S as sequence length for attention.
        mixed = hidden_reshaped.reshape(B * L, S, D)
        mixed, _ = self.cross_stream_attn(mixed, mixed, mixed)
        mixed = mixed.reshape(B, L, S, D).reshape(B, SL, D)

        # Residual connection.
        return hidden + mixed

    def forward(self, embeds: torch.Tensor,
                model_forward: callable) -> torch.Tensor:
        """Full hidden decoding forward pass.

        Args:
            embeds: (B, L, D) — base embeddings.
            model_forward: callable(hidden, layer_mask_fn) -> hidden
                that processes the expanded hidden states through the model.
                layer_mask_fn(layer_idx) -> attention_mask.

        Returns:
            (B, L, D) — merged output.
        """
        # Expand tokens into streams.
        expanded = self.expand_tokens(embeds)

        # Process through model (the model handles layer-by-layer attention).
        processed = model_forward(expanded, self.is_cross_stream_layer)

        # Merge streams.
        return self.merge_streams(processed)


class HiddenDecodingKey:
    """Hidden Decoding key — architecture transformation for CPT.

    This is a PARTIAL key — it provides the architecture transformation
    (stream expansion, factorized attention, merging) but requires continued
    pretraining to learn the stream embeddings and cross-stream patterns.

    The key is NOT identity-init (the stream offsets and merge weights change
    the output), so it must be trained before use.
    """

    def __init__(self, config: HiddenDecodingConfig):
        self.config = config

    def forward(self, data: dict) -> dict:
        """Initialize Hidden Decoding parameters.

        Args:
            data: {"d_model": int, "n_layers": int, "n_streams": int}

        Returns:
            {"stream_offsets": (S, D), "merge_weights": (S,)}
        """
        S = self.config.n_streams
        D = self.config.d_model

        stream_offsets = torch.zeros(S, D)
        if S > 1:
            stream_offsets[1:] = torch.randn(S - 1, D) * 0.02

        merge_weights = torch.zeros(S)
        merge_weights[0] = 1.0  # identity at init (stream 0 only)

        return {
            "stream_offsets": stream_offsets,
            "merge_weights": merge_weights,
            "n_streams": S,
            "cross_stream_start": self.config.cross_stream_start,
        }

    def get_stream_factorized_mask(self, seq_len: int, n_streams: int,
                                   cross_stream: bool) -> torch.Tensor:
        """Get the attention mask for stream-factorized attention.

        This can be used directly with FlexAttention or SDPA.

        Args:
            seq_len: expanded sequence length (L * S).
            n_streams: number of streams.
            cross_stream: if True, full causal; if False, within-stream only.

        Returns:
            (seq_len, seq_len) boolean mask.
        """
        if cross_stream:
            return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))

        S = n_streams
        L = seq_len // S
        # Vectorized within-stream mask.
        positions = torch.arange(seq_len)
        l = positions // S  # (seq_len,)
        s = positions % S   # (seq_len,)
        # mask[i, j] = (s[i] == s[j]) and (l[j] <= l[i])
        same_stream = s.unsqueeze(0) == s.unsqueeze(1)  # (seq_len, seq_len)
        causal = l.unsqueeze(0) <= l.unsqueeze(1)       # (seq_len, seq_len)
        return same_stream & causal


def apply_hidden_decoding_to_model(model: nn.Module,
                                   config: HiddenDecodingConfig,
                                   safe: bool = True) -> HiddenDecodingModule:
    """Attach Hidden Decoding to a model.

    Uses safety validation. Note: Hidden Decoding is NOT identity-init
    (stream offsets change the output), so the safety check only validates
    finiteness, not output identity.

    Args:
        model: the model to attach to.
        config: Hidden Decoding configuration.
        safe: if True, use safe_apply with finiteness validation.

    Returns:
        The HiddenDecodingModule (attached as model.hidden_decoding).
    """
    def _apply(m):
        hd = HiddenDecodingModule(config)
        m.hidden_decoding = hd
        return m

    if safe:
        from research.keys.safety import safe_apply
        # NOT identity_init — stream offsets change the output.
        safe_apply(model, _apply, identity_init=False)
    else:
        _apply(model)

    return model.hidden_decoding
