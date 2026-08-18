"""Multi-Token Prediction (MTP) heads with shared weights.

Implements the MTP architecture from Nemotron 3 Super:
- Shared-weight design: a single prediction head applied recursively
- During training: predicts next K tokens at each position (0.3 loss scaling)
- During inference: recursive drafting for native speculative decoding

Unlike standard MTP with N independent heads, Nemotron shares parameters
across MTP heads. This yields a unified prediction head that can be applied
recursively at inference to generate longer drafts with stable acceptance.

Architecture per MTP step:
  h_t (hidden) + embed(token_{t+1}) → concat → project → norm → shared_head → logits_{t+2}

The shared head is the SAME as the main model output head (weight sharing),
reducing parameter count and ensuring consistent token distributions.

Integration with our keys:
- Hidden Decoding (5C): MTP heads use sequence-length scaling for test-time
  compute — the draft can expand into S streams for richer speculative decode.
- PIT: if use_pit=True, the MTP head uses PIT instead of weight tying.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MTPHead(nn.Module):
    """Single shared-weight MTP head.

    Predicts the next token given:
    - Current hidden state h_t (from main model)
    - Predicted token embedding embed(token_{t+1}) from previous step

    The shared design means this SAME module is applied recursively:
    Step 1: h_t + embed(pred_1) → pred_2
    Step 2: h_t + embed(pred_2) → pred_3
    etc.

    Args:
        d_model: model hidden dimension
        vocab_size: vocabulary size
        n_heads: number of MTP heads (shared weight, applied recursively)
        loss_weight: scaling factor for MTP loss (Nemotron uses 0.3)
        identity_init: if True, zero-init the projection so MTP starts lossless
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        n_heads: int = 2,
        loss_weight: float = 0.3,
        identity_init: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_heads = n_heads
        self.loss_weight = loss_weight

        # Combine hidden state + token embedding: 2*d → d
        self.combine = nn.Linear(2 * d_model, d_model, bias=False)
        # Norm before head
        self.norm = nn.RMSNorm(d_model)
        # Shared prediction head (will be tied to model head externally)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        if identity_init:
            # Zero-init combine so MTP output = head(norm(0 + embed)) initially
            nn.init.zeros_(self.combine.weight)

    def forward(
        self,
        hidden: torch.Tensor,
        prev_token_embeds: torch.Tensor,
        targets: torch.Tensor | None = None,
    ):
        """Single MTP step.

        Args:
            hidden: (batch, T, d_model) — hidden states from main model
            prev_token_embeds: (batch, T, d_model) — embeddings of predicted
                tokens from previous step (or ground truth during training)
            targets: (batch, T) — target token IDs for loss computation

        Returns:
            logits: (batch, T, vocab_size)
            loss: scalar MTP loss (or None if no targets)
        """
        # Combine hidden + previous token embedding
        combined = torch.cat([hidden, prev_token_embeds], dim=-1)  # (B, T, 2d)
        projected = self.combine(combined)  # (B, T, d)
        normed = self.norm(projected)  # (B, T, d)
        logits = self.head(normed)  # (B, T, vocab)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
            )

        return logits, loss

    @torch.no_grad()
    def draft(
        self,
        hidden: torch.Tensor,
        first_token: torch.Tensor,
        embed_layer: nn.Embedding,
        n_draft: int = 4,
    ):
        """Generate a draft sequence of n_draft tokens for speculative decoding.

        Recursively applies the shared head to produce a sequence of candidate
        tokens. The main model then verifies these in a single forward pass.

        Args:
            hidden: (batch, 1, d_model) — hidden state at current position
            first_token: (batch, 1) — first predicted token from main model
            embed_layer: embedding layer for looking up token embeddings
            n_draft: number of draft tokens to generate

        Returns:
            draft_tokens: (batch, n_draft) — candidate token IDs
            draft_logits: (batch, n_draft, vocab) — logits for each draft step
        """
        batch = hidden.shape[0]
        device = hidden.device

        draft_tokens = [first_token.squeeze(-1)]  # list of (batch,)
        draft_logits = []

        current_embed = embed_layer(first_token)  # (batch, 1, d)
        current_hidden = hidden  # (batch, 1, d)

        for step in range(n_draft - 1):
            logits, _ = self.forward(current_hidden, current_embed)
            next_token = logits.argmax(dim=-1)  # (batch, 1)
            draft_tokens.append(next_token.squeeze(-1))
            draft_logits.append(logits)

            # Update for next step: use predicted token embedding
            current_embed = embed_layer(next_token)
            # Hidden state stays the same (shared weight design)

        # Include logits for the first draft token too
        first_logits, _ = self.forward(hidden, embed_layer(first_token))
        draft_logits = [first_logits] + draft_logits

        draft_tokens = torch.stack(draft_tokens, dim=1)  # (batch, n_draft)
        draft_logits = torch.stack(draft_logits, dim=1)  # (batch, n_draft, vocab)

        return draft_tokens, draft_logits


class MTPModule(nn.Module):
    """Multi-head MTP module wrapping multiple shared-weight MTP heads.

    During training, computes MTP loss for all heads:
    Head 1: predicts token t+2 from h_t + embed(token_{t+1})
    Head 2: predicts token t+3 from h_t + embed(token_{t+2})
    etc.

    Total MTP loss = loss_weight * sum(head_losses) / n_heads

    During inference, uses the heads recursively for speculative drafting.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        n_heads: int = 2,
        loss_weight: float = 0.3,
        identity_init: bool = True,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.loss_weight = loss_weight
        # Single shared head (Nemotron design) applied recursively
        self.head = MTPHead(
            d_model, vocab_size, n_heads=1,
            loss_weight=loss_weight, identity_init=identity_init,
        )

    def tie_head_to_model(self, model_head_weight):
        """Tie the MTP head's output projection to the main model's head."""
        self.head.head.weight = model_head_weight

    def forward(
        self,
        hidden: torch.Tensor,
        token_embeds: torch.Tensor,
        targets: torch.Tensor | None = None,
    ):
        """Compute MTP loss for all heads during training.

        Args:
            hidden: (batch, T, d_model) — hidden states from main model
            token_embeds: (batch, T, d_model) — ground truth token embeddings
            targets: (batch, T) — target token IDs (shifted by 1 for each head)

        Returns:
            total_loss: scalar MTP loss (or None if no targets)
            all_logits: list of (batch, T, vocab) per head
        """
        if targets is None:
            return None, []

        batch, T, d = hidden.shape
        total_loss = torch.tensor(0.0, device=hidden.device, dtype=hidden.dtype)
        all_logits = []

        # For head k (0-indexed): predict token t+k+2 from h_t + embed(token_{t+k+1}).
        # Main head predicts t+1; each MTP head adds one more step of lookahead.
        for k in range(self.n_heads):
            # Head k (0-indexed) predicts token t+k+2 from h_t + embed(token_{t+k+1}).
            # targets[i] = idx[i+1] (already shifted by 1), so the target for input
            # position t is targets[t+k+1] = idx[t+k+2].
            if T - k - 2 < 1:
                continue  # not enough tokens for this head

            h_input = hidden[:, :T - k - 2, :]           # (B, T-k-2, d)
            embed_input = token_embeds[:, k + 1:T - 1, :]  # (B, T-k-2, d)
            target_k = targets[:, k + 1:T - 1, :].squeeze(-1) if targets.dim() == 3 else targets[:, k + 1:T - 1]

            if h_input.shape[1] < 1 or target_k.shape[1] < 1:
                continue

            logits, loss = self.head(h_input, embed_input, target_k)
            all_logits.append(logits)
            if loss is not None:
                total_loss = total_loss + loss

        if len(all_logits) > 0:
            total_loss = total_loss * self.loss_weight / len(all_logits)

        return total_loss, all_logits

    @torch.no_grad()
    def draft_tokens(
        self,
        hidden: torch.Tensor,
        first_token: torch.Tensor,
        embed_layer: nn.Embedding,
        n_draft: int = 4,
    ):
        """Generate draft tokens for speculative decoding.

        Args:
            hidden: (batch, 1, d_model) — last hidden state
            first_token: (batch, 1) — first predicted token from main model
            embed_layer: model embedding layer
            n_draft: number of draft tokens to generate

        Returns:
            draft_tokens: (batch, n_draft) — candidate token IDs
            draft_logits: (batch, n_draft, vocab) — logits per draft step
        """
        return self.head.draft(hidden, first_token, embed_layer, n_draft)
