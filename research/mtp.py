"""MTP — Multi-Token Prediction head for training + speculative decoding.

Trains the model to predict N future tokens simultaneously (not just next token).
At inference, this enables self-speculative decoding: predict N tokens, verify
in one forward pass → 2-3x speedup.

Architecture:
    MTPHead: shared trunk + N output heads, each predicting token at position t+k
    Loss: sum of CE losses over all N heads (curriculum: start with N=1, increase)

Training modes:
1. Standard MTP: predict tokens t+1, t+2, ..., t+N from hidden state at t
2. L-MTP (leap): predict t+1, t+3, t+5 (non-adjacent, captures longer dependencies)
3. Curriculum: gradually increase N during training (small models benefit most)

Usage:
    from research.mtp import MTPHead, MTPTrainer

    # Add MTP head to model
    head = MTPHead(d_model=1024, vocab_size=151665, n_predict=4)
    # Train
    trainer = MTPTrainer(model, head, n_predict=4, curriculum=True)
    loss = trainer.compute_loss(input_ids)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MTPHead(nn.Module):
    """Multi-token prediction head.

    Predicts N future tokens from a single hidden state.
    Uses a shared projection + N independent output heads.

    Args:
        d_model: model hidden dimension
        vocab_size: vocabulary size
        n_predict: number of future tokens to predict (default 4)
        leap: if True, predict non-adjacent tokens (L-MTP mode)
        share_embedding: if True, share weights with model's token embedding
    """

    def __init__(self, d_model, vocab_size, n_predict=4, leap=False,
                 share_embedding: Optional[nn.Embedding] = None):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_predict = n_predict
        self.leap = leap

        # Shared trunk: project hidden state to a representation for prediction.
        self.trunk = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # N independent output heads (can share with model's lm_head).
        self.heads = nn.ModuleList([
            nn.Linear(d_model, vocab_size, bias=False)
            for _ in range(n_predict)
        ])

        # Optional: share embedding weights with output heads.
        if share_embedding is not None:
            for head in self.heads:
                head.weight = share_embedding.weight  # tied weights

    def forward(self, hidden_states):
        """Predict N future tokens from hidden states.

        Args:
            hidden_states: (B, T, d_model) from the main model

        Returns:
            logits: list of N tensors, each (B, T, vocab_size)
                logits[k] predicts token at position t+k+1
        """
        trunk_out = self.trunk(hidden_states)
        return [head(trunk_out) for head in self.heads]

    def predict_tokens(self, hidden_states, temperature=0.0):
        """Generate N draft tokens from the last hidden state.

        Args:
            hidden_states: (B, T, d_model)
            temperature: 0 for greedy, >0 for sampling

        Returns:
            tokens: (B, N) predicted token ids
            logits: (B, N, vocab_size) logits for each prediction
        """
        last_hidden = hidden_states[:, -1:, :]  # (B, 1, d_model)
        all_logits = self.forward(last_hidden)  # list of (B, 1, vocab)

        tokens = []
        logits_list = []
        for k, logits in enumerate(all_logits):
            l = logits[:, -1, :]  # (B, vocab)
            if temperature == 0:
                tok = l.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(l / temperature, dim=-1)
                tok = torch.multinomial(probs, num_samples=1)
            tokens.append(tok)
            logits_list.append(l)

        tokens = torch.cat(tokens, dim=1)  # (B, N)
        logits_out = torch.stack(logits_list, dim=1)  # (B, N, vocab)
        return tokens, logits_out


class MTPTrainer:
    """Trains a model with MTP head.

    Combines standard next-token prediction (NTP) with multi-token prediction (MTP).
    Curriculum learning: gradually increase n_predict during training.

    Args:
        model: the main LLM (frozen or trainable)
        mtp_head: MTPHead instance
        n_predict: current number of tokens to predict
        curriculum: if True, ramp up n_predict from 1 to max
        curriculum_steps: steps to reach max n_predict
        mtp_weight: weight of MTP loss vs NTP loss (default 0.5)
        leap: if True, use L-MTP (leap predictions)
    """

    def __init__(self, model, mtp_head, n_predict=4,
                 curriculum=True, curriculum_steps=1000,
                 mtp_weight=0.5, leap=False):
        self.model = model
        self.mtp_head = mtp_head
        self.n_predict = n_predict
        self.curriculum = curriculum
        self.curriculum_steps = curriculum_steps
        self.mtp_weight = mtp_weight
        self.leap = leap
        self.current_step = 0
        self.current_n = 1 if curriculum else n_predict

    def _get_current_n(self):
        """Get current n_predict based on curriculum schedule."""
        if not self.curriculum:
            return self.n_predict
        # Linear ramp from 1 to n_predict over curriculum_steps.
        progress = min(1.0, self.current_step / self.curriculum_steps)
        return max(1, int(1 + progress * (self.n_predict - 1)))

    def compute_loss(self, input_ids, ignore_index=-100):
        """Compute combined NTP + MTP loss.

        Args:
            input_ids: (B, T) token ids

        Returns:
            total_loss = ntp_loss + mtp_weight * mtp_loss
        """
        self.current_step += 1
        self.current_n = self._get_current_n()

        B, T = input_ids.shape
        device = input_ids.device

        # Forward pass through main model.
        model_out = self.model(input_ids)
        hidden = model_out[0] if isinstance(model_out, tuple) else model_out

        # Standard NTP loss (next-token prediction).
        # Detect if model returns hidden states or logits:
        # - If output dim == vocab_size, it's already logits
        # - If output dim == d_model, it's hidden states (need lm_head)
        vocab_size = self.mtp_head.vocab_size
        if hidden.size(-1) == vocab_size:
            ntp_logits = hidden  # already logits
        elif hasattr(self.model, "lm_head"):
            ntp_logits = self.model.lm_head(hidden)
        else:
            ntp_logits = hidden  # assume hidden IS logits

        ntp_targets = input_ids[:, 1:].contiguous()
        ntp_loss = F.cross_entropy(
            ntp_logits[:, :-1, :].contiguous().view(-1, ntp_logits.size(-1)),
            ntp_targets.view(-1),
            ignore_index=ignore_index,
        )

        # Get hidden states for MTP head (needs d_model dim, not vocab_size).
        mtp_input = hidden
        if hidden.size(-1) == vocab_size:
            # Output was logits — project back to d_model.
            # Check for embedding layer (different models use different names).
            has_embed = (hasattr(self.model, "wte") or
                        hasattr(self.model, "embed") or
                        hasattr(self.model, "embed_tokens"))
            if has_embed:
                if not hasattr(self, "_vocab_proj"):
                    self._vocab_proj = nn.Linear(vocab_size, self.mtp_head.d_model,
                                                bias=False).to(hidden.device)
                mtp_input = self._vocab_proj(hidden)
            else:
                mtp_input = hidden  # hope for the best

        # MTP loss: predict tokens t+2, t+3, ..., t+N from hidden at t.
        mtp_logits_list = self.mtp_head(mtp_input)  # list of (B, T, vocab)
        mtp_loss = 0.0
        n_active = min(self.current_n, len(mtp_logits_list))

        for k in range(n_active):
            # Head k predicts token at position t + k + 2 (offset by 1 for NTP + k for MTP).
            offset = k + 2 if not self.leap else 2 * (k + 1)
            if offset >= T:
                break
            # MTP logits at position t predict target at t + offset.
            mtp_logits_k = mtp_logits_list[k][:, :-offset, :].contiguous()
            mtp_targets_k = input_ids[:, offset:].contiguous()

            if mtp_logits_k.shape[1] == 0 or mtp_targets_k.shape[1] == 0:
                continue

            loss_k = F.cross_entropy(
                mtp_logits_k.view(-1, mtp_logits_k.size(-1)),
                mtp_targets_k.view(-1),
                ignore_index=ignore_index,
            )
            mtp_loss = mtp_loss + loss_k

        if n_active > 0:
            mtp_loss = mtp_loss / n_active  # average over heads

        total_loss = ntp_loss + self.mtp_weight * mtp_loss

        return total_loss, {"ntp_loss": ntp_loss.item(),
                           "mtp_loss": float(mtp_loss) if isinstance(mtp_loss, float) else mtp_loss.item(),
                           "current_n": self.current_n}

    def stats(self):
        return {"step": self.current_step, "current_n": self.current_n,
                "target_n": self.n_predict}
