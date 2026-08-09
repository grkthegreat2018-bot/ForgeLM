"""Decoding strategy backends.

Pluggable decoding strategies selectable at runtime:
  - StandardDecoding: autoregressive token-by-token (baseline)
  - SpeculativeDecoding: draft model + verify (wraps speculative_decode.py)
  - MedusaDecoding: parallel prediction heads (wraps medusa.py)
  - DSparkDecoding: semi-autoregressive + confidence scheduling (wraps dspark.py)
  - MTPSelfSpecDecoding: use MTP heads from checkpoint for self-speculative decoding

All implement the DecodingStrategy interface:
  generate(model, input_ids, max_new_tokens, temperature, top_p) -> output_ids
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch
import torch.nn.functional as F


class DecodingStrategy(ABC):
    """Base interface for decoding strategies."""

    @abstractmethod
    def generate(self, model, input_ids: torch.Tensor,
                 max_new_tokens: int = 100,
                 temperature: float = 0.0,
                 top_p: float = 1.0) -> torch.Tensor:
        """Generate tokens. Returns full sequence [1, prompt_len + gen_len]."""
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__


class StandardDecoding(DecodingStrategy):
    """Standard autoregressive decoding with KV cache."""

    def generate(self, model, input_ids, max_new_tokens=100,
                 temperature=0.0, top_p=1.0):
        ids = input_ids.clone()
        device = input_ids.device
        # EOS detection: check model attr, config, then Qwen defaults
        eos = getattr(model, "eos_token_id", None)
        if eos is None:
            cfg = getattr(model, "config", None)
            eos = getattr(cfg, "eos_token_id", None) if cfg else None
        # Qwen2.5 EOS tokens: <|endoftext|>=151643, <|im_end|>=151645
        eos_set = {151643, 151645}
        if eos is not None:
            eos_set.add(eos)
        eos_tensor = torch.tensor(list(eos_set), device=device)
        # Pinned memory for async D2H (reduces CPU sync spikes).
        token_pinned = torch.zeros(1, dtype=torch.long, pin_memory=True)

        # Prefill — model returns (logits, loss, presents) when use_cache=True
        with torch.inference_mode():
            out = model(ids, use_cache=True)
            if isinstance(out, tuple):
                logits = out[0]
                # KV cache is at index 2 (after loss at index 1), or index 1 if no loss
                past_kv = out[2] if len(out) > 2 else out[1]
            else:
                logits = out
                past_kv = None

        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_p < 1.0:
                next_logits = self._top_p(next_logits, top_p)
            if temperature == 0:
                next_token = next_logits.argmax(-1, keepdim=True)
            else:
                next_token = torch.multinomial(
                    F.softmax(next_logits, dim=-1), num_samples=1)

            # GPU-side EOS check: single sync only if token matches EOS.
            is_eos = (next_token == eos_tensor).any()
            token_pinned.copy_(next_token, non_blocking=True)
            if is_eos.item():
                break

            ids = torch.cat([ids, next_token], dim=-1)
            with torch.inference_mode():
                out = model(next_token, past_key_values=past_kv, use_cache=True)
                if isinstance(out, tuple):
                    logits = out[0]
                    past_kv = out[2] if len(out) > 2 else out[1]
                else:
                    logits = out

        return ids

    def _top_p(self, logits, top_p):
        from research.sampling_utils import top_p_filter_logits
        return top_p_filter_logits(logits, top_p)


class SpeculativeDecoding(DecodingStrategy):
    """Speculative decoding with separate draft model."""

    def __init__(self, draft_model, k=4):
        self.draft_model = draft_model
        self.k = k

    def generate(self, model, input_ids, max_new_tokens=100,
                 temperature=0.0, top_p=1.0):
        from research.speculative_decode import speculative_generate
        return speculative_generate(
            model, self.draft_model, input_ids,
            max_new_tokens=max_new_tokens, k=self.k,
            temperature=temperature, device=str(input_ids.device),
        )


class MedusaDecoding(DecodingStrategy):
    """Medusa parallel prediction heads."""

    def __init__(self, medusa_heads):
        self.medusa = medusa_heads

    def generate(self, model, input_ids, max_new_tokens=100,
                 temperature=0.0, top_p=1.0):
        from research.decoding.medusa import medusa_generate
        return medusa_generate(
            model, self.medusa, input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature, device=str(input_ids.device),
        )


class DSparkDecoding(DecodingStrategy):
    """DSpark semi-autoregressive speculative decoding."""

    def __init__(self, dspark_head=None):
        self.dspark = dspark_head

    def generate(self, model, input_ids, max_new_tokens=100,
                 temperature=0.0, top_p=1.0):
        from research.decoding.dspark import dspark_generate
        return dspark_generate(
            model, self.dspark, input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature, device=str(input_ids.device),
        )


class MTPSelfSpecDecoding(DecodingStrategy):
    """Self-speculative decoding using MTP heads from KeyStack checkpoint.

    The XP model checkpoint already contains MTP heads (mtp_head.heads.0-3.weight)
    initialized from the LM head. These predict tokens at positions t+1, t+2,
    t+3, t+4 in parallel. We use them as a draft model for self-speculative
    decoding — no separate draft model needed.

    Flow:
    1. Main model generates token t (with full attention)
    2. MTP heads predict draft tokens t+1..t+k (parallel, no attention)
    3. Main model verifies all k+1 tokens in one forward pass
    4. Accept matching prefix, reject and resample at first mismatch

    This is "free" speculative decoding — the MTP heads are already in the
    checkpoint and add zero inference cost for the draft phase.
    """

    def __init__(self, k=4, mtp_module=None):
        self.k = k
        self.mtp = mtp_module  # Optional: pre-loaded MTP module

    def generate(self, model, input_ids, max_new_tokens=100,
                 temperature=0.0, top_p=1.0):
        ids = input_ids.clone()
        device = input_ids.device
        eos = getattr(model, "eos_token_id", None)

        # Get MTP module from model if not provided
        mtp = self.mtp
        if mtp is None and hasattr(model, "mtp_head"):
            mtp = model.mtp_head
        if mtp is None:
            # No MTP heads — fall back to standard
            return StandardDecoding().generate(
                model, input_ids, max_new_tokens, temperature, top_p)

        # Prefill — need both KV cache (presents) and hidden states for MTP
        with torch.inference_mode():
            out = model(ids, use_cache=True, return_hidden=True)
            # Returns (logits, loss, presents, hidden) when use_cache + return_hidden
            logits = out[0]
            past_kv = out[2]  # presents
            hidden = out[3]   # hidden states

        generated = 0
        while generated < max_new_tokens:
            # Step 1: Main model predicts token t (using KV cache, single token)
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)
            if temperature == 0:
                main_token = next_logits.argmax(-1, keepdim=True)
            else:
                main_token = torch.multinomial(
                    F.softmax(next_logits, dim=-1), num_samples=1)

            if eos and (main_token == eos).any().item():
                ids = torch.cat([ids, main_token], dim=-1)
                break

            # Step 2: MTP heads predict draft tokens (parallel, from last hidden)
            if hidden is not None:
                with torch.inference_mode():
                    mtp_out = mtp(hidden[:, -1:, :])  # Use last hidden state
                draft_tokens = [main_token]  # [1, 1]
                if isinstance(mtp_out, (list, tuple)):
                    for head_out in mtp_out[:self.k]:
                        draft_tok = head_out.argmax(-1)[:, -1:]  # [B, 1]
                        draft_tokens.append(draft_tok)
                else:
                    draft_tokens.append(mtp_out.argmax(-1)[:, -1:])
            else:
                draft_tokens = [main_token]

            # Step 3: Verify draft tokens ONE AT A TIME with main model
            # This avoids KV cache rollback issues — process each token
            # individually and stop on first mismatch.
            n_accepted = 0  # main_token is always accepted
            for i in range(len(draft_tokens) - 1):
                # Feed draft_tokens[i] (already accepted) to get prediction for next
                verify_input = draft_tokens[i]  # [B, 1]
                with torch.inference_mode():
                    out = model(verify_input, past_key_values=past_kv,
                                use_cache=True, return_hidden=True)
                    verify_logits = out[0]
                    past_kv = out[2]
                    verify_hidden = out[3]

                pred = verify_logits[:, -1, :].argmax(-1, keepdim=True)  # [B, 1]
                # GPU-side comparison: single sync instead of two .item() calls.
                match = (pred == draft_tokens[i + 1]).any()
                if match.item():
                    n_accepted += 1
                else:
                    # Replace rejected draft with main model's prediction
                    draft_tokens[i + 1] = pred
                    # Update hidden for next MTP prediction
                    hidden = verify_hidden
                    break
                hidden = verify_hidden

            accepted = torch.cat(draft_tokens[:n_accepted + 1], dim=-1)
            ids = torch.cat([ids, accepted], dim=-1)
            generated += n_accepted + 1

            # logits already updated from last verify step
            # (verify_logits is from the last processed token)
            logits = verify_logits

            if eos:
                # GPU-side EOS check: stack accepted tokens, single sync.
                accepted_stack = torch.cat(draft_tokens[:n_accepted + 1], dim=-1)
                if (accepted_stack == eos).any().item():
                    break
                break

        return ids


def build_decoding(strategy: str = "standard", **kwargs) -> DecodingStrategy:
    """Factory: build decoding strategy by name."""
    strategies = {
        "standard": StandardDecoding,
        "speculative": SpeculativeDecoding,
        "medusa": MedusaDecoding,
        "dspark": DSparkDecoding,
        "mtp_selfspec": MTPSelfSpecDecoding,
    }
    cls = strategies.get(strategy, StandardDecoding)
    return cls(**kwargs)
