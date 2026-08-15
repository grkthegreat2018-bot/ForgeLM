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

from research.model_loader import unpack_output_with_kv


class DecodingStrategy(ABC):
    """Base interface for decoding strategies."""

    @abstractmethod
    def generate(self, model, input_ids: torch.Tensor,
                 max_new_tokens: int = 100,
                 temperature: float = 0.0,
                 top_p: float = 1.0,
                 top_k: int = 80,
                 repetition_penalty: float = 1.05) -> torch.Tensor:
        """Generate tokens. Returns full sequence [1, prompt_len + gen_len].

        top_k and repetition_penalty are LFM2.5-recommended defaults; they are
        only applied when temperature > 0 (ignored for greedy decoding).
        """
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__


class StandardDecoding(DecodingStrategy):
    """Standard autoregressive decoding with KV cache."""

    def generate(self, model, input_ids, max_new_tokens=100,
                 temperature=0.0, top_p=1.0,
                 top_k=80, repetition_penalty=1.05):
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
        token_pinned = torch.zeros(1, 1, dtype=torch.long, pin_memory=True)
        # Track generated token ids for repetition penalty
        generated_ids: list[int] = []

        # Prefill — model returns (logits, loss, presents) when use_cache=True
        with torch.inference_mode():
            out = model(ids, use_cache=True)
            logits, past_kv = unpack_output_with_kv(out)

        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)
            if temperature == 0:
                next_token = next_logits.argmax(-1, keepdim=True)
            else:
                # Repetition penalty: penalize tokens already generated
                # (look at last 64 tokens to limit compute)
                if generated_ids:
                    for tid in set(generated_ids[-64:]):
                        next_logits[:, tid] /= repetition_penalty
                # Top-k filtering: keep only top_k logits before softmax
                if top_k > 0:
                    indices_to_remove = next_logits < torch.topk(
                        next_logits, top_k)[0][..., -1, None]
                    next_logits.masked_fill_(indices_to_remove, float('-inf'))
                if top_p < 1.0:
                    next_logits = self._top_p(next_logits, top_p)
                next_token = torch.multinomial(
                    F.softmax(next_logits, dim=-1), num_samples=1)

            # GPU-side EOS check: single sync only if token matches EOS.
            tok_id = next_token.item()
            generated_ids.append(tok_id)
            is_eos = (next_token == eos_tensor).any()
            token_pinned.copy_(next_token, non_blocking=True)
            if is_eos.item():
                break

            ids = torch.cat([ids, next_token], dim=-1)
            with torch.inference_mode():
                out = model(next_token, past_key_values=past_kv, use_cache=True)
                logits, past_kv = unpack_output_with_kv(out)

        # Expose final KV cache state for _finish_to_stop fast path
        model._forge_last_kv = past_kv
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
                 temperature=0.0, top_p=1.0,
                 top_k=80, repetition_penalty=1.05):
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
                 temperature=0.0, top_p=1.0,
                 top_k=80, repetition_penalty=1.05):
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
                 temperature=0.0, top_p=1.0,
                 top_k=80, repetition_penalty=1.05):
        from research.decoding.dspark import dspark_generate
        return dspark_generate(
            model, self.dspark, input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature, device=str(input_ids.device),
        )


class Eagle3Decoding(DecodingStrategy):
    """EAGLE-3 feature-level speculative decoding.

    Uses a lightweight draft head that operates on the target model's
    multi-layer hidden states (low/mid/high) to predict draft tokens.
    No separate draft model needed — the head is attached to the target.

    Args:
        eagle_head: trained Eagle3Head module (optional, can be loaded later)
        draft_length: number of tokens to draft per iteration (default 4)
    """

    def __init__(self, eagle_head=None, draft_length: int = 4):
        self.eagle_head = eagle_head
        self.draft_length = draft_length

    def generate(self, model, input_ids, max_new_tokens=100,
                 temperature=0.0, top_p=1.0,
                 top_k=80, repetition_penalty=1.05):
        if self.eagle_head is None:
            # Fallback to standard if no head loaded
            return StandardDecoding().generate(
                model, input_ids, max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p,
                top_k=top_k, repetition_penalty=repetition_penalty,
            )
        from research.decoding.eagle import eagle3_generate as _eagle_gen
        # eagle3_generate takes a prompt string; here we work with token ids
        # so we use the internal generation loop directly.
        return _eagle_generate_from_ids(
            model, self.eagle_head, input_ids,
            max_new_tokens=max_new_tokens,
            draft_length=self.draft_length,
            temperature=temperature, top_k=top_k,
            repetition_penalty=repetition_penalty,
            device=str(input_ids.device),
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
    3. Main model verifies all k+1 tokens in ONE forward pass (EAGLE-style
       tree verification — a linear chain's tree mask is the standard causal
       mask, so no custom mask is needed). Predictions at all k positions are
       compared against the drafts in parallel; the first mismatch is found
       and the accepted prefix is kept. KV cache is rolled back by slicing
       the returned presents tensors to the accepted length.
    4. Accept matching prefix, reject and resample at first mismatch

    This is "free" speculative decoding — the MTP heads are already in the
    checkpoint and add zero inference cost for the draft phase. The batch
    verification makes it actually faster than standard decoding (one forward
    pass verifies k drafts, vs k forward passes for one-at-a-time).
    """

    def __init__(self, k=4, mtp_module=None):
        self.k = k
        self.mtp = mtp_module  # Optional: pre-loaded MTP module

    def generate(self, model, input_ids, max_new_tokens=100,
                 temperature=0.0, top_p=1.0,
                 top_k=80, repetition_penalty=1.05):
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
                model, input_ids, max_new_tokens, temperature, top_p,
                top_k=top_k, repetition_penalty=repetition_penalty)

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

            # Step 3: Batch-verify ALL draft tokens in ONE forward pass
            # (EAGLE-style tree verification for a linear chain — the tree
            # mask for a chain is just the standard causal mask, so no custom
            # mask is needed). This replaces the previous one-at-a-time loop
            # which defeated the purpose of speculative decoding (each token
            # did a full forward pass, making it slower than standard decoding).
            #
            # verify_seq = [main_token, draft_1, ..., draft_k]  (length n_draft)
            # logits[:, i, :] predicts the token after position i.
            # We compare the model's prediction at position i against draft_{i+1}
            # for all i in parallel, find the first mismatch, accept the prefix,
            # and resample at the mismatch.
            n_draft = len(draft_tokens) - 1  # number of MTP-drafted tokens
            if n_draft == 0:
                # No drafts — just accept the main token.
                ids = torch.cat([ids, main_token], dim=-1)
                generated += 1
                # logits already set for the next iteration.
                continue

            verify_seq = torch.cat(draft_tokens, dim=-1)  # [B, n_draft+1]
            past_len = past_kv[0][0].shape[-2] if past_kv is not None else 0
            with torch.inference_mode():
                out = model(verify_seq, past_key_values=past_kv,
                            use_cache=True, return_hidden=True)
                verify_logits = out[0]   # [B, n_draft+1, V]
                new_past_kv = out[2]     # list of (k, v) with len past_len + n_draft + 1
                verify_hidden = out[3]   # [B, n_draft+1, d]

            # Predictions for positions 0..n_draft-1 (what should follow each
            # accepted token) vs the drafted tokens at 1..n_draft.
            preds = verify_logits[:, :n_draft, :].argmax(-1)        # [B, n_draft]
            draft_stack = torch.cat(draft_tokens[1:], dim=-1)       # [B, n_draft]
            matches = (preds == draft_stack)                        # [B, n_draft]

            # First mismatch index (per batch). For B=1 this is a scalar.
            # n_accepted = number of accepted drafts (main_token + first n_accepted drafts).
            # We take the per-batch min of the first-mismatch index so the
            # accepted prefix is valid for every sequence in the batch.
            # first_false[i] = index of first False in matches[i], or n_draft if all True.
            not_match = ~matches
            # argmax returns first True index; if row is all False, argmax returns 0.
            any_mismatch = not_match.any(dim=-1)                    # [B]
            first_mismatch = torch.where(
                any_mismatch,
                not_match.float().argmax(dim=-1),
                torch.full_like(any_mismatch, n_draft, dtype=torch.long),
            )                                                       # [B]
            n_accepted = int(first_mismatch.min().item())           # accept the common prefix

            # KV cache rollback: keep only past_len + n_accepted + 1 entries
            # (the accepted prefix). The new_past_kv tensors have length
            # past_len + n_draft + 1; slice off the rejected tail.
            keep_len = past_len + n_accepted + 1
            past_kv = []
            for layer_kv in new_past_kv:
                k, v = layer_kv
                past_kv.append((k[:, :, :keep_len, :], v[:, :, :keep_len, :]))

            # Accepted tokens: main_token + drafts 1..n_accepted.
            accepted = verify_seq[:, :n_accepted + 1]
            ids = torch.cat([ids, accepted], dim=-1)
            generated += n_accepted + 1

            # Set up logits + hidden for the NEXT iteration:
            #  - If a mismatch occurred at index j=n_accepted, the model's
            #    prediction at position j (verify_logits[:, j, :]) is the
            #    corrected next token — use it as the next main_token's logits.
            #  - If all drafts accepted, use the prediction after the last
            #    draft (verify_logits[:, n_draft, :]).
            next_idx = min(n_accepted, n_draft)
            logits = verify_logits[:, next_idx:next_idx + 1, :]    # [B, 1, V]
            hidden = verify_hidden[:, next_idx:next_idx + 1, :]    # [B, 1, d]

            if eos:
                # GPU-side EOS check on the accepted prefix (single sync).
                if (accepted == eos).any().item():
                    break

        return ids


def build_decoding(strategy: str = "standard", **kwargs) -> DecodingStrategy:
    """Factory: build decoding strategy by name."""
    strategies = {
        "standard": StandardDecoding,
        "speculative": SpeculativeDecoding,
        "medusa": MedusaDecoding,
        "dspark": DSparkDecoding,
        "eagle3": Eagle3Decoding,
        "mtp_selfspec": MTPSelfSpecDecoding,
    }
    cls = strategies.get(strategy, StandardDecoding)
    return cls(**kwargs)


@torch.inference_mode()
def _eagle_generate_from_ids(
    model, head, input_ids, max_new_tokens=100, draft_length=4,
    temperature=0.0, top_k=0, repetition_penalty=1.0, device="cuda",
):
    """EAGLE-3 generation from token ids (used by Eagle3Decoding strategy)."""
    from research.decoding.eagle import extract_hidden_states, Eagle3Head

    model.eval()
    head.eval()

    extract_layers = [head.low_layer, head.mid_layer, head.high_layer]
    eos_id = getattr(model, 'eos_token_id', None) or 7

    # Prefill
    hidden_list, final_hidden, presents = extract_hidden_states(
        model, input_ids, extract_layers, use_cache=True,
    )
    fused = head.fuse_hidden_states(hidden_list)
    target_logits = model.head(final_hidden)
    last_logits = target_logits[:, -1, :]

    if temperature <= 0:
        next_token = last_logits.argmax(dim=-1, keepdim=True)
    else:
        l = last_logits / temperature
        if top_k > 0:
            idx_rm = l < torch.topk(l, top_k)[0][..., -1, None]
            l.masked_fill_(idx_rm, float('-inf'))
        probs = F.softmax(l, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

    generated = [next_token.item()]
    ids = torch.cat([input_ids, next_token], dim=1)

    while len(generated) < max_new_tokens:
        if generated[-1] == eos_id:
            break

        # Draft tokens autoregressively
        draft_tokens = []
        cur_fused = fused[:, -1:, :]
        cur_token = next_token
        for _ in range(draft_length):
            draft_tok, _ = head.predict_next(cur_fused, cur_token, temperature=temperature, top_k=top_k)
            draft_tokens.append(draft_tok.item())
            cur_token = draft_tok

        # Verify with target
        draft_tensor = torch.tensor([draft_tokens], device=device, dtype=ids.dtype)
        verify_hidden, verify_final, presents = extract_hidden_states(
            model, draft_tensor, extract_layers,
            past_key_values=presents, use_cache=True,
        )
        verify_logits = model.head(verify_final)

        # Accept longest prefix
        n_accepted = 0
        for i, dt in enumerate(draft_tokens):
            target_tok = verify_logits[:, i, :].argmax(dim=-1).item()
            if target_tok == dt:
                n_accepted += 1
            else:
                break

        for i in range(n_accepted):
            generated.append(draft_tokens[i])
            if len(generated) >= max_new_tokens:
                break

        if n_accepted < draft_length:
            next_token = verify_logits[:, n_accepted, :].argmax(dim=-1, keepdim=True)
        else:
            next_token = verify_logits[:, -1, :].argmax(dim=-1, keepdim=True)

        fused = head.fuse_hidden_states(verify_hidden)
        if next_token.item() == eos_id:
            break
        generated.append(next_token.item())
        ids = torch.cat([ids, next_token], dim=1)

    return ids[:, input_ids.shape[1]:]
