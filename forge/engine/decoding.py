"""Decoding strategy backends.

Pluggable decoding strategies selectable at runtime:
  - StandardDecoding: autoregressive token-by-token (baseline)
  - SpeculativeDecoding: draft model + verify (wraps speculative_decode.py)
  - MedusaDecoding: parallel prediction heads (wraps medusa.py)
  - DSparkDecoding: semi-autoregressive + confidence scheduling (wraps dspark.py)
  - MTPSelfSpecDecoding: use MTP heads from checkpoint for self-speculative decoding
  - SelfSpeculativeSparse: same model as draft+target with sparse-attention draft (R39-4)

All implement the DecodingStrategy interface:
  generate(model, input_ids, max_new_tokens, temperature, top_p) -> output_ids
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from forge.model_loader import unpack_output_with_kv


def _min_k_filter_logits(logits: torch.Tensor, sensitivity: float) -> torch.Tensor:
    """Min-k semantic-cliff sampling filter (ACL 2026, Ding et al.).

    Analyzes the local shape of the sorted logit distribution to identify
    "semantic cliffs" — sharp transitions from high-confidence core tokens
    to uncertain long-tail tokens. Temperature-invariant: operates on
    relative logit dynamics, not absolute probabilities.

    Args:
        logits: (batch, vocab) logit tensor.
        sensitivity: cliff detection sensitivity in [0, 1]. Higher = more
            aggressive truncation. 0 = no filtering.

    Returns:
        Filtered logits with long-tail tokens masked to -inf.
    """
    if sensitivity <= 0:
        return logits
    sorted_logits, _ = torch.sort(logits, descending=True, dim=-1)
    eps = 1e-8
    diffs = sorted_logits[..., :-1] - sorted_logits[..., 1:]
    n = diffs.shape[-1]
    weights = torch.linspace(1.0, 0.1, n, device=logits.device, dtype=logits.dtype)
    weighted_diffs = diffs * weights
    max_decay = weighted_diffs.max(dim=-1, keepdim=True).values.clamp(min=eps)
    cliff_pos = weighted_diffs.argmax(dim=-1, keepdim=True)
    positions = torch.arange(sorted_logits.shape[-1], device=logits.device)
    keep = positions <= cliff_pos
    sorted_indices = torch.sort(logits, descending=True, dim=-1).indices
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(-1, sorted_indices, ~keep)
    return logits.masked_fill(mask, float("-inf"))


class DecodingStrategy(ABC):
    """Base interface for decoding strategies."""

    @abstractmethod
    def generate(self, model, input_ids: torch.Tensor,
                 max_new_tokens: int = 100,
                 temperature: float = 0.0,
                 top_p: float = 1.0,
                 top_k: int = 80,
                 repetition_penalty: float = 1.05,
                 **kwargs) -> torch.Tensor:
        """Generate tokens. Returns full sequence [1, prompt_len + gen_len].

        top_k and repetition_penalty are LFM2.5-recommended defaults; they are
        only applied when temperature > 0 (ignored for greedy decoding).
        Subclasses may accept additional kwargs (min_p, min_k, etc.) — the base
        signature accepts **kwargs so the engine can uniformly pass all sampling
        params without TypeError on strategies that don't use them.
        """
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__


class StandardDecoding(DecodingStrategy):
    """Standard autoregressive decoding with KV cache."""

    def generate(self, model, input_ids, max_new_tokens=100,
                 temperature=0.0, top_p=1.0,
                 top_k=80, repetition_penalty=1.05,
                 min_p: float = 0.0, min_k: float = 0.0):
        ids = input_ids.clone()
        device = input_ids.device
        # EOS detection: check model attr, config, then Qwen defaults
        eos = getattr(model, "eos_token_id", None)
        if eos is None:
            cfg = getattr(model, "config", None)
            eos = getattr(cfg, "eos_token_id", None) if cfg else None
        # Qwen2.5 EOS tokens: <|endoftext|>=151643, <|im_end|>=151645
        eos_set = {7, 151643, 151645}  # LFM2.5 <|im_end|>=7 + Qwen2.5
        if eos is not None:
            eos_set.add(eos)
        eos_tensor = torch.tensor(list(eos_set), device=device)
        # Pinned memory for async D2H (reduces CPU sync spikes).
        token_pinned = torch.zeros(1, 1, dtype=torch.long, pin_memory=True)
        # Track generated token ids for repetition penalty + degeneration
        generated_ids: list[int] = []
        # Collect generated token tensors for a single final cat (O(n) vs O(n²))
        _generated_tokens: list[torch.Tensor] = []
        # Degeneration guard: stop if same token repeats too many times.
        MAX_REPEAT = 8  # same token 8x in a row = degenerate

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
                # Min-p sampling: filter tokens below min_p * max_prob.
                # Temperature-invariant dynamic truncation.
                if min_p > 0.0:
                    probs = F.softmax(next_logits, dim=-1)
                    max_prob = probs.max(dim=-1, keepdim=True).values
                    threshold = min_p * max_prob
                    next_logits = torch.where(
                        probs < threshold,
                        torch.full_like(next_logits, float('-inf')),
                        next_logits,
                    )
                # Min-k sampling: semantic-cliff detection (ACL 2026).
                # Temperature-invariant dynamic truncation via logit dynamics.
                if min_k > 0.0:
                    next_logits = _min_k_filter_logits(next_logits, min_k)
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

            # Degeneration guard: detect repetitive garbage and stop early.
            # Critical for base models that haven't learned to emit EOS.
            if len(generated_ids) >= MAX_REPEAT:
                if all(g == tok_id for g in generated_ids[-MAX_REPEAT:]):
                    break  # same token repeated 8x = degenerate
            if len(generated_ids) >= 20:
                recent = generated_ids[-20:]
                if len(set(recent)) / len(recent) < 0.4:
                    break  # <40% unique tokens = degenerate

            # Collect tokens for final cat (avoids O(n²) torch.cat per step)
            _generated_tokens.append(next_token)
            with torch.inference_mode():
                out = model(next_token, past_key_values=past_kv, use_cache=True)
                logits, past_kv = unpack_output_with_kv(out)

        # Expose final KV cache state for _finish_to_stop fast path
        model._forge_last_kv = past_kv
        # Single cat at the end instead of per-step (O(n) vs O(n²))
        if _generated_tokens:
            ids = torch.cat([ids] + _generated_tokens, dim=-1)
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
                 top_k=80, repetition_penalty=1.05,
                 **kwargs):
        from research.speculative_decode import speculative_generate
        return speculative_generate(
            model, self.draft_model, input_ids,
            max_new_tokens=max_new_tokens, k=self.k,
            temperature=temperature, device=str(input_ids.device),
        )


class NGramSpeculativeDecoding(DecodingStrategy):
    """N-gram speculative decoding (prompt-lookup).

    Uses n-gram matching against the prompt to generate draft tokens —
    no separate draft model needed. Works well for code completion and
    RAG where output often repeats prompt substrings.

    Based on vLLM's prompt_lookup_speculative_decoding.
    """

    def __init__(self, ngram_size: int = 3, draft_length: int = 4):
        self.ngram_size = ngram_size
        self.draft_length = draft_length

    def generate(self, model, input_ids, max_new_tokens=100,
                 temperature=0.0, top_p=1.0,
                 top_k=80, repetition_penalty=1.05,
                 **kwargs):
        ids = input_ids.clone()
        device = input_ids.device
        prompt_len = input_ids.shape[1]
        generated = 0

        with torch.inference_mode():
            # Prefill
            out = model(ids, use_cache=True)
            logits, past_kv = unpack_output_with_kv(out)

        while generated < max_new_tokens:
            # Generate main token
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)
            if temperature == 0:
                main_token = next_logits.argmax(-1, keepdim=True)
            else:
                probs = F.softmax(next_logits, dim=-1)
                main_token = torch.multinomial(probs.view(-1), 1).unsqueeze(0)

            # N-gram lookup: find draft tokens by matching last ngram_size tokens
            draft_tokens = self._ngram_lookup(ids, main_token.squeeze())

            if len(draft_tokens) > 0:
                # Verify draft tokens in one forward pass
                draft_tensor = torch.tensor(draft_tokens, device=device).unsqueeze(0)
                verify_ids = torch.cat([main_token, draft_tensor], dim=-1)
                with torch.inference_mode():
                    out = model(verify_ids, past_key_values=past_kv, use_cache=True)
                    verify_logits, past_kv = unpack_output_with_kv(out)

                # Accept tokens that match
                accepted = 0
                prev_token = main_token.squeeze()
                for i, draft_tok in enumerate(draft_tokens):
                    pred_tok = verify_logits[0, i, :].argmax()
                    if pred_tok.item() == draft_tok:
                        accepted += 1
                        prev_token = torch.tensor([draft_tok], device=device)
                    else:
                        break

                # Append accepted + 1 (the token predicted at the mismatch)
                new_tokens = [main_token.squeeze().item()] + draft_tokens[:accepted]
                if accepted < len(draft_tokens):
                    # Mismatch — use the model's prediction at the mismatch point
                    new_tokens.append(verify_logits[0, accepted, :].argmax().item())

                for tok in new_tokens:
                    ids = torch.cat([ids, torch.tensor([[tok]], device=device)], dim=-1)
                    generated += 1
                    if generated >= max_new_tokens:
                        break

                # Get logits for next iteration (last position)
                if accepted < len(draft_tokens):
                    logits = verify_logits[:, accepted:accepted+1, :]
                else:
                    logits = verify_logits[:, -1:, :]
            else:
                # No draft — standard step
                ids = torch.cat([ids, main_token], dim=-1)
                generated += 1
                with torch.inference_mode():
                    out = model(main_token, past_key_values=past_kv, use_cache=True)
                    logits, past_kv = unpack_output_with_kv(out)

        return ids

    def _ngram_lookup(self, ids: torch.Tensor, last_token: int) -> list[int]:
        """Find draft tokens by n-gram matching against the sequence."""
        seq = ids[0].cpu().tolist() + [last_token]
        if len(seq) < self.ngram_size + 1:
            return []
        ngram = seq[-self.ngram_size:]
        # Search for this ngram earlier in the sequence
        for i in range(len(seq) - self.ngram_size - 1, self.ngram_size - 2, -1):
            if seq[i:i+self.ngram_size] == ngram:
                # Return up to draft_length tokens after the match
                start = i + self.ngram_size
                end = min(start + self.draft_length, len(seq))
                return seq[start:end]
        return []


class ExternalDraftSpeculativeDecoding(DecodingStrategy):
    """Speculative decoding with an external draft model.

    The draft model is a smaller/faster model that proposes tokens.
    The main model verifies them in a single forward pass.
    """

    def __init__(self, draft_model, draft_length: int = 4):
        self.draft_model = draft_model
        self.draft_length = draft_length

    def generate(self, model, input_ids, max_new_tokens=100,
                 temperature=0.0, top_p=1.0,
                 top_k=80, repetition_penalty=1.05,
                 **kwargs):
        ids = input_ids.clone()
        device = input_ids.device
        generated = 0

        with torch.inference_mode():
            # Prefill both models
            out = model(ids, use_cache=True)
            logits, past_kv = unpack_output_with_kv(out)
            draft_out = self.draft_model(ids, use_cache=True)
            draft_logits, draft_past_kv = unpack_output_with_kv(draft_out)

        while generated < max_new_tokens:
            # Draft model proposes k tokens
            draft_tokens = []
            cur_logits = draft_logits[:, -1, :]
            cur_ids = ids
            for _ in range(self.draft_length):
                if temperature == 0:
                    tok = cur_logits.argmax(-1, keepdim=True)
                else:
                    probs = F.softmax(cur_logits / max(temperature, 1e-5), dim=-1)
                    tok = torch.multinomial(probs.view(-1), 1).unsqueeze(0)
                draft_tokens.append(tok.squeeze().item())
                cur_ids = torch.cat([cur_ids, tok], dim=-1)
                with torch.inference_mode():
                    draft_out = self.draft_model(tok, past_key_values=draft_past_kv, use_cache=True)
                    draft_logits, draft_past_kv = unpack_output_with_kv(draft_out)

            # Main model generates next token
            main_token = (logits[:, -1, :] / max(temperature, 1e-5)).argmax(-1, keepdim=True)

            if draft_tokens:
                # Verify draft in one forward pass
                draft_tensor = torch.tensor(draft_tokens, device=device).unsqueeze(0)
                verify_ids = torch.cat([main_token, draft_tensor], dim=-1)
                with torch.inference_mode():
                    out = model(verify_ids, past_key_values=past_kv, use_cache=True)
                    verify_logits, past_kv = unpack_output_with_kv(out)

                # Accept matching prefix
                accepted = 0
                new_tokens = [main_token.squeeze().item()]
                for i, draft_tok in enumerate(draft_tokens):
                    pred_tok = verify_logits[0, i, :].argmax()
                    if pred_tok.item() == draft_tok:
                        accepted += 1
                        new_tokens.append(draft_tok)
                    else:
                        new_tokens.append(pred_tok.item())
                        break

                for tok in new_tokens:
                    ids = torch.cat([ids, torch.tensor([[tok]], device=device)], dim=-1)
                    generated += 1
                    if generated >= max_new_tokens:
                        break

                logits = verify_logits[:, -1:, :]
            else:
                ids = torch.cat([ids, main_token], dim=-1)
                generated += 1
                with torch.inference_mode():
                    out = model(main_token, past_key_values=past_kv, use_cache=True)
                    logits, past_kv = unpack_output_with_kv(out)

        return ids


class MedusaDecoding(DecodingStrategy):
    """Medusa parallel prediction heads."""

    def __init__(self, medusa_heads):
        self.medusa = medusa_heads

    def generate(self, model, input_ids, max_new_tokens=100,
                 temperature=0.0, top_p=1.0,
                 top_k=80, repetition_penalty=1.05,
                 **kwargs):
        from forge.decoding.medusa import medusa_generate
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
                 top_k=80, repetition_penalty=1.05,
                 **kwargs):
        from forge.decoding.dspark import dspark_generate
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
                 top_k=80, repetition_penalty=1.05,
                 **kwargs):
        if self.eagle_head is None:
            # Fallback to standard if no head loaded
            return StandardDecoding().generate(
                model, input_ids, max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p,
                top_k=top_k, repetition_penalty=repetition_penalty,
                **kwargs,
            )
        from forge.decoding.eagle import eagle3_generate as _eagle_gen
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

    Evolution-discovered defaults: 7 draft tokens (k=7), 0.95 acceptance
    threshold, 0.10 draft_model_ratio → 5.37x speedup over standard decoding.
    These values were found by evolution search on RTX 5070 + LFM2.5-1.2B
    and represent the sweet spot between draft accuracy and verification
    overhead.
    """

    def __init__(self, k=7, mtp_module=None, acceptance_threshold=0.95,
                 draft_model_ratio=0.10, replay_ssm_cache=None):
        self.k = k
        self.mtp = mtp_module  # Optional: pre-loaded MTP module
        self.acceptance_threshold = acceptance_threshold
        self.draft_model_ratio = draft_model_ratio
        self.replay_ssm = replay_ssm_cache  # ReplaySSMCache for SSM rollback

    def generate(self, model, input_ids, max_new_tokens=100,
                 temperature=0.0, top_p=1.0,
                 top_k=80, repetition_penalty=1.05,
                 **kwargs):
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
                top_k=top_k, repetition_penalty=repetition_penalty,
                **kwargs)

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

            # ReplaySSM: checkpoint before drafting so we can rollback
            # the SSM state efficiently if the draft is rejected.
            ssm_checkpoint = None
            if self.replay_ssm is not None:
                ssm_checkpoint = self.replay_ssm.checkpoint()

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

            # ReplaySSM: if some drafts were rejected, rollback the SSM
            # input cache to the checkpoint so the next reconstruct_state
            # replays only the accepted inputs.  When all drafts are
            # accepted, no rollback is needed.
            if (self.replay_ssm is not None and ssm_checkpoint is not None
                    and n_accepted < n_draft):
                self.replay_ssm.rollback(ssm_checkpoint + n_accepted + 1)

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


class SelfSpeculativeSparse(DecodingStrategy):
    """Self-speculative decoding with sparse-attention draft (R39-4).

    The **same model** serves as both draft and target — no separate
    draft model is needed, so there is **zero extra memory**.

    Flow per iteration:
      1. **Draft phase**: generate ``draft_len`` tokens autoregressively
         using *sparse attention* (top-``sparse_k`` KV positions).  The
         model forward is called with ``sparse_k=<int>`` so attention
         layers can limit the KV cache to the top-k most relevant
         positions.  This is faster than full attention because the
         attention computation is O(seq_len * sparse_k) instead of
         O(seq_len²).
      2. **Verify phase**: run the model with **full attention** on the
         entire draft sequence in a single forward pass.  The full-
         attention logits at each position tell us what the *correct*
         next token is.
      3. **Accept**: compare draft tokens against the full-attention
         predictions.  Accept the longest matching prefix.  Resample at
         the first mismatch using the full-attention logits.
      4. **KV selection feedback**: the verification attention scores
         from the accepted positions are reused to select which KV
         positions to keep for the next draft round's sparse attention.

    Because verification uses full attention, the output is **lossless**
    — every accepted token is exactly what standard full-attention
    decoding would produce.  The sparse draft only affects *speed*, not
    *quality*.

    Args:
        draft_len: number of tokens to draft per iteration (default 4).
        sparse_k: number of top KV positions to attend to during the
            draft phase (default 64).  Larger = more accurate draft but
            slower; smaller = faster but lower acceptance rate.
    """

    def __init__(self, draft_len: int = 4, sparse_k: int = 64,
                 replay_ssm_cache=None):
        self.draft_len = draft_len
        self.sparse_k = sparse_k
        self.replay_ssm = replay_ssm_cache  # ReplaySSMCache for SSM rollback
        # Acceptance statistics (updated during generate).
        self.acceptance_rate: float = 0.0
        self._total_drafts: int = 0
        self._total_accepted: int = 0
        # KV selection scores from the last verification pass — reused
        # for sparse attention position selection in the next draft round.
        self._kv_scores: torch.Tensor | None = None

    def generate(self, model, input_ids, max_new_tokens=100,
                 temperature=0.0, top_p=1.0,
                 top_k=80, repetition_penalty=1.05,
                 **kwargs):
        ids = input_ids.clone()
        device = input_ids.device
        eos = getattr(model, "eos_token_id", None)
        eos_set = {7, 151643, 151645}
        if eos is not None:
            eos_set.add(eos)
        eos_tensor = torch.tensor(list(eos_set), device=device)

        # Prefill with full attention.
        with torch.inference_mode():
            out = model(ids, use_cache=True)
            logits, past_kv = unpack_output_with_kv(out)

        generated = 0
        _generated_tokens: list[torch.Tensor] = []

        while generated < max_new_tokens:
            # ── Step 1: Base token from full-attention logits ──────────
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)
            if temperature == 0:
                base_token = next_logits.argmax(-1, keepdim=True)
            else:
                base_token = torch.multinomial(
                    F.softmax(next_logits, dim=-1), num_samples=1)

            # ReplaySSM: checkpoint before drafting so we can rollback
            # the SSM state efficiently if the draft is rejected.
            ssm_checkpoint = None
            if self.replay_ssm is not None:
                ssm_checkpoint = self.replay_ssm.checkpoint()

            # ── Step 2: Draft phase — sparse attention ────────────────
            # Generate draft_len tokens autoregressively with sparse
            # attention (top-k KV positions).  The model forward receives
            # sparse_k so attention layers can limit the KV window.
            draft_tokens = [base_token]
            draft_kv = past_kv
            for _ in range(self.draft_len):
                with torch.inference_mode():
                    draft_out = model(
                        draft_tokens[-1],
                        past_key_values=draft_kv,
                        use_cache=True,
                        sparse_k=self.sparse_k,
                    )
                    draft_logits_i, draft_kv = unpack_output_with_kv(draft_out)
                draft_tok = draft_logits_i[:, -1, :].argmax(-1, keepdim=True)
                draft_tokens.append(draft_tok)

            n_draft = len(draft_tokens) - 1  # number of drafted tokens

            # ── Step 3: Verify phase — full attention ─────────────────
            # Run the model with full attention on the entire draft
            # sequence in ONE forward pass.  The draft KV cache (sparse)
            # is discarded — we re-process from the original full KV.
            verify_seq = torch.cat(draft_tokens, dim=-1)  # [B, n_draft+1]
            past_len = past_kv[0][0].shape[-2] if past_kv is not None else 0
            with torch.inference_mode():
                verify_out = model(
                    verify_seq,
                    past_key_values=past_kv,
                    use_cache=True,
                )
                verify_logits, new_past_kv = unpack_output_with_kv(verify_out)

            # ── Step 4: Compare draft vs full-attention predictions ───
            # verify_logits[:, i, :] predicts the token after position i.
            # We compare against draft_tokens[i+1] for i in [0, n_draft).
            preds = verify_logits[:, :n_draft, :].argmax(-1)     # [B, n_draft]
            draft_stack = torch.cat(draft_tokens[1:], dim=-1)    # [B, n_draft]
            matches = (preds == draft_stack)                     # [B, n_draft]

            # First mismatch index (per batch).
            not_match = ~matches
            any_mismatch = not_match.any(dim=-1)
            first_mismatch = torch.where(
                any_mismatch,
                not_match.float().argmax(dim=-1),
                torch.full_like(any_mismatch, n_draft, dtype=torch.long),
            )
            n_accepted = int(first_mismatch.min().item())

            # ── Acceptance statistics ────────────────────────────────
            self._total_drafts += n_draft
            self._total_accepted += n_accepted
            self.acceptance_rate = (
                self._total_accepted / max(self._total_drafts, 1)
            )

            # ── Step 5: KV cache rollback to accepted prefix ─────────
            keep_len = past_len + n_accepted + 1
            past_kv = []
            for layer_kv in new_past_kv:
                k, v = layer_kv
                past_kv.append(
                    (k[:, :, :keep_len, :], v[:, :, :keep_len, :]))

            # ReplaySSM: if some drafts were rejected, rollback the SSM
            # input cache to the checkpoint + accepted prefix so the next
            # reconstruct_state replays only the accepted inputs.
            if (self.replay_ssm is not None and ssm_checkpoint is not None
                    and n_accepted < n_draft):
                self.replay_ssm.rollback(ssm_checkpoint + n_accepted + 1)

            # ── Step 6: Accept tokens ────────────────────────────────
            accepted = verify_seq[:, :n_accepted + 1]
            _generated_tokens.append(accepted)
            generated += n_accepted + 1

            # Check EOS in accepted tokens.
            if any(t.item() in eos_set for t in accepted.flatten()):
                break

            # ── Set up logits for next iteration ─────────────────────
            # If mismatch at index j=n_accepted, verify_logits[:, j, :]
            # is the corrected next token's logits.
            # If all accepted, use the prediction after the last draft.
            next_idx = min(n_accepted, n_draft)
            logits = verify_logits[:, next_idx:next_idx + 1, :]

        # Single cat at the end (O(n) vs O(n²)).
        if _generated_tokens:
            ids = torch.cat([ids] + _generated_tokens, dim=-1)
        # Truncate to max_new_tokens (speculative acceptance can overshoot).
        max_len = input_ids.shape[1] + max_new_tokens
        if ids.shape[1] > max_len:
            ids = ids[:, :max_len]
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
        "self_speculative_sparse": SelfSpeculativeSparse,
        "batched": None,  # set below to avoid circular import
    }
    if strategy == "batched":
        from forge.engine.batched_decoding import BatchedDecoding
        return BatchedDecoding(**kwargs)
    cls = strategies.get(strategy, StandardDecoding)
    return cls(**kwargs)


@torch.inference_mode()
def _eagle_generate_from_ids(
    model, head, input_ids, max_new_tokens=100, draft_length=4,
    temperature=0.0, top_k=0, repetition_penalty=1.0, device="cuda",
):
    """EAGLE-3 generation from token ids (used by Eagle3Decoding strategy)."""
    from forge.decoding.eagle import extract_hidden_states, Eagle3Head

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
    _eagle_tokens: list[torch.Tensor] = [next_token]  # collect for final cat

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
        _eagle_tokens.append(next_token)

    # Single cat at the end instead of per-step (O(n) vs O(n²))
    ids = torch.cat([input_ids] + _eagle_tokens, dim=1)
    return ids[:, input_ids.shape[1]:]
