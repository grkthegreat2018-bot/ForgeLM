"""Shared forward-only decoder for training-free inference techniques.

Implements KV-cached autoregressive generation (pre-allocated cache, O(1)
append per token) shared by RAIN rewind loops and the TrainingFreeSolver.
No gradients, no optimizer — the entire module is inference-only.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def generate_with_cache(
    model,
    tokenizer,
    prompt: str,
    device: str = "cuda",
    max_tokens: int = 128,
    temperature: float = 0.0,
    stop_strings: tuple[str, ...] = (),
    collect_logprobs: bool = False,
) -> tuple[str, list[float] | None]:
    """Greedy/temperature KV-cached generation with a pre-allocated cache.

    Args:
        model: ConfigurableResearchLLM (or compatible).
        tokenizer: HF-style tokenizer.
        prompt: text to condition on.
        device: "cuda" or "cpu".
        max_tokens: max tokens to generate (excluding prompt).
        temperature: sampling temperature (0 = greedy).
        stop_strings: stop generating when any appears in the output.
        collect_logprobs: also return per-token log-probs of generated tokens.

    Returns:
        (generated_text, logprobs_or_None).
    """
    from research.model_loader import create_kv_cache

    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    eos_id = tokenizer.eos_token_id

    cache = create_kv_cache(
        model, input_ids.shape[1] + max_tokens, batch=1, device=device)
    cache.reset()

    log_probs: list[float] = []
    generated_ids: list[int] = []

    with torch.inference_mode():
        logits, _ = model(input_ids, preallocated_cache=cache)
        next_logits = logits[0, -1]

        for _ in range(max_tokens):
            if temperature > 0:
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax().unsqueeze(0)

            if collect_logprobs:
                lp = F.log_softmax(next_logits.float(), dim=-1)
                log_probs.append(lp[next_token].item())

            token_id = next_token.item()
            generated_ids.append(token_id)

            if eos_id is not None and token_id == eos_id:
                break

            decoded = tokenizer.decode(
                torch.tensor(generated_ids), skip_special_tokens=True)
            if any(s in decoded for s in stop_strings):
                break

            cur = next_token.view(1, 1)
            logits, _ = model(cur, preallocated_cache=cache)
            next_logits = logits[0, -1]

    text = tokenizer.decode(
        torch.tensor(generated_ids), skip_special_tokens=True)
    return text, (log_probs or None)
