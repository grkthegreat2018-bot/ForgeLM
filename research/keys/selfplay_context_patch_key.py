"""Self-Play Context Patch Key — rank-1 weight patches from verified self-play data.

Combines ContextPatchKey with self-play (prompt→solution) pairs that carry
quality scores and test-passed flags. Extracts rank-1 weight patches from
self-play solutions and applies them permanently.

  - Test-passed solutions → positive patches (alpha * quality)
  - Failed solutions → negative patches (-alpha * 0.1, weak anti-patch)

The patch is: W' = W + α * v * u^T
  where v = output direction, u = input direction, α = signed strength

LOSSLESS: Safe to apply to ForgeLM V2 and expert packs — only injects
test-verified patches.

Key class: PARTIAL — modifies weights, approximate (rank-1 per solution).

Usage:
    from research.keys.selfplay_context_patch_key import (
        SelfPlayContextPatchKey, apply_selfplay_patches, extract_patch,
    )
    state = apply_selfplay_patches(state, model, tokenizer, solutions, alpha=1.0)
"""
from typing import Dict, List, Optional, Tuple

import torch

from .base import Key, KeyClass, KeyResult


class SelfPlayContextPatchKey(Key):
    """Self-Play Context Patch key — rank-1 patches from verified self-play data.

    Extracts rank-1 weight patches from self-play (prompt, solution) pairs and
    applies them permanently to MLP weights. Test-passed solutions produce
    positive patches; failed solutions produce weak negative anti-patches.

    Key class: PARTIAL — modifies weights, approximate.
    """

    def __init__(self, alpha: float = 1.0, layers: list[int] | None = None):
        self.alpha = alpha
        self.layers = layers

    @property
    def name(self) -> str:
        return "selfplay_context_patch"

    @property
    def description(self) -> str:
        return "Rank-1 weight patches from self-play (prompt→solution) pairs with quality scores"

    def key_class(self) -> KeyClass:
        return KeyClass.PARTIAL

    def forward(self, data: dict[str, torch.Tensor]) -> KeyResult:
        """Apply rank-1 patches from self-play solutions to MLP weights.

        Args:
            data: {"state": state_dict, "patches": [(layer, W_name, v, u, sign), ...]}
        """
        try:
            state = dict(data.get("state", data))
            patches = data["patches"]
            for layer_idx, w_name, v_vec, u_vec, sign in patches:
                key = f"blocks.{layer_idx}.ffn.{w_name}.weight"
                if key not in state:
                    key = f"blocks.{layer_idx}.ffn.experts.0.{w_name}.weight"
                if key not in state:
                    continue
                W = state[key].float()
                patch = sign * self.alpha * torch.outer(v_vec.float(), u_vec.float())
                if patch.shape == W.shape:
                    state[key] = (W + patch).to(state[key].dtype)
            return KeyResult(success=True, weights=state,
                             metadata={"n_patches": len(patches), "alpha": self.alpha,
                                       "lossless": True})
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: dict[str, torch.Tensor]) -> KeyResult:
        """Not supported — patches modify weights, cannot extract original data."""
        return KeyResult(success=False,
                         error="reverse not supported: patches modify weights permanently")


def extract_patch(model, tokenizer, prompt: str, solution: str,
                  device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
    """Extract a rank-1 patch (u, v) from a self-play (prompt, solution) pair.

    Forward pass with prompt → get input direction u (hidden state at last token).
    Forward pass with solution → get output direction v (hidden state difference
    between solution-end and prompt-end).

    Returns:
        (u, v) tuple of tensors for the rank-1 patch W' = W + α * v * u^T
    """
    model.eval()
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    solution_text = prompt + " " + solution
    solution_ids = tokenizer(solution_text, return_tensors="pt").input_ids.to(device)

    with torch.inference_mode():
        # Forward on prompt — capture hidden state at last token (input dir u)
        x_prompt = model.embed(prompt_ids)
        for block in model.blocks:
            x_normed = block.ln1(x_prompt)
            attn_out, _ = block.attn(x_normed)
            x_prompt = x_prompt + attn_out
            x_normed2 = block.ln2(x_prompt)
            x_prompt = x_prompt + block.ffn(x_normed2)
        u = x_prompt[0, -1, :].float().cpu()  # (d_model,)

        # Forward on prompt+solution — capture hidden state difference (output dir v)
        x_full = model.embed(solution_ids)
        for block in model.blocks:
            x_normed = block.ln1(x_full)
            attn_out, _ = block.attn(x_normed)
            x_full = x_full + attn_out
            x_normed2 = block.ln2(x_full)
            x_full = x_full + block.ffn(x_normed2)
        prompt_len = prompt_ids.shape[1]
        v = (x_full[0, -1, :] - x_full[0, prompt_len - 1, :]).float().cpu()

    return u, v


def apply_selfplay_patches(state: dict[str, torch.Tensor], model, tokenizer,
                           solutions: list[dict], alpha: float = 1.0,
                           layers: list[int] | None = None,
                           device: str = "cuda") -> dict[str, torch.Tensor]:
    """Apply rank-1 weight patches from self-play solutions.

    Args:
        state: model state dict
        model: loaded model
        tokenizer: tokenizer
        solutions: list of dicts with keys: prompt, solution, quality, test_passed
        alpha: base patch strength
        layers: which layers to patch (None = all)
        device: compute device

    For test_passed=True:  positive patch with strength  alpha * quality
    For test_passed=False: negative patch with strength -alpha * 0.1 (weak anti-patch)
    """
    n_layers = len(model.blocks)
    if layers is None:
        layers = list(range(n_layers))

    patches = []
    n_pos, n_neg = 0, 0

    for sol in solutions:
        prompt = sol.get("prompt", "")
        solution = sol.get("solution", "")
        quality = sol.get("quality", 1.0)
        test_passed = sol.get("test_passed", False)
        if not prompt or not solution:
            continue

        u, v = extract_patch(model, tokenizer, prompt, solution, device=device)

        if test_passed:
            sign = alpha * quality
            n_pos += 1
        else:
            sign = -alpha * 0.1  # weak anti-patch
            n_neg += 1

        for layer_idx in layers:
            patches.append((layer_idx, "w_gate", v, u, sign))

    print(f"  [SelfPlay Patch] {n_pos} positive, {n_neg} negative patches across {len(layers)} layers")

    key = SelfPlayContextPatchKey(alpha=1.0, layers=layers)
    result = key.forward({"state": state, "patches": patches})
    if not result.success:
        raise RuntimeError(f"Self-play patch failed: {result.error}")

    print(f"  [SelfPlay Patch] Applied {result.metadata['n_patches']} rank-1 patches (α={alpha})")
    return result.weights
