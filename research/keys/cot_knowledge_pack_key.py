"""CoT Knowledge Pack Key — pre-computed KV cache from chain-of-thought reasoning.

Combines KnowledgePackKey with chain-of-thought (CoT) reasoning traces from
self-play. Pre-computes KV caches from reasoning traces and injects them as a
prefix at inference time, so the model gets the "thinking" for free without
generating it.

LOSSLESS: Safe to apply to ForgeLM V2 and expert packs — no weight changes,
runtime injection only.

Key class: TRIVIAL — runtime injection, no weight changes.

Usage:
    from research.keys.cot_knowledge_pack_key import (
        CoTKnowledgePackKey, CoTKnowledgePack, build_cot_packs_from_selfplay,
    )
    packs = build_cot_packs_from_selfplay("logs/selfplay", model, tokenizer)
    packs[0].inject(model, kv_cache, alpha=1.0)
"""
import json
import os
from typing import Dict, List, Optional, Tuple

import torch

from .base import Key, KeyClass, KeyResult


class CoTKnowledgePackKey(Key):
    """CoT Knowledge Pack key — zero-token reasoning via KV cache injection.

    Pre-computes KV caches from chain-of-thought reasoning traces and injects
    them at inference. No weight changes, no token cost, no context window usage.

    Key class: TRIVIAL — runtime injection strategy.
    """

    @property
    def name(self) -> str:
        return "cot_knowledge_pack"

    @property
    def description(self) -> str:
        return "Zero-token chain-of-thought delivery via pre-computed KV cache injection"

    def key_class(self) -> KeyClass:
        return KeyClass.TRIVIAL

    def forward(self, data: dict[str, torch.Tensor]) -> KeyResult:
        """data -> KV cache (CoT reasoning trace encoded as prefix)."""
        return KeyResult(success=True, weights=data,
                         metadata={"runtime": True, "zero_token": True,
                                   "lossless": True})

    def reverse(self, weights: dict[str, torch.Tensor]) -> KeyResult:
        """Not applicable — no weight changes to reverse."""
        return KeyResult(success=True, data=weights,
                         metadata={"reversible": True,
                                   "note": "runtime injection, no weights changed"})


class CoTKnowledgePack:
    """A pre-computed KV cache from a chain-of-thought reasoning trace.

    Created by running a forward pass on a CoT reasoning trace through the model.
    The resulting KV cache can be injected as a prefix at inference time, giving
    the model the "thinking" for free without generating it.

    LOSSLESS: Safe to apply to ForgeLM V2 and expert packs — no weight changes,
    runtime injection only.
    """

    def __init__(self, kv_caches: list[tuple[torch.Tensor, torch.Tensor]],
                 reasoning_text: str = "", task_prompt: str = "",
                 metadata: dict | None = None):
        self.kv_caches = kv_caches  # per-layer (K, V) tuples
        self.reasoning_text = reasoning_text
        self.task_prompt = task_prompt
        self.metadata = metadata or {}
        self.n_tokens = kv_caches[0][0].shape[-2] if kv_caches else 0

    @classmethod
    def from_cot(cls, model, tokenizer, reasoning_text: str,
                 task_prompt: str, device: str = "cuda") -> "CoTKnowledgePack":
        """Pre-compute KV cache from a chain-of-thought reasoning trace.

        Runs a forward pass on the reasoning trace (prefixed by the task prompt)
        and captures the KV cache. The model will "have thought" about the task
        without generating the CoT at inference time.
        """
        full_text = f"{task_prompt}\n{reasoning_text}"

        if hasattr(tokenizer, 'apply_chat_template'):
            messages = [{"role": "system", "content": full_text}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False)
            input_ids = tokenizer(formatted, return_tensors="pt").input_ids.to(device)
        else:
            input_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(device)

        model.eval()
        with torch.inference_mode():
            past_key_values = [None] * len(model.blocks)
            x = model.embed(input_ids)
            for i, block in enumerate(model.blocks):
                x, present = block(x, past_key_value=past_key_values[i],
                                   use_cache=True)
                past_key_values[i] = present

        kv_caches = []
        for present in past_key_values:
            if present is not None:
                k, v = present
                kv_caches.append((k.clone().cpu(), v.clone().cpu()))
            else:
                kv_caches.append((None, None))

        return cls(kv_caches, reasoning_text=reasoning_text, task_prompt=task_prompt,
                   metadata={"n_tokens": input_ids.shape[1], "n_layers": len(kv_caches)})

    def inject(self, model, kv_cache: list, alpha: float = 1.0,
               device: str = "cuda") -> list:
        """Inject this CoT Knowledge Pack at inference time.

        Prepends the reasoning KV entries to the existing cache so the model
        "has already thought" about the task before generating.
        """
        for i, (k_pack, v_pack) in enumerate(self.kv_caches):
            if k_pack is None or v_pack is None:
                continue
            k_pack = k_pack.to(device)
            v_pack = v_pack.to(device)
            if kv_cache[i] is not None:
                existing_k, existing_v = kv_cache[i]
                v_scaled = v_pack * alpha
                new_k = torch.cat([k_pack, existing_k], dim=-2)
                new_v = torch.cat([v_scaled, existing_v], dim=-2)
                kv_cache[i] = (new_k, new_v)
            else:
                kv_cache[i] = (k_pack, v_pack * alpha)
        return kv_cache

    def save(self, path: str):
        """Save CoT Knowledge Pack to disk."""
        from safetensors.torch import save_file
        state = {}
        for i, (k, v) in enumerate(self.kv_caches):
            if k is not None:
                state[f"layer_{i}_k"] = k
                state[f"layer_{i}_v"] = v
        state["_metadata"] = torch.tensor([self.n_tokens], dtype=torch.int32)
        save_file(state, path)

    @classmethod
    def load(cls, path: str) -> "CoTKnowledgePack":
        """Load CoT Knowledge Pack from disk."""
        from safetensors import safe_open
        kv_caches = []
        with safe_open(path, framework="pt") as f:
            keys = sorted(f.keys())
            layer_keys = [k for k in keys if k.startswith("layer_")]
            n_layers = max(int(k.split("_")[1]) for k in layer_keys) + 1
            for i in range(n_layers):
                k_key, v_key = f"layer_{i}_k", f"layer_{i}_v"
                if k_key in keys and v_key in keys:
                    kv_caches.append((f.get_tensor(k_key), f.get_tensor(v_key)))
                else:
                    kv_caches.append((None, None))
        return cls(kv_caches, metadata={"loaded_from": path, "n_layers": n_layers})


def build_cot_packs_from_selfplay(log_dir: str, model, tokenizer,
                                   min_quality: float = 0.7,
                                   device: str = "cuda") -> list[CoTKnowledgePack]:
    """Build CoT Knowledge Packs from self-play logs.

    Reads self-play logs, extracts reasoning traces from successful attempts,
    filters by quality score, and returns a list of CoTKnowledgePack objects.
    """
    packs = []
    for fname in sorted(os.listdir(log_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(log_dir, fname)) as f:
            entries = json.load(f)
        if isinstance(entries, dict):
            entries = [entries]
        for entry in entries:
            quality = entry.get("quality", 0.0)
            if quality < min_quality:
                continue
            reasoning_text = entry.get("reasoning", entry.get("cot", ""))
            task_prompt = entry.get("prompt", entry.get("task", ""))
            if not reasoning_text or not task_prompt:
                continue
            pack = CoTKnowledgePack.from_cot(
                model, tokenizer, reasoning_text, task_prompt, device=device)
            pack.metadata["quality"] = quality
            pack.metadata["source"] = fname
            packs.append(pack)
    return packs
