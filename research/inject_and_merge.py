"""Inject-and-Merge — unified pipeline for passing new params into a target model.

Combines KeyStack knowledge injection with model merging into one self-
handling flow:

  1. Load target model checkpoint (the "main" model)
  2. Clone it as the injection base (same arch = same keys/shapes)
  3. Run knowledge injection keys on the clone (facts, context patches,
     self-play solutions, spectral injection, test-gated facts)
  4. Compute task vector: injected_clone - target  (= pure knowledge delta)
  5. Merge delta into target via task_arith / TIES / DARE / SLERP / SVD
  6. Save the merged model

Because the clone starts from the target checkpoint, the task vector
isolates exactly what the keys injected — no random-init noise, no
architecture mismatch. All keys modify existing weight matrices in-place
(same keys, same shapes), so every merge method in merge_models.py works.

Usage (Python API):
    from research.inject_and_merge import InjectMergePipeline, InjectConfig

    pipe = InjectMergePipeline(
        target_checkpoint="research/checkpoints/ForgeLM_V2_LFM25-1.2B.safetensors",
        config_name="forgelm_v3",
        merge_method="task_arith",
    )
    merged = pipe.run(
        inject_type="facts",
        facts=[("Paris", "capital of France"),
               ("Einstein", "physicist")],
    )
    pipe.save(merged, "research/checkpoints/merged.safetensors")

Usage (CLI):
    python -m research.inject_and_merge \\
        --target research/checkpoints/ForgeLM_V2_LFM25-1.2B.safetensors \\
        --config lfm25_1.2b \\
        --inject-type facts \\
        --merge-method task_arith \\
        --facts-file facts.json \\
        --out research/checkpoints/merged.safetensors
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")

import torch

from research.config import get_config, ModelConfig
from research.checkpoint_io import load_checkpoint, save_checkpoint
from research.merge_models import (
    _load_state_dict, _task_vectors,
    merge_task_arith, merge_ties, merge_dare, merge_slerp, merge_svd,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class InjectConfig:
    """Configuration for the inject-and-merge pipeline."""
    # Injection type: which knowledge key to run
    inject_type: str = "facts"
    # Merge method: how to combine the injected delta with the target
    merge_method: str = "task_arith"
    # Merge hyperparams
    merge_scale: float = 1.0          # task_arith / svd scale
    density: float = 0.5              # TIES density
    drop_rate: float = 0.1            # DARE drop rate
    rank_ratio: float = 0.5           # SVD rank ratio
    slerp_t: float = 0.5              # SLERP interpolation factor
    seed: int = 0                     # DARE seed
    # Injection hyperparams
    inject_alpha: float = 1.0         # patch / injection strength
    inject_layer: int = -1            # which layer to inject into (-1 = last)
    inject_layers: list[int] | None = None  # specific layers for patches
    # Device
    device: str = "cuda"


# ---------------------------------------------------------------------------
# Injection handlers — each wraps a knowledge key into a uniform interface
# ---------------------------------------------------------------------------

def _inject_facts(
    state: dict[str, torch.Tensor],
    model: torch.nn.Module,
    tokenizer,
    config: ModelConfig,
    inject_cfg: InjectConfig,
    facts: list[tuple[str, str]] | list[tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Inject fact pairs via FactInjectionKey (closed-form MLP write).

    Args:
        facts: list of (input_text, output_text) or (input_vec, output_vec).
    """
    from research.keys.knowledge.fact_injection_key import (
        FactInjectionKey, create_fact_from_text,
    )

    # Convert text facts to vectors if needed
    fact_pairs = []
    for fact in facts:
        if isinstance(fact[0], str):
            x, y = create_fact_from_text(model, tokenizer, fact[0], fact[1],
                                         device=inject_cfg.device)
            fact_pairs.append((x, y))
        else:
            fact_pairs.append(fact)

    key = FactInjectionKey(layer_idx=inject_cfg.inject_layer)
    result = key.forward({
        "state": state, "facts": fact_pairs,
        "n_layers": config.n_layers,
        "d_model": config.d_model,
        "d_ff": config.intermediate_size,
        "layer_idx": inject_cfg.inject_layer,
    })
    if not result.success:
        raise RuntimeError(f"Fact injection failed: {result.error}")
    return result.weights


def _inject_test_gated(
    state: dict[str, torch.Tensor],
    model: torch.nn.Module,
    tokenizer,
    config: ModelConfig,
    inject_cfg: InjectConfig,
    solutions: list[dict],
) -> dict[str, torch.Tensor]:
    """Inject test-verified solutions via TestGatedFactInjectionKey."""
    from research.keys.knowledge.test_gated_injection_key import (
        TestGatedFactInjectionKey,
    )

    key = TestGatedFactInjectionKey(layer_idx=inject_cfg.inject_layer)
    result = key.forward({
        "state": state,
        "verified_solutions": solutions,
        "n_layers": config.n_layers,
        "d_model": config.d_model,
        "d_ff": config.intermediate_size,
        "layer_idx": inject_cfg.inject_layer,
        "tokenizer": tokenizer,
    })
    if not result.success:
        raise RuntimeError(f"Test-gated injection failed: {result.error}")
    return result.weights


def _inject_context_patch(
    state: dict[str, torch.Tensor],
    model: torch.nn.Module,
    tokenizer,
    config: ModelConfig,
    inject_cfg: InjectConfig,
    context_text: str,
    query_text: str,
) -> dict[str, torch.Tensor]:
    """Inject context as rank-1 weight patches via ContextPatchKey."""
    from research.keys.knowledge.context_patch_key import (
        ContextPatchKey, extract_context_patches,
    )

    patches = extract_context_patches(
        model, tokenizer, context_text, query_text,
        device=inject_cfg.device, layers=inject_cfg.inject_layers,
    )
    key = ContextPatchKey(alpha=inject_cfg.inject_alpha,
                          layers=inject_cfg.inject_layers)
    result = key.forward({"state": state, "patches": patches})
    if not result.success:
        raise RuntimeError(f"Context patch failed: {result.error}")
    return result.weights


def _inject_selfplay_patch(
    state: dict[str, torch.Tensor],
    model: torch.nn.Module,
    tokenizer,
    config: ModelConfig,
    inject_cfg: InjectConfig,
    solutions: list[dict],
) -> dict[str, torch.Tensor]:
    """Inject self-play solutions as rank-1 patches via SelfPlayContextPatchKey."""
    from research.keys.knowledge.selfplay_context_patch_key import (
        SelfPlayContextPatchKey, extract_patch,
    )

    n_layers = config.n_layers
    layers = inject_cfg.inject_layers or list(range(n_layers))
    patches = []
    for sol in solutions:
        prompt = sol.get("prompt", "")
        solution = sol.get("solution", "")
        quality = sol.get("quality", 1.0)
        test_passed = sol.get("test_passed", False)
        if not prompt or not solution:
            continue
        u, v = extract_patch(model, tokenizer, prompt, solution,
                             device=inject_cfg.device)
        sign = (inject_cfg.inject_alpha * quality if test_passed
                else -inject_cfg.inject_alpha * 0.1)
        for layer_idx in layers:
            patches.append((layer_idx, "w_gate", v, u, sign))

    key = SelfPlayContextPatchKey(alpha=1.0, layers=layers)
    result = key.forward({"state": state, "patches": patches})
    if not result.success:
        raise RuntimeError(f"Self-play patch failed: {result.error}")
    return result.weights


def _inject_spectral(
    state: dict[str, torch.Tensor],
    model: torch.nn.Module,
    tokenizer,
    config: ModelConfig,
    inject_cfg: InjectConfig,
    facts: list[tuple[str, str]] | list[torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Inject facts via SpectralInjectionKey (SVD-space, avoids forgetting).

    Applies to the last layer's w_gate (or specified layer).
    """
    from research.keys.knowledge.spectral_injection_key import SpectralInjectionKey
    from research.keys.knowledge.fact_injection_key import create_fact_from_text

    layer = (config.n_layers - 1 if inject_cfg.inject_layer < 0
             else inject_cfg.inject_layer)
    key_name = f"blocks.{layer}.ffn.w_gate.weight"
    if key_name not in state:
        # Try MoE expert
        key_name = f"blocks.{layer}.ffn.experts.0.w_gate.weight"
    if key_name not in state:
        raise RuntimeError(f"No w_gate found at layer {layer}")

    # Convert text to fact embeddings
    fact_embs = []
    for fact in facts:
        if isinstance(fact, tuple) and isinstance(fact[0], str):
            x, _ = create_fact_from_text(model, tokenizer, fact[0], fact[1],
                                         device=inject_cfg.device)
            fact_embs.append(x)
        elif isinstance(fact, torch.Tensor):
            fact_embs.append(fact)
        elif isinstance(fact, tuple):
            fact_embs.append(fact[0])

    spectral_key = SpectralInjectionKey(
        gamma=1.0, alpha=inject_cfg.inject_alpha)
    W = state[key_name]
    W_new = spectral_key.inject_facts(
        W, torch.stack(fact_embs) if fact_embs else torch.tensor([]),
        mode="new_knowledge", alpha=inject_cfg.inject_alpha)
    state[key_name] = W_new.to(state[key_name].dtype)
    return state


# Registry of injection handlers
INJECT_HANDLERS = {
    "facts": _inject_facts,
    "test_gated": _inject_test_gated,
    "context_patch": _inject_context_patch,
    "selfplay_patch": _inject_selfplay_patch,
    "spectral": _inject_spectral,
}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class InjectMergePipeline:
    """Unified inject-and-merge pipeline.

    Loads a target checkpoint, clones it, runs a knowledge injection key
    on the clone, computes the task vector (injected - target), and merges
    the delta back into the target.

    The clone starts from the target checkpoint, so the task vector is a
    pure knowledge delta — no random-init noise, no architecture mismatch.
    All injection keys modify existing weight matrices in-place (same keys,
    same shapes), so every merge method works.
    """

    def __init__(self, target_checkpoint: str, config_name: str = "forgelm_v3",
                 merge_method: str = "task_arith", device: str = "cuda"):
        self.target_checkpoint = target_checkpoint
        self.config_name = config_name
        self.config = get_config(config_name)
        self.merge_method = merge_method
        self.device = device if torch.cuda.is_available() else "cpu"

        # Lazy-loaded
        self._target_state: dict[str, torch.Tensor] | None = None
        self._model: torch.nn.Module | None = None
        self._tokenizer = None

    def _load_target(self) -> dict[str, torch.Tensor]:
        """Load the target checkpoint state dict (cached)."""
        if self._target_state is None:
            print(f"[InjectMerge] Loading target: {self.target_checkpoint}")
            self._target_state = _load_state_dict(self.target_checkpoint)
        return self._target_state

    def _get_model(self) -> torch.nn.Module:
        """Build and load the model for injection keys that need forward passes."""
        if self._model is None:
            from research.model_loader import ModelLoader
            print(f"[InjectMerge] Building model ({self.config_name})...")
            cfg = self.config
            cfg.device = self.device
            self._model = ModelLoader.build_model(
                cfg, checkpoint_path=self.target_checkpoint).to(self.device).eval()
        return self._model

    def _get_tokenizer(self):
        """Load the tokenizer (cached)."""
        if self._tokenizer is None:
            from research.tokenizer_cache import get_tokenizer
            self._tokenizer = get_tokenizer()
        return self._tokenizer

    def inject(self, inject_type: str, inject_cfg: InjectConfig,
               **inject_kwargs) -> dict[str, torch.Tensor]:
        """Run an injection key on a clone of the target checkpoint.

        Args:
            inject_type: one of "facts", "test_gated", "context_patch",
                "selfplay_patch", "spectral".
            inject_cfg: injection + merge hyperparams.
            **inject_kwargs: handler-specific data (facts, solutions, etc.)

        Returns:
            Injected state dict (same keys/shapes as target).
        """
        if inject_type not in INJECT_HANDLERS:
            raise ValueError(
                f"Unknown inject_type: {inject_type}. "
                f"Choose from: {list(INJECT_HANDLERS.keys())}")

        # Clone the target state dict (deep copy of tensors)
        target = self._load_target()
        state = {k: v.clone() for k, v in target.items()}
        print(f"[InjectMerge] Cloned target ({len(state)} tensors)")

        # Some injection keys need the model + tokenizer for forward passes
        model = None
        tokenizer = None
        if inject_type in ("facts", "context_patch", "selfplay_patch", "spectral"):
            model = self._get_model()
            tokenizer = self._get_tokenizer()

        handler = INJECT_HANDLERS[inject_type]
        print(f"[InjectMerge] Running injection: {inject_type}")
        injected = handler(state, model, tokenizer, self.config,
                           inject_cfg, **inject_kwargs)
        print(f"[InjectMerge] Injection complete ({len(injected)} tensors)")
        return injected

    def merge(self, injected: dict[str, torch.Tensor],
              inject_cfg: InjectConfig) -> dict[str, torch.Tensor]:
        """Merge the injected state dict into the target.

        Computes task_vector = injected - target, then applies the selected
        merge method to combine the delta with the target.

        Args:
            injected: the injected state dict (same keys/shapes as target).
            inject_cfg: merge hyperparams.

        Returns:
            Merged state dict.
        """
        target = self._load_target()

        # Verify shape compatibility
        mismatched = [k for k in target if k in injected
                      and target[k].shape != injected[k].shape]
        if mismatched:
            raise RuntimeError(
                f"Shape mismatch on {len(mismatched)} keys: {mismatched[:5]}")

        method = self.merge_method
        print(f"[InjectMerge] Merging via: {method}")

        if method == "task_arith":
            # task_vector = injected - target, then target + scale * task_vector
            tv = _task_vectors(injected, target)
            merged = merge_task_arith(target, [tv], [inject_cfg.merge_scale])

        elif method == "ties":
            tv = _task_vectors(injected, target)
            merged = merge_ties(target, [tv], density=inject_cfg.density)

        elif method == "dare":
            tv = _task_vectors(injected, target)
            merged = merge_dare(target, [tv], drop_rate=inject_cfg.drop_rate,
                                seed=inject_cfg.seed)

        elif method == "svd":
            tv = _task_vectors(injected, target)
            merged = merge_svd(target, [tv], rank_ratio=inject_cfg.rank_ratio,
                               scales=[inject_cfg.merge_scale])

        elif method == "slerp":
            # SLERP between target and injected directly
            merged = merge_slerp(target, injected, t=inject_cfg.slerp_t)

        elif method == "linear":
            # Simple weighted average: (1-t)*target + t*injected
            t = inject_cfg.slerp_t
            merged = {}
            for k in target:
                if k in injected:
                    merged[k] = ((1 - t) * target[k].float() +
                                 t * injected[k].float()).to(target[k].dtype)
                else:
                    merged[k] = target[k].clone()

        else:
            raise ValueError(f"Unknown merge_method: {method}")

        print(f"[InjectMerge] Merge complete ({len(merged)} tensors)")
        return merged

    def run(self, inject_type: str, inject_cfg: InjectConfig | None = None,
            **inject_kwargs) -> dict[str, torch.Tensor]:
        """Full pipeline: inject on clone, then merge into target.

        Args:
            inject_type: injection key type.
            inject_cfg: config (defaults to InjectConfig with pipeline defaults).
            **inject_kwargs: handler-specific data.

        Returns:
            Merged state dict.
        """
        if inject_cfg is None:
            inject_cfg = InjectConfig(
                inject_type=inject_type, merge_method=self.merge_method)

        injected = self.inject(inject_type, inject_cfg, **inject_kwargs)
        merged = self.merge(injected, inject_cfg)
        return merged

    def save(self, state: dict[str, torch.Tensor], path: str) -> str:
        """Save a merged state dict."""
        out = path
        if not out.endswith(".safetensors") and not out.endswith(".pt"):
            out = out + ".safetensors"
        save_checkpoint(state, out)
        print(f"[InjectMerge] Saved to {out}")
        return out

    def cleanup(self):
        """Free model and tokenizer from VRAM."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_inject_data(args) -> dict:
    """Load injection data from files based on inject_type."""
    data = {}

    if args.inject_type == "facts":
        if args.facts_file:
            with open(args.facts_file, encoding="utf-8") as f:
                data["facts"] = [tuple(pair) for pair in json.load(f)]
        elif args.fact:
            # Single fact from CLI: --fact "Paris" "capital of France"
            data["facts"] = [tuple(args.fact)]
        else:
            p.error("--facts-file or --fact required for facts injection")

    elif args.inject_type == "test_gated":
        if not args.solutions_file:
            p.error("--solutions-file required for test_gated injection")
        with open(args.solutions_file, encoding="utf-8") as f:
            data["solutions"] = json.load(f)

    elif args.inject_type == "context_patch":
        if not args.context_text or not args.query_text:
            p.error("--context-text and --query-text required for context_patch")
        data["context_text"] = args.context_text
        data["query_text"] = args.query_text

    elif args.inject_type == "selfplay_patch":
        if not args.solutions_file:
            p.error("--solutions-file required for selfplay_patch injection")
        with open(args.solutions_file, encoding="utf-8") as f:
            data["solutions"] = json.load(f)

    elif args.inject_type == "spectral":
        if args.facts_file:
            with open(args.facts_file, encoding="utf-8") as f:
                data["facts"] = [tuple(pair) for pair in json.load(f)]
        elif args.fact:
            data["facts"] = [tuple(args.fact)]
        else:
            p.error("--facts-file or --fact required for spectral injection")

    return data


def main():
    global p
    p = argparse.ArgumentParser(
        description="Inject-and-Merge: pass new params into a target model via KeyStack + merging")
    p.add_argument("--target", required=True,
                   help="Target model checkpoint (the main model)")
    p.add_argument("--config", default="forgelm_v3",
                   help="Model config name (lfm25_1.2b, lfm25_tiny, forgelm_v3, forgelm_v4)")
    p.add_argument("--inject-type", required=True,
                   choices=list(INJECT_HANDLERS.keys()),
                   help="Which knowledge injection key to run")
    p.add_argument("--merge-method", default="task_arith",
                   choices=["task_arith", "ties", "dare", "svd", "slerp", "linear"],
                   help="How to merge the injected delta into the target")
    p.add_argument("--out", required=True, help="Output checkpoint path")

    # Injection data sources
    p.add_argument("--facts-file", default=None,
                   help="JSON file: [[\"input\", \"output\"], ...]")
    p.add_argument("--fact", nargs=2, default=None, metavar=("INPUT", "OUTPUT"),
                   help="Single fact pair")
    p.add_argument("--solutions-file", default=None,
                   help="JSON file: [{prompt, solution, quality, test_passed}, ...]")
    p.add_argument("--context-text", default=None, help="Context for context_patch")
    p.add_argument("--query-text", default=None, help="Query for context_patch")

    # Injection hyperparams
    p.add_argument("--inject-alpha", type=float, default=1.0,
                   help="Injection strength (patch alpha / spectral alpha)")
    p.add_argument("--inject-layer", type=int, default=-1,
                   help="Layer to inject into (-1 = last)")
    p.add_argument("--inject-layers", nargs="*", type=int, default=None,
                   help="Specific layers for patches (default: all)")

    # Merge hyperparams
    p.add_argument("--merge-scale", type=float, default=1.0,
                   help="task_arith/svd: delta scaling weight")
    p.add_argument("--density", type=float, default=0.5, help="TIES density")
    p.add_argument("--drop-rate", type=float, default=0.1, help="DARE drop rate")
    p.add_argument("--rank-ratio", type=float, default=0.5, help="SVD rank ratio")
    p.add_argument("--slerp-t", type=float, default=0.5,
                   help="SLERP/linear interpolation factor")
    p.add_argument("--seed", type=int, default=0, help="DARE seed")

    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    inject_data = _load_inject_data(args)

    inject_cfg = InjectConfig(
        inject_type=args.inject_type,
        merge_method=args.merge_method,
        merge_scale=args.merge_scale,
        density=args.density,
        drop_rate=args.drop_rate,
        rank_ratio=args.rank_ratio,
        slerp_t=args.slerp_t,
        seed=args.seed,
        inject_alpha=args.inject_alpha,
        inject_layer=args.inject_layer,
        inject_layers=args.inject_layers,
        device=args.device,
    )

    pipe = InjectMergePipeline(
        target_checkpoint=args.target,
        config_name=args.config,
        merge_method=args.merge_method,
        device=args.device,
    )

    try:
        merged = pipe.run(args.inject_type, inject_cfg, **inject_data)
        pipe.save(merged, args.out)
    finally:
        pipe.cleanup()


if __name__ == "__main__":
    main()
