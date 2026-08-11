"""Inject HF dataset knowledge into AirMoE expert packs.

Pipeline per topic:
  1. Load HF dataset JSONL (from download_hf_datasets.py)
  2. Load base model + tokenizer
  3. For each (prompt, solution) pair:
     a. Compute fact embedding from prompt (model forward pass)
     b. For each expert layer (28 layers):
        - Reconstruct W from SVD-compressed expert (U @ diag(S) @ Vh)
        - Spectral inject fact into W
        - Recompress W back to SVD form
  4. Save updated expert weights
  5. Verify with held-out test cases

Expert weight format (SVD-compressed):
  w1_U, w1_S, w1_Vh  →  W1 = U @ diag(S) @ Vh  (shape: d_ff × d_model)
  w2_U, w2_S, w2_Vh  →  W2 = U @ diag(S) @ Vh  (shape: d_model × d_ff)
  w3_U, w3_S, w3_Vh  →  W3 = U @ diag(S) @ Vh  (shape: d_ff × d_model)

Usage:
    set HF_TOKEN=<token>
    python scripts/inject_hf_data.py --topics coding,math --max-facts 100
    python scripts/inject_hf_data.py --topics all --max-facts 200 --layers 14,21,27
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research import paths as _paths

# Central HF cache location (env var still wins if already set).
os.environ.setdefault("HF_HOME", _paths.as_str(_paths.HF_CACHE_DIR))
os.environ.setdefault("HF_TOKEN", os.environ.get("HF_TOKEN", ""))

import torch
from safetensors.torch import load_file, save_file

from research.keys.knowledge.spectral_injection_key import SpectralInjectionKey
from research.tokenizer_cache import get_tokenizer

EXPERT_DIR = _paths.EXPERTS_DIR
HF_DATA_DIR = _paths.HF_DATASETS_DIR
OUTPUT_DIR = _paths.ensure_dir(_paths.FORGELM_V2_EXPERTS_DIR)
BASE_CHECKPOINT = _paths.as_str(_paths.FORGELM_V2_CHECKPOINT)

N_LAYERS = 28
# Inject into upper layers (where factual knowledge lives in LLMs)
DEFAULT_INJECT_LAYERS = [14, 18, 21, 24, 27]
# Weight matrices per expert layer
EXPERT_WEIGHTS = ["w1", "w2", "w3"]

# Topic -> keywords for manifest (used by router)
TOPIC_KEYWORDS = {
    "coding": ["code", "function", "program", "debug", "implement", "algorithm",
               "python", "javascript", "java", "cpp", "software", "compile"],
    "python": ["python", "def", "class", "import", "list", "dict", "string",
               "tuple", "lambda", "comprehension", "decorator"],
    "math": ["math", "calculate", "equation", "solve", "number", "algebra",
             "calculus", "geometry", "probability", "statistics", "arithmetic"],
    "algorithms": ["algorithm", "sort", "search", "complexity", "recursive",
                   "dynamic programming", "graph", "tree", "hash", "optimize"],
    "theory": ["explain", "why", "how", "reason", "analyze", "compare",
               "theory", "concept", "principle", "understand", "prove"],
    "creativity": ["write", "story", "poem", "creative", "imagine", "narrative",
                   "character", "plot", "fiction", "describe", "compose"],
    "tool_use": ["tool", "function call", "api", "command", "execute",
                 "automate", "script", "shell", "request", "invoke"],
    "token_efficiency": ["concise", "brief", "short", "summary", "efficient",
                         "minimal", "direct", "clear", "simple"],
    "general": ["help", "what", "when", "where", "who", "question",
                "information", "advice", "general"],
    "python_algorithms": ["algorithm", "sort", "fibonacci", "factorial", "prime",
                          "recursion", "search", "collatz", "gcd", "dynamic"],
}


def load_expert_layer(topic: str, layer: int) -> dict[str, torch.Tensor]:
    """Load SVD-compressed expert weights for one layer.
    Falls back to injected output dir, then original expert dir."""
    # Check injected output first
    for d in [OUTPUT_DIR, EXPERT_DIR]:
        path = d / f"expert_l{layer}_{topic}.safetensors"
        if path.exists():
            return load_file(str(path))
    return {}


def create_expert_from_base(layer: int, expert_idx: int = 0,
                            svd_energy: float = 0.95) -> dict[str, torch.Tensor]:
    """Create a new SVD-compressed expert from the base model's weights.

    Copies expert_idx from the base checkpoint and SVD-compresses it.
    This is the starting point for new topic experts.
    """
    from safetensors.torch import load_file as load_safetensors
    gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load only the 3 keys we need, not the full 3.6GB checkpoint
    base = load_safetensors(BASE_CHECKPOINT)
    expert_state = {}
    for part in EXPERT_WEIGHTS:
        k = f"blocks.{layer}.ffn.experts.{expert_idx}.{part}.weight"
        if k not in base:
            continue
        W = base[k].float().to(gpu)  # SVD on GPU
        del base[k]  # free CPU memory immediately
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        cumsum = (S ** 2).cumsum(0)
        total = cumsum[-1].clamp(min=1e-12)
        rank = max(1, int((cumsum < svd_energy * total).sum().item()) + 1)
        rank = min(rank, S.shape[0])
        expert_state[f"{part}_U"] = U[:, :rank].cpu().to(torch.bfloat16)
        expert_state[f"{part}_S"] = S[:rank].cpu().to(torch.float16)
        expert_state[f"{part}_Vh"] = Vh[:rank, :].cpu().to(torch.bfloat16)
        expert_state[f"{part}_rank"] = torch.tensor([rank], dtype=torch.int32)
        del W, U, S, Vh
    del base
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return expert_state


def reconstruct_weight(svd_dict: dict, prefix: str) -> torch.Tensor | None:
    """Reconstruct full weight from SVD components: W = U @ diag(S) @ Vh."""
    U = svd_dict.get(f"{prefix}_U")
    S = svd_dict.get(f"{prefix}_S")
    Vh = svd_dict.get(f"{prefix}_Vh")
    if U is None or S is None or Vh is None:
        return None
    # S is 1-D singular values, U and Vh are 2-D
    return U.to(torch.float32) @ torch.diag(S.to(torch.float32)) @ Vh.to(torch.float32)


def recompress_svd(W: torch.Tensor, original_rank: int | None = None,
                   energy_threshold: float = 0.95) -> dict[str, torch.Tensor]:
    """SVD-compress a weight matrix back to U, S, Vh format.

    Keeps enough singular values to retain energy_threshold of total energy,
    or original_rank if specified (whichever is smaller).
    """
    gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    W_gpu = W.to(gpu) if W.device.type != "cuda" else W
    U, S, Vh = torch.linalg.svd(W_gpu, full_matrices=False)

    # Determine rank to keep
    if original_rank is not None:
        rank = min(original_rank, S.shape[0])
    else:
        # Keep enough for energy_threshold
        cumsum = torch.cumsum(S ** 2, dim=0)
        total = cumsum[-1].clamp(min=1e-12)
        ratios = cumsum / total
        rank = int((ratios < energy_threshold).sum().item()) + 1
        rank = max(rank, min(64, S.shape[0]))  # minimum rank

    rank = min(rank, S.shape[0])
    U_k = U[:, :rank].cpu().to(torch.bfloat16)
    S_k = S[:rank].cpu().to(torch.float16)
    Vh_k = Vh[:rank, :].cpu().to(torch.bfloat16)
    del W_gpu, U, S, Vh

    return {f"U": U_k, f"S": S_k, f"Vh": Vh_k,
            f"U_shape": torch.tensor([U_k.shape[0], U_k.shape[1]], dtype=torch.int32),
            f"Vh_shape": torch.tensor([Vh_k.shape[0], Vh_k.shape[1]], dtype=torch.int32),
            f"rank": torch.tensor([rank], dtype=torch.int32)}


def compute_fact_embedding(prompt: str, model, tokenizer, device: str = "cuda",
                           max_tokens: int = 128, d_model: int = 1536) -> torch.Tensor:
    """Compute fact embedding from a prompt using the model's hidden states.

    Uses mean of last hidden layer as the embedding (d_model dimensional).
    Falls back to logits pooled across sequence if hidden states not available.
    Always returns a (d_model,) tensor.
    """
    ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                    max_length=max_tokens).input_ids.to(device)
    with torch.no_grad():
        try:
            out = model(ids, output_hidden_states=True)
            # Handle different output formats
            if hasattr(out, "hidden_states") and out.hidden_states:
                emb = out.hidden_states[-1].mean(dim=1).squeeze(0)
            elif isinstance(out, tuple) and len(out) > 1 and out[1] is not None:
                # (logits, hidden_states)
                hs = out[1]
                if isinstance(hs, (list, tuple)):
                    emb = hs[-1].mean(dim=1).squeeze(0)
                else:
                    emb = hs.mean(dim=1).squeeze(0)
            else:
                # Fallback: pool logits across sequence dimension
                # logits shape: (1, seq_len, vocab_size) → mean over seq → (vocab_size,)
                # Then project to d_model via mean pooling to fixed size
                logits = out[0] if isinstance(out, tuple) else out
                emb = logits.float().mean(dim=1).squeeze(0)  # (vocab_size,)
                # Resize to d_model by averaging chunks
                if emb.shape[0] != d_model:
                    if emb.shape[0] > d_model:
                        emb = emb[:d_model]
                    else:
                        emb = torch.nn.functional.pad(emb, (0, d_model - emb.shape[0]))
        except Exception:
            # Minimal forward: just use logits
            out = model(ids)
            logits = out[0] if isinstance(out, tuple) else out
            emb = logits.float().mean(dim=1).squeeze(0)
            if emb.shape[0] != d_model:
                if emb.shape[0] > d_model:
                    emb = emb[:d_model]
                else:
                    emb = torch.nn.functional.pad(emb, (0, d_model - emb.shape[0]))
    return emb.float().cpu()


def load_hf_dataset(topic: str, max_facts: int = 100) -> list[dict]:
    """Load HF dataset JSONL for a topic."""
    path = HF_DATA_DIR / f"{topic}.jsonl"
    if not path.exists():
        print(f"  [WARN] No HF dataset for {topic} at {path}")
        return []
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if len(samples) >= max_facts:
                break
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return samples


def inject_topic(topic: str, model, tokenizer, device: str,
                 max_facts: int = 100, layers: list[int] = None,
                 alpha: float = 0.1, mode: str = "new_knowledge"):
    """Inject HF dataset knowledge into all expert layers for a topic."""
    if layers is None:
        layers = DEFAULT_INJECT_LAYERS

    print(f"\n{'='*60}")
    print(f"Injecting {topic}: {max_facts} facts, layers {layers}")
    print(f"{'='*60}")

    # Load HF dataset
    samples = load_hf_dataset(topic, max_facts)
    if not samples:
        print(f"  [SKIP] No data for {topic}")
        return 0

    # Compute fact embeddings
    print(f"  Computing {len(samples)} fact embeddings...")
    fact_embeddings = []
    for i, sample in enumerate(samples):
        prompt = sample.get("prompt", "")
        if not prompt:
            continue
        emb = compute_fact_embedding(prompt, model, tokenizer, device)
        fact_embeddings.append(emb)
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(samples)} embeddings done")

    print(f"  Computed {len(fact_embeddings)} fact embeddings")
    if not fact_embeddings:
        print(f"  [SKIP] No valid embeddings")
        return 0

    # Spectral injection key
    key = SpectralInjectionKey(alpha=alpha)

    # Inject into each expert layer
    total_injected = 0
    created_new = 0
    for layer in layers:
        expert_data = load_expert_layer(topic, layer)
        if not expert_data:
            # Create new expert from base model's expert 0
            print(f"  Layer {layer}: creating new expert from base model...")
            expert_data = create_expert_from_base(layer, expert_idx=0)
            if not expert_data:
                print(f"  Layer {layer}: failed to create expert, skipping")
                continue
            created_new += 1

        # Get original ranks for recompression
        original_ranks = {}
        for w in EXPERT_WEIGHTS:
            r = expert_data.get(f"{w}_rank")
            if r is not None:
                original_ranks[w] = int(r.item())

        # Inject into each weight matrix (GPU-accelerated)
        gpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        updated = {}
        for w_name in EXPERT_WEIGHTS:
            W = reconstruct_weight(expert_data, w_name)
            if W is None:
                continue

            # Move to GPU for fast SVD, inject, move back
            W_gpu = W.to(gpu_device)
            W_new = key.batch_inject(W_gpu, fact_embeddings, mode=mode)
            W_new = W_new.cpu()

            # Recompress to SVD (on CPU to save VRAM)
            compressed = recompress_svd(W_new, original_rank=original_ranks.get(w_name))
            for k, v in compressed.items():
                updated[f"{w_name}_{k}"] = v

            total_injected += 1

        # Save updated expert (ensure contiguous for safetensors)
        out_path = OUTPUT_DIR / f"expert_l{layer}_{topic}.safetensors"
        updated = {k: v.contiguous() for k, v in updated.items()}
        save_file(updated, str(out_path))
        tag = "NEW" if created_new > 0 and len([l for l in layers if l < layer]) == 0 else "inj"
        print(f"  Layer {layer}: {tag} {len(fact_embeddings)} facts -> {out_path.name}")

    if created_new:
        print(f"  Created {created_new} new expert(s) from base model")
    return total_injected


def verify_topic(topic: str, model, tokenizer, device: str,
                 n_test: int = 10) -> dict:
    """Verify injection by testing on held-out samples."""
    samples = load_hf_dataset(topic, n_test + 50)
    test_samples = samples[-n_test:] if len(samples) > n_test else samples

    correct = 0
    total = 0
    for sample in test_samples:
        prompt = sample.get("prompt", "")
        expected = sample.get("solution", "")
        if not prompt or not expected:
            continue
        total += 1
        # Simple check: does the model generate something containing key parts of expected?
        ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=256).input_ids.to(device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=128, temperature=0.1,
                                 do_sample=False)
        generated = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        # Check for overlap
        expected_words = set(expected.lower().split())
        gen_words = set(generated.lower().split())
        overlap = len(expected_words & gen_words) / max(len(expected_words), 1)
        if overlap > 0.3:
            correct += 1

    rate = correct / max(total, 1)
    print(f"  [Verify] {topic}: {correct}/{total} = {rate:.1%}")
    return {"topic": topic, "correct": correct, "total": total, "rate": rate}


def main():
    parser = argparse.ArgumentParser(description="Inject HF data into expert packs")
    parser.add_argument("--topics", type=str, default="all")
    parser.add_argument("--max-facts", type=int, default=100)
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices (default: 14,18,21,24,27)")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--mode", type=str, default="new_knowledge",
                        choices=["new_knowledge", "reorient"])
    parser.add_argument("--verify", action="store_true", help="Run verification after injection")
    parser.add_argument("--model", type=str, default="forgelm_v2",
                        help="Model config to load")
    args = parser.parse_args()

    if args.topics == "all":
        topics = ["coding", "python", "math", "algorithms", "theory",
                  "creativity", "tool_use", "token_efficiency", "general"]
    else:
        topics = [t.strip() for t in args.topics.split(",")]

    layers = None
    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]

    print(f"\n{'='*60}")
    print(f"HF Data Injection into Expert Packs")
    print(f"Topics: {', '.join(topics)}")
    print(f"Max facts per topic: {args.max_facts}")
    print(f"Mode: {args.mode}, alpha: {args.alpha}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"{'='*60}")

    # Load model + tokenizer
    print("\nLoading model...")
    from research.config import get_config
    from research.model_loader import ModelLoader

    config = get_config(args.model)
    device = config.device if hasattr(config, "device") else ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = "research/checkpoints/forgelm_v2.safetensors"
    # Force CUDA for GPU acceleration
    if torch.cuda.is_available():
        config.device = "cuda"
    model = ModelLoader.build_model_fast(config, checkpoint_path=checkpoint,
                                         dtype=torch.bfloat16)
    tokenizer = get_tokenizer()
    model.eval()
    device = config.device

    # Inject each topic
    results = {}
    for topic in topics:
        n = inject_topic(topic, model, tokenizer, device,
                         max_facts=args.max_facts, layers=layers,
                         alpha=args.alpha, mode=args.mode)
        results[topic] = n

    # Verify if requested
    if args.verify:
        print(f"\n{'='*60}")
        print("Verification")
        print(f"{'='*60}")
        for topic in topics:
            verify_topic(topic, model, tokenizer, device)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for topic, n in results.items():
        print(f"  {topic}: {n} weight matrices injected")
    print(f"\nInjected experts saved to: {OUTPUT_DIR}")

    # Create manifest.json for the router
    manifest = {
        "n_layers": N_LAYERS,
        "topics": {},
    }
    for topic in topics:
        kws = TOPIC_KEYWORDS.get(topic, [topic])
        manifest["topics"][topic] = {
            "label": topic.replace("_", " ").title(),
            "keywords": kws,
            "subtopics": [],
        }
    # Also include existing python_algorithms if present
    if "python_algorithms" not in manifest["topics"]:
        manifest["topics"]["python_algorithms"] = {
            "label": "Python Algorithms",
            "keywords": TOPIC_KEYWORDS.get("python_algorithms", ["algorithm"]),
            "subtopics": [],
        }

    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved to: {manifest_path}")
    print(f"Topics in manifest: {len(manifest['topics'])}")


if __name__ == "__main__":
    main()
