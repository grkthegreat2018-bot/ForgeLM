"""Convert ForgeLM V2 -> V4 AirMoE format.

Re-runnable: updates base model + seed experts from V2, preserves trained experts.

Usage:
    python convert_to_v4.py
    python convert_to_v4.py --src research/checkpoints/forgelm_v2.safetensors --out research/checkpoints/forgelm_v4
    python convert_to_v4.py --svd-energy 0.95 --no-int4-base
    python convert_to_v4.py --preserve-trained   (default: True)
"""
import os
import sys
import time
import json
import hashlib
import argparse
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

N_LAYERS = 28
N_EXPERTS = 4

# Seed topics: original V2 experts mapped to topic names
SEED_MAP = {
    0: "math_arithmetic",
    1: "python_general",
    2: "reasoning_general",
    3: "general",
}

# Full topic library (keywords for router)
TOPIC_LIBRARY = {
    "python_general":      {"label": "Python General",      "keywords": ["python", "def", "print", "list", "dict", "tuple", "set", "loop", "for", "while", "function"]},
    "python_math":         {"label": "Python Math",         "keywords": ["fibonacci", "factorial", "prime", "gcd", "sqrt", "matrix", "sum", "average", "quadratic", "polynomial"]},
    "python_strings":      {"label": "Python Strings",      "keywords": ["string", "reverse", "palindrome", "regex", "substring", "concat", "split", "join", "replace", "uppercase"]},
    "python_algorithms":   {"label": "Python Algorithms",   "keywords": ["sort", "search", "binary search", "recursive", "dynamic programming", "tree", "graph", "bfs", "dfs"]},
    "python_oop":          {"label": "Python OOP",          "keywords": ["class", "object", "inheritance", "method", "constructor", "self.", "__init__"]},
    "python_file_io":      {"label": "Python File I/O",     "keywords": ["file", "open", "read", "write", "csv", "json", "path", "directory"]},
    "math_arithmetic":     {"label": "Arithmetic",          "keywords": ["add", "subtract", "multiply", "divide", "sum", "product", "average", "remainder", "modulo"]},
    "math_algebra":        {"label": "Algebra",             "keywords": ["solve", "equation", "algebra", "variable", "linear", "polynomial", "factor"]},
    "math_geometry":       {"label": "Geometry",            "keywords": ["triangle", "circle", "angle", "area", "perimeter", "volume", "geometry", "polygon"]},
    "math_probability":    {"label": "Probability",         "keywords": ["probability", "combinatoric", "permutation", "combination", "dice", "coin", "bayes"]},
    "math_theory":         {"label": "Math Theory",         "keywords": ["proof", "theorem", "lemma", "corollary", "axiom", "mathematical proof"]},
    "reasoning_general":   {"label": "General Reasoning",   "keywords": ["reason", "explain", "why", "how", "analyze", "deduce", "infer", "think"]},
    "logic":               {"label": "Logic",               "keywords": ["logic", "syllogism", "deductive", "premise", "conclusion", "inference", "valid", "sound"]},
    "general":             {"label": "General Knowledge",   "keywords": ["what", "how", "why", "describe", "explain", "science", "history", "concept", "define"]},
}


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()[:16]


def _quantize_int4(w: torch.Tensor, group_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-group symmetric int4 quantization (stored as int8)."""
    original_shape = w.shape
    n_elements = w.numel()
    if n_elements % group_size != 0:
        pad_size = group_size - (n_elements % group_size)
        w = w.reshape(-1)
        w = torch.cat([w, torch.zeros(pad_size, dtype=w.dtype, device=w.device)])
    w = w.reshape(-1, group_size)
    max_abs = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = max_abs / 7.0
    q = (w / scale).round().clamp(-8, 7).to(torch.int8)
    scale = scale.squeeze(-1).to(torch.float16)
    return q, scale


def compress_expert_svd(w: torch.Tensor, svd_energy: float = 0.9) -> Dict[str, torch.Tensor]:
    """SVD compress an expert weight matrix."""
    U, S, Vh = torch.linalg.svd(w.float(), full_matrices=False)
    cumsum = (S ** 2).cumsum(0)
    total = cumsum[-1]
    rank = max(1, (cumsum < svd_energy * total).sum().item() + 1)
    return {
        "U": U[:, :rank].contiguous().to(torch.bfloat16),
        "U_shape": torch.tensor(U[:, :rank].shape, dtype=torch.int32),
        "S": S[:rank].to(torch.float16),
        "Vh": Vh[:rank, :].contiguous().to(torch.bfloat16),
        "Vh_shape": torch.tensor(Vh[:rank, :].shape, dtype=torch.int32),
        "rank": torch.tensor([rank], dtype=torch.int32),
    }


def quantize_base_int4(state: Dict[str, torch.Tensor], group_size: int = 128) -> Dict[str, torch.Tensor]:
    """Apply Hadamard rotation + int4 to base model weights (not experts)."""
    from research.keys.spinquant_key import hadamard_matrix

    quantized = {}
    skip_keywords = ["embed", "head", "norm", "router", "rotor", "dwa",
                     "_runtime", "mtp", "value_residual", "bias"]
    for name, tensor in state.items():
        if tensor.dim() != 2 or tensor.numel() < 10000:
            quantized[name] = tensor
            continue
        if any(s in name for s in skip_keywords):
            quantized[name] = tensor
            continue
        if "experts." in name:
            continue
        w = tensor.clone()
        hdim = w.shape[0]
        if hdim & (hdim - 1) == 0:
            H = hadamard_matrix(hdim).to(w.dtype).to(w.device)
            w = (H.float() @ w.float()).to(w.dtype)
        q, scale = _quantize_int4(w, group_size)
        quantized[f"{name}__q"] = q
        quantized[f"{name}__scale"] = scale
    return quantized


def find_trained_experts(experts_dir: Path) -> Set[str]:
    """Find trained expert files (hidden files starting with .trained_)."""
    trained = set()
    if not experts_dir.exists():
        return trained
    for f in experts_dir.iterdir():
        if f.name.startswith(".trained_"):
            # .trained_python_algorithms -> python_algorithms
            trained.add(f.name.replace(".trained_", ""))
    return trained


def list_existing_expert_topics(experts_dir: Path) -> Dict[str, List[int]]:
    """Scan existing expert files and return {topic: [layers]}."""
    import re
    topics = {}
    if not experts_dir.exists():
        return topics
    for f in experts_dir.iterdir():
        if f.name.startswith("."):
            continue
        m = re.match(r'expert_l(\d+)_(.+)\.safetensors$', f.name)
        if m:
            layer = int(m.group(1))
            topic = m.group(2)
            if topic not in topics:
                topics[topic] = []
            topics[topic].append(layer)
    return topics


def convert(src: str, out_dir: str, svd_energy: float = 0.9,
            int4_base: bool = True, preserve_trained: bool = True,
            skip_existing: bool = True):
    """Convert V2 checkpoint to V4 AirMoE format."""
    from safetensors.torch import save_file, load_file
    from safetensors import safe_open

    out = Path(out_dir)
    experts_dir = out / "experts"
    out.mkdir(parents=True, exist_ok=True)
    experts_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Convert V2 -> V4 AirMoE")
    print(f"  Source: {src}")
    print(f"  Output: {out_dir}")
    print(f"  SVD energy: {svd_energy}, int4 base: {int4_base}")
    print("=" * 70)

    # --- Phase 1: Load V2 state dict ---
    print(f"\n[1/5] Loading V2 checkpoint...")
    t0 = time.time()
    state = load_file(src)
    print(f"  Loaded {len(state)} tensors in {time.time()-t0:.1f}s")

    # --- Phase 2: Separate base from experts ---
    print(f"\n[2/5] Separating base weights from experts...")
    base_state = {}
    expert_state = {}  # {layer: {expert_idx: {w1, w2, w3}}}

    for name, tensor in state.items():
        if "ffn.experts." in name:
            parts = name.split(".")
            layer = int(parts[1])
            ei = int(parts[4])
            part = parts[5]  # w1, w2, w3
            expert_state.setdefault(layer, {}).setdefault(ei, {})[part] = tensor.clone()
        else:
            base_state[name] = tensor.clone()

    n_expert_tensors = sum(len(v) for layer in expert_state.values() for v in layer.values())
    print(f"  Base tensors: {len(base_state)}")
    print(f"  Expert tensors: {n_expert_tensors} ({len(expert_state)} layers x {N_EXPERTS} experts)")

    # --- Phase 3: Quantize base model ---
    print(f"\n[3/5] Quantizing base model...")
    t0 = time.time()
    if int4_base:
        base_out = quantize_base_int4(base_state, group_size=128)
        base_orig_mb = sum(t.numel() * t.element_size() for t in base_state.values()) / 1e6
        base_comp_mb = sum(t.numel() * t.element_size() for t in base_out.values()) / 1e6
        print(f"  int4: {base_orig_mb:.0f} MB -> {base_comp_mb:.0f} MB ({base_orig_mb/base_comp_mb:.1f}x)")
        base_filename = "base_model_int4.safetensors"
    else:
        base_out = base_state
        base_filename = "base_model.safetensors"
        print(f"  No quantization (bf16)")
    base_path = out / base_filename
    save_file(base_out, str(base_path))
    base_file_size = base_path.stat().st_size
    print(f"  Saved: {base_filename} ({base_file_size/1e6:.0f} MB) in {time.time()-t0:.1f}s")

    # --- Phase 4: Build expert library ---
    print(f"\n[4/5] Building expert library...")

    # Check what already exists
    existing_topics = list_existing_expert_topics(experts_dir) if skip_existing else {}
    trained_topics = find_trained_experts(experts_dir) if preserve_trained else set()
    if trained_topics:
        print(f"  Preserving {len(trained_topics)} trained topic sets: {sorted(trained_topics)}")
    if existing_topics:
        print(f"  Existing expert topics on disk: {len(existing_topics)}")

    expert_manifest = []
    total_expert_orig = 0
    total_expert_comp = 0
    t0 = time.time()

    for topic_name in TOPIC_LIBRARY:
        # Determine seed expert
        seed_ei = None
        for ei, seed_topic in SEED_MAP.items():
            if seed_topic == topic_name:
                seed_ei = ei
                break
        if seed_ei is None:
            seed_ei = 3  # fallback to general

        # Skip if trained expert exists for this topic
        if topic_name in trained_topics:
            print(f"  {topic_name}: PRESERVED (trained)")
            # Re-add existing files to manifest
            for layer in range(N_LAYERS):
                shard_name = f"expert_l{layer}_{topic_name}.safetensors"
                shard_path = experts_dir / shard_name
                if shard_path.exists():
                    expert_manifest.append({
                        "id": f"l{layer}_{topic_name}",
                        "layer": layer,
                        "topic": topic_name,
                        "file": f"experts/{shard_name}",
                        "size_bytes": shard_path.stat().st_size,
                        "sha256": _sha256_file(str(shard_path)),
                        "compressed": True,
                        "compression": "svd",
                        "seed_expert": seed_ei,
                    })
            continue

        # Skip if files already exist and skip_existing is True
        if skip_existing and topic_name in existing_topics:
            existing_layers = set(existing_topics[topic_name])
            if len(existing_layers) == N_LAYERS:
                print(f"  {topic_name}: EXISTS ({N_LAYERS} files)")
                for layer in range(N_LAYERS):
                    shard_name = f"expert_l{layer}_{topic_name}.safetensors"
                    shard_path = experts_dir / shard_name
                    expert_manifest.append({
                        "id": f"l{layer}_{topic_name}",
                        "layer": layer,
                        "topic": topic_name,
                        "file": f"experts/{shard_name}",
                        "size_bytes": shard_path.stat().st_size,
                        "sha256": _sha256_file(str(shard_path)),
                        "compressed": True,
                        "compression": "svd",
                        "seed_expert": seed_ei,
                    })
                continue

        # Build from seed expert
        topic_size = 0
        topic_orig = 0
        for layer in range(N_LAYERS):
            if layer not in expert_state or seed_ei not in expert_state[layer]:
                continue

            weights = expert_state[layer][seed_ei]
            compressed = {}
            for part, w in weights.items():
                comp = compress_expert_svd(w, svd_energy)
                for k, v in comp.items():
                    compressed[f"{part}_{k}"] = v

            orig_size = sum(w.numel() * w.element_size() for w in weights.values())
            comp_size = sum(t.numel() * t.element_size() for t in compressed.values())
            total_expert_orig += orig_size
            total_expert_comp += comp_size
            topic_orig += orig_size
            topic_size += comp_size

            shard_name = f"expert_l{layer}_{topic_name}.safetensors"
            shard_path = experts_dir / shard_name
            save_file(compressed, str(shard_path))

            expert_manifest.append({
                "id": f"l{layer}_{topic_name}",
                "layer": layer,
                "topic": topic_name,
                "file": f"experts/{shard_name}",
                "size_bytes": shard_path.stat().st_size,
                "sha256": _sha256_file(str(shard_path)),
                "compressed": True,
                "compression": "svd",
                "seed_expert": seed_ei,
            })

        print(f"  {topic_name}: {N_LAYERS} files, "
              f"{topic_orig/1e6:.1f} MB -> {topic_size/1e6:.1f} MB "
              f"({topic_orig/max(topic_size,1):.1f}x)")

    expert_ratio = total_expert_orig / max(total_expert_comp, 1) if total_expert_comp > 0 else 0
    print(f"\n  Total experts: {len(expert_manifest)} files")
    print(f"  Time: {time.time()-t0:.1f}s")

    # --- Phase 5: Save manifest + config ---
    print(f"\n[5/5] Saving manifest + config...")

    manifest = {
        "name": "ForgeLM-v4",
        "description": "V2 base (int4) + infinite AirMoE expert library",
        "source_checkpoint": src,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_layers": N_LAYERS,
        "base_model_file": base_filename,
        "base_model_size_mb": base_file_size / 1e6,
        "expert_compression": "svd",
        "expert_svd_energy": svd_energy,
        "topics": TOPIC_LIBRARY,
        "experts": expert_manifest,
        "n_topics": len(TOPIC_LIBRARY),
        "n_expert_files": len(expert_manifest),
        "total_expert_size_mb": total_expert_comp / 1e6,
        "total_size_mb": (base_file_size + total_expert_comp) / 1e6,
        "infinite": True,
        "note": "Expert library is infinite -- add new topics anytime. "
                "Only VRAM limits how many topics can be cached simultaneously.",
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    config = {
        "name": "forgelm_v4",
        "base_config": "forgelm_v2",
        "architecture": "infinite_airmoe",
        "optimizations": [
            "int4_quantization (Hadamard + GPTQ) on base weights" if int4_base else "bf16 base weights",
            "svd expert compression",
            "infinite_expert_library (VRAM-limited hotswap)",
        ],
        "checkpoint_dir": str(out),
        "expert_library": "experts/ (unlimited topics, add anytime)",
        "router": "keyword-based, maps query to topic to expert files",
    }
    (out / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # --- Summary ---
    total_size = base_file_size + total_expert_comp
    per_topic_mb = total_expert_comp / max(len(TOPIC_LIBRARY), 1) / 1e6

    print(f"\n{'='*70}")
    print(f"Done! V4 AirMoE at {out_dir}")
    print(f"{'='*70}")
    print(f"  base: {base_filename} ({base_file_size/1e6:.0f} MB)")
    print(f"  experts: {len(expert_manifest)} files ({total_expert_comp/1e6:.0f} MB)")
    print(f"  topics: {len(TOPIC_LIBRARY)} ({per_topic_mb:.0f} MB/topic)")
    print(f"  total: {total_size/1e6:.0f} MB")
    if preserve_trained and trained_topics:
        print(f"  trained preserved: {len(trained_topics)} topic sets")
    print(f"  manifest: manifest.json")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ForgeLM V2 to V4 AirMoE format")
    parser.add_argument("--src", default="research/checkpoints/forgelm_v2.safetensors",
                        help="Source V2 checkpoint path")
    parser.add_argument("--out", default="research/checkpoints/forgelm_v4",
                        help="Output V4 directory")
    parser.add_argument("--svd-energy", type=float, default=0.9,
                        help="SVD energy retention for expert compression (default 0.9)")
    parser.add_argument("--no-int4-base", action="store_true",
                        help="Skip int4 quantization on base model (keep bf16)")
    parser.add_argument("--no-preserve-trained", action="store_true",
                        help="Overwrite trained expert files")
    parser.add_argument("--no-skip-existing", action="store_true",
                        help="Rebuild all expert files even if they exist")
    parser.add_argument("--rebuild-all", action="store_true",
                        help="Same as --no-preserve-trained --no-skip-existing")
    args = parser.parse_args()

    if args.rebuild_all:
        args.no_preserve_trained = True
        args.no_skip_existing = True

    convert(
        src=args.src,
        out_dir=args.out,
        svd_energy=args.svd_energy,
        int4_base=not args.no_int4_base,
        preserve_trained=not args.no_preserve_trained,
        skip_existing=not args.no_skip_existing,
    )
