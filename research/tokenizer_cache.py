"""Centralized tokenizer cache — loads once, wraps with gigatoken for ~35x speedup.

All modules should use `get_tokenizer()` instead of `AutoTokenizer.from_pretrained()`.
This avoids the 4.3s import + load overhead repeated across 15+ call sites, and
wraps with gigatoken's HFCompat for ~35x faster encode on bulk operations.

Fast path: uses the `tokenizers` Rust library directly (190ms total) instead of
going through `transformers` (4254ms). Falls back to `transformers` if the fast
path fails (e.g. missing tokenizer.json or special token config).

Usage:
    from research.tokenizer_cache import get_tokenizer

    tokenizer = get_tokenizer("research/checkpoints/lfm25_tokenizer")
    tokens = tokenizer("print('hello')", return_tensors="pt")
    decoded = tokenizer.decode(tokens.input_ids[0])
"""
import json
import os
from functools import lru_cache
from typing import Optional


def _load_fast_tokenizer(path: str):
    """Fast path: load via `tokenizers` Rust library + gigatoken wrap.

    Bypasses `transformers` entirely (saves ~4s import time).
    Reads special token IDs from tokenizer_config.json.
    """
    from tokenizers import Tokenizer as RustTokenizer

    tokenizer_json = os.path.join(path, "tokenizer.json")
    if not os.path.exists(tokenizer_json):
        raise FileNotFoundError(f"tokenizer.json not found in {path}")

    rust_tok = RustTokenizer.from_file(tokenizer_json)

    # Read special token strings from tokenizer_config.json.
    # gigatoken's eos_token_id property reads from self.eos_token (the string),
    # so we set the strings and the ID properties resolve automatically.
    config_path = os.path.join(path, "tokenizer_config.json")
    cfg = {}
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)

    # Wrap with gigatoken for fast encode
    import gigatoken as gt
    fast_tok = gt.Tokenizer(rust_tok).as_hf()

    # Set special token strings — gigatoken's eos_token_id / bos_token_id /
    # pad_token_id properties read from these attributes. Without setting
    # them, the properties return None (the raw tokenizers.Tokenizer doesn't
    # carry HF special token metadata).
    if cfg.get("eos_token"):
        fast_tok.eos_token = cfg["eos_token"]
    if cfg.get("bos_token"):
        fast_tok.bos_token = cfg["bos_token"]
    if cfg.get("pad_token"):
        fast_tok.pad_token = cfg["pad_token"]

    return fast_tok


def _load_hf_tokenizer(path: str):
    """Slow fallback: load via transformers AutoTokenizer.

    Takes ~4s due to the transformers import overhead.
    """
    from transformers import AutoTokenizer
    hf_tok = AutoTokenizer.from_pretrained(path)
    try:
        import gigatoken as gt
        return gt.Tokenizer(hf_tok).as_hf()
    except ImportError:
        return hf_tok


@lru_cache(maxsize=8)
def get_tokenizer(path: str = "research/checkpoints/lfm25_tokenizer"):
    """Load and cache a tokenizer, wrapped with gigatoken if available.

    Fast path: uses `tokenizers` Rust library directly (190ms).
    Fallback: uses `transformers` AutoTokenizer (4254ms).

    Args:
        path: HuggingFace tokenizer directory path.

    Returns:
        A tokenizer object with HF-compatible API (encode, decode, eos_token_id, etc.).
    """
    try:
        return _load_fast_tokenizer(path)
    except Exception:
        return _load_hf_tokenizer(path)


@lru_cache(maxsize=8)
def get_tokenizer_no_wrap(path: str = "research/checkpoints/lfm25_tokenizer"):
    """Load and cache a raw HF tokenizer (no gigatoken wrapping).

    Use this when you need the exact HF tokenizer object (e.g. for save_pretrained).
    """
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path)
