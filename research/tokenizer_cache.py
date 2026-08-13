"""Centralized tokenizer cache — loads once, wraps with gigatoken for ~35x speedup.

All modules should use `get_tokenizer()` instead of `AutoTokenizer.from_pretrained()`.
This avoids the 4.3s import + load overhead repeated across 15+ call sites, and
wraps with gigatoken's HFCompat for ~35x faster encode on bulk operations.

Usage:
    from research.tokenizer_cache import get_tokenizer

    tokenizer = get_tokenizer("research/checkpoints/lfm25_tokenizer")
    tokens = tokenizer("print('hello')", return_tensors="pt")
    decoded = tokenizer.decode(tokens.input_ids[0])
"""
import os
from functools import lru_cache
from typing import Optional


@lru_cache(maxsize=8)
def get_tokenizer(path: str = "research/checkpoints/lfm25_tokenizer"):
    """Load and cache a tokenizer, wrapped with gigatoken if available.

    Args:
        path: HuggingFace tokenizer directory path.

    Returns:
        A tokenizer object with HF-compatible API (encode, decode, eos_token_id, etc.).
        Uses gigatoken.HFCompat (~35x faster) when gigatoken is installed,
        otherwise falls back to transformers AutoTokenizer.
    """
    from transformers import AutoTokenizer

    hf_tok = AutoTokenizer.from_pretrained(path)

    try:
        import gigatoken as gt

        fast_tok = gt.Tokenizer(hf_tok).as_hf()
        return fast_tok
    except ImportError:
        return hf_tok


@lru_cache(maxsize=8)
def get_tokenizer_no_wrap(path: str = "research/checkpoints/lfm25_tokenizer"):
    """Load and cache a raw HF tokenizer (no gigatoken wrapping).

    Use this when you need the exact HF tokenizer object (e.g. for save_pretrained).
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path)
