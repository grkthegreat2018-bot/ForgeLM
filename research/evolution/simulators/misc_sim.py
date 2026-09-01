"""Misc domain simulators — pure metric computation, no scoring logic.

These domains (KVEviction, HqeKV, SparseAttn, KARA) have complex internal
state (synthetic KV tensors, etc). The simulators delegate to the legacy
Python domain's evaluate() and return raw metrics for JSON-based scoring.
"""
from __future__ import annotations
from . import register
from research.evolution.domains.kara import KARADomain
from research.evolution.domains.hqe_kv import HqeKVDomain
from research.evolution.domains.sparse_attn import SparseAttentionDomain
from research.evolution.domains.kv_eviction import KVEvictionDomain

# Cache of legacy domain instances (they hold expensive synthetic KV state).
# Keyed by (class name, seq_len, seed, device).
_LEGACY_CACHE: dict = {}


def _delegate_to_legacy(domain, legacy_cls, config: dict) -> dict:
    """Delegate to the legacy domain's evaluate().

    JSONSpecDomain passes itself as ``domain``; in that case build (or reuse)
    the legacy domain instance with matching seq_len/seed/device. Passing a
    real legacy domain instance delegates to it directly.
    """
    if domain is not None and not hasattr(domain, "spec"):
        result = domain.evaluate(config)
    else:
        seq_len = int(getattr(domain, "seq_len", 2048) or 2048)
        seed = int(getattr(domain, "_seed", 42) or 42)
        device = getattr(domain, "_device", None)
        key = (legacy_cls.__name__, seq_len, seed, str(device))
        legacy = _LEGACY_CACHE.get(key)
        if legacy is None:
            legacy = legacy_cls(seq_len=seq_len, seed=seed, device=device)
            _LEGACY_CACHE[key] = legacy
        result = legacy.evaluate(config)
    return result


def _delegation_metrics(result: dict) -> dict:
    """Extract raw metrics from a legacy domain's evaluate() result."""
    score = result["score"]
    behavioral = result.get("behavioral", (0, 0))
    b0 = behavioral[0] if isinstance(behavioral, (tuple, list)) else 0
    b1 = behavioral[1] if isinstance(behavioral, (tuple, list)) and len(behavioral) > 1 else 0
    return {"base_score": score,
            "compression": b0, "fwd_error": b1,
            "behavioral_0": b0, "behavioral_1": b1}


@register("kv_eviction_simulate")
def kv_eviction_simulate(config: dict, domain=None) -> dict:
    """KVEvictionDomain metrics: compression, fwd_err, cache_ms."""
    result = _delegate_to_legacy(domain, KVEvictionDomain, config)
    return {"compression": result["metadata"].get("compression", 1.0),
            "fwd_err": result["metadata"].get("fwd_err", 1.0),
            "cache_ms": result["metadata"].get("cache_ms", 0.3),
            "behavioral_0": result["behavioral"][0],
            "behavioral_1": result["behavioral"][1]}


@register("hqe_kv_simulate")
def hqe_kv_simulate(config: dict, domain=None) -> dict:
    """HqeKVDomain metrics: delegate to the legacy domain's evaluate."""
    result = _delegate_to_legacy(domain, HqeKVDomain, config)
    return _delegation_metrics(result)


@register("sparse_attn_simulate")
def sparse_attn_simulate(config: dict, domain=None) -> dict:
    """SparseAttentionDomain metrics: delegate to the legacy domain's evaluate."""
    result = _delegate_to_legacy(domain, SparseAttentionDomain, config)
    return _delegation_metrics(result)


@register("kara_simulate")
def kara_simulate(config: dict, domain=None) -> dict:
    """KARADomain metrics: delegate to the legacy domain's evaluate."""
    result = _delegate_to_legacy(domain, KARADomain, config)
    return _delegation_metrics(result)


# NOTE: quant_domain_simulate is defined in quant_sim.py with proper sqnr/compression metrics.
# The old delegation-based version here was shadowed by the quant_sim.py registration and
# always returned base_score=0.0 because domain was None in the JSON spec system.
