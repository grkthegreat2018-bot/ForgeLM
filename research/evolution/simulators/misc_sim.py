"""Misc domain simulators — pure metric computation, no scoring logic.

These domains (KVEviction, HqeKV, SparseAttn, KARA, QuantDomain) have complex
internal state (synthetic KV tensors, etc). The simulators delegate to the
domain's internal computation but return raw metrics for JSON-based scoring.
"""
from __future__ import annotations
import torch
import numpy as np
from . import register


@register("kv_eviction_simulate")
def kv_eviction_simulate(config: dict, domain=None) -> dict:
    """KVEvictionDomain metrics: compression, fwd_err, cache_ms."""
    # Delegate to domain's internal eviction + attention computation
    if domain is not None and hasattr(domain, '_run_eviction'):
        k_comp, v_comp = domain._run_eviction(config)
        with torch.no_grad():
            from research.evolution.domains.kv_utils import full_attention_output
            y_comp = full_attention_output(domain.q, k_comp, v_comp, domain.N_KV_HEADS)
            fwd_err = (domain.y_ref - y_comp).norm().item() / domain.y_ref.norm().item()
        comp_ratio = domain.seq_len / max(k_comp.shape[2], 1)
        cache_ms = comp_ratio * 0.3
        return {"compression": comp_ratio, "fwd_err": fwd_err, "cache_ms": cache_ms,
                "behavioral_0": comp_ratio, "behavioral_1": fwd_err}
    # Fallback: no domain context
    return {"compression": 1.0, "fwd_err": 1.0, "cache_ms": 0.3,
            "behavioral_0": 1.0, "behavioral_1": 1.0}


@register("hqe_kv_simulate")
def hqe_kv_simulate(config: dict, domain=None) -> dict:
    """HqeKVDomain metrics: delegate to domain evaluate, extract metrics."""
    if domain is not None and hasattr(domain, 'evaluate'):
        result = domain.evaluate(config)
        score = result["score"]
        behavioral = result.get("behavioral", (0, 0))
        return {"base_score": score,
                "behavioral_0": behavioral[0] if isinstance(behavioral, (tuple, list)) else 0,
                "behavioral_1": behavioral[1] if isinstance(behavioral, (tuple, list)) and len(behavioral) > 1 else 0}
    return {"base_score": 0.0, "behavioral_0": 0.0, "behavioral_1": 0.0}


@register("sparse_attn_simulate")
def sparse_attn_simulate(config: dict, domain=None) -> dict:
    """SparseAttentionDomain metrics: delegate to domain evaluate, extract metrics."""
    if domain is not None and hasattr(domain, 'evaluate'):
        result = domain.evaluate(config)
        score = result["score"]
        behavioral = result.get("behavioral", (0, 0))
        return {"base_score": score,
                "behavioral_0": behavioral[0] if isinstance(behavioral, (tuple, list)) else 0,
                "behavioral_1": behavioral[1] if isinstance(behavioral, (tuple, list)) and len(behavioral) > 1 else 0}
    return {"base_score": 0.0, "behavioral_0": 0.0, "behavioral_1": 0.0}


@register("kara_simulate")
def kara_simulate(config: dict, domain=None) -> dict:
    """KARADomain metrics: delegate to domain evaluate, extract metrics."""
    if domain is not None and hasattr(domain, 'evaluate'):
        result = domain.evaluate(config)
        score = result["score"]
        behavioral = result.get("behavioral", (0, 0))
        return {"base_score": score,
                "behavioral_0": behavioral[0] if isinstance(behavioral, (tuple, list)) else 0,
                "behavioral_1": behavioral[1] if isinstance(behavioral, (tuple, list)) and len(behavioral) > 1 else 0}
    return {"base_score": 0.0, "behavioral_0": 0.0, "behavioral_1": 0.0}


# NOTE: quant_domain_simulate is defined in quant_sim.py with proper sqnr/compression metrics.
# The old delegation-based version here was shadowed by the quant_sim.py registration and
# always returned base_score=0.0 because domain was None in the JSON spec system.
