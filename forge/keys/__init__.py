"""Key package — architecture-variant keys for ForgeLM models.

Each key class implements a checkpoint-conversion path for one architectural
variant.  The canonical loader (research/model_loader.py) imports specific key
classes directly; this package re-exports only the base classes and the
KeyStack composer for backward compatibility.

Canonical keys (imported by model_loader.py / forge_engine.py / sft_train.py):
  architecture: attn_residual, factorized_embed, gated_residual, hyperloop,
                mhc, mod_router, titan_memory
  attention:    csa, differential_attn, gla, gta, lisa, qsa
  compression:  kron_ffn, monarch_ffn, nlrq_ffn, tt_ffn
  knowledge:    ngram_embedding
  misc:         pit
  moe:          expert_tying
  position:     lerope
  quantization: bitnet_b158, fused_gemm
  speculative:  mtp
"""
from .misc.base import Key, KeyClass, KeyResult
from .misc.keystack import KeyStack

__all__ = [
    'Key',
    'KeyClass',
    'KeyResult',
    'KeyStack',
]
