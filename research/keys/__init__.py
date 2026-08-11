"""Key package — all discovered keys and the KeyStack builder.

Re-exports from functional subdirectories (reorganized 2026-08).
All imports should use direct module paths (e.g. from research.keys.attention.gqa_key import GQAKey).
These re-exports are kept for backward compatibility.
"""
from .attention.attn_scale_fold_key import AttnScaleFoldKey, apply_attn_scale_fold
from .attention.causal_mask_key import CausalMaskKey
from .attention.gqa_key import GQAKey
from .compression.dead_weight_key import DeadWeightKey, apply_dead_weight_prune
from .compression.tensor_dedup_key import TensorDedupKey, apply_tensor_dedup, restore_aliases
from .misc.base import Key, KeyClass, KeyResult
from .misc.embedding_key import EmbeddingKey
from .misc.keystack import KeyStack, build_qwen2_keystack
from .misc.linear_key import LinearMSEKey
from .misc.lm_head_tied_key import LMHeadTiedKey
from .normalization.norm_gated_mod_key import NormGatedMoDKey, apply_norm_gated_mod
from .normalization.norm_gated_mod_key import calibrate as calibrate_norm_gated_mod
from .normalization.rmsnorm_key import RMSNormKey
from .position.rope_key import RoPEKey
from .position.rope_share_key import RoPEShareKey, apply_rope_sharing
from .quantization.slicegpt_key import SliceGPTKey, apply_slicegpt_to_model, compute_pca_transform
from .cache.streaming_key import StreamingKVCache, StreamingLLMKey
from .activation.swiglu_key import SwiGLUKey

# Registry of all available keys
KEY_REGISTRY = {
    'linear_mse': LinearMSEKey,
    'embedding': EmbeddingKey,
    'rmsnorm': RMSNormKey,
    'rope': RoPEKey,
    'gqa_attention': GQAKey,
    'swiglu_ffn': SwiGLUKey,
    'causal_mask': CausalMaskKey,
    'lm_head_tied': LMHeadTiedKey,
    'slicegpt': SliceGPTKey,
    'streaming_llm': StreamingLLMKey,
    'norm_gated_mod': NormGatedMoDKey,
    'tensor_dedup': TensorDedupKey,
    'attn_scale_fold': AttnScaleFoldKey,
    'dead_weight': DeadWeightKey,
    'rope_share': RoPEShareKey,
}

# Registry of KeyStacks
KEYSTACK_REGISTRY = {
    'qwen2': build_qwen2_keystack,
}

__all__ = [
    'KEYSTACK_REGISTRY',
    'KEY_REGISTRY',
    'AttnScaleFoldKey',
    'CausalMaskKey',
    'DeadWeightKey',
    'EmbeddingKey',
    'GQAKey',
    'Key',
    'KeyClass',
    'KeyResult',
    'KeyStack',
    'LMHeadTiedKey',
    'LinearMSEKey',
    'NormGatedMoDKey',
    'RMSNormKey',
    'RoPEKey',
    'RoPEShareKey',
    'SliceGPTKey',
    'StreamingKVCache',
    'StreamingLLMKey',
    'SwiGLUKey',
    'TensorDedupKey',
    'apply_attn_scale_fold',
    'apply_dead_weight_prune',
    'apply_norm_gated_mod',
    'apply_rope_sharing',
    'apply_slicegpt_to_model',
    'apply_tensor_dedup',
    'build_qwen2_keystack',
    'calibrate_norm_gated_mod',
    'compute_pca_transform',
    'restore_aliases',
]
