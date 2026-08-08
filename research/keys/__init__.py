"""Key package — all discovered keys and the KeyStack builder."""
from .base import Key, KeyClass, KeyResult
from .linear_key import LinearMSEKey
from .embedding_key import EmbeddingKey
from .rmsnorm_key import RMSNormKey
from .rope_key import RoPEKey
from .gqa_key import GQAKey
from .swiglu_key import SwiGLUKey
from .causal_mask_key import CausalMaskKey
from .lm_head_tied_key import LMHeadTiedKey
from .slicegpt_key import SliceGPTKey, compute_pca_transform, apply_slicegpt_to_model
from .streaming_key import StreamingLLMKey, StreamingKVCache
from .norm_gated_mod_key import NormGatedMoDKey, apply_norm_gated_mod, calibrate as calibrate_norm_gated_mod
from .tensor_dedup_key import TensorDedupKey, apply_tensor_dedup, restore_aliases
from .attn_scale_fold_key import AttnScaleFoldKey, apply_attn_scale_fold
from .dead_weight_key import DeadWeightKey, apply_dead_weight_prune
from .rope_share_key import RoPEShareKey, apply_rope_sharing
from .keystack import KeyStack, build_qwen2_keystack

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
    'Key', 'KeyClass', 'KeyResult', 'KeyStack',
    'LinearMSEKey', 'EmbeddingKey', 'RMSNormKey', 'RoPEKey',
    'GQAKey', 'SwiGLUKey', 'CausalMaskKey', 'LMHeadTiedKey',
    'SliceGPTKey', 'compute_pca_transform', 'apply_slicegpt_to_model',
    'StreamingLLMKey', 'StreamingKVCache',
    'NormGatedMoDKey', 'apply_norm_gated_mod', 'calibrate_norm_gated_mod',
    'TensorDedupKey', 'apply_tensor_dedup', 'restore_aliases',
    'AttnScaleFoldKey', 'apply_attn_scale_fold',
    'DeadWeightKey', 'apply_dead_weight_prune',
    'RoPEShareKey', 'apply_rope_sharing',
    'KEY_REGISTRY', 'KEYSTACK_REGISTRY', 'build_qwen2_keystack',
]
