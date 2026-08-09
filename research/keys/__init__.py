"""Key package — all discovered keys and the KeyStack builder."""
from .attn_scale_fold_key import AttnScaleFoldKey, apply_attn_scale_fold
from .base import Key, KeyClass, KeyResult
from .causal_mask_key import CausalMaskKey
from .dead_weight_key import DeadWeightKey, apply_dead_weight_prune
from .embedding_key import EmbeddingKey
from .gqa_key import GQAKey
from .keystack import KeyStack, build_qwen2_keystack
from .linear_key import LinearMSEKey
from .lm_head_tied_key import LMHeadTiedKey
from .norm_gated_mod_key import NormGatedMoDKey, apply_norm_gated_mod
from .norm_gated_mod_key import calibrate as calibrate_norm_gated_mod
from .rmsnorm_key import RMSNormKey
from .rope_key import RoPEKey
from .rope_share_key import RoPEShareKey, apply_rope_sharing
from .slicegpt_key import SliceGPTKey, apply_slicegpt_to_model, compute_pca_transform
from .streaming_key import StreamingKVCache, StreamingLLMKey
from .swiglu_key import SwiGLUKey
from .tensor_dedup_key import TensorDedupKey, apply_tensor_dedup, restore_aliases

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
