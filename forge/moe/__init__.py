"""MoE package — core MoE conversion and AirMoE expert library."""
from .airmoe_hotswap import AirMoEHotswapLoader
from .airmoe_infinite import InfiniteAirMoE
from .routers import KeywordRouter, SemanticRouter, DEFAULT_TOPIC_DESCRIPTIONS, LASERRouter, METRORouter
from .moe import (
    MoELayer, Router, Expert, collect_aux_loss, replace_ffn_with_moe,
    update_moe_biases, disable_dense_bypass, set_intra_expert_sparsity,
)

__all__ = [
    'AirMoEHotswapLoader',
    'InfiniteAirMoE',
    'KeywordRouter',
    'SemanticRouter',
    'DEFAULT_TOPIC_DESCRIPTIONS',
    'LASERRouter',
    'METRORouter',
    'MoELayer',
    'Router',
    'Expert',
    'collect_aux_loss',
    'replace_ffn_with_moe',
    'update_moe_biases',
    'disable_dense_bypass',
    'set_intra_expert_sparsity',
]
