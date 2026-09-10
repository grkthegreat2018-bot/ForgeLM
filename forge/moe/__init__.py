"""MoE package — core MoE conversion and AirMoE expert library."""
from .airmoe_hotswap import AirMoEHotswapLoader
from .airmoe_infinite import InfiniteAirMoE
from .moe import (
    Expert,
    MoELayer,
    Router,
    collect_aux_loss,
    disable_dense_bypass,
    replace_ffn_with_moe,
    set_intra_expert_sparsity,
    update_moe_biases,
)
from .routers import DEFAULT_TOPIC_DESCRIPTIONS, KeywordRouter, LASERRouter, METRORouter, SemanticRouter

__all__ = [
    'DEFAULT_TOPIC_DESCRIPTIONS',
    'AirMoEHotswapLoader',
    'Expert',
    'InfiniteAirMoE',
    'KeywordRouter',
    'LASERRouter',
    'METRORouter',
    'MoELayer',
    'Router',
    'SemanticRouter',
    'collect_aux_loss',
    'disable_dense_bypass',
    'replace_ffn_with_moe',
    'set_intra_expert_sparsity',
    'update_moe_biases',
]
