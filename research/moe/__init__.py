"""MoE package — core MoE conversion and AirMoE expert library."""
from .airmoe_hotswap import AirMoEHotswapLoader
from .airmoe_infinite import InfiniteAirMoE
from .keyword_router import KeywordRouter
from .moe import MoELayer, Router, collect_aux_loss, replace_ffn_with_moe

__all__ = [
    'AirMoEHotswapLoader',
    'InfiniteAirMoE',
    'KeywordRouter',
    'MoELayer',
    'Router',
    'collect_aux_loss',
    'replace_ffn_with_moe',
]
