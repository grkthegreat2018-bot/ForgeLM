"""MoE package — core MoE conversion and AirMoE expert library."""
from .moe import replace_ffn_with_moe, collect_aux_loss, MoELayer, Router
from .airmoe_infinite import InfiniteAirMoE
from .airmoe_hotswap import AirMoEHotswapLoader

__all__ = [
    'replace_ffn_with_moe', 'collect_aux_loss', 'MoELayer', 'Router',
    'InfiniteAirMoE', 'AirMoEHotswapLoader',
]
