"""Inference backend package — unified strategy-based runtime.

Integrates all ForgeAI runtime optimizations:
  - KV cache: standard, paged, rotorquant, hadamard INT4, compressed
  - Decoding: standard, speculative, medusa, dspark, MTP self-spec
  - Quantization: none, INT8, INT4 weight-only
  - Acceleration: none, CUDA graphs, AirLLM layer-streaming
  - Innovations: MRL-AdaptiveContext, QuaRot-KV, V0-WarmStart, ProgressiveKV
"""
from .kv_backend import (
    KVCacheStrategy, StandardKVCache, PagedKVCacheStrategy,
    RotorQuantKVCache, HadamardKVCache, CompressedKVCacheStrategy,
)
from .decoding import (
    DecodingStrategy, StandardDecoding, SpeculativeDecoding,
    MedusaDecoding, DSparkDecoding, MTPSelfSpecDecoding,
)
from .innovations import (
    MRLAdaptiveContext, QuaRotKV, V0WarmStart, ProgressiveKV,
)
