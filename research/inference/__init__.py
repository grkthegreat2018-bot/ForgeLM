"""Inference backend package — unified strategy-based runtime.

Integrates all ForgeAI runtime optimizations:
  - KV cache: standard, paged, rotorquant, hadamard INT4, compressed
  - Decoding: standard, speculative, medusa, dspark, MTP self-spec
  - Quantization: none, INT8, INT4 weight-only
  - Acceleration: none, CUDA graphs, AirLLM layer-streaming
  - Innovations: MRL-AdaptiveContext, QuaRot-KV, V0-WarmStart, ProgressiveKV
"""
from .decoding import (
    DecodingStrategy,
    DSparkDecoding,
    MedusaDecoding,
    MTPSelfSpecDecoding,
    SpeculativeDecoding,
    StandardDecoding,
)
from .innovations import (
    MRLAdaptiveContext,
    ProgressiveKV,
    QuaRotKV,
    V0WarmStart,
)
from .kv_backend import (
    CompressedKVCacheStrategy,
    HadamardKVCache,
    KVCacheStrategy,
    PagedKVCacheStrategy,
    RotorQuantKVCache,
    StandardKVCache,
)
