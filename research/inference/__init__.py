"""Inference backend package — unified strategy-based runtime.

Integrates all ForgeAI runtime optimizations:
  - KV cache: standard, paged, rotorquant, hadamard INT4, compressed,
    streaming, snapkv, snapkv_4bit, paged_eviction, xquant, cpu_offload,
    s4r (15x compression), rotorquant (default, Givens rotation + Lloyd-Max),
    hqe_kv, 2bit
  - Decoding: standard, speculative, medusa, dspark, eagle3, MTP self-spec
  - Quantization: none, INT8, INT4, FP8, W8A8, NVFP4, BitNet ternary
    (auto-selected by VRAM + GPU capability)
  - Acceleration: none, CUDA graphs, AirLLM layer-streaming, megakernel,
    flex_decoding
  - Innovations: MRL-AdaptiveContext, QuaRot-KV, V0-WarmStart, ProgressiveKV
  - 42 feature-registry flags (attention, prefill, KV, scheduling, MoE, etc.)
  - Auto-activation: from_checkpoint(auto_activate=True) picks optimal
    strategies based on detected KeyStack features + VRAM capacity
  - Hybrid CPU/GPU offload for LFM2.5 hybrid architectures
  - LRU-bounded prefix cache, AirLLM shard caching
  - Session-aware generation: begin/continue/end session with KV cache
    persistence across turns (O(Δt) per turn instead of O(n))
  - KV cache TTL: pin KV during tool calls, auto-evict on expiry
  - Crash recovery: signal handlers + atexit + zstd-compressed disk snapshots
  - OOM recovery: auto-degrade to lower KV bits / CPU offload on OOM
  - Structured error handling: typed exception hierarchy with error codes
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
