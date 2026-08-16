# ForgeAI — Agent Notes

## Current Architecture: LFM2.5-1.2B

**Base model**: Liquid AI LFM2.5-1.2B-Instruct, ported 100% lossless into ForgeAI framework.
- 16 layers: 10 double-gated conv + 6 GQA attention (layers 2,5,8,10,12,14)
- d_model=2048, 32 heads, 8 KV heads (GQA 4x), head_dim=64
- SwiGLU FFN (intermediate=8192), RMSNorm, QK-layernorm on attention
- RoPE theta=1M, 128K context (32K for VRAM budget)
- Vocab=65536, tied embeddings
- 1.17B params, 2.34GB bf16

**Base checkpoint**: `research/checkpoints/ForgeLM_V2_LFM25-1.2B.safetensors`
**Self-play starting checkpoint**: `research/checkpoints/ForgeLM_V2_BSP.safetensors` (sft4: 600 steps, 1657 examples, proper tool use + format)
**Tokenizer**: `research/checkpoints/lfm25_tokenizer/`

## Config Presets

Only two configs exist now:
- `lfm25_1.2b` — Full LFM2.5-1.2B model
- `lfm25_tiny` — 4-layer tiny model for fast testing

## Key Files

### Core
- `research/config.py` — ModelConfig dataclass + presets
- `research/model_loader.py` — ConfigurableResearchLLM, ModelLoader, PreAllocatedKVCache
- `research/checkpoint_io.py` — Safetensors checkpoint I/O
- `research/paths.py` — Central path management
- `research/tokenizer_cache.py` — Tokenizer loading + caching

### Architecture (only MTP kept)
- `research/architecture/mtp.py` — Multi-Token Prediction heads (planned key)
- `research/architecture/port_lfm25_to_forgeai.py` — Porting script (reference)

### Inference
- `research/inference/forge_engine.py` — Unified inference engine
- `research/inference/decoding.py` — Decoding strategies (standard, speculative, MTP)
- `research/inference/kv_backend.py` — KV cache strategies (standard, paged, rotorquant, hadamard, compressed, streaming, snapkv)
- `research/inference/int4_quant.py` — INT4 weight-only quantization
- `research/inference/innovations.py` — Runtime innovations (MRL, QuaRot, V0, ProgressiveKV)

### Decoding implementations
- `research/decoding/dspark.py` — DSpark speculative decoding
- `research/decoding/eagle.py` — EAGLE-3 speculative decoding (feature-level, multi-layer fusion, TTT training)
- `research/decoding/medusa.py` — Medusa parallel prediction heads
- `research/decoding/mtp.py` — MTP speculative decoding

### Quantization
- `research/quantization/` — BitNet, FP8, RotorQuant, SpinQuant, WANDA, paged KV, KV compress

### Model Merging
- `research/merge_models.py` — SLERP, TIES, DARE, SVD, Task Arithmetic, Linear (model soup) on safetensors state dicts. CLI: `python -m research.merge_models --method <slerp|ties|dare|svd|task_arith|linear> ...`
- `research/inject_and_merge.py` — Unified pipeline: inject new params via KeyStack knowledge keys (facts, context patches, self-play, spectral, test-gated) then merge delta into target model. Auto-clones target as injection base for clean task vector. CLI: `python -m research.inject_and_merge --target <ckpt> --inject-type <facts|test_gated|context_patch|selfplay_patch|spectral> --merge-method <task_arith|ties|dare|svd|slerp|linear> ...`

### Keys (70+ files in research/keys/)
All keys preserved. Planned for LFM2.5 integration:
- mHC (Manifold Hyper-Connections), MTP, Safety, PIT, LeRoPE, CSA, AttnRes, QK-Norm

### Self-play & Training
- `research/self_play/` — Recursive self-play, sandbox, curriculum, GRPO
- `research/training/` — DPO alignment, training utils, chunked CE
- `research/evaluation/` — Prompt tests, reasoning benchmarks, LiveCodeBench

### Runtime
- `research/runtime/` — CUDA graphs, flex attention, VRAM manager, signal capture

## Build & Test Commands

```powershell
# Run tests
$env:PYTHONPATH="D:\windsurf\ForgeAI"; D:\windsurf\ForgeAI\venv\Scripts\python.exe -m pytest tests/ --tb=short -q

# Verify model loads
$env:PYTHONPATH="D:\windsurf\ForgeAI"; D:\windsurf\ForgeAI\venv\Scripts\python.exe -c "from research.model_loader import ConfigurableResearchLLM; print('OK')"

# Benchmark INT4 quantization
D:\windsurf\ForgeAI\venv\Scripts\python.exe D:\windsurf\ForgeAI\.devin\benchmark_int4.py
```

## Removed (cleanup 2025-01)

- All Qwen-based configs (qwen25_coder, xp_1.5b, forgelm_v1/v2, 360m_mla, etc.)
- All Nemotron configs and architecture files (mamba2, latent_moe, nemotron_lightning)
- Dead attention types (DifferentialAttention, MultiHeadLatentAttention, StandardSDPA)
- Dead FFN types (ReLUSquaredFFN, latent_moe)
- EAGLESpeculativeDraftHead
- serving/ directory (superseded by inference/forge_engine.py)
- convert_keys.py, convert_key_svd.py (Qwen-specific weight transforms)
- o1_generation/ (thinking model, dead)
- 6.5GB of dead checkpoints (forgelm_v2, nemotron, dspark, qwen_hf tokenizer)
- Dead test files referencing old configs

## Environment

- OS: Windows 11, GPU: RTX 5070 12GB
- Python venv: `D:\windsurf\ForgeAI\venv\`
- Key packages: torch, transformers, safetensors, bitsandbytes, pytest
