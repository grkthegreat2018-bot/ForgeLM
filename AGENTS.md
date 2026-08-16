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

Three configs:
- `forgelm_v3` — **ForgeLM V3: the labeled default for fresh self-play loops.** Same LFM2.5-1.2B skeleton + full 2025/2026 stack: Differential Attention (`attn_type="diff"`, identity warm start), BitNet b1.58 QAT FFN, TITAN memory (rank 64), MoD router (keep 1.0). 1206M params. Loads `ForgeLM_V2_BSP.safetensors` (or the base) **bit-exact** via automatic GQA→diff conversion. All loaders/trainers/self-play entry points default to this (`load_default_model()` with no args).
- `lfm25_1.2b` — Reference LFM2.5-1.2B port (plain GQA, no keys).
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

## Training-Free Alignment (research/training_free/)

Forward-only adaptation — no gradients, no optimizer, no weight updates:
- `urial.py` — URIAL in-context alignment (`build_prompt`: system + 3 style examples).
- `reflexion.py` — `ReflexionBuffer`: bounded episodic memory rendered into the prompt.
- `steering.py` — `ActivationSteerer`: capture residual activations, extract task vectors (`positive - negative`), inject via pre-hooks.
- `rain.py` — `RAINGenerator`: self-eval + rewind-and-regenerate loop.
- `solver.py` — `TrainingFreeSolver`: frozen-solver adapter combining the above; `record(task, output, error, success)` collects activations, `build_task_vector()` + `apply_steering(alpha)` steer inference.
- `SelfPlaySandbox(...)` accepts `training_free=TrainingFreeSolver(...)` (or call `sandbox.enable_training_free()`): run_task styles prompts with ICA + memory and records every outcome — replaces GRPO weight updates for self-play adaptation.
- Tests: `tests/unit/test_training_free.py` (CPU, tiny model).

## 2025/2026 Architecture Keys (research/keys/)

All config-driven, dimension-generic (work at LFM2.5-1.2B scale, d_model=2048). **Main `lfm25_1.2b` preset now enables ALL of them losslessly** — verified bit-exact (max logit diff 0.0) vs the plain GQA model on the real BSP checkpoint, for both plain and KV-cached prefill+decode:
- `quantization/bitnet_b158_key.py` — BitNet b1.58 ternary QAT: `BitNetLinear` (STE, learned per-layer `qscale`, ternary ONLY in training; eval = full-precision master weights until `bitnet_force_quant`). Enable: `use_bitnet=True` (main preset: on).
- `attention/differential_attn_key.py` — Diff-Transformer: dual-softmax subtraction, per-head λ, per-head RMSNorm+scale. **Identity warm start** (`lambda=0`): group-1 rows extracted contiguously (`_group1_weights`) so GEMM shapes match GQA exactly → bit-exact conversion; training moves λ off 0 to activate the real mechanism. `attn_type="diff"` (main preset: on; GQA checkpoints auto-convert at load). KV cache stores 2× head_dim.
- `attention/differential_attn_key.py` — Diff-Transformer: dual-softmax subtraction, per-head λ (paper init), per-head RMSNorm+scale. `attn_type="diff"`; KV cache stores 2× head_dim. `DifferentialAttentionKey` = GQA→diff weight transform (dup rows, warm start).
- `architecture/titan_memory_key.py` — TITAN neural memory: gated memory, zero-init gate => **lossless at start** (ported checkpoint loads identically). `TitanMemory.update()` = Hebbian surprise step (test-time training). Enable: `use_titan_memory=True`.
- `architecture/mod_router_key.py` — Mixture-of-Depths: per-block top-k token router (STE hard mask, soft grad). keep_fraction=1.0 => **lossless**. <1.0 gates tokens in training; inference keeps all (KV alignment). Enable: `use_mod=True`.
- Main `lfm25_1.2b` preset: `attn_type="diff"`, `use_bitnet=True`, `use_titan_memory=True` (rank 64), `use_mod=True` (keep_fraction=1.0) — all lossless at load; training activates each mechanism. `get_config` returns FRESH copies (preset mutation no longer leaks).
- Tests: `tests/unit/test_arch_keys.py` (incl. main 1.2B build forward).

## AirMoE Expert Consolidation (research/training_free/expert_bake.py)

Static offline counterpart of AirMoE runtime hotswap: fold topic experts into dense FFN weights.
- `decompress_expert(state)` — decodes raw / SVD-only / SVD+INT4 expert files (mirrors `research/moe/airmoe_infinite.py` formats; LatentMoE up/down skipped).
- `bake_expert(target, expert_paths, alpha, layers, out)` — per-layer task arithmetic: `target += alpha * mean(expert − base_ffn)` per `w_gate/w_up/w_down`. Multiple experts per layer are averaged. Output is a normal dense .safetensors — no router, no disk I/O at inference.
- Layer parsed from `expert_l{layer}_{topic}.safetensors` filenames; override with `--layers`.
- CLI: `python -m research.training_free.expert_bake --target T --expert experts/expert_l0_math.safetensors --alpha 0.8 --out O.safetensors`
- Tests: `tests/unit/test_expert_bake.py`.

## Offline Weight Baking (research/training_free/bake.py)

Permanent weight modification without backpropagation:
- `bake_task_vector(target, finetuned, base, alpha, out)` — task arithmetic: adds `alpha*(finetuned - base)` onto an arbitrary target checkpoint (e.g. self-play checkpoint). Offline tensor math, output is a normal .safetensors.
- `extract_distill_dataset(packets_jsonl, out_jsonl, ...)` — context distillation: from self-play packet logs, keeps only (task, final correct code) pairs in sft_train JSONL format for a low-epoch SFT pass.
- `fuse_lora(base_ckpt, adapter_dir, out, alpha_override)` — offline PEFT LoRA fusion (`W' = W + (alpha/r)*B@A`), standalone output, no PEFT dependency at inference. (sft_train also fuses via `merge_and_unload` at save time.)
- CLI: `python -m research.training_free.bake {task-vector|distill|fuse-lora} ...`
- Tests: `tests/unit/test_bake.py` (CPU, tensor-level).

Constant-memory note: full SSM blocks (Mamba-2) were removed in the 2025-01 cleanup; the LFM2.5 arch already gets constant-memory O(1) state from its 10 conv layers (`DoubleGatedConvLayer`) — use `hybrid_offload` (conv→CPU, attention→GPU) to trade compute for VRAM instead of re-adding SSM code.

## I/O & VRAM Notes (2026-08)

- `StreamingDataLoader(ds, ..., num_workers=N, prefetch_factor=2)` — multi-worker path is picklable on Windows (module-level `_ParquetWorkerDataset`); `pin_memory=True` + device now does non-blocking H2D.
- `ParquetDataset` pickles by re-opening the file (workers re-open by path).
- `HadamardKVCache` pre-allocates its max buffer on first append (no per-token `torch.cat`).
- `StreamedGenerator` no longer double-syncs per decode step (`.item()` is the single sync).
- `load_checkpoint(..., map_location="cpu")` returns memory-mapped tensors (zero-copy).
- New config: `selective_gradient_checkpointing` — `"all"` (default), `"ffn"` (recompute only FFN, big VRAM save), `"attn"`, `"none"`. Enable with `use_gradient_checkpointing=True`.
- `model.cache_devices()` runs once on first forward (16+ `next(param).device` scans cached); `hybrid_offload` invalidates it.

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
