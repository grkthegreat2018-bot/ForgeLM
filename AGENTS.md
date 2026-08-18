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
- `forgelm_v3` — **ForgeLM V3: the labeled default for fresh self-play loops.** Same LFM2.5-1.2B skeleton + full 2025/2026 stack: Differential Attention (`attn_type="diff"`, identity warm start), BitNet b1.58 QAT FFN, TITAN memory (rank 64), MoD router (keep 1.0), MHC hyper-connections (rank=512, gate=0), AttnRes cross-layer retrieval (k=4, gates=0). 1256M params. Loads `ForgeLM_V2_BSP.safetensors` (or the base) **bit-exact** via automatic GQA→diff conversion. All loaders/trainers/self-play entry points default to this (`load_default_model()` with no args). Triton kernels active: BitNet b1.58 (`FORGE_BITNET_KERNEL=triton`), Fused RoPE+QKNorm (`FORGE_FUSED_ROPE_QKNORM=1`). Benchmark: 165-328 tok/s, 2.69 GB VRAM (RTX 5070).
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
All keys preserved. Wired into V3:
- mHC (Manifold Hyper-Connections, DeepSeek-V4) — `use_mhc=True`, rank=512, gate=0 lossless
- AttnRes (Attention Residuals, Kimi K3) — `use_attn_residual=True`, k=4, gates=0 lossless
- PIT (Pseudo-Inverse Tying) — `use_pit=True`, L=I lossless (replaces weight tying)
- Differential Attention, BitNet b1.58, TITAN memory, MoD router — all lossless at load
Planned for further integration:
- MTP, Safety, LeRoPE, CSA, QK-Norm, SandwichNorm, LearnedSink, ValueResidual, SwiGLU Clamp, MRL

### Self-play & Training
- `research/self_play/infinite_loop.py` — **Unified AZR self-play loop** (entry point). Propose → solve → verify → SFT → eval → promote. CLI: `python -m research.self_play.infinite_loop --checkpoint <ckpt> --epochs 50`
- `research/self_play/infinite_curriculum.py` — AZR curriculum engine (task proposal, validation, solving, ELO difficulty tracking)
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
- `quantization/bitnet_b158_key.py` — BitNet b1.58 ternary QAT: `BitNetLinear` (STE, learned per-layer `qscale` re-anchored on checkpoint load, ternary ONLY in training; eval = full-precision master weights until `bitnet_force_quant`). **True BitNet integer kernels on CUDA**: default = int8 @ int8 tensor-core GEMM (`torch._int_mm`, a4.8-style activation quant); `FORGE_BITNET_KERNEL=triton` selects the b1.58 add-only Triton kernel (fp activations, zero-skip, no weight multiplies — verified bit-exact vs fp on small shapes). Applies to FFN + attention q/k/v/o projections. Enable: `use_bitnet=True` (main preset: on).
- `attention/differential_attn_key.py` — Diff-Transformer: dual-softmax subtraction, per-head λ, per-head RMSNorm+scale. **Identity warm start** (`lambda=0`): group-1 rows extracted contiguously (`_group1_weights`) so GEMM shapes match GQA exactly → bit-exact conversion; training moves λ off 0 to activate the real mechanism. `attn_type="diff"` (main preset: on; GQA checkpoints auto-convert at load). KV cache stores 2× head_dim.
- `attention/differential_attn_key.py` — Diff-Transformer: dual-softmax subtraction, per-head λ (paper init), per-head RMSNorm+scale. `attn_type="diff"`; KV cache stores 2× head_dim. `DifferentialAttentionKey` = GQA→diff weight transform (dup rows, warm start).
- `architecture/titan_memory_key.py` — TITAN neural memory: gated memory, zero-init gate => **lossless at start** (ported checkpoint loads identically). `TitanMemory.update()` = Hebbian surprise step (test-time training). Enable: `use_titan_memory=True`.
- `architecture/mod_router_key.py` — Mixture-of-Depths: per-block top-k token router (STE hard mask, soft grad). keep_fraction=1.0 => **lossless**. **TRUE skip in training** (no cache/mask): skipped tokens genuinely bypass attention+FFN (per-row gather/scatter, FLOPs scale with keep_fraction — verified: 0.5 fraction processes exactly 50% of tokens); router trained via aux loss (`ModRouter.aux_loss`). Inference keeps all tokens (KV alignment). Enable: `use_mod=True`.
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

## Recommended Improvements (2026-08)

Four improvements implemented based on architectural critique:

### 1. SnapKV + 4-bit KV Combined Cache (`research/inference/kv_backend.py`)
- `SnapKV4BitCache` — composes SnapKV eviction + Hadamard INT4 quantization.
- Total compression = eviction_ratio × bit_ratio (up to ~16× vs fp16 full cache).
- Strategy name: `"snapkv_4bit"` in `build_kv_cache()`.
- Evicts low-attention tokens first (SnapKV), then quantizes survivors (Hadamard INT4).

### 2. Golden Trajectory Injection (`research/self_play/grpo_trainer.py`)
- `GRPOTrainer` now accepts `replay_buffer` parameter (FOREVER-style `ReplayBuffer`).
- `GRPOConfig.replay_ratio` (default 0.15) — fraction of each batch from golden replays.
- `_inject_golden_replays()` — mixes previously successful trajectories into training batch.
- `_record_golden_trajectories()` — stores verified-successful completions for future replay.
- Prevents catastrophic forgetting in continual self-play (anti-regression).

### 3. ELO-Driven Curriculum (`research/self_play/elo_tracker.py`)
- `EloTracker` — ELO rating system for both model and individual prompts.
- Targets ~50% expected success (Goldilocks zone, max learning signal).
- `select_prompts()` — picks prompts closest to model's skill boundary.
- `select_mixed_prompts()` — Goldilocks-matched + exploration prompts.
- Integrated into `InfiniteCurriculum.record_result()` and `get_training_batch_elo()`.
- Zero-sum rating updates, K-factor decays with prompt attempts (stabilizes ratings).

### 4. Fused QK-Norm + RoPE Triton Kernel (`research/decoding/fused_rope_qknorm.py`)
- Fuses RMSNorm + RoPE into a single Triton kernel for Q and K preprocessing.
- Halves HBM traffic (1 load + 1 store vs 2+2 for separate ops).
- `fused_qk_norm_rope()` — public API, auto-falls back to PyTorch on CPU/compile-fail.
- Opt-in via `FORGE_FUSED_ROPE_QKNORM=1` env var in `GroupedQueryAttention.forward`.
- Attention itself stays on FlashAttention-2 (FA2) via SDPA — already fused.
- Tests: `tests/unit/test_recommended_improvements.py`.

## GRPO-λ Dynamic Length Penalty (2026-08)

Prevents the "CoT length penalty trap" (arXiv 2509.01155): static length penalties
cause accuracy collapse early in training when the model is still learning to reason.

### GRPOTrainer (`research/self_play/grpo_trainer.py`)
- `GRPOConfig.use_grpo_lambda` — enable dynamic length penalty (default False).
- `_group_correctness_ratio()` — fraction of completions with reward >= 0.99.
- `_apply_grpo_lambda_penalty()` — penalty only when correctness_ratio >= threshold.
  - Low correctness → NO penalty (pure 0/1 rewards, prioritize reasoning).
  - High correctness → penalty = -λ * n_tokens (encourage efficiency).
- `length_penalty_warmup` — delay penalty activation for first N steps.
- Stats track `correctness_ratios` and `length_penalty_active_count`.

### GoalScorer (`research/evaluation/goal_scorer.py`)
- `minimalism_active` flag (default True) — toggle minimalism/length penalty.
- `set_minimalism_active(False)` — redistributes minimalism weight to efficiency/diversity.
- Called by training loop based on GRPO-λ correctness ratio.
- Tests: `tests/unit/test_grpo_lambda.py` (17 tests).

## DiffusionBlocks (`research/diffusion_blocks.py`)

Block-wise training via diffusion interpretation of residual connections
(Sakana AI, ICLR 2026). First successful test on a >1B parameter model.

### How It Works
- Partitions 16 layers into B blocks (default B=4, 4 layers/block)
- Each training step trains ONE block independently as a denoising step
- B× memory reduction: only L/B layers need gradients per step
- AdaLN noise conditioning (shift/scale, zero-init = lossless at start)
- EDM-style loss weighting: w(σ) = (σ² + σ_data²) / (σ·σ_data)²

### V3 Benchmark Results (2026-08-18)
- **Standard training**: 5.74 GB, 8.14s/step (batch=2, seq=512)
- **DiffusionBlocks middle blocks**: 4.98 GB, 1.72-1.82s/step (13% less memory, 4x faster)
- **Batch scaling**: 4× larger batch (8 vs 2) fits in 8.70 GB
- **Key fix**: Removed gates from AdaLN (zero-init gates block gradients; shift/scale only)

### Usage
```python
from research.diffusion_blocks import DiffusionBlocks, DiffusionBlockConfig
db_config = DiffusionBlockConfig(num_blocks=4, use_noise_conditioning=True)
dblock = DiffusionBlocks(model, db_config, d_model=2048, num_layers=16)
# Train one block per step
result = dblock.train_step(input_ids, labels, optimizer)
```

### Config
- `DiffusionBlockConfig`: num_blocks, sigma_min/max, gamma (overlap), cond_dim
- `freeze_all_except_block(b)`: freezes all params except block b
- `unfreeze_all()`: restores standard training mode
- Model forward supports `layer_indices`, `noisy_embeds`, `modulation` kwargs

## Multi-Provider Distillation Client (`research/distillation/distill_client.py`)

Generates verified training data from free-tier API providers. Only uses
Apache 2.0 / MIT licensed models (distillation-safe). Llama and Gemma are
EXCLUDED due to license restrictions.

### Supported Providers (11 providers, 32 model entries, 13 canonical models)
- **Groq** (free, permanent): Qwen3-32B, gpt-oss-120b, gpt-oss-20b
- **DeepSeek** (MIT license): deepseek-reasoner (R1 CoT), deepseek-chat (V3)
- **NVIDIA NIM** (free, 40 RPM, no daily cap): DeepSeek R1/V3, Qwen3.5-122B, gpt-oss-120b
- **Cerebras** (free, permanent, 30 RPM, 1M tok/day): gpt-oss-120b/20b, Qwen3-32B, Qwen3-235B, GLM-4.7
- **SambaNova** (free, permanent, 20 RPM/20 RPD): gpt-oss-120b, DeepSeek-V3.1
- **Cloudflare Workers AI** (free, 10K neurons/day): gpt-oss-120b/20b, GLM-4.7
- **SiliconFlow** (free forever): Qwen3-8B, DeepSeek-R1-Distill-Qwen-7B
- **HuggingFace** (free, $0.10/mo credits): gpt-oss-120b/20b, DeepSeek-R1
- **Mistral AI** (free experiment plan): Mistral Small 4, Magistral Small (reasoning)
- **Z AI / Zhipu** (free, unlimited): GLM-4.7-Flash, GLM-4.5-Flash
- **OpenRouter** (free tier): Qwen3-235B MoE, gpt-oss-120b/20b, DeepSeek R1

### Multi-Provider Rate-Limit Bypass (max redundancy)
Same model served by multiple providers — client rotates through them:
- **gpt-oss-120b**: 7 providers (groq, nvidia, cerebras, sambanova, cloudflare, openrouter, huggingface)
- **gpt-oss-20b**: 5 providers (groq, cerebras, cloudflare, openrouter, huggingface)
- **deepseek-r1**: 4 providers (deepseek, nvidia, openrouter, huggingface)
- **deepseek-v3**: 3 providers (deepseek, nvidia, sambanova)
- **glm-4.7**: 3 providers (zai, cerebras, cloudflare)
- **qwen3-32b**: 2 providers (groq, cerebras)
- **qwen3-235b**: 2 providers (openrouter, cerebras)

### NVIDIA NIM Filter
`_nvidia_filter()` excludes NVIDIA's own models (Nemotron etc.) per Eval
Agreement §2.6. Only third-party MIT/Apache models on NIM are allowed.

### Key Features
- **Randomized model-per-goal**: shuffles model pool per goal for max quality diversity
- **Multi-distill**: different teacher models per sample → diverse training data
- **Auto-detects API keys**: only uses providers with credentials in env
- **Verification**: optional `verify_fn(solution, test_cases) -> bool` filters correct solutions
- **ReplayBuffer integration**: `distill_into_buffer()` stores verified results as golden trajectories
- **Temperature randomization**: (0.3, 1.0) range for GRPO group diversity
- **Reasoning traces**: captures CoT from R1/Qwen3 thinking mode for reasoning distillation

### Pipeline
```
DistillationClient → generate verified (prompt, solution) pairs
    ↓
ReplayBuffer.add() → store as golden trajectories
    ↓
GRPOTrainer._inject_golden_replays() → mix into training batches
```

### Usage
```python
from research.distillation.distill_client import DistillationClient
from research.self_play.replay_buffer import ReplayBuffer

client = DistillationClient(verify_fn=my_verify_fn)
buf = ReplayBuffer(max_size=10000)
stats = client.distill_into_buffer(
    goals=["Write fibonacci function", "Sort a list"],
    replay_buffer=buf,
    n_samples_per_goal=4,  # 4 diverse completions per goal
)
```

### License Safety
- ✅ Apache 2.0 (Qwen3, gpt-oss, Mistral): "prepare Derivative Works" explicitly allowed
- ✅ MIT (DeepSeek R1, GLM, Phi-4): "distill & commercialize freely"
- ❌ Llama Community: "will not use output to improve any other LLM" (BANNED)
- ❌ Gemma Terms: distilled model becomes "Model Derivative" (BANNED)
- ❌ Gemini TOS: "may not use Services to develop models that compete" (BANNED)
- ❌ NVIDIA-own: Evaluation Agreement §2.6 prohibits (third-party on NIM = OK)
- ❌ OpenAI GPT: "may not use Output to develop models that compete" (BANNED)
- ❌ Anthropic: "may not use Outputs to train models that compete" (BANNED)
- ❌ Grok/xAI: "weights cannot be used to train other models" (BANNED)
- ❌ Cohere: non-commercial use only (BANNED for commercial distillation)

### BANNED providers (do not re-add)
- Google AI Studio — TOS prohibits competing model development
- GitHub Models — RETIRED July 30, 2026
- Hyperbolic/Together/Novita/Chutes — trial credits, not permanent

### Research doc
- `docs/GROQ_DISTILLATION_RESEARCH.md` — full provider analysis, rate limits, pricing
- Tests: `tests/unit/test_distill_client.py` (26 tests)

## Tool-Call + Code-Format Distillation (`research/distillation/distill_tool_calls.py`)

Cold-start SFT data generation for ForgeLM V3. Uses API teachers (gpt-oss-120b,
DeepSeek, etc.) to generate training data with **thinking tokens captured**:
1. **Direct answers**: Simple Qs ("5*10=50") — no tools, just answer from knowledge
2. **Code generation**: task -> Python function + test cases
3. **Tool-call trajectories**: Multi-turn with `[end]` marker — teacher writes code,
   marks `[end]`, system executes it, returns result, teacher continues
4. **Reasoning**: Problem decomposition with step-by-step thinking

Thinking tokens (`reasoning_content` from gpt-oss/DeepSeek R1) are captured and
wrapped in ` IMD... IMD` blocks so the model learns to think before answering.

Multi-response format: each data point is a list of turns:
```
[{"role": "user", "content": "Task: ..."},
 {"role": "assistant", "content": " IMD\n<thinking>\n IMD\n<code>[end]"},
 {"role": "tool_result", "content": "stdout: ..."},
 {"role": "assistant", "content": " IMD\n<thinking>\n IMD\n<final answer>"}]
```

CLI: `python -m research.distillation.distill_tool_calls --n-code 200 --n-tool 100 --n-reason 50 --n-direct 50 --output research/data/finetune/v3_distill.jsonl`

## FFN-SkipLLM (Speculative Compute Reduction)

Based on EMNLP 2024 paper. Skips FFN computation on "saturated" layers
(high cosine similarity between FFN input and output) during eval.

- Config: `ffn_skip_threshold` (0.0 = disabled)
- **NOT applicable to ForgeLM V3** — calibration shows no saturation region:
  all 16 layers have low/negative cosine similarity (max +0.10, most negative)
  meaning the FFN is actively transforming representations in every layer
- FFN-SkipLLM requires 32+ layer models (LLaMa-2 7B/13B) where middle-layer
  FFNs become redundant (cosine similarity 0.95+)
- Infrastructure kept in codebase for future larger models
- Full analysis: `docs/FFN_RESEARCH.md`

### What DOES work for V3 inference acceleration:
- **Speculative decoding**: EAGLE-3, DSpark, MTP (already in `research/decoding/`)
- **Attention skipping in top layers**: "Attend First, Consolidate Later" (2024)
  — skip attention in top 30% for non-math tasks, keep all FFNs
- **middle_70 architecture**: for V4 — concentrate FFN in middle 70% of layers
  (+1.29% improvement at 1.2B scale per COLM 2025 paper)

## Agentic Distillation Client (`research/distillation/agentic_distill.py`)

Takes the self-play loop process from the former `tool_use_loop.py` (now removed — merged into the unified AZR loop) and applies it to the
distillation model router. Teacher API models call tools (run_script, web_search,
think, calculate, etc.) in an agentic loop to generate rich training trajectories.

**Fine-tuning is DISABLED** — pure data collection. Trajectories can later be
used for SFT or GRPO training of the local ForgeLM model.

### Architecture
```
Teacher API model (gpt-oss, DeepSeek, Qwen, etc.)
    ↓ receives task + tool definitions (OpenAI function-calling format)
    ↓ emits tool calls → we execute (run_script, web_search, think, etc.)
    ↓ results injected back → teacher continues
    ↓ loop until final answer or max_turns
    ↓
AgenticTrajectory (messages, tool_calls, reward, final_answer)
    ↓
save_trajectories() → JSONL for SFT training
```

### Key Components
- **`AgenticDistillClient`** — extends `DistillationClient` with agentic tool-use
- **`run_agentic_task()`** — runs the tool-use loop with a teacher model
- **`run_agentic_batch()`** — runs multiple tasks with multiple teachers per task
- **`generate_tasks()`** — teachers generate their own coding tasks with test cases
- **`save_trajectories()`** — saves full tool-use trajectories as JSONL
- **`compute_reward()`** — reuses the same multi-component reward from the former `tool_use_loop.py` (now legacy; import guarded)
  (format, tool selection, execution, completion, planning, self-verification, etc.)

### Tool-Capable Filtering
Not all providers support OpenAI function calling. The client filters on init:
- **Tool-capable**: Groq, DeepSeek, NVIDIA, Mistral, SambaNova, Cerebras, HuggingFace
- **Tool-capable (partial)**: OpenRouter (only gpt-oss and deepseek models)
- **No tool support**: Cloudflare, Z AI, SiliconFlow (auto-excluded from agentic pool)

### Task Generation
Teachers can generate their own tasks (like `GoalGenerator` in the former `infinite_tool_loop`):
- Teacher model proposes coding tasks with test cases
- Tasks filtered for quality (non-filler, requires tool use, has verifiable output)
- Filtered tasks added to the pool for agentic execution

### Usage
```python
from research.distillation.agentic_distill import AgenticDistillClient

client = AgenticDistillClient(max_turns=8)
# Run agentic tasks (teachers call tools)
trajectories = client.run_agentic_batch(
    tasks=["Write is_prime(n)", "Implement binary search"],
    n_samples_per_task=3,  # 3 different teachers per task
    min_reward=0.3,  # only keep good trajectories
)
# Let teachers generate their own tasks
new_tasks = client.generate_tasks(n_tasks=20)
# Save for SFT training
client.save_trajectories(trajectories, "agentic_distill_data.jsonl")
```

### CLI
```powershell
# Agentic mode with predefined tasks
python -m research.distillation.run_data_gen --agentic --n-samples 3

# Generate new tasks then run them
python -m research.distillation.run_data_gen --agentic --gen-tasks 20

# Only run generated tasks (skip predefined)
python -m research.distillation.run_data_gen --agentic --gen-tasks 20 --gen-only
```

### Tests
- `tests/unit/test_agentic_distill.py` (23 tests: schema conversion, task filtering,
  tool-capability filtering, trajectory serialization, save/load)

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

## Novel Discovery Protocol (R&D methodology)

When asked to do R&D or develop novel systems, do NOT just design on paper.
Follow this iterative empirical loop:

1. **Isolated test script first** — write a minimal script with the smallest
   possible input that exercises the core idea (e.g. tiny model, 4 layers,
   100 steps). Run it BEFORE researching. Get a baseline number.
2. **Research + think long** — web_search the topic, read 3-5 papers, think
   hard about what's known vs unknown. Write findings to `.devin/scratchpad.md`.
3. **Apply novel ideas, compare to documented results** — implement 2-3 novel
   variations in the isolated script. Run them. Compare numbers to documented
   baselines from research. Most novel ideas will LOSE to known answers —
   that's expected and informative.
4. **Iterate once more before defaulting** — if novel ideas lost, adjust the
   angle (not just parameters). Try a different combination. Only after a
   second failed iteration should you default to the best known answer.
5. **Cross-domain risky combinations** — if still stuck, try combining
   something that barely relates to the topic but might have novel effects.
   E.g. diffusion scheduling ideas applied to optimizer step timing, or
   compression theory applied to gradient sparsity.
6. **Record what worked AND what failed** — failed novel ideas are still
   valuable; document them in scratchpad so the next session doesn't repeat.
7. **Sometimes, leave it to luch** - sometimes, leaving things to chance can be a good modivator. if you need more ideas, take all known and loosly related systems, throw them into a randomizer script, and see what it adds together. ie; ##### + #####. This could help you find new ideas that you wouldn't have thought of otherwise.

Key principles:
- **Isolated scripts over integration** — test the core mechanism on a toy
  problem before touching the real training loop. Faster iteration, clearer
  signal.
- **Numbers over theory** — a 5-line script that runs in 10 seconds beats
  a 500-line design doc. Get a number, then explain it.
- **Lose fast** — most novel ideas don't work. Find out in 30 seconds with
  a toy script, not 30 hours of integration.
- **Cross-domain is where novelty lives** — the best novel discoveries
  combine techniques from fields that don't usually interact.

## Environment

- OS: Windows 11, GPU: RTX 5070 12GB
- Python venv: `D:\windsurf\ForgeAI\venv\`
- Key packages: torch, transformers, safetensors, bitsandbytes, pytest
