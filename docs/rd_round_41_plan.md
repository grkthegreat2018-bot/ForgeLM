# R41 Implementation Plan — Latent-Space Supervision + Training-Memory Unblocks

Status: PLAN (approved direction 2026-09-07). Derived from the 6-topic research
agenda + 3 parallel codebase audits (full findings in `.devin/scratchpad.md`,
"R38 Candidate Survey" section). Builds on R40's research sweep per the
build-on-the-prior rule; reconciliation with R40's matrix is below.

## Scope decision

**R41 core = Latent-prediction auxiliary loss (topic 5) + Phase-0 quick wins.**

Rationale:
- Only fully greenfield area of the six (zero existing latent-supervision code).
- ~0 VRAM cost, CPU-testable (closed-loop verification, directive B/C).
- Cross-feeds two existing production paths: BitNet ternary QAT (topic 2) and
  self-play training; also composes with MTP (R40's top pick) — MTP predicts
  *tokens*, the latent head predicts *hidden states*; both hook the same
  hidden-state plumbing.
- Does not conflict with R40's R41 items.

### Reconciliation with R40's R41-R43 matrix
| R40 item | Status |
|---|---|
| MTP activation in training | Complementary — latent head shares its hook; keep both |
| MatryoshkaKV | **Already implemented + wired** (`forge/engine/kv/matryoshka_kv.py`) — R40 doc stale |
| Quamba2 W4A8, Mamba-2/3 ports, expert streaming | Deferred to R42/R43, unchanged |
| BTC-LLM, ToMe token merging | Still parked (R40 verdicts stand) |

---

## Phase 0 — Quick wins (unblock existing orphaned code)

1. **CLI exposure of orphaned optimizers** — `forge/training/runners/sft_train.py`
   `--optimizer` choices += `apollo`, `galore`, `flashoptim`, `nvme_muon_4bit`
   (all already handled by `configure_optimizer` in
   `forge/training/training_utils.py`; also add to `cpt_train.py`). Add unit
   test asserting each choice builds on a tiny model.
2. **Wire `BitNetResidualLinear` into ModelLoader** — `use_bitnet_residual`
   config flag exists (`forge/config.py:369`) but `forge/model_loader.py` never
   calls the builder (`forge/keys/quantization/bitnet_residual_key.py`). R24
   data says ternary+10% residual = 0.33 err vs 0.80 pure ternary; this key is
   a validated winner that is currently unreachable.
3. *(moved to R42)* `train_kvpop_scorer` wiring — belongs with the context-
   manager round.

## Phase 1 — R41-1: Latent-prediction auxiliary loss

**New file**: `forge/training/losses/latent_predict.py` (losses dir currently
has only chunked_ce / sigreg / adaln_zero / advanced_rl — no latent loss).

**Baseline (community)**: EPFL/JHU-style latent prediction — predictor head
predicts hidden state h_{L+k} from h_L; target = detached h_{L+k}; loss =
cosine + normalized MSE. Exponentially more data-efficient than token CE.

**Novel variations (implement alongside baseline, directive C):**
- **N1 — Ternary-gap latent distillation**: teacher = bf16 master forward,
  student = ternary (BitNet STE) forward; latent matching loss bridges the
  QAT quality gap directly (targets the exact error BitNet introduces).
  Cross-domain: latent prediction × BitNet — no prior work does this.
- **N2 — Entropy-gated latent loss**: apply the loss weighted by token
  entropy (already computed for sampling); high-uncertainty tokens get more
  latent supervision. Ties into ForgeAdapter's entropy mechanism + self-play.

**Design constraints:**
- One shared predictor head (d_model→d_model MLP + layer-embedding
  conditioning) across hooked layers: ~8-16 MB fp32 for d_model=2048.
  **VRAM budget: < 50 MB total** (directive D).
- λ=0 must be **bit-exact lossless** at init (preset lineage check, rule A).
- Hidden-state plumbing: reuse whatever `ConfigurableResearchLLM.forward`
  already exposes for MTP; extend only if output_hidden_states is missing.
- CLI: `--latent-predict-weight`, `--latent-predict-horizon k`,
  `--latent-predict-mode {baseline,ternary_gap,entropy_gated}`.
- Tests (CPU, `tests/unit/`): lossless-at-λ=0, loss decreases on tiny model,
  head shape/dtype, ternary-gap mode requires BitNet enabled, config wiring.

## Phase 2 — R42/R43 backlog (from the remaining agenda topics)

Ordered by impact × effort × novelty on RTX 5070:

1. **R42-A Trainable context manager (AdaCoM-style, topic 4)** — biggest KV
   gap: all 26 wired strategies are heuristic. STE hard-decision per-token
   gate (keep/compress/evict) reusing `ModRouter` machinery
   (`mod_router_key.py`); wire `train_kvpop_scorer` first as the learned
   scorer. Baseline to beat: `auto_context` + `snapkv`; KV quality bar:
   s4r/spectral reconstruction error at equal memory.
2. **R42-B Full-ternary training stack (topic 2)** — from-scratch ternary
   pretraining path (today `train_8b_all.py` *disables* QAT at init), wire
   `TernaryOptimizer` (2-bit flip-direction states, R20, orphaned) into
   `configure_optimizer` + CLI, ternary-native ForgeLM pretrain run. VRAM:
   int8 GPU + bf16 CPU master (existing path, 8GB/16GB at 8B).
3. **R43-A Optimizer round (topics 1+3)** — AdEMAMix (slow momentum on CPU,
   fast on GPU — mixed per directive D), Sophia-H, native GaLore with
   per-layer adaptive rank + rank annealing + NLRQ-factor-subspace
   projection (APOLLO already covers SVD-free scaling; GaLore must beat it,
   not just AdamW). Baseline rule: beat `muon_sf_plain` (V3) / `cpu_offload`
   (V4).
4. **R43-B Fine-grained BitNet MoE (topic 6)** — many-small experts
   (n_experts↑, d_ff↓, constant active params) + BitNet ternary experts +
   shared expert + CPU-RAM expert hotload (ExpertHotload sim domain exists);
   wire AirMoE hotswap into `forge_engine.py` (currently orphaned).

## Verification requirements (all phases)

- Bit-exact lossless check at identity config (max logit diff 0.0) before any
  training experiment.
- Optimizer work: converge on real BSP base, beat `muon_sf_plain`/`cpu_offload`.
- VRAM: profile with `torch.cuda.max_memory_allocated()`; state budget in the
  round notes; mixed CPU/GPU fallback mandatory past 12GB.
- Tests in `tests/unit/`, CPU where possible; suite must stay green
  (currently 1225 passed).

---

# Web Research Additions (2026-09-07 sweep)

Full annotated findings (≈60 techniques, sources + numbers) in
`.devin/scratchpad.md` → "Web Research Sweep". Sources are from search
snippets — verify arXiv IDs before implementing. Top picks per focus area:

## F1 — Low-mem full train (adds to R43-A optimizer round)
| Technique | Source | Why it wins here |
|---|---|---|
| SubTrack-Grad/++ | arXiv:2502.01586 (NeurIPS 25) | Rank-1 Grassmannian subspace tracking replaces GaLore's periodic SVD; 65% wall-time cut, 3B overhead 31% vs 157% |
| Adam-mini / Q-Adam-mini | arXiv:2406.16793 (ICLR 25) | Hessian-block shared LR kills dense v; −45-50% opt mem; INT8 m variant = 8× total cut |
| SOLO | arXiv:2505.00347 | 2-3-bit Adam states; pairs with TernaryOptimizer → ternary weights + 2-bit momentum |
| GradLite | arXiv:2510.22467 | Low-rank Jacobian + error feedback tolerates dropped/compressed activations; −50% activation mem |
| ZenFlow | arXiv:2505.12242 | Importance-aware CPU offload; 5× vs ZeRO-Offload; direct CPUAdamW successor |
| Batch-1 token-half-life β | arXiv:2507.07101 | Batch-1 stable training kills grad accumulation; longest seq per 12GB |
| Q-GaLore | arXiv:2407.08296 | INT4 projections + lazy subspace; 7B on 16GB consumer card |

## F2 — Small-model potential (new R42/R43 candidates)
| Technique | Source | Why |
|---|---|---|
| Overtraining scaling laws | arXiv:2403.08540 | 0.1-0.4B wind-tunnel runs → extrapolate 1.2B token budget; overtraining has a knee — find it cheaply |
| Dispersion loss | arXiv:2602.00217 | Zero-param embedding-geometry aux loss for small LMs; slots next to R41 latent head |
| Prefix on-policy distillation + EOPD | ACL 26 / arXiv:2603.07079 | Distill 3-7B teacher into 1.2B at 2-40× less FLOP; reverse+forward KL mix |
| Looped transformers / LOTUS | arXiv:2502.17416, 2606.31779 | k layers looped L× ≈ kL depth on reasoning; zero extra params |
| Mixture-of-Recursions | arXiv:2507.10524 | Token-level dynamic recursion + recursion-wise KV; 2.18× throughput at 135M-1.7B |
| MoDA depth-attention | arXiv:2603.15619 | Heads read prior-layer KV; +2.11% downstream @ +3.7% FLOPs (1.5B) |
| MoE upcycling (MoEsturizer) | ICLR 26 | Dense→sparse MoE via ~150k SFT samples; replaces from-scratch MoE in R43-B |
| Sherry 3:4 sparse ternary | ACL 26 | 1.25 bpw, zero accuracy loss on 1B — extends BitNet stack |
| T1 + PA-Tool | arXiv:2504.04718, 2510.07248 | Tool-filtered verification (1B>8B on MATH); training-free schema alignment for tool SFT |
| Jet-Long bifocal RoPE | arXiv:2607.07740 | Zero-shot 32-128K ctx via dynamic RoPE rescaling, ≤4% overhead, no retrain |

## F3 — Context mem + decode speed (adds to R42-A KV round)
| Technique | Source | Why |
|---|---|---|
| KV-Direct | arXiv:2603.19664 | Residual-only cache: 5KB/token vs 136KB; recompute up to 5× faster than reading KV — validates engine's `residual_stream`/`capture` strategies; promote + benchmark |
| ShadowKV | arXiv:2410.21465 (ICML 25) | Low-rank keys on GPU + values offloaded + sparse retrieval; 3.04× throughput, 6× batch — layers onto cpu_offload tier |
| STAR-KV | arXiv:2606.08382 (ICML 26) | Learned per-head soft-threshold rank + fused Triton; 75% compression, 6.9× attn speedup — S4R successor |
| SelKV / SemantiCache | arXiv:2607.16213, 2603.14303 | Training-free token merging with attention compensation; 25% retention near-lossless |
| FreeKV | arXiv:2505.13109 | Speculative KV retrieval overlapping decode; 13× — upgrade for offload tiers |
| MoBA/FlashMoBA | arXiv:2502.13189, 2511.11571 | Block-sparse attention; 14.7× over FA2 at small blocks; add to auto_context pool |
| HeteroSpec / SpecPV / HiSpec | arXiv:2505.13254, 2512.02337, 2510.01336 | Training-free speculative controllers; up to 4.24× over EAGLE-3 / 6× partial-KV self-spec |
| KVTuner / PatternKV / QJL / NVFP4-KV | see F4 | Mixed-precision + residual KV quant paths for existing 2bit/hqe selectors |

## F4 — Quants beyond BitNet (new quant round)
| Technique | Source | Why |
|---|---|---|
| BTC-LLM | arXiv:2506.12040 (ACL 26) | 0.7-1.11 bpw binary codebook; 13B @ 0.8bpw −3.1% — beats TernLC Pareto at sub-1-bit |
| OptRot | arXiv:2512.24124 | Data-free learned rotations (kurtosis min); beats SpinQuant/QuaRot; fuses into weights — upgrade hadamard_int4/rotorquant |
| QoQ/QServe W4A8KV4 | arXiv:2405.04532 (MLSys 25) | 4w/8a/4kv co-design; the exact missing slot in the quant stack (have W8A8/FP8, not W4A8KV4) |
| LittleBit | arXiv:2506.13771 | 0.1 bpw latent factorization; 31× mem cut — risky, prototype at 1.2B |
| KVTuner | arXiv:2502.04420 | Layer-wise KV precision MOO; 3.25-bit effective lossless |
| QJL | AAAI 25 | 1-bit JL KV, zero metadata overhead; >5× KV mem cut |
| Sparse-BitNet | arXiv:2603.05168 | 1.58-bit + N:M joint; ternary tolerates sparsity better than BF16 |
| NVFP4 KV | NVIDIA 12/25 | Blackwell FP4 KV; −50% vs FP8 KV, <1% loss |

## F5 — Training-time minimization (new speed round)
| Technique | Source | Why |
|---|---|---|
| NVFP4 end-to-end pretraining | arXiv:2509.25149 | Hadamard + 2D block + stochastic rounding; FP8-parity at ½ mem, Blackwell-native |
| µS (µnit Scaling) FP8 | arXiv:2502.05967 | FP8 without dynamic scaling; 1B-13B parity, +33% faster |
| FlashAttention-4 | arXiv:2603.05451 (MLSys 26) | Blackwell-native; 2.7× vs Triton — watch SM120 consumer support |
| u-µP | arXiv:2407.17465 | µP + unit scaling: tune 100M proxy → transfer HPs to 1.2B/8B; FP8-stable defaults |
| ScheduleFree+ | arXiv:2605.19095 | Fixes SF averaging/large-batch issues; +31% vs SOTA schedules @ 1000 tok/param; direct `muon_sf` upgrade |
| WSM checkpoint merging | arXiv:2507.17634 | Replace WSD decay with checkpoint merging; no fixed training length |
| AdEMAMix | arXiv:2409.03137 (ICLR 25) | 1.3B @ 101B tokens = AdamW @ 197B; already on R43 backlog |
| MuToR register MTP | arXiv:2505.10518 | Register-token multi-token prediction, no extra heads; composes with existing MTP |
| SUS sparse backward | arXiv:2505.15080 | Attention backward O(n²)→O(nc) for long-seq training |
| Influence Distillation | arXiv:2505.19051 | 3.5× faster data selection for curriculum building |

## Revised round mapping
- **R41** (unchanged): latent-prediction loss + Phase-0 quick wins. ADD: dispersion
  loss as a second zero-param aux loss (same hidden-state hook).
- **R42**: trainable context manager — baselines to beat now include ShadowKV,
  STAR-KV, SelKV (not just internal s4r/spectral). ADD candidates: KV-Direct
  promotion of `residual_stream`, FreeKV speculative retrieval, MoBA in
  auto_context pool, Jet-Long RoPE.
- **R43**: optimizer/memory round — scope now: SubTrack-Grad (SVD replacement),
  Adam-mini/Q-Adam-mini, SOLO, ZenFlow, batch-1 half-life β, AdEMAMix,
  ScheduleFree+, u-µP proxy tuning.
- **R44 (new)**: quantization round — BTC-LLM, OptRot, W4A8KV4, LittleBit vs
  internal TernLC/TernPack/NanoQuant; Sparse-BitNet N:M extension.
- **R45+**: small-model architecture round (looped depth, MoR, MoDA, Matryoshka
  nesting, MoE upcycling) + Blackwell speed adoption (NVFP4/µS/FA4 as kernels
  land) + agentic recipe (PA-Tool + T1 + tool-GRPO).

---

# 0.5B Time-to-Model Reality Check (2026-09-07 web sweep)

## Measured anchors (public, single-consumer-GPU)
| Datapoint | Hardware | Result |
|---|---|---|
| modded-nanogpt 124M speedrun | 1× RTX 4090 | 130-163k tok/s, val 3.25 in ~90-115 min |
| Same, 2×4090 | 124M | 1.88B tokens to 3.28 target (baseline needed 6.44B → **3.4× token efficiency**) |
| 1× RTX 5090 (community) | 124M | ~42 min to 3.28 |
| TinyLlama 1.1B | 1× 4090 | 17k tok/s |
| ZeroShot-500M | 0.53B, 7.9B tokens | RTX 5090, ~51 h |
| NPC Nano 0.5B | 0.5B | 8.93B tokens, ~6.7 days (A40) |

RTX 5070 planning number for 0.5B: **12-16k tok/s** well-tuned BF16 (25-45% MFU
on the 61.7 TFLOPS BF16 peak). FP8 on SM120 is experimental (nanochat: ~1% e2e
gain from lm_head-only FP8; MXFP8 unreliable on SM120) — do not bank on it.

## The honest conclusion
From-scratch generalist at 0.5B costs **~5-10 days minimum** (public 0.5B runs:
7.9-10B tokens) and lands well below Qwen2.5-0.5B (capacity wall: NPC Nano
+15B extra math tokens → GSM8K still ~2%). From-scratch generalist pretraining
on this GPU is the wrong tool. Proven fast paths:

| Path | Budget on RTX 5070 | Quality outcome |
|---|---|---|
| **A. Warm-start Qwen2.5-0.5B-Instruct + 25k-200k curated examples** (SFT/distill, 1-3 ep) | **0.5-6 h** | DistilQwen2.5-0.5B: IFEval 42.8→52.6, MT-Bench 5.49→5.78; math SFT: GSM8K 21.6→36.2 |
| **B. White-box distillation (DistiLLM-2-style)** | 1-5 h student + one-time teacher logit gen | 0.5B student wins 66-77% vs base vs 1.8-14B teachers |
| **C. Continued pretraining from strong 0.5B ckpt** | 0.1B tok ≈ 4 h; 1B ≈ 1.5 d; 2B ≈ 3 d | domain gains, general stable w/ 10-20% replay |
| **D. From-scratch 0.5B base, 7-10B curated tokens** | ~5-10 days | base-level only (HS ~37-39%, GSM8K ~2%) — far below Qwen2.5-0.5B |
| **E. Agent/tool distillation + test-time compensation** | 0.5-2 h SFT + 0 for PA-Tool | 0.5B ≈ 1.5B CoT tier; 1B+tools > 8B on MATH |

## Implication for ForgeAI
- The from-scratch ForgeLM 0.5B generalist is the *only* multi-week path — and
  public evidence says it lands well below Qwen2.5-0.5B anyway. Deprioritize.
- Highest-value use of the 12GB card: **warm-start from Qwen2.5-0.5B via the
  existing `--hf-model` path**, then ForgeAI post-training (distillation,
  self-play, tool SFT, BitNet QAT bake). Hours, not weeks.
- Speedrun techniques worth porting into `sft_train`/`cpt_train` regardless:
  fused/cut cross-entropy (logits 5GB→135MB), Muon (already wired as
  `muon_sf`), compile+static shapes (currently OFF by user directive — revisit
  as opt-in flag), value embeddings/U-Net skips (token-efficiency 3.4× in
  speedrun), packing (already present).
- R&D angle preserved: the custom-architecture R&D (BitNet, NLRQ, KV keys)
  runs on the **from-scratch small-scale track** (100-400M wind-tunnel models,
  hours each) — not on the 0.5B generalist, which should be warm-started.
