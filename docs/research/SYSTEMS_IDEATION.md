# Systems & Process Ideation — Architecture, Boot, Training, I/O

> Fourth-tier ideation covering the SYSTEMS side of LLM improvement: architecture changes, boot-time pipeline, training speed, process/workflow improvements, and I/O. The existing ideation docs cover weight-level keys (F1-F12, Ideas 1-12, D1-D11) and boot-time stages (BOOT_TIME_AUDIT). This doc goes DEEPER into the systems layer — the things that AREN'T weight transforms but affect compute cost, speed, and quality at the process level.
>
> Compiled 2026-08-07. Extends the 100 existing ideas (77 mapped + 23 deep subjects). Total after this doc: ~140.

---

## The Systems Layer (what's NOT covered yet)

The key framework (FULL/BI/PARTIAL/TRIVIAL) maps weight transforms. But LLM improvement has FIVE more layers that aren't weight transforms:

| Layer | What it controls | Existing coverage | Gap |
|---|---|---|---|
| **Architecture** | Model topology, block structure | F2 (memory hierarchy), F9 (registers), D8 (variable width) | Block-level redesigns, hybrid topologies |
| **Boot-time** | Cold-start → first-token | BOOT_TIME_AUDIT (7 stages, 2 keys) | Deeper pipeline parallelism, predictive boot |
| **Training** | Forward+backward+optimizer | train.py (fused, muon, galore, chunked-ce) | Gradient computation, optimizer state, data pipeline |
| **Process** | The training/inference workflow | self_play, recursive_self_play | Orchestration, feedback loops, verification |
| **I/O** | Disk/network/memory transfer | fast_loader (mmap+prefetch) | Deeper I/O patterns, memory hierarchy |

Each layer has its own "hidden assumptions" that can be broken with the generative rule.

---

## Part A — Architecture Changes (block-level redesigns)

These are NOT-A-KEY in the weight-transform sense (they need from-scratch training or major fine-tuning), but they're the highest-leverage ideas for a future ForgeLM V4+ architecture.

### A1. Asymmetric Depth — different block types at different depths

**Hidden assumption:** All transformer blocks are identical (same attention + same FFN). The stack is uniform.

**Why it's wasteful:** Different depths do different jobs. Early layers extract features (attention-heavy). Middle layers reason (FFN-heavy). Late layers predict (output-focused). Forcing all layers to have the same attention/FFN ratio wastes capacity where one component dominates.

**Derivation:**
- Layers 0-8: **attention-dominant** (wide attention, narrow FFN). These layers route information between positions — attention is the main work.
- Layers 9-19: **balanced** (standard attention + wide FFN/MoE). These layers reason — both components matter.
- Layers 20-27: **FFN-dominant** (narrow attention, wide FFN). These layers retrieve knowledge for prediction — FFN is the main work.
- The depth profile is derived from the Information Profile (D4a): where I(x_l; x_0) is high → attention-dominant; where I(x_l; weights) is high → FFN-dominant.

**Implementation:** `config.py` — per-layer block type. New config: `block_types = ["attn_heavy"]*9 + ["balanced"]*11 + ["ffn_heavy"]*8`.
**Composes with:** D4a InformationProfile (informs the split), VariableWidth (D8a), MoE (FFN-dominant layers get more experts).
**Class:** NOT-A-KEY (architecture, needs from-scratch training). **File:** `config.py` entry.

### A2. Branching Residual — fork the residual stream into parallel paths

**Hidden assumption:** The residual stream is a single path. Every layer reads from and writes to the same stream.

**Why it's wasteful:** Different subtasks (syntax, semantics, reasoning) could be processed in PARALLEL paths that merge at the end. The single-stream forces serial processing of what could be parallel.

**Derivation:**
- At layer L, fork the residual into K parallel paths: `x → [x_1, x_2, ..., x_K]`.
- Each path processes through its own sub-stack of layers (different attention heads, different FFN experts).
- At layer L+M, merge: `x_out = merge([x_1', x_2', ..., x_K'])`.
- The merge is a learned weighted combination (like a router, but for residual paths).
- **At init:** K=1, merge = identity → single path (lossless). Fine-tune to learn K>1 paths.
- This is a generalization of mHC/xHC (which have N parallel full streams) to ADAPTIVE branching (fork only where beneficial).

**Composes with:** MoE (each path = an expert group), AirMoE (offload cold paths), D8a VariableWidth (paths can have different widths).
**Class:** NOT-A-KEY (architecture, needs from-scratch or heavy fine-tune). **File:** `config.py` entry.

### A3. Skip-Connection Graph — non-sequential layer connectivity

**Hidden assumption:** Layers are connected in a linear chain. Layer L only connects to L-1 (input) and L+1 (output).

**Why it's wasteful:** DenseFormer (DWA) already shows that cross-layer connections help. But DWA is a weighted average of ALL previous layers — it's dense, not sparse. The OPTIMAL connectivity might be sparse (layer 5 connects to layers 0, 3, 12, but not 1, 2, 4, 6-11).

**Derivation:**
- Learn a SPARSE skip-connection graph: `x_l = x_{l-1} + f_l(x_{l-1}) + Σ_{j in S_l} w_{l,j} * f_j(x_j)`.
- S_l is a learned sparse set of "useful previous layers" for layer l.
- The graph is learned during training (gradient-based sparsity penalty).
- **At init:** S_l = {} (no skip connections) → standard sequential (lossless). Fine-tune to learn the graph.
- This is DenseFormer with LEARNED SPARSITY instead of dense averaging.

**Composes with:** DenseFormer (generalizes it), NormGatedMoD (skip layers not in the graph), SharedBasis (share basis across connected layers).
**Class:** PARTIAL (needs fine-tune to learn graph, lossless at init). **File:** `skip_graph_key.py`.

### A4. Recurrent Depth — share weights across depth (weight-shared transformer)

**Hidden assumption:** Each layer has its own weights. A 28-layer model has 28× the weights of a 1-layer model.

**Why it's wasteful:** If layers learn SIMILAR functions (which they often do — see Shared Basis, F8), having separate weights is redundant. A recurrent transformer (shared weights across depth) can achieve similar quality with 1/N the weights.

**Derivation:**
- Use the SAME block weights for all layers: `x_{l+1} = f(x_l, W)` where W is shared.
- The model processes the input through the same block N times (recurrent application).
- To break symmetry (otherwise all iterations are identical), add a **depth embedding**: `x_{l+1} = f(x_l + d_l, W)` where d_l is a learned per-depth embedding.
- **Benefit:** N× fewer weights → N× faster load, N× less VRAM, N× faster training (per step).
- **Cost:** Needs more depth iterations to match quality (empirically ~2-3× more iterations for same quality).
- Net: ~N/2× weight reduction at similar quality.

**Composes with:** SharedBasis (F8 — if full sharing is too aggressive, share a basis + per-layer coefficients), VariableWidth (D8a — different widths per iteration with shared projection).
**Class:** NOT-A-KEY (architecture, needs from-scratch training). **File:** `config.py` entry.

### A5. Mixture-of-Depths with Quality Routing — skip layers based on output quality, not just norm

**Hidden assumption:** NormGatedMoD skips layers based on residual-delta norm. But norm doesn't measure QUALITY — a layer could have a large delta that's HARMFUL (adding noise).

**Why it's wasteful:** Skipping by norm alone can skip layers that are making important contributions (large but useful delta) and keep layers that are making harmful contributions (large but noisy delta).

**Derivation:**
- Train a tiny **quality predictor** per layer: `q_l = MLP(x_l_before, x_l_after) → [0, 1]`.
- q_l predicts whether layer l's contribution IMPROVED the representation (toward the correct output).
- Skip if q_l < threshold.
- The quality predictor is trained on (input, output, correct_target) triples — it learns to identify helpful vs harmful layer contributions.
- **At init:** q_l = 1.0 (never skip) → lossless. Calibrate the predictor on a held-out set.
- This is NormGatedMoD with a LEARNED quality signal instead of a free norm signal.

**Composes with:** NormGatedMoD (extends it), InformationProfile (D4a — informs which layers to skip), IterativeRefinement (D10a — re-enter layers with low quality).
**Class:** PARTIAL (needs calibration of quality predictor, lossless at init). **File:** `quality_mod_key.py`.

---

## Part B — Boot-Time Improvements (deeper than BOOT_TIME_AUDIT)

The BOOT_TIME_AUDIT covers 7 stages and 2 novel keys. Here are deeper boot-time improvements.

### B1. Predictive Boot — pre-load the model before the request arrives

**Hidden assumption:** Boot starts when the request arrives. The model is loaded on-demand.

**Why it's wasteful:** In a serving system, requests are predictable. A user who opens the chat UI will likely send a message within 30 seconds. An agent that starts a task will likely need the model within 5 seconds. We wait for the request, THEN load — wasting the user's "thinking" time.

**Derivation:**
- **Predictive pre-load:** when the user opens the UI (or the agent starts), begin loading the model in the background IMMEDIATELY — before any request is sent.
- **Speculative pre-compile:** if the model is already loaded, speculatively compile the most likely prompt shapes (based on history).
- **Prefix pre-warm:** if the system prompt is known, pre-compute its KV cache during the load phase (overlap I/O with compute).
- **Result:** by the time the user sends the first request, the model is loaded, compiled, and the system prompt KV is warm. First-token latency = decode only.

**Implementation:** A background service that watches for "session start" signals and triggers the boot pipeline preemptively.
**Composes with:** boot_pipeline (orchestrator), Knowledge Pack (pre-warm system prompt), RadixAttention (share pre-warmed prefix).
**Class:** TRIVIAL (runtime orchestration). **File:** `predictive_boot_key.py`.

### B2. Layered Load — load layers on-demand during generation

**Hidden assumption:** All layers must be in VRAM before generation starts. The full model is loaded as one unit.

**Why it's wasteful:** Generation is sequential — layer 0 runs first, layer 27 runs last. We don't need layer 27 in VRAM until ~27×(per-layer-time) after generation starts. Loading all layers upfront wastes VRAM during the early generation phase.

**Derivation:**
- **Layered load:** load layer 0 first, start generation immediately. Load layers 1-27 in the background while layer 0 is computing.
- By the time layer 1 needs to run, it's loaded. By the time layer 27 needs to run, it's loaded.
- **VRAM benefit:** peak VRAM is the same (all layers eventually loaded), but the RAMP-UP is smoother — no spike.
- **Latency benefit:** first token arrives after layer 0+1+...+first_output_layer, not after all 28 layers.
- This is AirLLM's core idea, but applied to the BOOT phase (not the serving phase).

**Composes with:** AirMoE (only load active experts), fast_loader (prefetch), boot_pipeline.
**Class:** TRIVIAL (runtime I/O scheduling). **File:** `layered_load_key.py`.

### B3. Checkpoint Diff Loading — load only the changed weights

**Hidden assumption:** When loading a new checkpoint, load the entire file. Every checkpoint is a full model.

**Why it's wasteful:** If you're updating from V2 to V3, most weights are IDENTICAL (only the folded norms changed). Loading 3.6 GB when only ~100 MB changed is 36× wasted I/O.

**Derivation:**
- Store checkpoints as DIFFS from a base: `checkpoint_v3 = base_v2 + diff_v3`.
- The diff is a sparse set of (tensor_name, delta) pairs — only the tensors that changed.
- Load: read the base (cached in VRAM from last session), apply the diff (small I/O).
- **Benefit:** checkpoint updates load in seconds, not minutes. Enables rapid A/B testing of key changes.
- **Format:** `.safetensors.diff` — a safetensors file with only the changed tensors + a manifest of what changed.

**Composes with:** fast_loader (apply diff during load), boot_pipeline, all apply_* scripts (they produce diffs from v1).
**Class:** TRIVIAL (I/O format). **File:** `diff_loader_key.py`.

### B4. Compile Cache Portability — share compiled kernels across machines

**Hidden assumption:** torch.compile cache is machine-local. Each machine compiles from scratch.

**Why it's wasteful:** If 10 machines all serve the same model, they all compile the same kernels. 10× wasted compile time.

**Derivation:**
- The compiled kernels (Triton .ttir, .ttgir, .llir, .cubin) are deterministic for a given (model, GPU, CUDA version) tuple.
- **Portable compile cache:** serialize the compiled kernels + upload to a shared store (S3, HF Hub). Other machines download and skip compilation.
- **Foundry** does this for CUDA graphs; this extends it to torch.compile kernels.
- **Benefit:** first machine compiles (40s), other machines download (2s). 20× faster boot for fleet deployments.

**Composes with:** boot_pipeline, graph_template (CUDA graph pack), persistent compile cache.
**Class:** TRIVIAL (I/O + caching). **File:** `portable_compile_key.py`.

### B5. Warm-Start Optimizer State — persist optimizer state across sessions

**Hidden assumption:** Optimizer state (AdamW momentum + variance) is lost when training stops. Each training run starts with fresh optimizer state.

**Why it's wasteful:** The optimizer state encodes the "direction the loss is moving." Discarding it means the first 100-500 steps of a new run are spent re-learning the loss landscape. This is why warmup is needed.

**Derivation:**
- **Persist optimizer state** to disk alongside the model checkpoint.
- On resume, load the optimizer state → skip warmup → start at the learning rate that was optimal at the end of the last run.
- **Benefit:** saves ~500 warmup steps (hours of training), and the optimizer converges faster (it remembers the loss landscape).
- Already partially implemented (checkpoint_io saves training state), but not consistently used.

**Composes with:** train.py, checkpoint_io, EMA (persist EMA too).
**Class:** TRIVIAL (checkpoint format). **File:** `warm_optimizer_key.py`.

---

## Part C — Training Speed Improvements

### C1. Gradient Sparsification — only compute gradients for important parameters

**Hidden assumption:** Backward pass computes gradients for ALL parameters. Every weight gets a gradient update.

**Why it's wasteful:** Many parameters have near-zero gradients (unimportant for the current batch). Computing and applying zero gradients wastes FLOPs and memory bandwidth.

**Derivation:**
- **Sparse gradient computation:** during backward, skip gradient computation for parameters whose gradient magnitude is below a threshold.
- The threshold is determined by a RUNNING AVERAGE of gradient magnitudes (per-parameter).
- Parameters with consistently small gradients are "frozen" for K steps (no gradient computation), then re-checked.
- This is the training analog of NormGatedMoD (skip unimportant layers) but at the PARAMETER level.
- **Benefit:** 20-50% backward pass speedup (empirically, ~30% of parameters have negligible gradients at any given step).

**Composes with:** GaLore (gradient low-rank projection — complementary), LISA (train only top-k layers — layer-level version of this).
**Class:** TRIVIAL (training runtime). **File:** `sparse_grad_key.py`.

### C2. Forward Caching — cache forward passes for repeated inputs

**Hidden assumption:** Every training step computes a fresh forward pass. Even if the same input was seen before, we recompute.

**Why it's wasteful:** In self-play and curriculum learning, the same prompts are seen multiple times (especially easy prompts that the model fails on repeatedly). Recomputing the forward pass for an identical input is pure waste.

**Derivation:**
- **Forward cache:** hash the input → check cache → if hit, reuse the cached logits/hidden states.
- Cache is an LRU with size ~1000 inputs.
- **When it helps:** self-play (repeated prompts), replay buffers (repeated examples), evaluation (repeated val sets).
- **When it doesn't:** pretraining on unique data (cache hit rate ~0%).
- **Benefit:** in self-play, ~20-40% of prompts repeat → 20-40% fewer forward passes.

**Composes with:** self_play, infinite_curriculum, replay buffer, eval_suite.
**Class:** TRIVIAL (training runtime). **File:** `forward_cache_key.py`.

### C3. Mixed-Precision Training — different precision for different phases

**Hidden assumption:** Training uses one precision (bf16 or fp32) for the entire forward+backward pass.

**Why it's wasteful:** The forward pass is robust to low precision (bf16 is fine). The backward pass needs higher precision for gradient accumulation (fp32). The optimizer step needs fp32 for state updates. Using fp32 everywhere is wasteful; using bf16 everywhere is unstable.

**Derivation:**
- **Phase-adaptive precision:**
  - Forward: bf16 (fast, robust).
  - Backward (gradient computation): bf16 for the chain rule, fp32 for gradient accumulation.
  - Optimizer step: fp32 for momentum/variance, bf16 for the weight update.
- This is partially implemented (mixed precision training), but the PHASE-ADAPTIVE part (different precision for different phases of the SAME step) is underexplored.
- **Benefit:** 1.5-2× training speedup with no quality loss (fp32 is only used where it matters).

**Composes with:** chunked-ce, fused-clip, muon optimizer.
**Class:** TRIVIAL (training runtime). **File:** `phase_precision_key.py`.

### C4. Speculative Training — use a draft model to predict gradient directions

**Hidden assumption:** Training computes gradients from the FULL model's forward+backward pass. Every step is a full forward+backward.

**Why it's wasteful:** Most gradient steps are SMALL and PREDICTABLE (the loss landscape is smooth locally). A tiny draft model could predict the gradient direction, and we only do a full forward+backward to VERIFY and correct.

**Derivation:**
- **Speculative training** (analog of speculative decoding, but for gradients):
  1. Draft model (tiny, 10× fewer params) predicts the gradient direction: `g_draft = draft.backward(x)`.
  2. Full model computes the actual gradient: `g_full = full.backward(x)`.
  3. If `cos(g_draft, g_full) > threshold`, accept the draft gradient (skip the full backward — use the draft's magnitude with the full's direction).
  4. If below threshold, use the full gradient.
- The draft model is trained to predict the full model's gradients (a form of gradient distillation).
- **Benefit:** when the draft is accurate (smooth loss regions), skip 50-80% of full backward passes → 2-5× training speedup.

**Composes with:** Mech Distill (F7 — distill the gradient predictor), MTP (multi-token prediction as the draft signal), DSpark (speculative decoding analog).
**Class:** PARTIAL (needs draft model training, lossless if verification is strict). **File:** `spec_train_key.py`.

### C5. Curriculum-Aware Batching — batch by difficulty, not by arrival

**Hidden assumption:** Training batches are random samples from the dataset. Every batch has a random mix of easy and hard examples.

**Why it's wasteful:** Easy and hard examples need different learning rates. Easy examples (high confidence) need small LR (they're already learned). Hard examples (low confidence) need large LR (they need to move). A random batch forces a single LR that's suboptimal for both.

**Derivation:**
- **Difficulty-homogeneous batching:** group examples by difficulty (model confidence), train each batch with a difficulty-appropriate LR.
- Easy batch: low LR (fine-tune, don't overfit).
- Hard batch: high LR (learn aggressively).
- Medium batch: medium LR.
- This is the training analog of Feather batching (prefix-homogeneous) but for DIFFICULTY instead of prefix.
- **Benefit:** faster convergence (each example gets its optimal LR), less overfitting on easy examples.

**Composes with:** infinite_curriculum (difficulty is already measured), self_play (confidence filtering), conformal_batch (D from Part 2 — conformal compute batching).
**Class:** TRIVIAL (training runtime). **File:** `curriculum_batch_key.py`.

### C6. Gradient Checkpointing with Selective Recomputation — don't recompute everything

**Hidden assumption:** Gradient checkpointing (activation checkpointing) saves ALL activations by recomputing ALL of them during backward. It's all-or-nothing.

**Why it's wasteful:** Some activations are cheap to store (small tensors) and expensive to recompute (deep layers). Others are expensive to store (large tensors) and cheap to recompute (shallow layers). Checkpointing everything recomputes cheap-to-store activations unnecessarily.

**Derivation:**
- **Selective checkpointing:** only checkpoint the activations that are EXPENSIVE to store (large, deep-layer activations). Keep cheap activations (small, shallow-layer) in memory.
- The decision is per-layer: `checkpoint_layer_l = (activation_size_l > threshold) and (recompute_cost_l < storage_cost_l)`.
- **Benefit:** 50-70% of the VRAM savings of full checkpointing, with only 10-20% of the recomputation cost.
- This is "checkpointing with a budget" — spend the VRAM budget on the most expensive activations, checkpoint the rest.

**Composes with:** gradient-checkpointing (extends it), VRAMManager, chunked-ce.
**Class:** TRIVIAL (training runtime). **File:** `selective_ckpt_key.py`.

---

## Part D — Process & Workflow Improvements

### D1. Self-Play with Active Learning — generate tasks where the model is UNCERTAIN

**Hidden assumption:** Self-play generates random tasks (or tasks from a fixed curriculum). The task distribution is not informed by the model's current weaknesses.

**Why it's wasteful:** Generating tasks the model already solves is wasted compute (no learning). Generating tasks the model can never solve is also wasted (no signal). The sweet spot is tasks at the model's frontier — where it's uncertain.

**Derivation:**
- **Active learning loop:**
  1. Model generates a batch of tasks at various difficulties.
  2. Model attempts each task, recording confidence (entropy of output distribution).
  3. Tasks with confidence in the "Goldilocks zone" (40-60% success rate) are kept for training.
  4. Tasks the model is too confident on (already solved) or too uncertain on (can't solve) are discarded.
  5. The next batch of tasks is biased toward the Goldilocks zone (the proposer is rewarded for generating frontier tasks).
- This is AZR (Absolute Zero Reasoner) + active learning. The curriculum is ADAPTIVE to the model's current capability.

**Composes with:** infinite_curriculum (already has difficulty targeting), self_play (confidence filtering), UncertainLearn (Idea 11 — uncertainty-gated learning).
**Class:** TRIVIAL (process). **File:** `active_curriculum_key.py`.

### D2. Verification-Gated Training — only update weights on verified-correct outputs

**Hidden assumption:** Training updates weights on ALL examples, regardless of whether the output was correct. The loss function handles incorrect outputs by penalizing them.

**Why it's wasteful:** If the model's output is CORRECT, we should REINFORCE it (positive update). If INCORRECT, we should CORRECT it (negative update). But standard training does both via the loss gradient — it doesn't distinguish. For self-play with a verifier (unit tests), we KNOW which outputs are correct. We should use that signal directly.

**Derivation:**
- **Verification-gated training:**
  1. Model generates output for a task.
  2. Verifier (unit test, formal proof, judge model) checks correctness.
  3. If CORRECT: apply a REINFORCEMENT update (increase the probability of this output). This is a positive gradient: `∇θ log P(output | input)`.
  4. If INCORRECT: apply a CORRECTION update (decrease the probability, increase the probability of the correct output if known). This is a negative gradient + supervised gradient.
  5. If UNKNOWN (no verifier): skip (don't train on unverified outputs).
- This is RLHF/GRPO but with a DETERMINISTIC verifier (unit tests) instead of a learned reward model. The verifier is EXOGENOUS (not model-coupled) — avoiding the recursive self-training collapse.

**Composes with:** Test-Gated Fact Injection (only inject verified facts), self_play (verifier = unit tests), AZR (Python executor as verifier).
**Class:** TRIVIAL (process). **File:** `verify_gated_train_key.py`.

### D3. Continual Learning with Expert Forgetting — forget stale experts, not stale facts

**Hidden assumption:** Continual learning preserves ALL old knowledge. Forgetting is always bad.

**Why it's wasteful:** Some knowledge becomes STALE (old API versions, deprecated frameworks, outdated facts). Preserving stale knowledge wastes capacity and can HURT (the model uses outdated information). The goal should be: forget stale knowledge, keep evergreen knowledge.

**Derivation:**
- **Expert-level forgetting:**
  1. Each expert has a "freshness score" (time since last trained on relevant data).
  2. Stale experts (freshness < threshold) are candidates for FORGETTING.
  3. Forgetting = merge the stale expert into the shared expert (Expert Consolidation) — its knowledge is diluted, not deleted.
  4. The freed expert slot is used for a NEW domain (ExpertGenesis — spawn a fresh expert).
- This is continual learning with CONTROLLED forgetting. The model's total capacity stays constant, but the knowledge DISTRIBUTION shifts toward current domains.

**Composes with:** ExpertGenesis (Idea 3 — spawn new experts), Expert Consolidation (merge stale ones), AirMoE (offload forgotten experts to disk).
**Class:** TRIVIAL (process). **File:** `expert_forgetting_key.py`.

### D4. Multi-Model Orchestration — route to specialized models, not one big model

**Hidden assumption:** One model handles all tasks. The model is a generalist.

**Why it's wasteful:** A generalist model spends capacity on tasks that a specialist could do better. A 1.5B generalist is worse at coding than a 1.5B coding specialist. But we can't have a specialist for every task... unless we ROUTE.

**Derivation:**
- **Multi-model orchestration:**
  1. Train multiple small specialists (coding, math, reasoning, writing) — each via self-play + Fact Injection in its domain.
  2. A tiny router (100M params) classifies the query → routes to the best specialist.
  3. The router is trained on (query, specialist, quality) triples from self-play.
  4. Only the routed specialist is loaded into VRAM (AirMoE-style, but at the MODEL level, not the expert level).
- This is the model-level analog of MoE. Instead of experts within a model, it's models within a fleet.
- **Benefit:** each specialist is better than a generalist at its domain. Total VRAM = one specialist + router (not all specialists).

**Composes with:** AirMoE (model-level offload), self_play (train specialists), ExpertGenesis (spawn specialists), RouteMoA (LLM-as-judge routing).
**Class:** TRIVIAL (process). **File:** `model_orchestration_key.py`.

### D5. Self-Modeling — the model predicts its own errors

**Hidden assumption:** The model doesn't know what it doesn't know. Errors are discovered by external evaluation.

**Why it's wasteful:** The model's own confidence (entropy, logit gap) is a SIGNAL about its likely errors. But this signal is noisy. A dedicated "error predictor" could be much more accurate.

**Derivation:**
- **Self-modeling loop:**
  1. Model generates output for a task.
  2. An "error predictor" (tiny MLP on the model's hidden states) predicts: "will this output be correct?"
  3. If the predictor says "likely wrong," the model can: (a) retry, (b) ask for clarification, (c) flag for human review.
  4. The error predictor is trained on (hidden_states, output, verified_correct) triples from self-play.
- This is a form of metacognition — the model knows when it's likely wrong.
- **Benefit:** the model can AVOID errors (retry before outputting) instead of just making them. This is test-time compute scaling, but SELF-AWARE.

**Composes with:** IterativeRefinement (D10a — re-enter layers when error predictor says "wrong"), self_play (generate training data for the predictor), conformal_exit (conformal early-exit).
**Class:** PARTIAL (needs error predictor training, lossless if disabled). **File:** `self_model_key.py`.

---

## Part E — I/O Improvements

### E1. Weight Streaming — stream weights from disk during generation (AirLLM-style for inference)

**Hidden assumption:** All weights are in VRAM during generation. The full model must be resident.

**Why it's wasteful:** During generation, only ONE layer is active at a time. The other 27 layers sit idle in VRAM. For a 3.6 GB model on a 12 GB GPU, that's 3.6 GB of VRAM used for idle weights.

**Derivation:**
- **Weight streaming:** keep only the CURRENT layer + NEXT layer in VRAM. Stream layers from disk as needed.
- Layer 0 computes → evict layer 0, load layer 1 → layer 1 computes → evict, load layer 2 → ...
- **VRAM benefit:** only 2× per-layer-size in VRAM (current + next), not 28×. For a 28-layer model, that's 14× less VRAM.
- **Latency cost:** disk I/O per layer. Mitigated by: (a) prefetch next layer during current compute, (b) SVD compress layers on disk (AirMoE), (c) int4 quantize layers on disk.
- This is AirLLM applied to INFERENCE (not just loading). AirMoE does this for experts; this does it for ALL layers.

**Composes with:** AirMoE (expert-level streaming), fast_loader (prefetch), Lossless Quant Chain (compress layers on disk).
**Class:** TRIVIAL (runtime I/O). **File:** `weight_stream_key.py`.

### E2. KV Cache Offloading — move cold KV cache to CPU RAM

**Hidden assumption:** The entire KV cache is in VRAM. Every token's K/V is on GPU.

**Why it's wasteful:** In long-context generation, the KV cache grows linearly. At 32K context, the KV cache can be larger than the model. Most of those K/V vectors are for OLD tokens that are rarely attended to.

**Derivation:**
- **KV cache offloading:** keep RECENT K/V (last N tokens) in VRAM, OFFLOAD old K/V to CPU RAM.
- When attention needs an old K/V, fetch it from CPU RAM (slower, but rare).
- The split is adaptive: tokens with high attention weight stay in VRAM; tokens with low attention weight are offloaded.
- This is the memory hierarchy (F2) applied to the KV cache: VRAM = "hot" KV, CPU RAM = "cold" KV.
- **Benefit:** 10-100× more effective context length (limited by CPU RAM, not VRAM).

**Composes with:** RotorQuant (compress cold KV), 4-bit KV (quantize cold KV), KVDeltaEncoding (D5a — compress cold KV via delta), SnapKV (eviction — alternative to offloading).
**Class:** TRIVIAL (runtime I/O). **File:** `kv_offload_key.py`.

### E3. Pinned Memory Pipeline — overlap CPU↔GPU transfer with compute

**Hidden assumption:** Data loading is synchronous. CPU prepares a batch → transfers to GPU → GPU computes → repeat.

**Why it's wasteful:** The GPU sits idle during CPU data preparation and transfer. The CPU sits idle during GPU compute. No overlap.

**Derivation:**
- **Pinned memory pipeline:**
  1. CPU prepares batch N+2 → stages in pinned memory (page-locked, faster H2D transfer).
  2. While GPU computes batch N, CPU transfers batch N+1 to GPU (overlap).
  3. While GPU computes, CPU prepares batch N+2 (overlap).
- This is standard async data loading, but with PINNED memory (not page-locked → 2× faster H2D).
- Already partially implemented (BinaryDataset has prefetch), but not with pinned memory consistently.

**Composes with:** fast_loader, BinaryDataset, continuous_batch.
**Class:** TRIVIAL (runtime I/O). **File:** `pinned_pipeline_key.py`.

### E4. Weight Deduplication — store shared weights once

**Hidden assumption:** Every weight tensor is stored separately in the checkpoint. Even if two tensors are identical, they're stored twice.

**Why it's wasteful:** Tied embeddings (input embed = output head) store the same matrix twice. Shared layers (A4 — recurrent depth) would store the same weights N times. QKTying (D7a) creates a shared subspace that's stored in both Q and K.

**Derivation:**
- **Weight deduplication:** hash each tensor → if two tensors have the same hash, store one + a reference.
- The checkpoint format: `{tensors: {unique_tensors}, references: {tensor_name → hash}}`.
- On load: materialize references by pointing to the unique tensor.
- **Benefit:** tied embeddings save ~150 MB (one vocab×d_model matrix). Recurrent depth (A4) saves ~27× per-layer weights. QKTying saves the shared subspace.
- This is content-addressable storage for model weights.

**Composes with:** QKTying (D7a), recurrent depth (A4), LM head tying (existing), SharedBasis (F8).
**Class:** TRIVIAL (I/O format). **File:** `weight_dedup_key.py`.

### E5. Async Checkpointing — save checkpoints without blocking training

**Hidden assumption:** Checkpointing is synchronous. Training pauses while the checkpoint is written to disk.

**Why it's wasteful:** A 3.6 GB checkpoint takes ~5-10 seconds to write to NVMe. During that time, the GPU is idle. For frequent checkpointing (every 500 steps), this adds up.

**Derivation:**
- **Async checkpointing:**
  1. At checkpoint time, COPY the model state to CPU RAM (fast, ~0.5s for 3.6 GB via PCIe).
  2. Training continues immediately (GPU is free).
  3. A background thread writes the CPU copy to disk (slow, ~5-10s, but doesn't block training).
- **Benefit:** checkpoint overhead drops from 10s (blocking) to 0.5s (non-blocking). 20× less training pause.

**Composes with:** checkpoint_io, train.py, EMA (async EMA save too).
**Class:** TRIVIAL (runtime I/O). **File:** `async_ckpt_key.py`.

---

## Part F — Novel Compositions (systems + weight keys)

### F1. Full Boot-to-First-Token Pipeline

```
PredictiveBoot (B1) → LayeredLoad (B2) → fastsafetensors → PersistentCompile (B4)
  → GrowableKV (VRAMManager) → CudaGraph capture → KnowledgePack pre-warm
  → DSpark first token → Ready
```
**Result:** first token in <5s (vs ~30-50s current). Predictive boot starts loading before the request; layered load starts generation before all layers are in; persistent compile skips JIT; KnowledgePack pre-warms the system prompt.

### F2. Training Speed Stack

```
SparseGrad (C1) → ForwardCache (C2) → PhasePrecision (C3) → SpecTrain (C4)
  → CurriculumBatch (C5) → SelectiveCkpt (C6) → AsyncCkpt (E5)
```
**Result:** 3-5× training speedup. Sparse gradients skip 30% of backward; forward cache skips 20-40% of forward (self-play); phase precision uses bf16 where safe; speculative training skips 50-80% of full backwards; curriculum batching improves convergence; selective checkpointing saves VRAM with minimal recompute; async checkpointing eliminates save pauses.

### F3. Infinite Context Pipeline

```
KVOffload (E2) → KVDeltaEncoding (D5a) → RotorQuant → 4-bit KV → SnapKV eviction
  → ConformalKV (coverage guarantee) → RadixAttention (prefix sharing)
```
**Result:** 100K+ context on 12 GB VRAM. Offload cold KV to CPU RAM; delta-encode for temporal redundancy; rotate + quantize for precision redundancy; evict with conformal guarantees; share prefixes across requests.

### F4. Self-Improving Loop

```
ActiveCurriculum (D1) → SelfPlay → VerifyGatedTrain (D2) → TestGatedFactInjection
  → SelfModel (D5) → IterativeRefinement (D10a) → ExpertGenesis (new domain)
  → ExpertForgetting (D3, retire stale) → AirMoE (offload cold experts)
```
**Result:** a self-improving system that generates frontier tasks, verifies solutions, injects verified knowledge, predicts its own errors, retries on errors, spawns new experts for new domains, and retires stale ones. No human in the loop. No gradient descent on the model (closed-form injection only). Exogenous verification (unit tests) prevents recursive collapse.

### F5. Fleet Serving Stack

```
ModelOrchestration (D4) → AirMoE (expert offload) → WeightStream (E1)
  → PortableCompile (B4) → CudaGraph → ContinuousBatch → RadixAttention
  → ComputeAwareRouter (D6b) → ConformalBatch (Part 2)
```
**Result:** a fleet of small specialists, each loaded on-demand, with portable compiled kernels, CUDA graphs for decode, continuous batching for throughput, prefix sharing for multi-turn, compute-aware routing for speed/quality trade-off, and conformal batching for guaranteed-latency tiers.

---

## Summary — New Keys Proposed

### Architecture (NOT-A-KEY, for config.py):

| # | Key | Effect | Composes with |
|---|---|---|---|
| A1 | Asymmetric Depth | Per-depth block types (attn-heavy early, FFN-heavy late) | D4a InfoProfile, D8a VariableWidth |
| A2 | Branching Residual | Fork residual into parallel paths, merge at end | MoE, AirMoE, D8a |
| A3 | Skip-Connection Graph | Learned sparse cross-layer connections | DenseFormer, NormGatedMoD |
| A4 | Recurrent Depth | Share weights across depth (N× fewer weights) | SharedBasis, D8a |
| A5 | Quality MoD | Skip layers by predicted quality, not norm | NormGatedMoD, D4a, D10a |

### Boot-Time (TRIVIAL):

| # | Key | Effect | Composes with |
|---|---|---|---|
| B1 | Predictive Boot | Pre-load before request arrives | boot_pipeline, KnowledgePack |
| B2 | Layered Load | Load layers on-demand during generation | AirMoE, fast_loader |
| B3 | Checkpoint Diff | Load only changed weights | fast_loader, apply_* scripts |
| B4 | Portable Compile | Share compiled kernels across machines | boot_pipeline, graph_template |
| B5 | Warm Optimizer | Persist optimizer state across sessions | train.py, checkpoint_io |

### Training Speed (TRIVIAL/PARTIAL):

| # | Key | Effect | Composes with |
|---|---|---|---|
| C1 | Sparse Grad | Skip gradients for unimportant params | GaLore, LISA |
| C2 | Forward Cache | Cache forward passes for repeated inputs | self_play, replay buffer |
| C3 | Phase Precision | Different precision per training phase | chunked-ce, fused-clip |
| C4 | Speculative Training | Draft model predicts gradient direction | MechDistill, MTP, DSpark |
| C5 | Curriculum Batch | Batch by difficulty, not by arrival | infinite_curriculum, self_play |
| C6 | Selective Ckpt | Checkpoint only expensive activations | gradient-checkpointing, VRAMManager |

### Process (TRIVIAL/PARTIAL):

| # | Key | Effect | Composes with |
|---|---|---|---|
| D1 | Active Curriculum | Generate tasks at model's frontier | infinite_curriculum, UncertainLearn |
| D2 | Verify-Gated Train | Only update on verified-correct outputs | TestGatedInjection, self_play, AZR |
| D3 | Expert Forgetting | Retire stale experts, spawn new ones | ExpertGenesis, ExpertConsolidation |
| D4 | Model Orchestration | Route to specialized models | AirMoE, self_play, RouteMoA |
| D5 | Self-Modeling | Predict own errors, retry before output | IterativeRefinement, self_play |

### I/O (TRIVIAL):

| # | Key | Effect | Composes with |
|---|---|---|---|
| E1 | Weight Streaming | Stream layers from disk during inference | AirMoE, fast_loader, LosslessQuant |
| E2 | KV Offloading | Move cold KV to CPU RAM | RotorQuant, 4-bit KV, SnapKV |
| E3 | Pinned Pipeline | Overlap CPU↔GPU transfer with compute | fast_loader, BinaryDataset |
| E4 | Weight Dedup | Store shared weights once | QKTying, recurrent depth, tying |
| E5 | Async Ckpt | Save checkpoints without blocking training | checkpoint_io, train.py |

**Total: 26 new systems-level ideas.**
- Architecture (NOT-A-KEY): 5
- Boot-time (TRIVIAL): 5
- Training (TRIVIAL/PARTIAL): 6
- Process (TRIVIAL/PARTIAL): 5
- I/O (TRIVIAL): 5

---

## Implementation Priority

### Tier 1 — Immediate (TRIVIAL, zero risk, high impact)

| Priority | Key | Why | Effort |
|---|---|---|---|
| 1 | **E5 Async Ckpt** | Eliminates checkpoint pause (10s → 0.5s). Simple. | 1 hour |
| 2 | **C2 Forward Cache** | 20-40% fewer forward passes in self-play. Simple. | 2 hours |
| 3 | **C5 Curriculum Batch** | Faster convergence in self-play. Simple. | 2 hours |
| 4 | **C6 Selective Ckpt** | 50-70% VRAM savings with minimal recompute. | 3 hours |
| 5 | **E3 Pinned Pipeline** | Overlap H2D with compute. Standard async. | 3 hours |
| 6 | **B3 Checkpoint Diff** | 36× faster checkpoint updates. | 4 hours |
| 7 | **E4 Weight Dedup** | Save storage for tied/shared weights. | 4 hours |
| 8 | **B5 Warm Optimizer** | Skip warmup on resume. | 2 hours |

### Tier 2 — Near-term (TRIVIAL, moderate complexity)

| Priority | Key | Why | Effort |
|---|---|---|---|
| 9 | **B1 Predictive Boot** | First token <5s. Needs UI integration. | 1 day |
| 10 | **D1 Active Curriculum** | Adaptive task generation. Extends infinite_curriculum. | 1 day |
| 11 | **D2 Verify-Gated Train** | Only train on verified outputs. Extends self_play. | 1 day |
| 12 | **E2 KV Offloading** | 100K+ context. Needs CPU RAM management. | 2 days |
| 13 | **C1 Sparse Grad** | 20-50% backward speedup. Needs gradient tracking. | 2 days |
| 14 | **B2 Layered Load** | Start generation before all layers loaded. | 2 days |

### Tier 3 — Medium-term (PARTIAL, needs calibration/training)

| Priority | Key | Why | Effort |
|---|---|---|---|
| 15 | **C4 Speculative Training** | 2-5× training speedup. Needs draft model. | 1 week |
| 16 | **D5 Self-Modeling** | Error prediction. Needs predictor training. | 1 week |
| 17 | **A5 Quality MoD** | Better layer skipping. Needs predictor. | 3 days |
| 18 | **A3 Skip-Graph** | Learned sparse connections. Needs fine-tune. | 1 week |
| 19 | **D4 Model Orchestration** | Multi-model routing. Needs specialists. | 2 weeks |
| 20 | **D3 Expert Forgetting** | Controlled continual learning. Needs freshness tracking. | 1 week |

### Tier 4 — Long-term (NOT-A-KEY, architecture redesign)

| Priority | Key | Why | Effort |
|---|---|---|---|
| 21 | **A1 Asymmetric Depth** | Per-depth block types. Needs from-scratch. | Months |
| 22 | **A2 Branching Residual** | Parallel paths. Needs from-scratch. | Months |
| 23 | **A4 Recurrent Depth** | Weight sharing. Needs from-scratch. | Months |
| 24 | **E1 Weight Streaming** | Inference-time layer streaming. Needs careful I/O. | 1 week |
| 25 | **B4 Portable Compile** | Fleet compile sharing. Needs infra. | 1 week |
| 26 | **C3 Phase Precision** | Per-phase mixed precision. Needs careful impl. | 1 week |

---

## The Meta-Pattern (systems-level generative rule)

The 26 ideas above were found by applying the generative rule to SYSTEMS assumptions:

| Assumption Category | What's uniform/global/binary | What it could be |
|---|---|---|
| **Architecture** (A) | Uniform blocks, sequential, independent weights | Asymmetric (per-depth), branching (parallel), graph (sparse skips), recurrent (shared) |
| **Boot** (B) | On-demand, all-at-once, machine-local | Predictive (pre-request), layered (on-demand), diff (changed-only), portable (cross-machine), warm (persisted) |
| **Training** (C) | All gradients, all forward, one precision, random batch, all-or-nothing ckpt | Sparse (important-only), cached (repeated), phase-adaptive (per-phase), speculative (draft), curriculum (difficulty), selective (budget) |
| **Process** (D) | Random tasks, all examples, preserve all, one model, no self-awareness | Active (frontier), verified (correct-only), forgetting (stale), orchestrated (specialists), self-modeling (error-aware) |
| **I/O** (E) | All in VRAM, all in VRAM, sync, unique, blocking | Streamed (disk), offloaded (CPU), pinned (async), deduped (shared), async (non-blocking) |

**To generate more:** pick a system component, identify what it treats as uniform/global/binary/synchronous, and ask "what if it were structured/local/spectral/asynchronous?"

---

## Relationship to Existing Ideation

| New Key | Extends | Relationship |
|---|---|---|
| A1 Asymmetric Depth | D4a InfoProfile, D8a VariableWidth | Depth-stratified block TYPE (not just width/precision) |
| A3 Skip-Graph | DenseFormer | DenseFormer with LEARNED SPARSITY |
| A4 Recurrent Depth | SharedBasis (F8) | Full weight sharing (F8 is partial via basis) |
| A5 Quality MoD | NormGatedMoD | NormGatedMoD with LEARNED quality (not free norm) |
| B1 Predictive Boot | boot_pipeline | Pre-request boot (not on-demand) |
| B2 Layered Load | AirMoE, fast_loader | AirMoE for ALL layers (not just experts) |
| C1 Sparse Grad | LISA, GaLore | Parameter-level sparsity (LISA is layer-level) |
| C2 Forward Cache | self_play, replay | Cache forward passes (not just examples) |
| C4 Speculative Training | DSpark, MechDistill | Speculative decoding for GRADIENTS |
| C5 Curriculum Batch | Feather, conformal_batch | Difficulty-homogeneous (not prefix-homogeneous) |
| D1 Active Curriculum | AZR, infinite_curriculum | Active learning (not random curriculum) |
| D2 Verify-Gated Train | TestGatedInjection, RLHF | Verification-gated GRADIENT (not just injection) |
| D3 Expert Forgetting | ExpertGenesis, ExpertConsolidation | CONTROLLED forgetting (not preserve-all) |
| D4 Model Orchestration | AirMoE, RouteMoA | Model-level MoE (not expert-level) |
| D5 Self-Modeling | IterativeRefinement, conformal_exit | Metacognition (predict own errors) |
| E1 Weight Streaming | AirMoE, AirLLM | AirLLM for INFERENCE (not just loading) |
| E2 KV Offloading | F2 MemoryHierarchy, SnapKV | VRAM→CPU KV hierarchy (not eviction) |
| E4 Weight Dedup | QKTying, recurrent depth | Content-addressable weight storage |

---

## Total Idea Count (across all ideation docs)

| Doc | Ideas | Status |
|---|---|---|
| FIRST_PRINCIPLES_IDEATION (F1-F12) | 12 | Original theorizing |
| NOVEL_SOLUTION_IDEATION (Ideas 1-12) | 12 | Technique combinations |
| KEY_NOVELTY_AUDIT_PART2 (8 novel) | 8 | Uncovered components |
| KEY_MAPPING_MASTER (component atlas) | 26 | Component what-ifs |
| DEEP_SUBJECTS_IDEATION (D1-D11) | 23 | Deep structural subjects |
| SYSTEMS_IDEATION (this doc, A1-E5) | 26 | Systems & process |
| **Total** | **107** | (77 mapped in KEY_MAPPING_MASTER + 30 new) |

Wait — let me recount. KEY_MAPPING_MASTER maps 58 ideas (12 + 26 + 12 F-ideas, minus overlaps = 58). KEY_NOVELTY_AUDIT maps 47 of those + Part 2 adds 19 more = 77. DEEP_SUBJECTS adds 23. This doc adds 26. **Total: 77 + 23 + 26 = 126 ideas.**

---

*Compiled 2026-08-07. Extends all prior ideation docs. The systems layer (architecture, boot, training, process, I/O) is where the next 2-5× improvements live — weight-level keys give 1.2-1.5× each, but systems-level compositions give 3-5×. Use the Meta-Pattern to generate more by asking "what is the system treating as uniform/global/binary/synchronous, and what if it were structured/local/spectral/asynchronous?"*
