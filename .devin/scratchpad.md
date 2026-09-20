# ForgeAI Scratchpad — LoRA Portability + Context Window Research
## Consolidated findings (2026-09-16)

Sources: codebase audit (`forge/engine/engine_lora.py`, `bitnet_lora.py`,
`sft_train.py`, `hotswap.py`, `prefix_cache.py`, `prefill/`, `llm.py`,
`forge/engine/kv/*`) + online research (NeurIPS'24/'25, ICLR'25, ICML'26,
arXiv).

---

## TOPIC 1 — Do LoRAs need retraining on every model change?

### How the system works today

- Adapters attach by **module-name substring** (`w_gate`, `q_proj`, `in_proj`…)
  via `add_lora_adapters()` in `forge/training/bitnet_lora.py`.
- Checkpoints save raw `{module_path.lora_adapter.lora_A/B: tensor}` pairs
  (`--save-lora-adapter` → `*.lora.safetensors`, sft_train.py ~L2209).
  **No base-model fingerprint is recorded.**
- `ForgeEngine.load_lora` (engine_lora.py L74-87) re-creates fresh adapters on
  the current model, then `copy_`s weights by exact name:
  - name missing → warning + skip (**silent partial load**)
  - name present, shape differs → `copy_` RuntimeError (**crash**)
  - names+shapes match but base weights drifted → loads silently, **gracefully degrades**

### Verdict per change type

| Model change | Retrain needed? |
|---|---|
| New key via zero/identity-init lossless port (v2→v12_jamba pattern) | **No** — shared weights bit-identical, new components are no-ops. Breaks only on module *renames* (partial apply) or *resizes* (crash). |
| Same-arch weight drift (merged self-play epoch, continued pretrain) | Loads fine; quality decays with ‖W_new − W_old‖. Soft problem. |
| Structural change (d_model, n_layers, vocab, renames, Mamba-2→3 swap) | **Yes today** — no transplant tooling exists. |

### Research-backed ways to avoid full retraining

- **Direct copy is a strong baseline** (ICML'26 "Trivial Baselines" paper):
  copying LoRA between *related* bases beats elaborate schemes (CrossLoRA,
  ProLoRA); success tracks weight similarity. MCQA transfers easily;
  generation degrades most.
- **Warm-start touch-up > from-scratch** — ReLoRA (arXiv 2606.02606; NOT the
  2023 ReLoRA): Bayesian-opt fusion init of old adapter + base delta, then
  short FT w/ scheduled regularization → 8.9× faster rollout, +4.6% acc.
- **LoRASuite (NeurIPS'25)** — handles breaking upgrades: transfer matrices
  from old+new weights for dim mismatch; CKA layer mapping for depth changes;
  small stabilizing FT. Beat full retrain on MiniCPM/Qwen (+1.4/+6.6 math),
  −78% compute. Code: github.com/YananLi18/LoRASuite
- **Trans-LoRA (NeurIPS'24, IBM)** — nearly data-free: source base+LoRA makes
  synthetic data → fresh adapter trained on new base. Works cross-family and
  cross-PEFT (LoRA↔DoRA). Fits our synthetic-data + self-play infra.
- **Task-vector re-basin (TransFusion, arXiv 2505.22697)** — training/data-free
  transfer of τ = θ_ft − θ_base via permutation alignment.
- **Exact delta rebase (closed-form)**: ΔW' = (W_old + ΔW_lora) − W_new
  reproduces merged model bit-exactly but *cancels* new-base gains in adapted
  matrices. Only for freezing behavior; SVD-truncate to stay rank-r.

### Concrete recommendations for ForgeAI

1. Keep lossless-port discipline; add `LORA_NAME_REMAP` dict in `load_lora`
   for the day a key renames `w_gate` → `ffn.gate` etc.
2. Fingerprint adapters: `metadata={"base_preset", "base_hash"}` into
   `.lora.safetensors` (safetensors `metadata=` arg; `lora_store.py` already
   parses headers stdlib-only). Warn in `load_lora` on mismatch.
3. Same-arch drift: zero-shot copy first, then warm-start touch-up
   (~10-20% of original steps), ReLoRA-style init.
4. Structural changes: LoRASuite transfer matrices + CKA layer mapping;
   zero-init adapters on new layers (missing-name path already tolerates
   partial transplants).
5. Large jumps: Trans-LoRA synthetic distillation (have teacher + data gen).
6. ForgeAI wrinkle: our targets include `in_proj` (Mamba SSM) — literature is
   transformer-only; for Mamba layers prefer zero-shot copy or distillation.

---

## TOPIC 2 — Massively increasing effective context (no training, ~3GB VRAM headroom)

### Reframe: KV cache is NOT the bottleneck on Jamba-3B

- Attention KV: 2 layers × MQA (1 KV head × 128) ≈ **1KB/token** → 1M tokens ≈ 1GB.
- Mamba-2 state: fixed ~0.3MB fp32/layer × 26 ≈ **~10MB total**, constant
  vs. length.
- Real transient cost: **logits** — vocab=65536 → 8K-token prefill chunk ≈
  1GB bf16. `ChunkedPrefiller` (engine/prefill/) already exists; keep chunks
  ≤8K in the ~3GB headroom.
- **Actual bottleneck = Mamba effective receptive field (ERF)**: SSM state is
  a lossy fixed-size compression; recall saturates long before 256K.

### What already exists (do NOT rebuild)

- `hotswap.set_infinite_context` (hotswap.py L179-198): max_context=1M +
  eviction strategy auto-select.
- `forge/engine/kv/`: 24 strategies (snapkv, s4r, streaming_llm,
  cpu_kv_offload, cacheblend, paged_eviction, spectral, matryoshka, …) +
  `auto_context.py` meta-manager (entropy-driven strategy swap).
- `prefill/`: ChunkedPrefiller + HybridChunkedPrefiller.
- `prefix_cache.py`: `capture_recurrent_state` / `apply_recurrent_state_prefix`
  — Marconi-style conv+SSM state snapshots ALREADY implemented; model stores
  `_last_prefill_recurrent` snap at prefill end (llm.py L578-591).
- `kv/replay_ssm.py`: ReplaySSMCache (input-replay state reconstruction).
- **Missing**: RAG/retrieval index, prompt compression, per-document SSM
  state library, ∆t calibration.

### Research findings (all training-free)

**Tier 1 — direct fit, cheap:**

- **MambaExtend (ICLR'25)** — STRONGEST MATCH. Mamba long-context failure =
  OOD discretization steps (∆t). Calibrate only per-layer ∆t scaling factors
  (~26 scalars for us) via gradient-free zeroth-order opt. **32× extension
  (2k→64k), minimal PPL increase**, ~5.42×10⁶× fewer param updates.
  Code: github.com/ArminAzizi98/LongContextMamba
- **SSM-state document library** — extend `capture_recurrent_state`: ingest
  doc once → save ~10-30MB state snapshot → restore + query later, no
  re-prefill. Megatron-LM shipped prod Mamba prefix caching same way
  (PR #3225). Per-doc state library = Mamba-native RAG; nobody has
  productized it → R&D-worthy.
- **Prompt compression** — LongLLMLingua (ACL'24): 4-6× compression, +21%
  RAG tasks; reordering matters MORE for us: Mamba is **recency-biased**
  (not lost-in-middle) → put query/key facts **at the END**.
  LLMLingua-2 = XLM-RoBERTa-large, pure CPU, zero VRAM.

**Tier 2 — orchestration, zero model changes:**

- **Chain-of-Agents (NeurIPS'24)** — sequential worker passes over chunks
  carrying a running note; manager synthesizes. +10% over RAG/full-context.
  Unlimited context via linear passes.
- **ReadAgent (ICML'24)** — pause points → gist memories → lookup raw
  passages on demand. 3.5-20× effective extension.
- **MemGPT self-paging** — model manages context via tool calls. We have
  tool-use infra + hot-swap LoRAs; prompting layer only.

**Tier 3 — heavier engine work:**

- **InfLLM (NeurIPS'24)** — block-level context memory: past context in
  CPU-RAM memory units, retrieve relevant blocks per step. Validated 1024K.
  For our hybrid: retrieval over offloaded KV blocks + SSM snaps; combines
  w/ existing `cpu_kv_offload`. (github.com/thunlp/InfLLM)
- **DeciMamba (ICLR'25)** — token decimation inside Mamba via ∆t-norm
  importance; longer ERF + faster inference. Written for Mamba-1/S6 —
  Mamba-2 port needed, longest pole. (github.com/assafbk/decimamba)

### Suggested order

1. `set_infinite_context` + chunked prefill already works — measure where
   recall actually breaks: passkey/ERF probe at increasing depths
   (`forge/keys/architecture/mamba_probe.py` exists for this).
2. **MambaExtend ∆t calibration** — ~26 learnable scalars, zeroth-order,
   afternoon-scale run. Highest leverage/effort.
3. **SSM-state document library** — extends existing code; the "throw more
   info at it" killer feature for a hybrid model.
4. **Query-at-end prompt policy + LLMLingua-2 on CPU** for bulk stuffing.
5. **Chain-of-Agents worker/manager** over forge_server for corpus reasoning.

### Open questions / next steps

- [ ] Write ERF probe: passkey retrieval vs depth at 4k/16k/64k/256k —
      gets the real degradation curve before committing to a tier.
- [ ] Verify Mamba-2 layer exposes ∆t post-softplus for scaling-factor hook
      (check `forge/engine/mamba3.py` / mamba-2 impl in model loader).
- [ ] Confirm `_ssm_state` snapshot works under quantized model paths
      (quamba2.py has its own `_ssm_state` handling).
- [ ] Sizing: what fraction of 32GB RAM can hold doc-state library
      (10-30MB/doc → 100+ docs trivially).

---

## TOPIC 3 — Tokenless byte-stream front-end for ForgeLM V2 (video: "Building an LLM replacement", jrz9761code)

### First-principles math (verified by hand, 2026-09-17)

Token I/O cost today (untied, vocab=65536, d_model=2560):
- embed_tokens: 65536×2560 = 167.77M params
- lm_head:      65536×2560 = 167.77M params
- Total ~335.5M ≈ 10.4% of 3.2B params ≈ 671MB bf16.

Byte I/O replacement:
- byte embed 256×2560 = 0.66M; byte head 2560×256 = 0.66M (+stop logit)
- Frees ~334M params / ~668MB bf16 against the 8GB server budget.

Sequence-length economics — MEASURED on forgelm_v2_tokenizer (2026-09-17):
- test_corpus.txt: 3.200 B/tok; test_corpus_large.txt: 3.746 B/tok → ~3.5×.
- Full byte granularity → ~3.5× backbone steps → ~3.5× decode latency;
  attention O(n²) → ~12× attn FLOPs (only 2/28 layers, KV/ctx grows 3.5×);
  Mamba-2 linear → ~3.5× SSM FLOPs for same text.
- BLT-style patching (avg patch ≈ 4 bytes) → backbone length ≤ token-length,
  compute ≈ neutral; byte work confined to small local encoder/decoder.
- Constrained decoding infra EXISTS: forge/engine/structured/xgrammar.py +
  logits-processor path in engine_generation.py → UTF-8 DFA mask slots in.

### RTU assessment (repo VERIFIED 2026-09-17 — github.com/jrz97619761/test-model-thing)

Actual impl (main.py, 252 lines, MLX, MIT):
- `Encoder` = nn.Embedding(256, dim) + `embedtrace` eligibility buffer.
- `Layer`: `state = σ(decay)·state + x + dummy`; `out = x + SiLU(W·LN(state))`.
  → diagonal-decay SISO SSM + residual readout. NOT the NeurIPS 2×2-rotation
  RTU — name borrowed, strictly weaker than Mamba-2 SSD.
- `dummy` trick: zero tensor added to state; ∂loss/∂dummy = ∂L/∂s_t per layer
  without BPTT ("dlds" signal).
- TRUE RTRL: `decaytrace = λ·trace + λ(1−λ)·s` = exact forward sensitivity for
  decay params; `embedtrace = λ·trace + onehot(byte)` = embedding eligibility.
  All other weights get 1-step myopic grads only.
- Loss: byte CE + stop MSE + latent MSE→sg(embed[next byte]) (JEPA-ish) +
  variance-floor anti-collapse.
- AdamW step EVERY BYTE; `layer.states` persist across calls + saved in
  checkpoint = the "memory". `notrace` mode skips state persistence →
  immediate breakdown (per README) — confirms state carryover IS the memory.
- Scale: 4.5M params (dim=512, L=16), ~12h simplewiki. Solo-dev PoC, no evals.

Verdict vs claims:
- "Continuous learning": REAL — genuine per-byte weight updates w/ partial
  exact forward-gradients. The only truly novel piece vs our stack.
- "Infinite memory": marketing — exponential decay, weaker than Mamba-2 state;
  same ERF ceiling (Topic 2).
- Per-byte serial fwd+bwd+AdamW: inherent scaling wall → why demo is 4.5M.
  Cannot run on 3.2B backbone; online updates only feasible on small
  adapter/decoder subset for us.

Extractable novelties for ForgeLM:
1. dlds trick → per-layer ∂L/∂state signal in one graph (diagnostics /
   forward-gradient research).
2. RTRL for diagonal decay generalizes to Mamba-2 per-head A:
   ds/dA = e^{∆A}(ds/dA + ∆·s) — O(n_state) trace → test-time plasticity of
   SSM timescales on frozen backbone, no activation storage. Genuinely novel,
   fits self_play/RSI infra.
3. Byte front-end (BLT-style patching) as planned.
4. UTF-8 validity-masked byte sampling — fixes their malformed-UTF8 issue.
5. Embedding eligibility trace → could augment MTP aux losses.

### Graft options (ranked)

1. **BLT-style byte front-end, backbone warm-start** (recommended):
   byte conv/local encoder → patch pool (≈4B) → existing 28L backbone
   (weights carry bit-exact, tokenization-agnostic) → patch→byte local
   decoder + stop head. Tokenizer excised, compute ≈ neutral, frees 668MB.
   Fits port-first directive: backbone identity, new modules zero-init.
2. **Full byte-level (video-faithful)**: 1 byte/step through backbone.
   Simplest, worst economics (~4× decode cost), MambaByte shows it works.
3. **Token+byte hybrid**: keep tokenizer, byte side-channel for OOV.
   Safest, least R&D value.

### Novel twists available (per directive C)

- Mamba-native dynamic chunking: use per-layer ∆t/state-norm as patch
  boundary signal instead of entropy model (H-Net uses a separate router).
- UTF-8 validity-masked byte sampling (finite-state constraint in
  decoding.py) — fixes video's malformed-UTF8 limitation for free.
- Tokenizer-excision port as a key class: `ByteFrontendKey` — backbone
  identity-init, encoder/decoder zero-init, bit-exact load test.
- RTU-style eligibility-trace test-time plasticity on byte decoder —
  dovetails with self_play/infinite_loop telemetry.

### Graft points (verified in codebase)

- `llm.py:61-62` — `self.embed = nn.Embedding(vocab, d_model)`,
  `self.head = nn.Linear(d_model, vocab)`; alt paths already exist
  (FactorizedEmbedding, KroneckerEmbedding, PIT, BitNet) → embed/head is a
  designed swap point.
- `llm.py:479` `x = self.embed(idx)`; `:649` `logits = self.head(hidden)` —
  the two backbone boundary calls.
- Tokenizer boundary is narrow: encode sites `engine_generation.py:352`,
  `engine_diagnostics.py:95`, `session_manager.py:651`, `async_d2h.py:150`;
  decode sites `engine_generation.py:187,1185` + per-token decode for
  streaming/tooling.
- `KroneckerEmbedding` (kronecker_embed_key.py) is token-id→bytes
  factorization, NOT byte-level text I/O — but shows the exact
  Key/KeyClass/KeyResult port pattern for a `ByteFrontendKey`.
- MTP head infra exists (`llm.py:158` tie_head_to_model) → precedent for an
  auxiliary stop head.

### Fact-check vs published SOTA (two research sweeps, 2026-09-17)

TMT trace math CONFIRMED exact RTRL — but per-layer only:
- `∂s_t/∂p = λ·∂s_{t-1}/∂p + λ(1−λ)·s_{t-1}` ✓ exact for decay params;
  embedtrace ✓ exact for embed rows. All cross-layer/other-param credit is
  1-step truncated. RTRL+AdamW combo is unanalyzed heuristic (lit uses SGD).
- TMT is a strict special case of **Zucchet et al., NeurIPS'23 (2305.15947)**:
  same diagonal-recurrence insight but traces ALL recurrent params at ~2×
  fwd cost, validated on LRA. RTU (2409.01449) = architectural twin.
- "Continuous learning" = **Dynamic Evaluation** (Krause'18, 1709.07432) /
  fast-weight lineage; the correct at-scale version is **TTT-E2E
  (2512.23675): whole-net test-time SGD, 3B params, 164B tokens, meta-learned
  init, 2.7× faster than FA @128K**. TMT lacks meta-init + parallelism + evals.
- Atlas (2505.23735) publishes the direct critique of TMT-style updates:
  myopic single-input memory writes are the documented defect → fix =
  windowed/pooled memory optimization (+80% @10M ctx BABILong vs Titans).
- Dohare/Sutton (Nature'24, s41586-024-07711-7): continual GD loses
  plasticity → any always-updating model needs reinit/perturbation budget.
- Persistent state rebutted: Illusion of State (2404.08819, TC⁰), PARITY
  (2405.17394), Stuffed Mamba (2410.07145), PackMamba (2408.03865 — SOTA
  practice RESETS state at doc boundaries). "Infinite memory" framing dropped.
- Anti-collapse = SimSiam-minus-predictor rescued by VICReg-style variance
  hinge. Published fragility: moving target (living teacher), unigram
  collapse, dead dims at hinge boundary. If used → VICReg verbatim or
  BYOL predictor+EMA.
- Entropy temp = linear variant of EDT (2403.14541); subsumed by
  min-p / adaptive-temp. Not worth extracting.

### Byte-level: graft recipe already published

- **Bolmo (2512.15586, Ai2)**: "byteified" OLMo-2-1B/OLMo-3-7B at <1% of
  pretrain budget, near-parity w/ source — EXACTLY our graft option 1,
  published + open. mLSTM encoder + non-causal boundary predictor.
- BLT (2412.09871): flop-parity with Llama-3 @8B, ≤50% fewer inference FLOPs.
- H-Net (2507.07955): 1-stage beats matched BPE transformer; 2-stage matches
  2×-size transformer; Mamba-2 local encoder — same family as our backbone.
- MambaByte (2401.13660): byte Mamba ≈ subword parity + speculative
  subword-draft/byte-verify 2.6× decode speedup.
- FastBLT (2605.08044): self-speculation fixes byte decode latency.
- UTF-8 validity masking unhandled in ALL flagship byte papers → real edge.
- T-FREE (2406.19223): tokenizer-free trigram-hash embeddings proven @3B —
  alternative to full byte I/O.

### Indie findings (community scan)

- **TRM** (2510.04871, 6.6k★): 7M params, recursion w/ deep supervision →
  45% ARC-AGI-1. "Recurse a shared block for depth" = cheapest transferable
  trick; MoR (2507.10524) is the production version w/ per-token depth.
- **HRM** (2506.21734, 12.6k★): 27M → ~40% ARC-AGI-1 — BUT ARC Prize teardown:
  hierarchy ≈ useless, outer refinement loop + augmentation = real driver.
  Lesson: community ablates everything; claims need public evals.
- **HRM-Text**: ~1B latent-recursion LM trained for ~$1–1.5k — hobbyist-scale
  recurrent-depth LM precedent.
- **RWKV-7** (2503.14456): generalized delta rule = in-context GD on state
  every token; proven @2.9B. Mamba-2 SSD is already a delta-rule variant →
  vector-valued in-context LR is the highest-leverage state-learning port.
- **Letta/MemGPT** (24.6k★): community's adopted "continual learning" =
  self-editing CONTEXT, weights frozen. Weight-plasticity is NOT what
  practitioners adopted — important reality check.
- **modded-nanogpt**: the verification-culture bar — public logs, one-command
  repro, named contributors. Spawned Muon.
- Forward-Forward graveyard: never scaled past MNIST-class — cautionary tale
  for local-learning rules.
- TMT community footprint: ZERO indexed discussion (no HN/Reddit/forks
  found). Gap is distribution (no checkpoint/metric/repro), not ideas.
- Nested Learning/HOPE (2512.24695): multi-timescale weight-update
  frequencies — principled version of "some weights online, most frozen."
- SEAL (2506.10943): episodic self-edits → persistent adaptation without
  per-step risk — pragmatic middle path.
- SnAp (2006.07232) / Sparse-RTRL (2603.15195): published cheap eligibility
  approximations (6% of Jacobian paths ≈ 84% adaptation) for non-diagonal
  weights.

### Ranked transferable ideas → ForgeLM V2 (post-verification)

1. RWKV-7 vector-valued in-context LR / decoupled add-remove keys on
   Mamba-2 state — proven @2.9B, highest impact/effort.
2. Exact RTRL on SSM decay params (Zucchet-style, per-head A_log traces) —
   cite Zucchet, not TMT. Test vs truncated-BPTT adapter + TTT-E2E-style
   meta-init.
3. Bolmo-style byteification of Jamba backbone (<1% pretrain cost) +
   UTF-8 FSM decode mask (unclaimed edge in all byte papers).
4. Atlas/MIRAS windowed surprise-gated memory writes — published fix for
   myopic updates.
5. TRM/MoR recursive depth on shared blocks — free effective depth.
6. T-FREE as cheaper tokenizer-free alternative (proven @3B).
7. MambaByte speculative byte/subword decode if byte gen is adopted.
8. SEAL episodic consolidation for self_play persistent adaptation.
9. Plasticity safeguards (continual-backprop reinit) — mandatory for any
   always-learning config.
10. bits-per-byte eval + public repro harness (nanochat norms) — required
    for any community-visible claim.
- SKIP: entropy-adaptive temp (subsumed), persistent-state-as-memory framing
  (rebutted), full-model per-byte updates (scaling wall).

### Open questions

- [x] Repo find → VERIFIED (section above). RTU = diagonal SSM + hand-rolled
      RTRL; 4.5M PoC.
- [x] Constrained decoding → xgrammar structured path exists; UTF-8 DFA
      mask = new logits processor on same hook.
- [ ] Entropy patcher vs fixed-4B patcher: measure bpb delta on test_corpus.
- [x] bytes/token measured: 3.2–3.75 → ~3.5× sequence blowup at byte level.

---

## TOPIC 4 — In-weight nano-training ("Experience LoRA") — gaps & solutions (2026-09-17)

User spec: passive learning during interaction + self-play; knowledge
survives context eviction; persists in weights OR tiny LoRA sidecar
("hypercompressed updater"); copy model file → knowledge maintained.

### Published near-exact analogs (the idea exists — convergent 2025-26 line)

- **TMEM** (2606.04536): fast LoRA Δₜ updated online within episodes via
  distilled QA supervision; π(θ₀+Δₜ); SVD-init. Beats retrieval/summary
  baselines on LoCoMo/LongMemEval across scales. = user's idea, published.
- **aTTT** (2607.03441): live LoRA updates in agent episodes via vLLM
  runtime LoRA API (1.9× cost); +5.0 ALFWorld. Failure found: drift on
  repeated update-text → fix = n-gram downweighting.
- **SCoL** (2605.07076): model LEARNS which layers to LoRA-update (meta-RL,
  Fisher-aligned sparse selection).
- **SEAL** (2506.10943): self-edit (data+hyperparams) → SFT → persistent
  weights; SQuAD 33.5→47.0% on 7B LoRA r16.
- **OPCD/context distillation** (2602.12275, 2503.08727): internalize context
  into per-doc LoRA by matching teacher-with-context hidden states — beats
  next-token CE which "performs poorly" for knowledge internalization.
- **Merge-before-Forget** (2512.23017): orthogonal LoRAs merged into ONE
  evolving LoRA — constant-memory single updater file.
- **Online-LoRA** (2411.05663): detects distribution shift → spawns new LoRA.
- **TTT-LoRA** (2411.07279): per-instance LoRA r128 at test time → 6× on ARC.
- **Dynamic Evaluation** (1709.07432): per-segment (~1K tok) updates w/
  decay-to-base prior; ~10-15% PPL gains. Cheapest proven cadence.

### Hard constraints (documented failure modes — design around these)

1. NEVER write base weights: sequential edits collapse — Mirage of Model
   Editing (2502.11177): ~10% success @1k edits; ROME/MEMIT catastrophic
   forgetting @1200 edits (2401.07453); single edit touching <1% params →
   ~0 on 8 downstream tasks (RECT 2401.04700); AlphaEdit bounded too.
   → additive sidecar ONLY; base frozen forever.
2. Raw text→weights DOESN'T WORK: verbatim writes are memorized but NOT
   extractable — 0% QA (Allen-Zhu Physics of LM 3.1, 2309.14316). MUST
   self-generate QA/implication pairs (augmentation mandatory). This is why
   SEAL/TMEM/SELF-PARAM all use self-distilled supervision.
3. Gekhman (2405.05904): finetuning on unknown facts → hallucination grows
   linearly w/ fraction of new facts → gate new-knowledge fraction per update.
4. Ripple-effect propagation unsolved for ALL parametric methods
   (RippleEdits 2307.12976) — stored facts don't propagate implications;
   in-context still wins there. Set expectations accordingly.
5. Evidence ceiling: nothing published beyond ~2k sequential edits or
   single-episode. Months-scale continual learning = unclaimed territory.
6. SRT warning (2505.21444): prolonged self-reward training → reward
   hacking → sudden collapse. Cap iterations, keep held-out verifier.

### Mechanics (cadence / gate / format / forgetting)

- Cadence: mini-batch ~256–1K tokens (Krause; TTT-E2E 1K; LaCT 2K–1M).
  Per-token unnecessary — no published benefit.
- Write gate: Titans surprise = ‖∇ℓ_assoc‖ w/ momentum S_t = ηS−θ∇ℓ +
  forgetting gate α·M (2501.00663). Atlas fix: optimize over last ~c tokens
  window, Muon/NS5-orthogonalized grads (2505.23735, +80% @10M ctx).
- MIRAS (2504.13173): any seq model = {memory arch, objective, retention
  gate, optimizer} — a LoRA updater IS "memory=adapter, retention=decay-to-
  init, optimizer=AdamW". Clean formalization.
- File size @ d=2560, 28L: LoRA r=4 on in/out_proj ≈ 2M params ≈ 4.7MB bf16
  / ~1.2MB int4. LoRA-XS (2405.17604): frozen SVD U,V + train r×r R only →
  ~16 params/matrix → KB-scale. VeRA (2310.11454): shared frozen random
  A,B + vectors → ~100-300KB. (IA)³ ~50-150KB. BitDelta (2402.10193) 1-bit
  deltas — only viable on param subsets. PiSSA (2404.02948): free +2-4pt
  upgrade to LoRA init.
- Forgetting: "LoRA Learns Less and Forgets Less" (2405.09673) — adapter
  isolation IS the mitigation, rank controls it. Replay 1–5% generic data
  (worth more than scaling, PMLR v330). O-LoRA (2310.14152): orthogonal
  subspaces, no replay needed. Merge many LoRAs → B-space cross-term
  interference ("Crowded in B-Space" 2604.16826) → keep versioned adapters,
  concat on load, merge only post-eval-gate.
- SSM-state-as-file: infra EXISTS (prefix_cache recurrent snapshots;
  Megatron #3225; Marconi 2411.19379 = ~34× hit rate) but ~15–30MB,
  prefix-locked, unproven as knowledge storage → warm-start cache only.

### Local feasibility (ForgeAI infra — verified in code)

- `.lora.safetensors` sidecar format exists (sft_train --save-lora-adapter).
- add_lora_adapters/merge_lora_adapters by module-name substring
  (bitnet_lora.py); load_lora re-creates + copy_ (engine_lora.py:74-87).
- self_play/infinite_loop.py ALREADY IS the consolidation loop: generate →
  SFT/GRPO LoRA finetune → merge → save epoch ckpt → _evaluate →
  promote/demote (lines 700-1160). The eval gate exists.
- GRPOTrainer + CPUAdamW proves LoRA training fits 12GB alongside 3.2B
  (~7.5GB total) → a ~1K-token backward micro-step ≈ seconds; amortizable
  to turn boundaries / idle ("sleep-time") slots.
- GAP: zero inference-time training hooks — all backward() in engine are
  quant calibration. The online updater is a genuinely new subsystem.

### Recommended architecture ("Experience LoRA" — convergent design)

1. Frozen base forever (constraint #1).
2. Persistent sidecar: versioned LoRA shards (MELO-style gated bank) OR one
   evolving LoRA (Merge-before-Forget). Start r=4–8 on in/out_proj+MLP
   subset ≈ 1–5MB; int4/int8 → sub-MB; LoRA-XS for KB extreme.
3. Write trigger: surprise-gated mini-batch (~512–1K tok) at turn/episode
   boundaries + idle-time slots. NOT per-token.
4. Supervision: self-generated QA/implication pairs + context distillation
   (teacher-with-context → student-without). NEVER raw stream CE for
   knowledge writes (constraint #2); stream CE ok for style/adaptation.
5. Per update: 1–5% replay mix + Gekhman cap on new-fact fraction + aTTT
   n-gram downweighting + optional O-LoRA orthogonality vs past shards.
6. Consolidation: periodic shard→accumulated-LoRA merge behind the EXISTING
   self_play eval gate (promote only on verified gain) → "improve not drift".
7. Retrieval gate for locality (GRACE ε-ball / MELO index / SCoL layer
   selection) so unrelated queries never see the delta.
8. Frontier option: dedicate one layer as Titans-style memory MLP, or
   persist DeltaNet-style state — checkpointable weight tensors; the SSM
   state channel is unvalidated lit-wise = real R&D white space.

### Open risks / open questions

- [ ] No published months-scale continual learning — we'd be past evidence.
- [ ] Ripple-effect generalization unsolved everywhere — knowledge stored
      ≠ knowledge usable transitively.
- [ ] Measure: micro-update latency on RTX 5070 (backward ~1K tok, LoRA-only
      grads, CPUAdamW vs GPU opt) — decide turn-boundary vs idle-only.
- [ ] Gate quality: does surprise-norm actually fire on knowledge-dense
      turns vs fluff? Probe script before committing.
- [ ] LoRA-vs-SSM-state capacity comparison for the same session knowledge.

### Addendum — THE FIX: minimal-training + minimal-context design (2026-09-17, two more sweeps)

#### Cost × durability spectrum (verified numbers)

| Write method | Cost/write | Durability evidence | Caveat |
|---|---|---|---|
| Steering vectors (ActAdd 2308.10248, ITI 2306.03341) | 1–2 fwd, ~KB | style/behavior only | CANNOT store facts |
| kNN-LM datastore (1911.00172 + 2109.04212) | ~1 fwd, ~4KB/entry | beats parametric on tail facts | retrieval-scope, no generalization; GBs |
| Doc-to-LoRA hypernet (2602.15902) | <1s → rank-8 file | near-perfect NIAH @4-5× ctx | meta-train ~GPU-weeks upfront |
| MEMIT closed-form + AlphaEdit null-space | ~2.7s/edit | 10k batched; 2k sequential | sequential collapse; Mamba LESS robust to edits → target GQA/MLP not SSM |
| RLSEdit (2601.15686) | per-edit O(1) via Woodbury | 10K sequential edits | newest, least replicated |
| LoCA closed-form ridge adapters (2608.03020) | ~seconds, fwd-only after 1 calibration bwd | 0.5–14B, beats LoRA CE 16/25 | generative-fact storage unproven |
| GRACE codebook (2211.11031) | ~100 GD steps ≈ s, KB/edit | thousands sequential | ε-ball coverage only |
| LoRA micro-SGD (~1K tok) | ~0.7s (local math: ~20 TFLOPs @5070) | LoRA-forgets-less | needs supervision data |
| S0 state files (2604.01168) | ~3min train, 48MB file | **VERIFIED on Mamba-2 hybrid** (FalconH1-7B); +23.6pp HumanEval (Qwen3.5-4B) | task-adaptation shown, fact-QA unproven |
| SSM/delta-rule state persist | 0 (forward IS the write) | Titans 2M NIAH; ReplaySSM systems | ~4MB/layer; no cross-session merge work exists = OPEN GAP |
| MeZO fwd-only (2305.17333) | ~30–100× MORE total compute | ~1% of FT | saves VRAM not time — wrong trade |

#### The context-cost fixes (verified)

- Supervision reuse: teacher KL pass can condition on **cached session KV**
  → marginal cost = generated suffix only (~5 diversified restatements per
  fact — Physics-3.1 multiplicity: 5 forms → 96% extractable vs 9.7% single).
- Replay-free that WORKS on generative: O-LoRA (T5 15-task), MIGU
  (2406.17245, +15.2%), Any-SSR (2503.13575 — RLS router ~100% acc on
  Llama2-7B). EWC-family does NOT hold up on generative.
- Always-on ungated sidecar = measurably harmful (2401.04700: <1% params →
  ~0 on 8 tasks; 2502.19416 Frobenius growth → subspace shift). GATE by
  activation router (WISE margin router) / ε-ball (GRACE) / index (MELO).
- Dedup mandatory: aTTT n-gram downweighting (repeated update-text → drift).
- Write gate = Titans surprise (‖∇ℓ‖ + momentum) + loss-plateau trigger
  (Online-LoRA) + learned policy option (Memory-R1 2508.19828: RL-trained
  ADD/UPDATE/DELETE/NOOP).
- Inference context tax: gated sidecar = 0 window tokens vs RAG's per-turn
  cost. KV-cartridges (2506.06266) = middle path (0 window, KV-resident).
- **"Do LMs Need Sleep" (2605.26099): DIRECT precedent on SSM-attention
  hybrids — context fills → offline passes write persistent fast weights
  into SSM blocks → clear KV. More sleep → better multi-hop/math.**
- Sparse Delta Memory (2607.07386): learned initial state as parametric
  memory measurably improves knowledge/reasoning — state-file evidence.
- WARNING: sequential edits on Mamba degrade faster than transformers
  (KTH thesis) → never write SSM projections; target GQA/MLP only.
- WARNING: "alignment tax" anecdote (sleeping-llm, non-peer): RLHF may
  suppress LoRA-injected knowledge at scale — verify on our model early.

#### The optimum (minimal-train + minimal-context)

1. Gate writes: surprise + dedup + plateau → most turns cost ~0.
2. Idle-time: generate ~5 diversified restatements/fact, suffix-only over
   cached session KV (SELF-PARAM/OPCD objective, KL with-ctx→without-ctx).
3. Write into gated LoRA sidecar: ~1s micro-SGD OR LoCA closed-form ridge
   OR D2L-style generated adapter (if hypernetwork amortizes).
4. Session S0 state file alongside (48MB, ~3min, 0 inference cost, VERIFIED
   on our arch family) — complementary channel at different timescale.
5. GRACE/MELO gate at inference → 0 window tokens.
6. Consolidate through existing self_play eval gate → merge on verified
   gain only.
7. Fallback for non-distilling facts: kNN datastore (free inserts, tail-
   fact SOTA) — the published "impossible triangle" resolution is hybrid:
   parametric for generalization + non-parametric for verbatim reliability.

---

## TOPIC 5 — R&D: cross-session SSM-state merging (the missing piece)

### The claim

Mamba-2 per-head state recurrence (sequential view):
  S_t = a_t·S_{t-1} + x_t·B_t^T     S ∈ R^{P×N} (P=headdim, N=d_state)
  read: y = S·C_t  →  y = Σ_i w_i·x_i·(B_i·C_t)   (w_i = Π a_j decay)

So the state IS an unnormalized associative moment matrix:
  M = Σ_i w_i·x_i·B_i^T      (value × write-key)

Augment the "experience file" with a per-head Gram:
  G = Σ_i w_i·B_i·B_i^T      (N×N — 64KB fp32 @d_state=128; cheap to
                              accumulate during prefill: += w_t·B_t⊗B_t)

Exact least-squares read:  y* = M·G⁺·c  → for c = B_i:
  y* = w_i·x_i  EXACTLY (B^T G⁺ B = I on the stored-key subspace, n≤N).
  Model's read path unchanged if we inject pre-normalized S' = M·G⁺.

Optimal closed-form merge of two session files:
  M_AB = M_A + M_B        (running sums add over the union multiset)
  G_AB = G_A + G_B
  S'_AB = M_AB·G_AB⁺      inject as merged SSM state

Why this is the right object: delta rule = online least squares; G is the
normalizer the SSM never materializes. Storing (M,G) makes the lossy
associative memory an EXACT least-squares memory (n≤N keys), and merging
becomes summation — no optimization, no interference beyond key-subspace
overlap.

### Caveats / failure modes to test

- Gram solves memory-side interference, NOT read-key alignment: retrieval
  still requires query's C_t to land in the stored-key subspace.
- n > N keys → G rank-deficient → ridge/pseudoinverse gives least-squares-
  best (graceful), not exact. Measure capacity curve.
- Correlated keys (similar facts/sessions) → G ill-conditioned → need ε.
- Decay weights w_i fold into both M and G consistently (weighted LS still
  exact); heavily-decayed writes vanish from both — consistent forgetting.
- Conv state + dt dynamics untouched — this is purely the memory matrix.

### Experiment plan

- [x] E1 toy (CPU): synthetic per-head SSM, measure recall RMSE vs #keys
      for merge ops: naive-add / weighted / Gram-exact / delta-rule seq.
      Orthogonal + clustered key regimes.
- [ ] E2 real: capture state via prefix_cache machinery after fact-session
      prefill; accumulate G via hook on B_t; merge two sessions; QA probe.
- [ ] E3 capacity: how many facts until merged recall degrades at
      d_state=128 vs naive baseline (expect ~N per head per layer ×26 layers).

### E1 toy results (toy_ssm_merge.py, 2026-09-18)

- GRAM merge EXACT (rel_err ~0.00–0.01, cos≈1.0) at n_total ≤ N keys;
  beats naive-add at EVERY capacity and both key regimes. Beyond n>N it
  degrades gracefully to least-squares-best while naive collapses
  (N=64: cos 0.68 vs 0.53 @2× overcap; N=16: 0.68 vs 0.54).
- Gram solve CANCELS decay: U^T G⁺ U = W⁻¹ (diagonal) → retrieves x_i
  unweighted — decay penalty removed, not just interference.
- delta-rule seq merge: unstable near n=N (pinv conditioning) — dropped.
- NOISE CRITICAL FINDING: real C_t ≠ B_i. Pure pinv amplifies read-key
  noise catastrophically (σ=0.05 → err 5.9 vs naive 1.3). Fix = Tikhonov
  λ: λ* ≈ σ²·N tracks read-key noise (σ=.05→λ~.1-.5, σ=.2→λ~2). At every
  σ, optimal-λ Gram beats naive on rel_err AND cos_sim.
- N=16 regime (real arch): exact only ≤16 keys/layer; Gram still optimal.

### E2 real-model results (e2_ssm_merge.py, ForgeLM_V2 3.2B, 2026-09-18)

Setup: FACT_A (56 tok) / FACT_B (42 tok) prefilled → per-layer `presents`
ssm_state; merged naive-add / avg; QA probe = teacher-forced logp margin
(correct vs a-priori-plausible distractor); GQA KV nulled in merged
conditions to isolate the SSM channel.

| cond | probeA | surpA (tail) |
|---|---|---|
| fresh | −3.93 | 2.42 |
| oracle (A+B seq KV) | −2.53 | **0.31** |
| A_only ssm-state | −4.09 | 2.33 |
| naive merge | −4.09 | 2.33 |
| avg merge | −4.16 | 2.33 |

**VERDICT: captured SSM state is ~4% of the oracle's verbatim surprisal
gain and ~0% on QA probes.** naive-add ≡ A_only identically — not because
merging works, but because a 16-dim-per-channel state after ~50 tokens is
a decaying summary; there is nothing to interfere with. The substrate is
the problem, not the merge op — Gram merge moot at this signal level.

**Redirect**: cross-session state *merging* is dead as a knowledge
carrier. The live variant is **S0-tuning** (2604.01168): state files
*trained* per session on distilled supervision, not captured — verified on
FalconH1-7B (same hybrid family). States only hold content if explicitly
optimized to. E3 capacity cancelled — meaningless at 4% signal.

### Real bugs found during E2 (all confirmed)

1. **`build_model_fast(cfg, checkpoint_path=None)` silently runs
   random-init weights** — `get_config('forgelm_v2')` carries no ckpt path;
   loader defaults to None, produces a uniform-output model (nll ≈ ln(vocab)
   = 11.09 on any text; top token prob 0.02%). No warning. Footgun: every
   "the model won't learn" symptom traces here first. Should hard-error or
   at least warn loudly when checkpoint_path is None for a preset that has
   a canonical checkpoint.
2. **`_ssm_state` capture path is dead for MambaLayer** —
   `_last_prefill_recurrent` + `capture_recurrent_state` read
   `attn._ssm_state`, which `MambaLayer.forward` never writes (attr exists,
   reset-only). Working path = `presents[i]['ssm_state']` via use_cache.
   `apply_recurrent_state_prefix` then restores only 'conv' — 'ssm' keys in
   snaps are silently ignored. Net: engine-level recurrent-state save/load
   is half-implemented for the production Mamba layer.
3. **Chunked-prefill needs `attention_mask`** — `model(ids, past_key_values=…)`
   without attention_mask leaves GQA on `is_causal` (top-left aligned) →
   cached KV unreachable, zero error. Engine always passes a ones-mask;
   model-level callers must too. (GQA concat path itself is correct.)
4. **forge/engine mixin layer is mid-refactor-broken** (untracked WIP, 48
   files dirty): `forge_engine` unimportable — `_FEATURE_REGISTRY` /
   `ActivationConfig` / `_apply_recurrent_state_prefix` /
   `_capture_recurrent_state` stale-import fixes applied
   (feature_registry.py / activation.py / prefix_cache.py are the real
   homes); remaining unresolved names incl. `nn` across several mixins —
   needs the shared-namespace decision finished, OR `model_loader`'s lazy
   `from forge.engine.forge_engine import _tokenizer_for_vocab` stays
   broken for non-canonical vocabs. Experiment bypassed via
   `ModelLoader.build_model_fast` + `get_tokenizer()` directly.

### Novelty verdict (research sweep): PARTIALLY CLAIMED, narrow moat

- Prior art: State Soup (2406.08423 linear state mixing), PICASO
  (2502.17605 CASO concat-simulation + sidecar weight matrix), document
  souping (2505.24033), Engrammics (naive superposition FAILS on shared
  key directions — direct motivation for G), axiom_engine (DARE-TIES on
  state files), RWKV blend_states ad-hoc.
- ACIL (2205.14922) OWNS the merge math — running R=ΣΦΦᵀ, Q=ΣΦY sums,
  pooled closed-form LS — but for frozen-feature classifier heads, never
  recurrent memory states. MUST cite + distinguish.
- Preconditioned DeltaNet (2604.21100) + mesa (2309.05858) + VLA
  (2605.11196) materialize inverse Gram INSIDE recurrence for exact
  recall — but not as persistent mergeable sidecar.
- OLAM (Kohonen) owns S = M·G⁻¹ classically.
- DEFENSIBLE CORE: (M,G) as first-class portable sufficient-statistic
  files → commutative order-free union-merge, provably pooled LS optimum,
  exact recall post-hoc, Marconi's exact-prefix constraint relaxed via
  union-join (vs PICASO concat-join vs CacheBlend recompute). Novelty =
  packaging/semantics + Mamba application, not the math.

### Real-arch correction (codebase report)

- ForgeAI "Mamba-2" = Mamba-1/Jamba per-channel scan: h (B,5120,16),
  B_t SHARED (16,) across channels, no heads. d_inner=5120, d_state=16,
  dt_rank=160, A_log (5120,16). Scan = _selective_scan_ref Python loop
  (mamba_probe.py:135-189) — pure torch, CPU-ok.
- Mamba layers: 26 (all except block idx 7,21 = GQA n_heads=20/kv=1/128).
- Per-channel structure: h[c,n] = Σ_t w_t[c,n]·B_t[n]·x_t[c],
  w_t[c,n] = dt[c,t]·Π_{s>t}exp(dt·A_neg[c,n]) — same P×N outer-product
  form as toy (P=5120, N=16) BUT effective per-channel key
  k̃_t[c] = w_t[c,:]⊙B_t (decay modulates each key-dim differently).
- Gram choice: per-channel G[c] = Σ_t (w̃_t[c]⊙B_t)(w̃_t[c]⊙B_t)^T or
  shared approximation — offline-computable from captured (B,delta,x,A).
- State file size: 5120×16 fp32 = 320KB/layer ×26 = ~8.5MB + G 5.2MB/L.
- Capture/inject: past_kv dicts {ssm_state, conv_state}; inject via
  past_key_values list arg (llm.py:380); conv_state needed for T=1 decode.
- capture_recurrent_state/_last_prefill_recurrent DEAD for MambaLayer
  (reads attrs never written) — use returned past_kv.
- ReplaySSMCache incompatible w/ MambaLayer (initial_state kwarg swallowed).
- LATENT BUG to verify: engine_sessions.py:9-10 imports
  _apply_recurrent_state_prefix/_capture_recurrent_state from
  engine_common — names don't exist there (live in prefix_cache.py).
  `import forge.engine.forge_engine` may raise ImportError.

### Merge operators under test

1. naive S_A + S_B           — baseline; interference ~ √n·key-corr
2. α·S_A + (1−α)·S_B       — weighted baseline
3. (M_A+M_B)·(G_A+G_B+λI)⁻¹ — proposed Gram merge (λ*≈σ²N noise-adaptive)
4. delta-rule seq merge      — DROPPED (pinstable near n=N)

## R&D: ForgeGate probes (2026-09-18)
Gate R route probe + Gate E doom monitor � single-model mode-switch
(direct vs think), ~60KB probes, ~0GB overhead, self-labeled training.

Pipeline (all temp scripts, C:\Users\tmk68\AppData\Local\Temp\):
- gate_r_dataset.py   build_large() ~1450 items (800 gsm8k + arith + wp + trivia + deceptive)
- gate_r_harvest.py   resumable batched greedy labels -> harvest_labels.jsonl / harvest_reasoned.jsonl
- gate_r_train.py     feature harvest (h_last+h_mean) -> harvest_feats.pt; probe -> gate_r_probe_v2.pt
- gate_r_v2.py        priming bakeoff + full-trace gen -> harvest_traces.jsonl
- gate_r_doom_eval.py teacher-forced per-pos hidden -> doom_feats.pt + doom_probe.pt
- gate_rt.py          live runtime: route -> monitored direct -> escalate-to-think

Numbers (held-out):
- Gate R: test AUC 0.798, per-cat p_easy ~= d_acc (calibrated)
- Gate E on routed items (K2 t0.85): 67% fail-detect @ ~6% FA; K3/t0.8 on p>=0.7: 66% @ 2.4% FA
- E2E simulated (1450): gated acc .480 vs think .441 vs direct .270; tok -9.5% vs think
- Live rt eval (hardest-quartile set): 149/150 correctly routed think, acc = always-think
- Harvest ~15min, features+train ~1.5min � inside 30min retrain cap

Negative results: gold-logprob/noul/12tok-truncation label proxies AUC<0.59;
raw early-exit draft agreement 1-24% (Mamba early layers dont decode);
doom probe unusable without hysteresis (FA 47-87% -> 2-6% w/ K-run + routed-prior).

Known limitation: direct mode emits ~96tok derivations despite </think> priming
(terse priming bakeoff: acc .30 tok 92 vs .23/95 � marginal). Next pools:
Gate C convergence early-exit (DEER-validated, biggest token pool),
terse-direct decoding, production wiring as generate(mode="auto").

## R50 missing-feature batch + R49-2 KDA (2026-09-19)

Shipped (survey -> implement -> CPU-tested; suite 2607 green):
- R50-1 DRY penalty: `_dry_penalties`/`_apply_dry` in engine_common (+ local
  copies in decoding.py to avoid import cycle); llama.cpp semantics �
  longest-repeated-suffix scan over last_n, penalty = mult*base^(r-allowed)
  subtracted in logit space. Wired through ALL gen paths + server models +
  model_registry. NOTE: temp==0 short-circuits before penalties (matches
  existing rep-penalty convention � DRY only applies when sampling).
- R50-2 DoLa: `DoLaDecoding`, `build_decoding("dola")`; dynamic premature-
  layer pick = argmax JSD over {n/4,n/2,3n/4}; needs return_hidden_states
  (already exposed for EAGLE). Novel angle: Mamba-vs-attn layer contrast
  on the hybrid � untested on real checkpoint yet.
- R50-3 PRM: `ProcessRewardHead` + `fit_prm`/`score_steps` in decision_head.py.
  Step-end positions via incremental encode (prefix+step+delim). Supports
  per-step labels OR scalar outcome broadcast (Math-Shepherd weak sup).
- R50-4 filler KV: `engine/kv/filler_kv.py` standalone cache + strategy
  wrapper + `build_kv_cache("filler")` + activation fallback map.
  `filler_token_ids(tokenizer)` derives English function-word/punct set.
- R50-5 depth upscale: `merge_models.py depth_upscale/parse_layer_map/
  depth_layer_types` + `--method depth` CLI; writes `.depth.json` sidecar.
  NOT lossless (documented � warm start for continued training).
- R49-2 KDA: `forge/keys/attention/kda_key.py` � KDALayer (per-key-dim
  decay, GDN init, conv k=4, fp32 naive scan, sigmoid out-gate, scalar
  gate=0) + KDAKey (BI; seeded deterministic port; convert_model_state
  whole-ckpt wrapper). Config flags: use_kda, kda_n_heads, kda_head_dim,
  kda_beta_gt1 (N3, off), kda_decay_floor. Wired: layers.py side-path on
  x0 (cached gate==0 skip), llm.py new-seq reset + prefill snapshot,
  prefix_cache capture/apply (KDA state continues across prefix hits �
  conv prefix + S state both restored). VRAM ~25M params/layer @2560/20H;
  state ~1.3MB fp32/layer; NO KV on KDA layers.

Bugs found+fixed (BUG_LOG 2026-09-19):
- snapkv/filler _evict bool-mask vs overgrown buffer (mask :total).
- _min_k_filter dead sensitivity mask � now rightmost-exceeding cliff
  (argmax fallback at sens=1). decoding.py copy had same bug.
- fit_prm inference-mode features/labels ? post-harvest re-clone.
- merge_models help-string unterminated quote (session-introduced).

Still open (next round candidates):
- KDA chunked-scan kernel (naive scan is O(T) python loop � fine for CPU
  tests/short seqs, needs WY-representation chunked path for GPU speed).
- KDA on real V2 checkpoint: bit-exact preset-lineage check vs V12.
- V13 preset (Rule A): V12 + use_uno/use_kda/use_mova/use_lightning_index
  /use_attn_output_gate/use_zc_rmsnorm � not created yet.
- MoVA key, DSA lightning indexer (R49-3/4), CALM cross-attn adapters,
  abliteration (guardrail eval needed � math-reasoning risk noted).
- DoLa/DRY/PRM on real checkpoint (all tested on stubs only so far).
- _forward_mod_skip path skips TITAN/MHC/KDA tail � check if KDA should
  apply there too (skipped tokens bypass block entirely by design � OK).

## R50 verification results (2026-09-19, smoke_r50_gpu.py on real V2)

- DRY on real checkpoint: repeated-suffix prompt ? token ' on' penalized
  -2526 logits; sampled continuation BROKE the repetition loop
  ("on the mat. The cat sat on the mat..." ? "with the mat... sat with
  the cat... on a mat... sat in"). DRY visibly changes behavior, not just
  plumbing.
- DoLa on V2: works after @torch.no_grad() fix on _contrast_logits
  (inference-tensor inputs + requires_grad params crashed; BUG_LOG'd +
  regression test). Output vs greedy: "Paris and the currency the euro.
  Population is ~67" vs "Paris. The capital of Germany is Berlin."
- KDA real-checkpoint port: convert_model_state on V2 sd ? 28 blocks ?
  strict=True load ? max|dlogit| = 0.0 vs baseline at 3.2B. Port-first
  verified at real scale, not just tiny.
- CPU smoke (smoke_r50_features.py): 23/23 � filler KV on real tokenizer
  (134 filler ids, 38?23 with 15 filler-first evictions), PRM harvest on
  real tokenizer + 65k-vocab tiny, depth upscale strict load + typed-
  layer negative check, KDA bit-exact + stateful decode on tiny.

## R50 feature benchmark (bench_r50_features.py, V2 bf16, RTX 5070, greedy decode)

| config | tok/s | peak GiB |
|---|---|---|
| baseline | 43.6 | 6.03 |
| +DRY | 44.6 | 6.03 |
| +DoLa fixed L8 | 43.1 | 6.03 |
| +DoLa dynamic (JSD all layers) | 37.4 | 6.03 |
| +KDA gate=0 | 39.8* | 7.75 (+1.72 weights) |
| +KDA gate=1 | 39.2* | 7.75 |

*KDA model loaded via plain ConfigurableResearchLLM, not FastBuild �
part of the ~4 tok/s gap vs baseline may be loader path; gate0 vs gate1
shows the delta-step itself is ~free at decode T=1. Naive-scan cost
shows at prefill (chunked path still pending).

- rep-3 on repetition prompt: 0.588 -> 0.000 with DRY (multiplier 1.0,
  base 1.75, allowed 2, last_n 512).
- DoLa dynamic -14% throughput = ~27 extra head projections/token.
- Filler-KV @4096 ctx, budget 1024: V2 MQA 4->1 MiB (marginal);
  qwen3_4b 36L/8KV 576->162 MiB (-414 MiB, -72%).
- KDA params: +922M (+28.8% params) for a KDA side path on all 28
  blocks � the future 3:1 hybrid preset would REPLACE layers, not add
  side paths; current port is per-block warm-start.

## R&D: checkpoint load speedup (2026-09-20) - SHIPPED

Goal: faster model load in GUI + ForgeEngine on RTX 5070 / ForgeLM_V2
6.39GB safetensors.

### Baseline profile (bench_load_time.py)
- total 5.22s: import 0.77, tokenizer 0.45, weight I/O **3.37s (dominant,
  ~1.9 GB/s)**, KV alloc 0.07, registry 0.22, warmup 0.94.

### Key finding
- `fastsafetensors` is installed but BROKEN on this box: "GPU runtime
  library not found (expected libcudart.so, libamdhip64.so, cudart64_XX.dll)"
  - silently swallowed by try/except fallbacks, so real loads used
  safetensors per-tensor pageable copies. Do NOT rely on that package here.
- Disk floor (warm page cache): ~1.96-1.99s for 6.39GB.

### Fix: `load_safetensors_pipelined` in forge/checkpoint_io.py
- Parse header (struct+json), one `torch.empty` CUDA alloc per tensor
  (same semantics as `get_tensor` - downstream weight surgery frees
  storage identically), N reader threads each with a 128MB pinned
  staging buffer, `readinto` file reads + per-tensor `copy_(
  non_blocking=True)` slices on a shared side stream; per-thread CUDA
  event guards staging-buffer reuse. Tensors never span groups.
- Wired in: `ModelLoader._load_safetensors_mmap` + `_load_sharded_
  safetensors` (between fastsafetensors attempt and safetensors fallback)
  and `load_checkpoint` CUDA path. All keep the safetensors fallback on
  exception. All `build_model_fast` callers benefit (GUI, sft_train,
  hotswap, lifecycle wake, self-play).
- `from_checkpoint` also starts `get_tokenizer` on a daemon thread and
  joins lazily - tokenizer (~0.45s) overlaps weight I/O.

### Measured (bench_weight_io.py / bench_load_time.py)
- safetensors per-tensor CUDA: 3.29s (1.94 GB/s)
- pipelined (8T/128MB pinned): **0.49-0.55s (~12-13 GB/s), 6.2x**
- 21/21 + 40/40 sampled tensors bit-identical; peak VRAM = model size.
- End-to-end `from_checkpoint`: **5.22s -> 2.42s** (weights 0.67s,
  warmup 0.92s and import 0.77s now dominate).
- GUI server preload (full HTTP path incl. uvicorn startup): ~3.4s to
  ready:true.

### GUI EngineService changes (engine_rt.py + deps.py)
- Preload resident model in background at `services.start()`:
  `FORGE_GUI_NO_PRELOAD=1` disables; `FORGE_GUI_PRELOAD` /
  `FORGE_GUI_PRELOAD_CONFIG` override checkpoint/config. Skips cleanly
  when checkpoint absent.
- `load()` during loading: identical req -> no-op; different req ->
  queued in `_queued_load`, runs via `_drain_queue` after settle.
- Bare `load()` for the already-resident model -> no-op (no wasteful
  unload+reload cycle).
- `unload()` during in-flight load: `_load_seq` bump invalidates the
  pending result; the superseded engine is freed via `_discard_engine`
  (sleep level=2 + empty_cache) instead of being installed.
- `unload()` uses `sleep(level=2)` - skips level-1's ~6.4GB GPU->CPU
  copy (~2s) since the engine is deleted anyway.

### Remaining load latency (not addressed)
- warmup ~0.92s, forge import ~0.77s (import-lazying or a warm daemon
  process would be the next levers), activation registry 0.22s.

## BUG-FIX: chat route probe vs tools block (2026-09-20)
Symptom: "Hello" in GUI chat -> ~150tok think parroting the master-prompt
guidelines, then a bland "Answer:". Root cause found + fixed same session.

Root cause (measured, ForgeLM V2 + gate_probes.pt):
- chat_loop scored _route_p_easy on the PRODUCTION render incl. the
  ~17-schema <tools> system block. Probe was trained tools-free; the
  ~1k-token schema block collapses h_mean -> p_easy drops ~0.25:
    q          tools   no-tools
    Hello      0.098   0.355
    hello      0.103   0.325
    hi         0.090   0.288
    hey        0.037   0.113
    thanks!    0.029   0.224
    2+2        0.234   0.549
    prime fn   0.007   0.069
- Threshold 0.2 was calibrated on tools=None renders -> every greeting
  fell under it in production. p_notools ~= p_qonly -> the system prompt
  contributes ~nothing; the tools block was the sole distractor.

Fix (chat_loop.py):
1. Probe input = tools-free render (same conv, tools=None). Matches the
   trained distribution, keeps system+history signal, immune to tool-set
   size. Generation still uses the tools render.
2. _is_trivial_turn: whole-message greeting/ack/closing regex (<=48 chars)
   short-circuits to direct BEFORE the probe � covers chit-chat the probe
   can't see (gate_r has no chit-chat class: hey=0.113, yes=0.135 even
   tools-free). Emits gate event p_easy=1.0.
3. tests/unit/test_gated.py::TestTrivialTurn (26 cases). bench docstring
   updated. 52/52 unit tests pass.

Residual / next-round candidates:
- Probe still blind to non-regex chit-chat ("got a sec?") -> gate_r
  retrain with a chit-chat class on the real render (30min pipeline,
  scripts in Temp may be gone � check).
- Think CONTENT quality is an SFT artifact: model parrots system-prompt
  guidelines in "we" voice, misassigns "You are ForgeLM V2" to the user.
  Mitigations: shorten/reword master prompt; ablate THINKING_PREFIX;
  hybrid SFT (think/no-think from DIFFERENT questions � Demystifying
  Hybrid Thinking 2510.12680) or DAST/AdaptThink (2503.04472/2505.13417).
- Fixed 160-tok cap -> DEER-style trial-answer confidence exit
  (2504.15895) or Dynasor certaindex (2412.20993), training-free.
- Expose per-request think_budget (Gemini thinkingBudget convention).

UPDATE (same session) � full fix set shipped + GPU-verified e2e:
- _conv_exit_observer: Gate C conv probe now wired into chat think path.
  generate_stream/hidden_observer plumbs per-step last-token hidden from
  _decode_tokens (return_hidden=True only when observer set). K=2
  consecutive conv>0.75 past 32 tok -> force-answer inject via
  _think_cap_processor(exit_flag=). Emits gate SSE {mode:"conv-exit"}.
  Fixed budget is now just the backstop.
- think_budget per-request: ChatSendRequest.think_budget -> run_chat ->
  cap (0 = no forced exit; clamps 8..4096). Chat settings "Think budget"
  NumInput (default 160, persisted in forge.chat.settings.v2).
- E2E (real engine + probes, run_chat directly):
    "Hello"     -> gate direct p=1.0 (trivial rule), clean greeting
    "hey"       -> direct (model still emits stray </think> mid-answer �
                   cosmetic; known direct-mode musing limitation)
    "what's 2+2"-> gate direct p=0.549 (probe on tools-free render)
    sqrt(2)     -> gate think p=0.02 -> conv-exit p=0.991 at ~35 think
                   tok -> full correct proof. Conv probe works live.
- 58 unit tests in test_gated.py (TestTrivialTurn + TestConvExitObserver
  + exit_flag case); 128 touched-area tests green; tsc clean.

UPDATE 2 � web-tool chat loop (e2e with real ToolHarness + DDG):
Symptoms reported: model "cannot use web tools" + think-voice text
leaking into the regular reply.

Findings (measured):
- web tools DO work end-to-end (defs in chat_tool_defs, dispatch at
  tool_harness.py:356, DDG returned results). The breakage was the LOOP:
- conv-exit fired on tool-CONTINUATION rounds (probe OOD there) ->
  forced "Answer:" made model restate its plan -> re-emitted the SAME
  web_search 5x, never synthesized.
- "check the news"/"search the web..." score p_easy~0.31 -> DIRECT path
  -> model writes musing in plain text then <tool_call> (direct mode
  still emits calls fine) -> musing persisted as the visible "reply".

Fixes shipped:
- fresh_turn gate: routing + conv observer only when conv[-1] is user;
  continuation rounds -> think + budget backstop only.
- _call_sig repeat guard: identical name+args repeat -> defs=None next
  round (forced synthesis); stray calls with defs=None are nulled.
- direct+tool_calls -> pre-call musing dropped from stored msg;
  _strip_direct_musing for text turns (answer = post-</think> tail).
- master_prompt capabilities line now mentions web tools.

Verified e2e: "check the news" -> direct -> web_search -> results ->
continuation synthesized a real headlines list. No loops, no gate spam.

Residual: DDG html parse returns ad-redirect URLs (y.js?ad_domain=...)
as results � filter ad/redirect links in forge/web_primitives.py later.

UPDATE 3 - news search fixed (the "agent cannot find today's news" pass):
Root cause: not a restriction problem - web tools were already
unrestricted (no safety gate, no approval gate, in chat_tool_defs +
agent harness; enabledTools UI filter is opt-in). The DATA was junk:
DDG html returned portal homepages (cnn.com/, nbcnews.com/) + y.js ad
redirectors for news queries - zero real headlines.

Fixes shipped (forge/web_primitives.py + web_tools.py):
- new news_search tool -> Google News RSS
  (news.google.com/rss/search?q=..&hl=en-US&gl=US&ceid=US:en; empty q
  -> top-headlines feed). Returns real {title,url,published,source,
  snippet}. Keyless. In chat_tool_defs + WebTools.NAMES automatically.
- parse_ddg_html now skips duckduckgo.com/ + ad_domain= links (y.js
  ads use u3= base64, not uddg= -> unwrap impossible -> drop).
- ddg_search falls back to google_news_search when 0 parseable results.
- agent_loop fallback ToolHarness now wires WebTools(enabled=True) so
  web tools survive even if deps.make_harness factory fails.
- agent DEFAULT_SYSTEM + master_prompt mention news_search.
- web_search description now says "for current news prefer news_search".

Regression found+fixed: _decode_tokens passed return_hidden=... to
model.forward unconditionally -> broke _EngineModel mocks in
test_recommended_improvements.py. Now only sends kwarg when
hidden_observer is set.

Verified LIVE e2e (real ForgeLM V2 + real ToolHarness, no mocks):
"check the news" -> gate direct (0.588) -> news_search({"query":"news"})
-> 5 real dated headlines (BBC/NBC/CNN/Fox, today) -> continuation
synthesized formatted headlines list w/ sources. 133 tests green.
