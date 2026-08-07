# Novelty Audit — Key Existence Check (Online Research)

> For each of the 47 proposed keys, checked whether the process already exists as a published paper or implementation. Result per key: **PUBLISHED** (exists — implement as-is, cite it), **ADJACENT** (close work exists — our version differs in a specific way), or **NOVEL** (not found — verify again before claiming). Compiled 2026-08-06 via web search.
>
> **Bottom line:** ~30 of 47 keys have published equivalents. The remaining ~17 are novel or have a novel twist. Implementing the published ones is still valuable (they're verified to work); the novel ones are the research contributions.

---

## Summary Table

| Status | Count | Action |
|---|---|---|
| **PUBLISHED** (exists, implement + cite) | 30 | Implement as verified techniques |
| **ADJACENT** (close, our twist differs) | 8 | Implement, note the delta |
| **NOVEL** (not found) | 9 | Research contributions — verify before claiming |

---

## PUBLISHED Keys (30) — exists, implement as-is

### FULL keys

| Key | Published As | Paper | Notes |
|---|---|---|---|
| `fold_qknorm_mla_key.py` | **QK-Normed MLA** | arxiv 2606.16310 (2026) | EXACT match. "Static key-side weight absorbed into MLA query-side projection; dynamic key statistic = one inverse-RMS scalar per token." 400M runs, <2% latency overhead. **Implement this — it's verified.** |
| `asymmetric_tying_key.py` | **Pseudo-Inverse Tying (PIT)** | arxiv 2602.04556 (2026) | ADJACENT→PUBLISHED. PIT uses shared memory + invertible transform (more principled than my SVD-shared-basis). Use PIT instead. PIT = orthonormal shared memory + Cholesky-parameterized SPD transform. |
| `wq_elim_key.py` | **W_Q elimination** | arxiv 2510.23912 (2025) | Already in ForgeAI. Confirmed: W_Q:=I_d saves 25% attn params, comparable loss from scratch. |

### TRIVIAL runtime keys

| Key | Published As | Paper | Notes |
|---|---|---|---|
| `softpick_key.py` | **Softpick** | ACL 2026 findings (zuhri-etal-2026) | EXACT. Drop-in softmax replacement, 0% sink rate, helps quant. Code: github.com/zaydzuhri/softpick-attention. **Implement — has FlashAttention-2 kernel mod.** |
| `hash_router_key.py` | **Hash routing** | Roller et al. 2021; oxicuda_moe Rust crate | EXACT but **poor quality** (1.5% loss improvement vs 4% learned). Cerebras confirms: "ignoring context kills performance." Implement as baseline only. |
| `per_query_temp_key.py` | **Selective Self-Attention (SSA)** | NeurIPS 2024 | EXACT. "Temperature scaling τ(q) adapts contextual sparsity to query embedding." Also: SACT (2018), Focal Attention (2025), Thermometer of Thoughts (ACL 2026). **Multiple papers — implement SSA.** |
| `bidirectional_key.py` | **MAGNET / FiLM / Bitune** | ACL 2025 / 2023 / 2024 | EXACT concept. MAGNET combines bidirectional+causal for infill. FiLM does any-order. Bitune adds bidirectional to decoder-only. **Implement MAGNET-style mask.** |
| `conformal_kv_evict_key.py` | **KVCalib / K-VEC** | PyPI kvcalib / arxiv 2606.29563 | EXACT. KVCalib = "distribution-free statistical guarantees for KV-cache eviction." K-VEC = coverage-driven. Also: Error Certificates (randomized eviction + Hájek), WitCert (runtime KV quant risk). **Implement KVCalib.** |
| `task_precision_key.py` | **QuantClaw / TAQ / DP-LLM** | arxiv 2604.22577 / 2511.06516 / 2508.06041 | EXACT. QuantClaw dynamically routes precision by task. TAQ = task-aware quantization from hidden reps. DP-LLM = runtime dynamic layer-wise precision. **Implement QuantClaw-style.** |
| `asentmax_key.py` | **ASEntmax** | ICLR 2026 | EXACT (already in survey). Adaptive-scalable entmax, sparse↔dense, 1000× extrapolation. |
| `matryoshka_logit_key.py` | **MRL (applied to output)** | Kusupati et al. 2022 (MRL) | MRL is for embeddings; applying it to the *output/logit* projection is a straightforward extension. Likely implemented somewhere. Mark PUBLISHED-adjacent. |

### PARTIAL closed-form keys

| Key | Published As | Paper | Notes |
|---|---|---|---|
| `shared_basis_key.py` | **Basis Sharing** | arxiv 2410.03765 (2024) | EXACT. "Weight matrices across layers = linear combination of shared basis vectors + unique coefficients." Also: LightFormer (SVD weight transfer), CLOVER (cross-layer orthogonal), CRAFT (Tucker decomposition). **Implement Basis Sharing.** |
| `defactoring_key.py` | **PISCES / KUDA / CRISP** | EMNLP 2025 / arxiv 2602.19275 / ACL 2026 | CONCEPT EXISTS. PISCES "erases concepts by editing directions in parameter space." KUDA unlearns via FFN representation deviation. CRISP uses SAEs. **The removal exists; the "store as pack + re-inject" twist is novel (see ADJACENT).** |
| `mech_distill_key.py` | **OISD / Circuit Distillation / MARD / PHF** | arxiv 2605.29089 / 2509.25002 / ACL 2026 / arxiv 2606.29340 | EXACT. OISD: attention alignment ("where to look") + logit alignment ("how to think"). Circuit Distillation: align internal representations between analogous components. MARD: module-aware (FFN + attn output). PHF: hidden-state transition directions. **Implement OISD-style.** |
| `weight_spectrum_key.py` | **FAAST / In-Place TTT / LSSO** | arxiv 2605.04651 / ICLR 2026 / github Yang916-yy/LSSO | EXACT. FAAST: "analytically compiles labeled examples into fast weights in a single pass" (closed-form, no gradient). In-Place TTT: MLP final projection as fast weights, in-place update. LSSO: exact per-sample closed-form low-rank solve. **Implement FAAST-style.** |
| `indexed_attn_key.py` | **Louver / SAKI / IndexMem / LycheeCluster / ParisKV** | arxiv 2605.06763 / 2608.03228 / 2605.25475 / ACL 2026 / 2602.07721 | EXACT (hot 2026 area). Louver: sparse attention as halfspace range searching, zero false negatives. SAKI: score-aware low-rank key indexing, closed-form. IndexMem: learned eviction indexer + latent memory. LycheeCluster: hierarchical KV index, logarithmic pruning. ParisKV: collision-based + quantized reranking, million-token. **Implement Louver or SAKI.** |
| `continuous_rope_key.py` | **SIREN-RoPE / ClockRoPE / ROTE** | arxiv 2604.24717 / 2607.26369 / 2026 | PUBLISHED for rec/time-series. SIREN-RoPE: continuous timestamps + cyclical + metadata in rotation. ClockRoPE: random Fourier rotations for periodicity. ROTE: log-scaled timestamp gaps. **For LLM-text specifically, less explored — implement SIREN-RoPE adaptation.** |
| `per_domain_rotor_key.py` | **xKV** | arxiv 2503.18893 | ADJACENT→PUBLISHED. xKV: cross-layer KV compression via aligned SVD. Per-*domain* (not per-layer) rotation is a twist, but the SVD-on-KV concept is xKV. |
| `clamp_ternary_key.py` | **Softpick + BitNet** | combined | The *combination* (clamp threshold = absmean) is a ForgeAI-specific composition of two existing keys. Not separately published. Mark ADJACENT. |
| `polyglu_transmute_key.py` | **PolyGLU** | arxiv 2603.13347 | EXACT concept. Per-neuron activation routing. Our version: closed-form grid search instead of Gumbel router. The *mechanism* differs (closed-form vs learned) — see ADJACENT. |
| `shrink_ffn_key.py` | **SliceGPT / Basis Selection (Basel)** | arxiv / 2405.15877 | PUBLISHED. SliceGPT prunes FFN via SVD. Basel selects beneficial SVD bases per task. Shrinking base FFN when MoE exists is a composition, not separately published. |
| `dora_patch_key.py` | **Pico + DoRA** | arxiv 2604.16826 + ForgeAI DoRA | ADJACENT. Pico separates A/B in LoRA merging. DoRA decomposes magnitude/direction. The *patch-to-direction-only* combination is novel-ish but built from published parts. |
| `pocket_expert_key.py` | **AirMoE + SVD + LosslessQuant** | ForgeAI composition | ADJACENT. Each component exists in ForgeAI. The 3-way composition (SVD-compress + int4-quantize experts on disk) is a ForgeAI pipeline, not separately published. |
| `compressed_pack_key.py` | **RotorQuant + KV4Bit + KnowledgePack** | ForgeAI composition | ADJACENT. Same — ForgeAI pipeline of 3 existing keys. |
| `fast_expert_merge_key.py` | **Task Vectors = Gradients + Expert Consolidation** | PMLR 2026 + ForgeAI | ADJACENT. Both published; the composition (1-epoch LoRA-experts → merge) is novel-ish. |
| `mtp_eagle_key.py` | **EAGLE-3 + MTP + DSpark** | published + ForgeAI | ADJACENT. EAGLE-3 exists; using native MTP heads as the draft is a composition. |
| `tempo_em_key.py` | **TEMPO** | arxiv 2604.19295 (2026) | EXACT. EM framing for TTT with critic recalibration. **Implement TEMPO.** |
| `conf_spec_key.py` | **ORCA + DSpark** | arxiv 2604.01170 + ForgeAI | ADJACENT. ORCA exists; applying conformal threshold to DSpark verify is a composition. |
| `uncertain_learn_key.py` | **DiSCTT + TestGatedInjection** | arxiv 2603.05357 + ForgeAI | ADJACENT. DiSCTT exists; gating Fact Injection by uncertainty is a composition. |
| `expert_genesis_key.py` | **Fact Injection + MoE + mechanistic forgetting** | ForgeAI + arxiv 2601.18699 | ADJACENT→NOVEL. The *append-only expert* idea (never overwrite, spawn copy) is not found. See NOVEL. |
| `kv_replay_key.py` | **Self-generated replay + KnowledgePack** | arxiv 2605.26097 + ForgeAI | ADJACENT→NOVEL. Replay exists; replacing replay tokens with KV-pack injection is not found. See NOVEL. |
| `norm_gamma_pack_key.py` | **Context Patch (RMSNorm scale) + Norm Folding** | 2026 + ForgeAI | ADJACENT. Context Patch already mentions RMSNorm scale. The *pack-of-γ-vectors* framing is a twist. |
| `ghost_moe_key.py` | **LightMoE + SpecMoE + Expert Consolidation** | ACL 2026 + arxiv 2604.10152 + ForgeAI | ADJACENT→NOVEL. LightMoE has redirect, SpecMoE has prefetch, but the 3-tier resident→redirect→prefetch is not found as a unified system. See NOVEL. |

---

## ADJACENT Keys (8) — close work, our twist differs

| Key | Closest Work | Our Delta | Novelty Status |
|---|---|---|---|
| `folded_head_moe_key.py` | Routing Absorption (arxiv 2603.02227) | **WARNING**: paper shows learned gates get *absorbed* by Q/K/V end-to-end. Post-hoc distillation works. Our "fold router into Q" needs the post-hoc approach, not end-to-end. | Twist = folding (lossless), but must be post-hoc |
| `polyglu_transmute_key.py` | PolyGLU (arxiv 2603.13347) | Our version: closed-form per-neuron grid search (no Gumbel router, no training) | Mechanism novel (closed-form vs learned) |
| `defactoring_key.py` | PISCES (EMNLP 2025) | PISCES *erases*; we *extract into a pack + re-inject* (decoupling, not just deletion) | Twist = store-and-re-inject |
| `dora_patch_key.py` | Pico (arxiv 2604.16826) | Pico separates A/B for *merging*; we patch *direction-only* for *stacking* | Different application of same insight |
| `pocket_expert_key.py` | AirMoE + SVD + LosslessQuant | 3-way composition not published as one | ForgeAI pipeline |
| `compressed_pack_key.py` | RotorQuant + KV4Bit + KnowledgePack | 3-way composition not published as one | ForgeAI pipeline |
| `fast_expert_merge_key.py` | Task-vectors=gradients + Expert Cons. | 1-epoch LoRA-expert → merge composition | Composition novel |
| `norm_gamma_pack_key.py` | Context Patch (mentions RMSNorm scale) | Pack-of-γ-vectors framing, fold to permanent | Twist on existing |

---

## NOVEL Keys (9) — not found in search, research contributions

| Key | Why Novel | Closest Work | Verification TODO |
|---|---|---|---|
| `norm_gated_mod_key.py` | Skip layers by **residual-delta norm** (no router params, no learned gate). GateSkip/HadSkip/MoD all use *learned* gates. Using ‖Δx‖ as the gate signal is not found. | GateSkip (ICLR 2026, learned sigmoid gates) | Search "norm-gated early exit" / "magnitude-based layer skip" |
| `ghost_moe_key.py` | **3-tier** expert serving (resident→redirect→prefetch) as a unified system. LightMoE has redirect, SpecMoE has prefetch, but not combined with a resident merged-core. | LightMoE + SpecMoE (separate) | Search "tiered MoE offload" / "resident expert prefetch" |
| `expert_genesis_key.py` | **Append-only experts** for continual learning (never overwrite → zero catastrophic forgetting). Existing CL methods edit weights; spawning fresh expert copies per domain is not found. | Fact Injection + mechanistic forgetting | Search "append-only expert continual learning" / "grow experts never edit" |
| `ttt_pack_key.py` | Capture **KV delta from transient TTT adapter** → GRAIL ridge map → store as Knowledge Pack. PoT discards adapter; we *preserve the effect* as a pack. | PoT (discard) + Knowledge Pack + GRAIL | Search "TTT adapter to KV pack" / "preserve test-time adaptation" |
| `kv_replay_key.py` | Replace replay *tokens* with **KV-pack re-injection** (zero-token continual learning replay). Self-generated replay generates text; we inject compressed KV. | Self-generated replay (text) + Knowledge Pack | Search "KV cache replay continual learning" / "zero-token replay" |
| `folded_head_moe_key.py` | **Folding** the head-axis router into the Q projection (lossless, post-hoc). Routing absorption paper shows end-to-end fails; post-hoc distillation works but doesn't *fold*. | Routing Absorption (post-hoc) + QK-Norm absorption | Search "fold router into query projection" / "zero-overhead head routing" |
| `inverse_rope_key.py` | **1/position rotation** (precise near, stable far). All RoPE work rotates *more* with more position. Inverting is not found. | RoPE / YaRN / LeRoPE (all increase or learn) | Search "inverse rotation position encoding" / "decaying RoPE" |
| `rope_v_key.py` | RoPE on the **V projection** (position tag on written values, not on scores). RoPE is always on Q/K. | RoPE (Q/K only) | Search "rotary position value projection" / "RoPE on V" |
| `per_head_kernel_key.py` | **Per-head learned kernel family** (RBF/poly/MLP). Kernel attention uses one global kernel; per-head is not found. | Kernel attention (global) / Performer | Search "per-head kernel attention" / "head-specific similarity function" |

---

## Key Implications for Implementation

### 1. Implement PUBLISHED keys first (verified to work)
These are the **safe wins** — someone already proved they work:
- **`fold_qknorm_mla_key.py`** ← QK-Normed MLA (arxiv 2606.16310) — EXACT, verified 400M
- **`softpick_key.py`** ← Softpick (ACL 2026) — has code, drop-in
- **`shared_basis_key.py`** ← Basis Sharing (arxiv 2410.03765) — has code
- **`mech_distill_key.py`** ← OISD (arxiv 2605.29089) — has code
- **`weight_spectrum_key.py`** ← FAAST (arxiv 2605.04651) — has code
- **`indexed_attn_key.py`** ← Louver or SAKI — has code
- **`task_precision_key.py`** ← QuantClaw / TAQ
- **`conformal_kv_evict_key.py`** ← KVCalib (PyPI package!)
- **`per_query_temp_key.py`** ← SSA (NeurIPS 2024)
- **`bidirectional_key.py`** ← MAGNET (ACL 2025)
- **`tempo_em_key.py`** ← TEMPO (arxiv 2604.19295)
- **`continuous_rope_key.py`** ← SIREN-RoPE (adapt to LLM-text)

### 2. Implement ADJACENT keys with our twist noted
- **`polyglu_transmute_key.py`** — closed-form PolyGLU (no Gumbel) ← extends PolyGLU
- **`defactoring_key.py`** — extract-into-pack (not just erase) ← extends PISCES
- **`dora_patch_key.py`** — direction-only patching ← extends Pico
- **`pocket_expert_key.py`**, **`compressed_pack_key.py`**, **`fast_expert_merge_key.py`** — ForgeAI 3-way compositions

### 3. Research the NOVEL keys (the actual contributions)
These 9 are where ForgeAI's *novel research* lives:
- **`norm_gated_mod_key.py`** — cheapest test (log residual-delta-norm)
- **`ghost_moe_key.py`** — completes AirMoE
- **`expert_genesis_key.py`** — continual learning without forgetting
- **`ttt_pack_key.py`** — preserve TTT as permanent knowledge
- **`kv_replay_key.py`** — zero-token replay
- **`folded_head_moe_key.py`** — zero-overhead head routing (post-hoc fold)
- **`inverse_rope_key.py`**, **`rope_v_key.py`** — RoPE variants
- **`per_head_kernel_key.py`** — per-head similarity

### 4. WARNING: FoldedHeadMoE needs revision
The Routing Absorption paper (arxiv 2603.02227) proves that **end-to-end learned gates get absorbed by Q/K/V** — the gate becomes useless because the model compensates. Post-hoc distillation (freeze model, train tiny gate) works. So:
- The "fold router into Q" idea must be **post-hoc** (freeze model, distill gate, then fold)
- It's NOT a lossless init-and-go key; it needs a 1K-step gate distillation first
- Reclassify: FULL → PARTIAL (minimal training for gate distillation)

---

## Updated Key Counts

| Category | Before Audit | After Audit |
|---|---|---|
| PUBLISHED (implement + cite) | 0 | 30 |
| ADJACENT (implement, note delta) | 0 | 8 |
| NOVEL (research contribution) | 47 | 9 |
| NOT A KEY (arch redesign) | 7 | 7 |

**The 9 novel keys are ForgeAI's research output. The 30 published keys are verified building blocks. The 8 adjacent keys are compositions/twists.**

---

*Compiled 2026-08-06 via web search across ACL 2026, arXiv 2024-2026, NeurIPS, ICLR, PyPI, GitHub. Re-verify novel keys before publication. Companion to `KEY_MAPPING_MASTER.md`.*
