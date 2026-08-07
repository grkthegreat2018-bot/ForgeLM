# Novel Solution Ideation — Combining LLM Techniques

> A working doc for generating **novel** LLM research ideas by combining existing techniques so that each one solves the other's weakness. Compiled 2026-08-06. Companion to `LLM_TECHNIQUES_SURVEY.md` (the technique catalog) and `docs/keys/KEY_DEFINITION.md` (the weight-extraction theory).

---

## 1. The Methodology

The ForgeAI project's `AirMoE` key is the canonical example of the ideation pattern used here. The pattern is **not** "apply technique X." It is:

```
Technique A  ──has problem──►  P_A
Technique B  ──has problem──►  P_B
A + B        ──solves both──►  P_A fixed by B's strength, P_B fixed by A's strength
```

### The AirMoE worked example

| | Technique | Its strength | Its problem |
|---|---|---|---|
| A | **AirLLM** | Runs huge models on tiny VRAM by loading only the bare minimum, streaming the rest from disk as generation progresses | Pointless for small models; the streaming overhead dominates when the model already fits |
| B | **MoE** | Only the active expert computes, so quality scales with total params while FLOPs stay small | All experts still must sit in VRAM; memory grows with expert count |
| **A+B** | **AirMoE** | Tiny dense base always resident; experts hot-loaded like LoRAs only when the router calls them | Disk-transfer/load-speed bottleneck |

The combination's **remaining** problem (disk transfer) is itself the seed for the *next* idea (see §3 Idea 1: GhostMoE). **A good novel idea leaves a clearly-named residual problem** — that residual is the next paper.

### Rules for a candidate idea to count as "novel"

1. **Both inputs must be real, documented techniques** (cite them — see the survey).
2. **The combination must address a *named weakness* of each input**, not just stack them for more speedup.
3. **State the residual problem** the combination still has. If there is none, the idea is probably already published.
4. **Prefer lossless or provably-bounded combinations** over "train it and hope." ForgeAI's KeyStack philosophy: a key is FULL only if reversible + data→weight + composable.
5. **Check ForgeAI fit**: does it compose with existing keys (MLA, MoE, BitNet, Knowledge Pack, Fact Injection, GRAIL, Norm Folding, Expert Consolidation, DSpark, RotorQuant)?

---

## 2. Residual-Problem Map (seeds for new ideas)

Every implemented ForgeAI key has a residual problem. These are the raw material for combinations.

| Key | Residual problem (the seed) |
|---|---|
| AirMoE | Disk-transfer / load-speed bottleneck for cold experts |
| MoE (dense→experts) | Router starts uniform → needs fine-tuning; load imbalance |
| Expert Consolidation | Merging loses rare-expert specialization; threshold is heuristic |
| BitNet (ternary) | Accuracy cliff on small models; not lossless |
| MLA | Custom kernels needed; RoPE-on-latent is fiddly |
| Knowledge Pack | KV packs are large; injecting many domains blows VRAM |
| Fact Injection | Rank-1 per fact → capacity ceiling; fact conflicts overwrite |
| Context Patch | Patches interfere when stacked (shared directions) |
| GRAIL | Reconstruction map adds runtime matmul; needs calibration data |
| Norm Folding | Can't fold norms that follow dynamic/conditional ops (MoE router) |
| DSpark | RNN verifier needs fine-tuning; fixed verify threshold |
| RotorQuant | Rotation chosen offline; not content-adaptive |
| Lossless Quant Chain | int4 GPTQ calibration cost; rotation is global, not per-layer |
| Self-Play | Generated data has noise; confidence filtering is heuristic |
| Test-Gated Injection | Only covers testable facts; non-testable knowledge excluded |
| Hybrid Linear Attention | Linear-attention layers lose long-range recall (lossy) |

---

## 3. Novel Idea Catalog

Each idea follows the §1 template. ForgeAI-fit notes which existing keys it composes with.

### Idea 1 — GhostMoE: three-tier expert serving
**A = AirMoE** (problem: cold-expert disk transfer latency).
**B = SpecMoE** (2026, self-assisted spec-decode for offloaded MoE; strength: hides load latency behind speculation).
**C = LightMoE** (2026, similarity-based redirection to a resident expert, no I/O).
**Combination:** A resident core of *merged* experts (built by ForgeAI Expert Consolidation) handles the majority of tokens with zero load. When the router calls a non-resident expert, LightMoE-style redirect first tries the nearest resident merge-target (no I/O). For genuinely novel experts, SpecMoE-style speculation runs ahead to **prefetch** them from disk before they're needed. Three tiers: **resident → redirect → prefetch**. Each tier catches the failure mode of the one below.
**Residual:** Redirect quality depends on merge fidelity; prefetch wastes bandwidth on mis-speculated paths.
**ForgeAI fit:** composes AirMoE + Expert Consolidation + DSpark (spec engine).

### Idea 2 — TTT-Pack: turn throwaway test-time compute into permanent KV knowledge
**A = Policy of Thoughts (2026, per-instance transient LoRA via GRPO, discarded after).**
**B = Knowledge Pack (ForgeAI, pre-computed KV injection, zero token cost).**
**C = GRAIL (ForgeAI, ridge-regression reconstruction heals lossy transforms).**
**Combination:** PoT's transient per-instance adapter is *not* discarded. Instead, run the adapted model on the solved instance's prompt and capture the **KV delta** (adapted KV − base KV). GRAIL computes a closed-form ridge map from the prompt to that KV delta. Store the map as a Knowledge Pack. Next time a similar prompt appears, inject the KV delta directly — zero tokens, zero gradient, zero recompute. Failed instances produce **anti-packs** (subtractive deltas) that steer away from wrong reasoning.
**Residual:** KV delta is prompt-specific; generalization across prompts requires clustering deltas into domain packs.
**ForgeAI fit:** composes Knowledge Pack + GRAIL + Self-Play + CoT Knowledge Pack.

### Idea 3 — ExpertGenesis: continual learning by *growing* experts, never editing them
**A = Fact Injection (ForgeAI, closed-form rank-1 MLP updates store facts).**
**B = Self-Generated Replay (2026, model samples own distribution to prevent forgetting; *fails when capacity saturated*).**
**C = Mechanistic Forgetting (2026, forgetting is layer-localized to mid-deep FFN/expert blocks).**
**Combination:** Continual learning never overwrites an existing expert. Each new knowledge domain spawns a **fresh expert copy**, initialized by closed-form Fact Injection (rank-1 updates into the copy's MLP). The router is the only thing trained (small, cheap). Because experts are **append-only**, the mid-deep representation collapse that drives forgetting never happens — there is nothing to overwrite. Self-generated replay is needed *only for the router*, not the experts. Capacity grows linearly with knowledge; AirMoE/GhostMoE keeps cold experts on disk.
**Residual:** Expert count grows unbounded → routing becomes a retrieval problem; router replay can still drift.
**ForgeAI fit:** composes Fact Injection + MoE + AirMoE + Expert Consolidation (merge stale experts back down).

### Idea 4 — FoldedHeadMoE: zero-overhead head-axis MoE attention
**A = MISA (2026, MoE on the attention *head* axis of DeepSeek Sparse Attention).**
**B = QK-Norm MLA (ForgeAI, RMSNorm on Q/K with projection absorption).**
**C = Norm Folding (ForgeAI, fold RMSNorm into adjacent Linear, lossless).**
**Combination:** MISA's router scores each head from the query. ForgeAI already QK-norms Q. Fold the QK-Norm scale *and* the MISA router projection into the Q projection matrix (lossless, like Norm Folding). Active-head selection becomes a single fused matmul — the router adds **zero extra ops** beyond the Q projection that already exists. Head-axis sparsity at no runtime cost.
**Residual:** Folding only works for static (non-conditional) projections; the top-k *selection* still needs a gather.
**ForgeAI fit:** composes QK-Norm MLA + Norm Folding + MISA-style routing.

### Idea 5 — ConfSpec: conformally-guaranteed adaptive speculative decoding
**A = ORCA (2026, conformal prediction + TTT gives valid confidence on when to stop).**
**B = DSpark (ForgeAI, speculative decoding with confidence-scheduled verification).**
**Combination:** DSpark uses a *fixed* verify threshold. Replace it with ORCA's per-prompt conformal threshold: high-confidence prompts accept long spec drafts with few verify passes; low-confidence prompts verify strictly. The conformal guarantee (risk ≤ δ) carries to the *end-to-end* decoded output, not just the calibration module. A static knob becomes a principled per-prompt adaptive schedule with a correctness proof.
**Residual:** Conformal validity assumes the calibration set matches deployment; OOD prompts need online recalibration (→ TEMPO).
**ForgeAI fit:** composes DSpark + Self-Play confidence filtering.

### Idea 6 — FastExpertMerge: converged-quality MoE at 1-epoch cost
**A = Task Vectors = Gradients (2026, 1-epoch finetune task vector ≈ -lr·∇loss ≈ converged for merging).**
**B = Expert Consolidation (ForgeAI, merge similar MoE experts).**
**Combination:** Train *many* cheap 1-epoch LoRA-experts (gradient-equivalent to full training *for merge purposes*), then Expert Consolidation merges the similar ones into a compact set. You get a converged-quality routed MoE at ~1-epoch cost per expert instead of full pretraining per expert. The theory paper proves the approximation is bounded.
**Residual:** 1-epoch equivalence is for *merging*, not standalone use; un-merged single experts are weak.
**ForgeAI fit:** composes Expert Consolidation + DoRA + MoE split.

### Idea 7 — BudgetGuard: forgetting-aware non-uniform expert budget
**A = Alloc-MoE (2026, non-uniform expert activation budget per layer via DP).**
**B = Mechanistic Forgetting (2026, early attention = entropic dispersion, mid-deep FFN = collapse).**
**Combination:** Derive Alloc-MoE's per-layer budget map *from the forgetting analysis*: allocate more expert capacity to the mid-deep FFN layers that collapse under continual updates, less to the stable early attention layers. During continual fine-tuning, only the high-budget layers are touched; low-budget layers are frozen. Targeted capacity exactly where forgetting hits, none where it doesn't.
**Residual:** Forgetting profile is task-dependent; the budget map must be re-profiled per domain shift.
**ForgeAI fit:** composes MoE + SandwichNorm (stability) + continual self-play.

### Idea 8 — PocketExpert: 16-40× expert compression for disk-hot-loading
**A = AirMoE (hot-load experts from disk).**
**B = SVD expert compression (ForgeAI AirMoE already does 4-10×).**
**C = Lossless Quant Chain (ForgeAI, SpinQuant→QuaRot→int4, ~4×).**
**Combination:** Experts are stored on disk as **SVD-compressed int4** (rotate → quantize the low-rank factors). Disk transfer per expert drops 16-40×. On load: dequantize + SVD-reconstruct. A 671B-equivalent MoE's expert pool fits in pocket storage; only the 37B-active slice ever touches VRAM. Directly attacks Idea 1's residual (prefetch bandwidth).
**Residual:** SVD-reconstruct + dequant adds load CPU cost; rotation is global not per-expert.
**ForgeAI fit:** composes AirMoE + Lossless Quant Chain + SVD resize.

### Idea 9 — DoRAPatch: interference-free stacking of context patches
**A = Context Patch (ForgeAI, ICL effect = rank-1 MLP patch).**
**B = Pico (2026, LoRA merge interference comes from the B/output-side matrix; A stays task-specific).**
**C = DoRA (ForgeAI, decomposes weight into magnitude + direction).**
**Combination:** Apply context patches *only to the direction component* of DoRA (the B-equivalent), keeping the magnitude (shared, A-equivalent) frozen. Pico's insight — protect the shared directions, calibrate the task-specific ones — becomes: patches live in direction space, magnitude stays shared. Multiple context patches stack without overemphasizing shared directions. Patch composition becomes near-additive.
**Residual:** Direction-only patches are weaker per-patch; need more patches for same effect.
**ForgeAI fit:** composes Context Patch + DoRA + Self-Play Context Patch.

### Idea 10 — CompressedPack: pocket-sized zero-token knowledge
**A = Knowledge Pack (ForgeAI, pre-computed KV injection).**
**B = RotorQuant (ForgeAI, 3.88× Givens-rotation KV compression).**
**C = 4-bit KV (ForgeAI, group-wise KV quant with scale absorption).**
**Combination:** Pre-compute a domain's KV pack, *then* compress it (RotorQuant rotation → 4-bit quant with scales absorbed into QK-Norm). A full domain knowledge pack shrinks to a few MB. Inject dozens of domain packs simultaneously within a small VRAM budget. Zero-token knowledge at pocket size.
**Residual:** RotorQuant rotation is offline-fixed; 4-bit KV is lossy → pack quality degrades with compression depth.
**ForgeAI fit:** composes Knowledge Pack + RotorQuant + 4-bit KV + QK-Norm (scale absorption).

### Idea 11 — UncertainLearn: uncertainty-gated permanent-vs-transient learning
**A = DiSCTT (2026, route by epistemic uncertainty: high-consensus→SFT, low→RL).**
**B = Test-Gated Injection (ForgeAI, only inject test-verified facts).**
**C = Self-Play (ForgeAI, generate Q→A → confidence filter).**
**Combination:** Self-play generates solutions; **uncertainty estimation decides their fate**: high-consensus *and* test-passed → closed-form Fact Injection (permanent, baked into weights); low-consensus → RL refinement loop (transient, PoT-style, discarded). Uncertainty is the gate between permanent and transient learning. Prevents noisy/wrong facts from being baked in while still capturing high-confidence knowledge permanently.
**Residual:** Uncertainty estimation itself has error; borderline cases misrouted.
**ForgeAI fit:** composes Test-Gated Injection + Self-Play + recursive_self_play.

### Idea 12 — MTP-EAGLE-DSpark: native-MTP draft at zero draft-model memory
**A = MTP heads (ForgeAI, predict t+1, t+2 tokens, copy LM head + temporal shift).**
**B = EAGLE-3 (6.5× spec-decode, feature-level multi-layer fusion).**
**C = DSpark (ForgeAI, semi-autoregressive spec decode + RNN verify).**
**Combination:** Use ForgeAI's native MTP heads as EAGLE's feature-level draft predictor — no separate draft model loaded. DSpark's RNN verifier accepts/rejects. Native MTP → EAGLE-3 quality (6.5×) at **zero draft-model VRAM**. The draft model *is* the main model's own future-token heads.
**Residual:** MTP heads need fine-tuning to reach EAGLE-3 fusion quality; DSpark RNN still needs training.
**ForgeAI fit:** composes MTP + DSpark + EAGLE.

---

## 4. Research Gaps & Unexplored Combinations

Open directions where two literatures haven't met yet. Each is a candidate §3 idea.

- **Conformal guarantees × MoE routing.** No work gives a coverage guarantee on *which expert* is correct. Conformal over router logits → guaranteed-correct expert subset.
- **Mechanistic forgetting × Norm Folding.** Forgetting is layer-localized; Norm Folding eliminates norms. Can folded-norm layers be *frozen* during continual updates without losing the fold's losslessness? If yes, continual learning touches only unfolded (dynamic) layers.
- **MISA head-axis MoE × MLA.** MLA compresses KV; MISA sparsifies heads. Composing them: sparse *latent* heads. The low-rank latent may make head-selection cheaper (score in latent space).
- **Self-generated replay × Knowledge Packs.** Replay re-generates old data (token cost). A Knowledge Pack *is* the compressed old context. Replace replay tokens with KV-pack re-injection → replay at zero tokens.
- **Task-vector=gradient × Context Patch.** Context patches are rank-1 weight updates; task vectors are gradients. Are context patches *literally* one-step gradients of an ICL loss? If so, patch extraction = one gradient step, no closed-form solve needed.
- **Alloc-MoE budget × Sparse Frontier length-awareness.** Alloc-MoE fixes budget per layer; Sparse Frontier shows budget should grow with sequence length. Combine: budget = f(layer, seq_len), DP over a 2D grid.
- **Pico B-separation × GRAIL.** GRAIL reconstructs lost weights via ridge regression on B-space. Pico says B is the interference axis. GRAIL-in-B-space may heal merged-LoRA interference directly.
- **TEMPO EM × Self-Play loops.** TEMPO's EM framing (policy step + critic recalibration) is exactly a self-play loop with a missing recalibration step. ForgeAI self-play currently has no critic recalibration → adding it is a direct TEMPO application.
- **OAKS streaming facts × Fact Injection.** OAKS shows RAG/memory-agents fail at evolving facts. Closed-form Fact Injection with *overwrite-aware* rank-1 updates (update the same hidden dim = fact revision, new dim = new fact) is a direct fix the benchmark hasn't tested.

---

## 5. How to Stress-Test a Novel Idea (before building a key)

1. **Name both input techniques and their cited weaknesses.** If you can't, it's not a combination — it's one technique with extra steps.
2. **Write the residual problem.** No residual → likely already published; search before claiming novelty.
3. **Lossless check.** Is any step lossy? If yes, can GRAIL compensate it? If GRAIL can't, mark it LOSSY and forbid on ForgeLM V2/expert packs (per AGENTS.md policy).
4. **Composability check.** Does it chain with ≥2 existing FULL/BI keys? A key that breaks the KeyStack pipeline is lower value.
5. **Data→weight check.** Does it produce weights from data without a traditional training loop? If yes → candidate FULL key.
6. **Reversibility check.** Can you recover the pre-transform weights? If yes → lossless; if no → PARTIAL.
7. **Compute the parameter/FLOPS/VRAM delta** symbolically before implementing. Novelty that costs 10× compute is usually not worth it.
8. **Search the 2026 literature** (survey Part 6 + arXiv) for the exact combination. The field moves fast; "novel" in 2025 is often published by 2026.

---

## 6. ForgeAI Implementation Priority

Ordered by (expected impact × composability × losslessness). Lossless ideas rank higher.

| Rank | Idea | Lossless? | Composes with | Priority |
|---|---|---|---|---|
| 1 | Idea 10 CompressedPack | No (4-bit KV) | Knowledge Pack, RotorQuant, QK-Norm | High — easy, big VRAM win |
| 2 | Idea 4 FoldedHeadMoE | Yes | QK-Norm MLA, Norm Folding | High — lossless, zero overhead |
| 3 | Idea 8 PocketExpert | Yes (chain) | AirMoE, Lossless Quant, SVD | High — solves AirMoE residual |
| 4 | Idea 1 GhostMoE | Yes | AirMoE, Expert Cons., DSpark | High — completes AirMoE |
| 5 | Idea 12 MTP-EAGLE-DSpark | Partial | MTP, DSpark, EAGLE | High — big speedup |
| 6 | Idea 6 FastExpertMerge | Yes | Expert Cons., DoRA, MoE | Med — training cost win |
| 7 | Idea 9 DoRAPatch | Yes | Context Patch, DoRA | Med — clean patch stacking |
| 8 | Idea 11 UncertainLearn | Yes | Test-Gated, Self-Play | Med — quality gate |
| 9 | Idea 5 ConfSpec | Yes | DSpark, Self-Play | Med — correctness guarantee |
| 10 | Idea 3 ExpertGenesis | Yes | Fact Inject, MoE, AirMoE | Med — research-heavy |
| 11 | Idea 2 TTT-Pack | Yes | Knowledge Pack, GRAIL | Med — research-heavy |
| 12 | Idea 7 BudgetGuard | Yes | MoE, SandwichNorm | Low — needs profiling |

---

## 7. Adding a New Idea (template)

```markdown
### Idea N — <Name>: <one-line description>
**A = <Technique>** (<cite>, <strength>; problem: <P_A>).
**B = <Technique>** (<cite>, <strength>; problem: <P_B>).
[**C = <Technique>** ...]
**Combination:** <how A's strength fixes P_B and B's strength fixes P_A, mechanism>.
**Residual:** <the new problem this leaves>.
**ForgeAI fit:** composes <key1> + <key2> + ...
```

When an idea matures into an implemented key, move it to `docs/keys/` as a per-key doc (format: see `key_swiglu_ffn.md`) and register it in `research/keys/keystack.py` + `AGENTS.md`.

---

## Key Mapping (see `KEY_MAPPING_MASTER.md`)

Each idea → a ForgeAI key. Training avoided wherever possible.

| Idea | Key | Class | Train? |
|---|---|---|---|
| 1 GhostMoE | `ghost_moe_key.py` | TRIVIAL | None |
| 2 TTT-Pack | `ttt_pack_key.py` | PARTIAL | Minimal¹ |
| 3 ExpertGenesis | `expert_genesis_key.py` | PARTIAL | None |
| 4 FoldedHeadMoE | `folded_head_moe_key.py` | **FULL** | None |
| 5 ConfSpec | `conf_spec_key.py` | TRIVIAL | None |
| 6 FastExpertMerge | `fast_expert_merge_key.py` | PARTIAL | Minimal² |
| 7 BudgetGuard | `budget_guard_key.py` | TRIVIAL | None |
| 8 PocketExpert | `pocket_expert_key.py` | PARTIAL | None |
| 9 DoRAPatch | `dora_patch_key.py` | PARTIAL | None |
| 10 CompressedPack | `compressed_pack_key.py` | PARTIAL | None |
| 11 UncertainLearn | `uncertain_learn_key.py` | PARTIAL | None |
| 12 MTP-EAGLE-DSpark | `mtp_eagle_key.py` | PARTIAL | Minimal³ |

¹ Pack creation needs transient LoRA run; injection is TRIVIAL. ² 1-epoch LoRA per expert (theory: ≈ converged for merge). ³ MTP head fine-tune for EAGLE quality.

**Full forward/reverse sketches + registration plan:** `docs/research/KEY_MAPPING_MASTER.md`

---

*Compiled 2026-08-06. Source techniques in `LLM_TECHNIQUES_SURVEY.md` (Parts 1-6). Implementation status in `AGENTS.md`. Weight-extraction theory in `docs/keys/KEY_DEFINITION.md`.*
