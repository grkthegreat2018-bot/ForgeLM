# First-Principles Ideation — Original Theorizing (No Literature Aggregation)

> Ideas derived by reasoning from the *mechanics* of an LLM, not by combining two papers. Each entry: the **hidden assumption** the field makes, **why it's wrong/wasteful**, the **derivation** of a fix, and a **novelty check** (what existing work it's *adjacent* to, so we can later verify it's actually new). Compiled 2026-08-06 by first-principles reasoning.
>
> **These are original thoughts, not literature summaries.** They may turn out to be published — the novelty-check column is where we'd verify. The point is the *reasoning method*: find the assumption, break it.
>
> **Status:** compute busy; all ideas are theory-only. Each has a "cheapest test" note.

---

## The Method (different from the combination method)

`NOVEL_SOLUTION_IDEATION.md` combines *two existing techniques* to fix each other's weaknesses. This doc uses a different generative process:

1. **Name the hidden assumption** the architecture makes (the thing everyone accepts without question).
2. **Show why it's wasteful or wrong** — find the concrete inefficiency.
3. **Derive a fix from the mechanics** — not "apply technique X," but "the math says we could do Y instead."
4. **Novelty check** — what's the closest existing work? (To verify later, not to claim novelty now.)

The best ideas come from assumptions so embedded nobody states them.

---

## Idea F1 — Indexed Attention: attention is a database query with no index

### The hidden assumption
Attention computes `softmax(Q·Kᵀ)·V` — a full scan of all K/V for every Q. The field accepts this as "what attention is." Sparse attention hacks around it (top-k, windows) but still *scans* to decide what to keep.

### Why it's wasteful
A database with a billion rows doesn't scan to find a match — it uses an **index** (B-tree, hash, learned index). The KV cache *is* a database (keys=K, values=V, query=Q) and we do a full scan every token. At 128K context that's 128K dot-products per query head per layer per token. The scan is the cost; the actual *useful* attention mass is on ~a handful of tokens.

### Derivation
Build a **learned index over K** (à la Kraska et al. "learned indexes" for DBs, but that's the inspiration, not the technique-combination). Concretely:
- During prefill, train a tiny model `I(K_i) → bucket` that maps each key vector to a bucket id, such that keys in the same bucket have high mutual attention with the same queries (cluster the K-space by "which queries attend to them").
- At decode, `I(Q) → bucket` (or top-b buckets), then attention only over K in those buckets.
- The index is **learned from the attention patterns themselves** — it's a compression of "which Q attends to which K."
- Cost: O(buckets_hit × bucket_size) instead of O(T). If the index is good, that's O(log T) or O(1).

The key insight: **the index is a lossy compression of the attention matrix A itself**, not of K/V. We're not compressing the data; we're compressing *the routing pattern*. A = which-Q-attends-to-which-K is sparse and structured; the index exploits that structure.

### ForgeAI angle
This is what sparse attention *should* have been — instead of heuristics (SnapKV: keep recent+top, StreamingLLM: sink+window), learn the routing structure. Pairs with MISA (head-axis) but operates on the *token axis via an index*, not the head axis.

### Novelty check
Adjacent to: Quest (page-aware, but scores pages by scanning them), NSA (learned top-k but still computes scores for all), learned indexes (DB world, never applied to attention K). The "index the attention pattern, not the data" framing appears novel.

### Cheapest test
On a ported model, cluster the K vectors by which Q-rows attend to them (k-means on the column-normalized attention matrix). Measure: if Q→cluster is predictable, the index is learnable. ~50 lines, no training.

---

## Idea F2 — Memory Hierarchy in the Residual Stream

### The hidden assumption
The residual stream is a single fixed-width bus of dimension `d`. Every layer reads and writes the *entire* `d`-vector. There's no hierarchy — no registers, no cache levels.

### Why it's wasteful
Real memory systems have hierarchy because **access cost ≠ storage cost** and **working set ≠ total state**. A layer that needs 3 numbers shouldn't read/write all 1536. Most layers touch a small subspace of the stream (this is why SVD/SliceGPT works — the effective rank is low). We pay full-bandwidth I/O for a small working set.

### Derivation
Give the residual stream **three tiers** with different access costs:
1. **Registers** (`r ∈ R^{d_r}`, `d_r ≪ d`, e.g. 64): per-layer private state, read/written every layer, cheap. Each layer has its own register file. This is where a layer keeps its "current thought."
2. **L1 cache** (`c ∈ R^{d_c}`, `d_c < d`, e.g. 512): shared across a *block* of layers, read/written by the block, medium cost. A block's shared working state.
3. **Main stream** (`x ∈ R^d`): the full residual, read/written only at block boundaries, expensive.

A layer does: read register + L1 → compute → write register + (maybe) L1. Only every K layers does a layer flush L1 → main stream. The bandwidth drops by ~d/d_c for most layers.

The math: this is a **block-diagonal residual structure** — layers within a block operate on a low-dim subspace, and only block-boundary layers project back to full-dim. It's a generalization of mHC/xHC (which have N parallel full-dim streams) to a *hierarchical, different-width* structure.

### Why it might work
SliceGPT proves the effective rank per layer is low. If layer l only uses a 512-dim subspace, forcing it through 1536-dim is pure bandwidth waste. The register/L1 lets it work in its natural subspace.

### Novelty check
Adjacent to: mHC/xHC (parallel streams, same width), SliceGPT (prunes to low-rank but keeps one stream), MoD (skips layers). The *hierarchical different-width tiers with access-cost semantics* framing appears novel.

### Cheapest test
Per-layer SVD of the residual stream on a ported model; plot effective rank per layer. If most layers are low-rank, the hierarchy pays off. Already half-done by SliceGPT's analysis.

---

## Idea F3 — The Weight/Activation Spectrum (semi-static weights)

### The hidden assumption
Weights are static; activations are dynamic. This is binary. A weight is frozen after training; an activation is computed per-token.

### Why it's wasteful
The distinction is artificial — a weight is just an activation that got frozen. Fact Injection and Context Patch already blur this (closed-form weight updates from data). But they're treated as *exceptions*. The real question: **which weights should be how dynamic?** Currently it's all-or-nothing: attention (Q/K/V/O) = fully dynamic, FFN = fully static. There's no middle.

### Derivation
Assign each weight matrix a **update frequency** on a spectrum:
- `f=0` (frozen): standard weights, never change.
- `f=batch` (training): standard gradient updates.
- `f=session` (per-conversation): a small LoRA that adapts to the user/domain over a session, discarded after.
- `f=token` (per-token): a tiny rank-1 update computed from the current token's context, applied and rolled back.

The novel part: **FFN weights at f=token**. A per-token LoRA on the FFN, rank-1, computed in closed form from the current hidden state (no gradient). It's a "just-in-time" specialization of the FFN to the current topic. Think of it as: the static FFN is the *prior*, the per-token LoRA is the *posterior* given the current context.

Mathematically: `FFN(x) = (W_down + ΔW_down(x)) · silu((W_gate + ΔW_gate(x)) · x) ⊙ (W_up · x)`, where `ΔW = a(x)·bᵀ` is rank-1, `a,b` computed from x by a tiny predictor. The FFN adapts to each token without any persistent change.

This is **localized test-time training** — but closed-form, per-token, rolled back, no gradient. It's what PoT (transient LoRA) would be if it were per-token and closed-form instead of per-instance and gradient-based.

### ForgeAI angle
Composes with Context Patch (which is rank-1 MLP updates) but makes them *ephemeral and per-token* rather than permanent. The Context Patch is the mechanism; the spectrum is the framing.

### Novelty check
Adjacent to: PoT (per-instance transient LoRA, gradient), Context Patch (permanent rank-1), TTT (gradient updates). Per-token, closed-form, rolled-back FFN adaptation appears novel.

### Cheapest test
On a frozen model, compute a rank-1 ΔW_down from the current hidden state via least-squares against the "ideal" output (from a teacher), apply, measure output delta. If the delta is consistently beneficial, the per-token adaptation helps.

---

## Idea F4 — Defactoring: separate knowledge from computation

### The hidden assumption
The FFN stores facts *and* does computation. ROME showed facts live in MLP. But the same matrices that store "Paris is the capital of France" also do the arithmetic of reasoning. Knowledge and computation are entangled in the same weights.

### Why it's wasteful
- To update a fact, you touch computation (and risk forgetting/collapse — the 2026 mechanistic-forgetting result).
- To compress the model, you can't compress facts and computation differently (they're in the same matrix).
- The same fact is stored redundantly across layers (ROME + follow-ups showed facts spread).
- Domain knowledge requires retraining; reasoning transfers.

### Derivation
A **defactoring pass**: extract facts from the FFN weights into a separate **fact store**, leaving the FFN as a pure reasoning engine.
1. Identify "fact-storing" directions in W_gate/W_down (the rank-1 fact directions from Fact Injection theory).
2. Project them *out* of the FFN weights: `W_down' = W_down - Σ fact_i·v_iᵀ` (remove the fact directions).
3. Store the facts as (key, value) pairs in a fact pack (Knowledge Pack format).
4. At inference, the FFN reasons; the fact pack is injected as KV.

Result: a **smaller pure-reasoning model** + a **swappable fact pack**. Swap the pack = swap domains without touching the reasoning weights. Compress the reasoning model aggressively (it's computation, quantizes well); keep the fact pack higher-precision (it's knowledge, lossy to quantize). Different compression for different natures.

This is the radical version of Knowledge Pack: not *adding* knowledge, but *extracting* it so the model and the knowledge are decoupled. The model becomes a CPU; the fact pack is the disk.

### ForgeAI angle
This is the inverse of Fact Injection. Fact Injection *adds* facts to weights; defactoring *removes* them. Together: inject → (model has facts) → defactor → (facts in pack, model pure) → re-inject only the needed domain. A fact lifecycle. Composes with ExpertGenesis (facts → experts → defactor → packs).

### Novelty check
Adjacent to: Knowledge Pack (inject, don't extract), ROME (locates facts, doesn't remove them), Fact Injection (adds). The *extraction/decoupling* direction appears novel — everyone adds knowledge, nobody removes it to decouple.

### Cheapest test
On a ported model, identify the top-k rank-1 fact directions in a mid-layer FFN (via the Fact Injection formula inverted), project them out, measure: (a) does general reasoning PPL hold? (b) do fact-recall prompts degrade? If (a) yes and (b) yes, defactoring works — the facts were isolated.

---

## Idea F5 — Per-Query Adaptive Temperature (attention sharpness = confidence)

### The hidden assumption
Attention temperature is `1/√d` — a single global constant for all queries, all heads, all layers, all tokens.

### Why it's wasteful
The temperature controls attention sharpness. A query that *knows what it wants* should attend sharply (low temp → peaky). A query that's *exploring* should attend broadly (high temp → flat). Using one temperature for all is like a search engine that can't distinguish "I want exactly X" from "show me things like X." Some queries are retrieval (sharp), some are mixing (flat) — and the model has no knob.

### Derivation
Make temperature a **function of the query**: `T(Q) = f(Q)` where `f` is a tiny learned head (d → 1 scalar). The query itself predicts how sharp its attention should be.
- `A = softmax(Q·Kᵀ / T(Q))` where `T(Q) = softplus(w·Q + b)` (always positive).
- A confident query (high `w·Q`) → low T → sharp.
- An uncertain query → high T → flat.

This is *not* just learned temperature (that's a constant). It's **per-query, content-dependent sharpness** — the attention distribution's entropy is controlled by the query's own confidence. It's the attention analog of adaptive temperature in sampling, but inside the model.

Deeper: this lets different *heads* do different things automatically. A head whose queries predict low T becomes a retrieval head (sharp); a head whose queries predict high T becomes a mixing head (flat). The head specialization emerges from the learned temperature predictor, not from training differences.

### ForgeAI angle
Composes with QK-Norm (the temperature can be folded into the QK-Norm scale — both are per-Q scalars). Lossless at init (T=√d).

### Novelty check
Adjacent to: ASEntmax (learnable *global* temperature), entmax (sparse via temperature), sigmoid attention (no temperature). *Per-query predicted* temperature appears novel — the query decides its own sharpness.

### Cheapest test
On a ported model, compute per-query the "ideal" temperature (the T that would maximize the gap between top-1 and top-2 attention scores). Check if ideal-T is predictable from Q via a linear probe. If yes, the head learns it.

---

## Idea F6 — Continuous-Time Position Encoding (time, not token index)

### The hidden assumption
Position = token index (0, 1, 2, ...). RoPE rotates by `pos · θ`. Time is discrete and uniform — every token is one "tick."

### Why it's wasteful
In real use (chat, agents, streaming), time is **continuous and irregular**. A user who pauses 10 seconds then types is in a different temporal context than rapid-fire typing. A token generated after a long "thinking" pause vs. immediate generation. Token-index ignores all of this. For agents that act over time (browse, compute, wait), the *wall-clock* or *logical-step* time is meaningful signal that's discarded.

### Derivation
Replace integer position with **continuous time**: `pos = t` where `t` is a real-valued timestamp (wall-clock, logical step, or turn index). RoPE rotates by `t · θ`. Now:
- A 10-second pause → larger rotation gap → the model "knows" time passed.
- Rapid tokens → small gaps → "same moment."
- Turn boundaries → configurable time jumps.

For training: assign synthetic timestamps (e.g., uniform within a turn, jump between turns). The model learns that *temporal gaps carry meaning*. For inference: use real timestamps from the application.

Deeper: this gives the model a notion of **temporal decay** for free — old tokens rotate far, new ones near. NoPE works because the model infers position from context; continuous-time RoPE gives it *explicit* temporal structure that NoPE lacks, without RoPE's discrete-index failure (the 2026 "RoPE fails at long context" result is about *integer* positions fraying; continuous time with real gaps may not fray the same way).

### ForgeAI angle
Composes with YaRN (rescale the time base), PartialRoPE (apply to subset). New key: `continuous_rope_key.py`.

### Novelty check
Adjacent to: ALiBi (linear bias by position, discrete), TiE (time-aware embeddings in some NLP), RoPE (discrete). *Continuous real-valued timestamp* RoPE for LLMs appears underexplored — most PE work is about *longer* context, not *irregular* time.

### Cheapest test
Train tiny_model with `pos = token_index + noise(0, σ)` vs uniform. If the model is robust to jitter, continuous time is learnable. If it degrades, the model relies on exact integer spacing (and continuous time needs training).

---

## Idea F7 — Mechanistic Distillation (distill the process, not the output)

### The hidden assumption
Distillation matches *outputs*: logits, hidden states, or sequences. The student is told *what* to output, not *how to think*.

### Why it's wasteful
Two models can produce the same output via different internal processes. A student that matches logits but uses different attention patterns / expert routes has learned a *different function* that happens to agree on the training set — it won't generalize the same way. The teacher's *reasoning structure* (which heads fire, which experts route, which layers matter) is discarded.

### Derivation
Add **mechanistic losses** alongside output losses:
1. **Attention-pattern loss**: match the teacher's attention matrix `A_teacher` with `‖A_student - A_teacher‖²`. The student learns to attend *where* the teacher attends.
2. **Routing loss** (if MoE): match the teacher's expert-routing distribution per token.
3. **Layer-importance loss**: match the teacher's per-layer residual-delta norm (which layers contribute most). The student learns *which layers matter for which tokens*.
4. **Head-specialization loss**: match which heads the teacher activates per token-type.

The student doesn't just *answer* like the teacher; it *thinks* like the teacher. The internal trajectory is matched, not just the endpoint.

This directly addresses a known failure: small distilled models "cheat" — they find shortcuts that match logits on the distillation set but fail OOD. Matching the *process* forbids shortcuts because the shortcut uses different internals.

### ForgeAI angle
ForgeAI already does hidden-state distillation (matching intermediate activations). This extends it to matching *dynamic* quantities (attention, routing) not just static ones (hidden states). Composes with top-K KL (match the top-K attention patterns, not all — cheaper).

### Novelty check
Adjacent to: hidden-state KD (matches a static intermediate, not the dynamic pattern), attention transfer (exists in CV, rarely in LLM distillation), MiniLLM (reverse KL on outputs). *Multi-mechanism* (attn + routing + layer-importance) distillation for LLMs appears underexplored.

### Cheapest test
Distill tiny_model with and without an attention-pattern loss on a teacher. Measure OOD generalization delta. If positive, the process matters.

---

## Idea F8 — Shared Basis Layers (decompose weights across layers)

### The hidden assumption
Each layer has its own independent weight matrices. Layer 3's W_gate has nothing to do with layer 7's W_gate. Layers are independent functions.

### Why it's wasteful
Layers learn *similar* transforms (this is why model merging, layer pruning, and DenseFormer work — there's redundancy across depth). We store L independent copies of ~the same kind of operation. The parameter count scales with L × (params per layer), but the *effective* number of distinct operations is smaller.

### Derivation
Decompose every layer's weights over a **shared basis** of "primitive operations":
- `W_l = Σ_i c_{l,i} · B_i` where `B_i` are shared basis matrices (learned, common to all layers) and `c_{l,i}` are per-layer coefficients (small, L × n_basis scalars).
- The model is a composition over a shared basis, not L independent layers.
- Params: `n_basis × (d²) + L × n_basis` instead of `L × d²`. If `n_basis < L`, big savings.
- The basis is **interpretable**: each `B_i` is a primitive transform; each layer is a recipe of coefficients.

Deeper: this is a **tensor decomposition of the depth axis**. The full weight tensor `W ∈ R^{L×d×d}` is decomposed as `W = C ×_1 B` (mode-1 product), `C ∈ R^{L×n_basis}`, `B ∈ R^{n_basis×d×d}`. Low rank along the *layer* dimension. We know layers are redundant (DenseFormer, layer merging); this makes the redundancy explicit and parameter-efficient.

Risk: expressiveness. If the true per-layer transforms don't share a low-rank basis, this loses capacity. But the success of DenseFormer (layers are averages of each other) and layer-pruning (drop a layer, fine) suggests they do.

### ForgeAI angle
Composes with DenseFormer (DWA is a special case where the basis = {the layers themselves} and coefficients = the DWA weights). This generalizes it: the basis is *learned*, not the layers. Composes with Norm Folding (fold the per-layer coefficient into the norm scale).

### Novelty check
Adjacent to: DenseFormer (layers average each other — basis = layers), tensorized weights (per-layer, not cross-layer), MoE (experts as basis, but routed per-token not composed per-layer). *Cross-layer shared basis with per-layer coefficients* appears novel in this framing.

### Cheapest test
SVD the weight tensor `W ∈ R^{L×d×d}` (stack all layers' W_gate) along the layer axis. If the top-k singular values capture most energy with small k, a shared basis exists. ~10 lines on the ported checkpoint.

---

## Idea F9 — Working Memory Registers (persistent scratchpad across context)

### The hidden assumption
The model's "memory" is the context window (KV cache). There's no persistent state that survives eviction or crosses context boundaries. The model is amnesiac.

### Why it's wasteful
Humans have working memory (~7 items) that's actively maintained and separate from long-term memory. The KV cache is long-term (big, evicted); there's no small, *always-present*, actively-maintained working memory. RAG retrieves from external store but doesn't give the model a *private* persistent state. The model can't "hold a thought" across a context reset.

### Derivation
Add a small set of **working-memory registers**: `M ∈ R^{m×d}` (m ≈ 8, e.g. 8 vectors of dim d) that:
- Are **not part of the KV cache** (not evicted, not positionally indexed).
- Are **read/written by special control tokens** (`<read_reg i>`, `<write_reg i>`).
- Persist across context window boundaries (carried in the model's state, not the prompt).
- Are initialized to zero and updated by the model itself during generation.

The model learns to *use* its registers: store intermediate results, hold a plan, track state across a long reasoning chain. At context reset, the registers survive — the model "remembers what it was doing."

This is between weights (permanent, global) and KV cache (ephemeral, per-context). It's a *session-level* memory that the model controls. Think of it as a tiny differentiable scratchpad that's part of the architecture, not the prompt.

### ForgeAI angle
Composes with Knowledge Pack (registers could be pre-loaded with domain KV), online_learn (registers as the online-learning state), ExpertGenesis (registers track which expert is "active"). Pairs with the memory-hierarchy idea (F2): registers in F2 are per-layer; these are cross-context.

### Novelty check
Adjacent to: RAG (external retrieval, not model-controlled), scratchpads (in-prompt tokens, not persistent state), recurrent memory (LSTM cell state — but that's per-step, not cross-context), xHC streams (parallel, not persistent-across-context). *Model-controlled persistent registers across context boundaries* appears novel.

### Cheapest test
Add 8 register vectors to tiny_model, add read/write tokens, train on a task requiring multi-step state (e.g., "count occurrences across two separate prompts"). If the model learns to use registers, accuracy on cross-context tasks beats no-register baseline.

---

## Idea F10 — Task-Adaptive Precision (quantize what doesn't matter *for this task*)

### The hidden assumption
Quantization precision is per-layer or per-weight (a static property of the model). We pick a precision scheme at deployment and it applies to all inputs.

### Why it's wasteful
A weight's importance is **task-dependent**. The weights critical for code are irrelevant for poetry. We quantize everything to the same bits because we don't know which weights the *current task* needs. We're carrying high-precision weights that are dead weight for this input.

### Derivation
**Per-task adaptive precision**: at inference, a cheap profiler runs on the first few tokens and determines which weight blocks are *active* (high gradient-like signal) for this task. Load those at high precision from disk; keep the rest at low precision (or don't load — AirMoE style).
- This is AirMoE on the **precision axis** instead of the expert axis.
- The "profiler" is a single forward pass: which layers/heads/experts fire strongly → those need precision.
- A code prompt → high-precision for code-related experts + logic layers; poetry prompt → different set.

Deeper: this means the model has a **full-precision version and a low-precision version on disk**, and inference dynamically composes a mixed-precision model per task. The "model" isn't one set of weights — it's a *precision schedule* over weights, instantiated per input.

### ForgeAI angle
Composes with AirMoE (load from disk), Lossless Quant Chain (the two precision versions), Expert Consolidation (merged experts = the low-precision fallback). The novel axis: precision is per-task, not per-weight.

### Novelty check
Adjacent to: AirMoE (expert offload, not precision), mixed-precision training (static, not per-task), Wanda (saliency-based, static). *Dynamic per-task precision loading* appears novel.

### Cheapest test
On a ported model, run code vs poetry prompts, log per-layer activation magnitudes. If the active-layer sets differ by task, task-adaptive precision has signal to exploit.

---

## Idea F11 — Bidirectional Generation Mode (infill is native, not a special case)

### The hidden assumption
Generation is strictly left-to-right. The causal mask is fixed. Fill-in-the-middle (FIM) is a hack with special tokens that pretends to be bidirectional.

### Why it's wasteful
Many real tasks are *infill*: complete the middle given both sides (code completion, document editing, template filling). The model has the right context but is forced to pretend it doesn't (FIM reorders the prompt). The causal mask throws away half the available information for infill tasks.

### Derivation
Make the **causal mask generation-mode-dependent**:
- **Autoregressive mode**: standard causal mask (left-to-right). For open-ended generation.
- **Infill mode**: bidirectional mask for the *known* context (both left and right), causal for the *generation* region. The model attends to both sides of the gap while generating the gap.
- A mode token or the prompt structure selects the mask. No FIM reordering — the model sees the true left-right order with a hole.

The mask is: known-left tokens attend bidirectionally to all known tokens; generation tokens attend causally to known-left + bidirectionally to known-right + causally to earlier generation tokens. This is the *correct* attention pattern for infill, and it's just a mask change — no architecture change, no reordering.

### ForgeAI angle
Composes with FlexAttention (custom masks at FA speed), ScoPE (mask-based positioning). The mask is already a "key" (causal_mask_key.py); this makes it *conditional*.

### Novelty check
Adjacent to: FIM (reorders prompt, doesn't change mask), UL2 (mix of objectives), bidirectional encoders (BERT, but not generative). *Native bidirectional-infill mask in an autoregressive decoder* appears underexplored — FIM is the dominant hack but it's a hack.

### Cheapest test
On a ported model, run infill tasks with (a) FIM reordering, (b) a custom bidirectional-around-gap mask. Measure quality delta. If (b) > (a), native infill wins.

---

## Idea F12 — Attention as a Kernel Method with Per-Head Learned Kernels

### The hidden assumption
All attention heads use the same similarity function: dot product `Q·Kᵀ`. Heads differ only in their *learned projections* (W_q, W_k), not in *how* they compute similarity.

### Why it's wasteful
Dot product is one kernel (linear). Different heads might benefit from fundamentally different similarity geometry — a head doing exact match wants a sharp kernel (RBF), a head doing fuzzy semantic match wants a broad one, a head doing ordering wants a polynomial. Forcing all heads into linear kernel means each head's W_q/W_k must *encode the kernel shape into the projection*, wasting projection capacity on what could be a free kernel choice.

### Derivation
Give each head a **learned kernel function** `κ_h(Q, K)` instead of dot product:
- Head h: `A_h = softmax(κ_h(Q_h, K_h) / T)`
- `κ_h` is from a small family: RBF `exp(-‖Q-K‖²/σ_h)`, polynomial `(Q·K+1)^p_h`, or a learned MLP kernel.
- The kernel *shape* (σ, p, or MLP weights) is per-head, learned.
- Keep it efficient via Nyström approximation or the kernel trick (for polynomial/RBF, expand into feature maps → still linear in features).

The heads now have genuinely different *similarity notions*, not just different subspaces of the same notion. A head can be an exact-match head (sharp RBF) or a semantic head (broad RBF) by *design*, not by hoping W_q/W_k learn it.

### ForgeAI angle
Composes with QK-Norm (normalize before the kernel), MLA (kernel in latent space). The kernel parameters are tiny (1-2 scalars per head).

### Novelty check
Adjacent to: kernel attention (exists, usually one global kernel), Performer (random-feature approximation of a kernel, but fixed), linear attention (feature map = one kernel). *Per-head learned kernel family* appears novel — kernel attention work uses a single kernel, not a per-head learned one.

### Cheapest test
On a ported model, replace one head's dot product with RBF and measure if that head's attention entropy changes meaningfully (it should get sharper/different). Check if downstream PPL holds.

---

## Meta-Observation: The Pattern Across All F-Ideas

Every F-idea breaks an **assumption that's invisible because it's structural**:
- F1: attention scans (assumption: no index)
- F2: one bus width (assumption: no hierarchy)
- F3: weights vs activations is binary (assumption: no spectrum)
- F4: knowledge and computation are entangled (assumption: FFN does both)
- F5: one temperature (assumption: sharpness is global)
- F6: time = token index (assumption: discrete uniform time)
- F7: distill outputs (assumption: process doesn't matter)
- F8: layers are independent (assumption: no shared basis)
- F9: memory = context window (assumption: no persistent private state)
- F10: precision is static (assumption: importance is task-independent)
- F11: always causal (assumption: one mask)
- F12: one kernel (assumption: dot product is the only similarity)

**The generative rule:** find the thing the architecture treats as uniform/global/binary, and ask "what if it were structured/local/spectral?" The answer is usually a novel idea. This rule itself is the most useful output — it's a *machine for generating ideas*, not just a list.

---

## Priority (by originality × ForgeAI fit × testability)

| Rank | Idea | Originality | Lossless? | Cheapest test |
|---|---|---|---|---|
| 1 | F5 Per-Query Temp | High | Yes (init T=√d) | Linear probe on ideal-T |
| 2 | F8 Shared Basis | High | No (capacity risk) | SVD weight tensor on layer axis |
| 3 | F4 Defactoring | High | No (extraction) | Project out fact directions, check PPL split |
| 4 | F1 Indexed Attn | High | No (lossy index) | Cluster K by attention columns |
| 5 | F3 Weight Spectrum | High | Yes (ΔW=0 init) | Rank-1 ΔW from teacher |
| 6 | F7 Mech Distill | Med-High | Yes (extra loss) | +attn-pattern loss, OOD delta |
| 7 | F10 Task Precision | Med-High | Yes (load choice) | Log per-layer activation by task |
| 8 | F2 Memory Hierarchy | High | No (arch) | Per-layer effective rank (done by SliceGPT) |
| 9 | F9 Working Memory | High | No (arch) | Registers + read/write tokens |
| 10 | F6 Continuous Time | Med | No (needs training) | Jitter-robustness test |
| 11 | F11 Bidirectional | Med | Yes (mask) | Infill mask vs FIM |
| 12 | F12 Per-Head Kernel | Med | No (kernel change) | RBF on one head |

---

## How to Use This Doc

- **For ideation:** read the Meta-Observation. Apply the rule ("find the uniform/global/binary assumption, make it structured/local/spectral") to a component not yet covered. Add the result as F13+.
- **For verification:** the novelty-check column is a TODO — search before implementing.
- **For implementation:** start with the top-ranked ideas. F5 and F8 have the cheapest tests (no training, just analysis of the ported checkpoint).

When an F-idea matures, cross-link it in `NOVEL_SOLUTION_IDEATION.md` (if it becomes a combination) or `LLM_COMPONENT_ATLAS.md` (if it becomes a component refinement), then implement as a key in `research/keys/`.

---

## Key Mapping (see `KEY_MAPPING_MASTER.md`)

Each F-idea → a ForgeAI key. Training avoided wherever possible.

| # | Idea | Key | Class | Train? |
|---|---|---|---|---|
| F1 | Indexed Attention | `indexed_attn_key.py` | PARTIAL | None⁷ |
| F2 | Memory Hierarchy | — | — | **Full (NOT A KEY)** |
| F3 | Weight Spectrum | `weight_spectrum_key.py` | PARTIAL | None⁸ |
| F4 | Defactoring | `defactoring_key.py` | PARTIAL | None |
| F5 | Per-Query Temp | `per_query_temp_key.py` | TRIVIAL | None⁹ |
| F6 | Continuous-Time RoPE | `continuous_rope_key.py` | TRIVIAL | Minimal¹⁰ |
| F7 | Mechanistic Distill | `mech_distill_key.py` | PARTIAL | None¹¹ |
| F8 | Shared Basis Layers | `shared_basis_key.py` | PARTIAL | None |
| F9 | Working Memory Reg | — | — | **Full (NOT A KEY)** |
| F10 | Task-Adaptive Precision | `task_precision_key.py` | TRIVIAL | None |
| F11 | Bidirectional Gen | `bidirectional_key.py` | TRIVIAL | None |
| F12 | Per-Head Kernel | `per_head_kernel_key.py` | PARTIAL | Minimal |

⁷ Index learned by clustering attention patterns (k-means, no gradient). ⁸ Closed-form per-token ΔW=a·bᵀ (needs teacher signal; if no teacher, predictor needs minimal training). ⁹ Init w=0, b=√d → identity = standard attention (lossless at init). ¹⁰ Config is TRIVIAL; model needs fine-tune to learn temporal meaning. ¹¹ Extra loss terms during existing distillation (no extra training, no weight transform).

**NOT A KEY (2):** F2 Memory Hierarchy (arch redesign), F9 Working Memory Registers (arch addition) → `config.py`, not `keys/`.

**Full forward/reverse sketches + registration plan:** `docs/research/KEY_MAPPING_MASTER.md`

---

*Compiled 2026-08-06 by first-principles reasoning. No literature aggregation — all ideas derived from mechanism analysis. Novelty-check columns are verification TODOs, not novelty claims. Companion to `NOVEL_SOLUTION_IDEATION.md` (combination method) and `LLM_COMPONENT_ATLAS.md` (component atlas).*
