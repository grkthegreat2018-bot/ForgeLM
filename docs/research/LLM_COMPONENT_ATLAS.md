# LLM Component Atlas — Mechanics, Optimizations & What-If Ideation

> A part-by-part walkthrough of an LLM: how each component works, what optimizations/research touch it, and "what if" ideation (push more / do the opposite / try it here). Companion to `LLM_TECHNIQUES_SURVEY.md` (technique catalog) and `NOVEL_SOLUTION_IDEATION.md` (combination methodology). Compiled 2026-08-06.
>
> **Status note:** a compute-heavy process is currently running, so all ideas here are *theory only* — no direct tests yet. Each idea ends with a "test when free" note describing the cheapest validation.

---

## How to read this doc

Each component section has four parts:
1. **Mechanics** — what the component actually computes, in one paragraph.
2. **Optimizations that touch it** — which techniques (from the survey + 2026 addendum) act on this part, and *how*.
3. **What-if ideation** — three flavors, tagged:
   - `[PUSH]` push an existing idea further
   - `[FLIP]` do the opposite of the conventional wisdom
   - `[TRY]` borrow an idea from elsewhere and apply it here
4. **ForgeAI fit** — which existing keys it composes with, lossless/lossy status.

---

## Component 1 — Token Embedding

### Mechanics
A lookup table `E ∈ R^{V×d}` maps a token id to a row vector. The same matrix (tied) or a separate `W_out ∈ R^{d×V}` projects the final hidden state back to vocab logits. For multilingual/large-vocab models this is **~25% of total params** (Qwen3-0.6B: embed = 1/4 of params).

### Optimizations that touch it
- **Weight tying** (share E and W_out) — saves V·d params but (ACL 2026) biases the matrix toward the *output* space because output gradients dominate early training, hurting early-layer input representation.
- **MRL / Matryoshka** — nested embeddings, truncate to any dim at inference. MIPIC (2026) adds self-distilled intra-relational alignment + depth-wise semantic chaining.
- **MEL** (2026) — factorized low-rank embedding, reduces *total* params (not just trainable). 3D-ML framework.
- **Untied head** — separate E and W_out, more expressive, +V·d params.
- **SVD resize** (ForgeAI) — shared SVD from embedding projects large→small model.

### What-if ideation

- **[FLIP] Gradient-scaled tying.** ACL 2026 showed scaling *input* gradients reduces the output-bias. What if we go further: **asymmetric tying** — tie only the *low-rank SVD core* of E and W_out, keep separate low-rank residuals. The shared core captures the common structure; the residuals capture input-vs-output specialization. Lossless if residuals initialized to zero (start = tied, drift apart).
  *Test when free:* take ported Qwen, SVD-decompose embed+head, tie top-k singular vectors, free the rest, measure logit cos vs original.

- **[TRY] Matryoshka logits, not just embeddings.** MRL truncates the *embedding* vector. What if the *output* projection is also Matryoshka — predict logits from a truncated hidden state, with a coarse-to-fine vocab head? At inference, cheap tokens (high-confidence) use the truncated head; hard tokens use the full head. Adaptive compute at the *output* layer, paired with MoD (skip layers) at the input.
  *Test when free:* truncate hidden to d/2, d/4 before lm_head, measure top-1 accuracy drop per truncation level on a held-out set.

- **[PUSH] Embedding as a Knowledge Pack source.** Embedding rows are token-level "knowledge." What if each token's embedding row is itself a compressed KV-pack entry (the row = a precomputed KV vector for that token's typical context)? Then the embedding lookup *is* a zero-cost KV injection for single-token queries. Conflates two currently-separate mechanisms.
  *Test when free:* compare embedding rows to actual KV cache entries for frequent tokens; measure cos sim.

- **[FLIP] No embedding at all — hash to hidden.** Replace the lookup with a learned hash `h(token) → d-dim` (e.g., a small MLP or even a fixed random projection of a one-hot/frequency encoding). Kills the V·d param block entirely. NoPE showed explicit position encodings are unnecessary; maybe explicit token embeddings are partially unnecessary too for subword-rich models. Extreme: a character-level hash + small conv to compose tokens.
  *Test when free:* replace embed with a 2-layer MLP on byte-trigram hashes, measure PPL delta on tiny_model.

**ForgeAI fit:** composes SVD resize, MRL key, Knowledge Pack. The hash idea is lossy/research; the asymmetric-tying idea is lossless at init.

---

## Component 2 — Position Encoding (RoPE)

### Mechanics
RoPE rotates 2D chunks of Q and K by an angle proportional to relative position. Each frequency band rotates at its own rate (geometric base). The Q·K inner product then depends only on relative offset. No learned params (just a base hyperparameter).

### Optimizations that touch it
- **YaRN / LongRoPE / 3D-RoPE** — rescale frequencies for longer context.
- **PartialRoPE** (ForgeAI) — apply RoPE to a subset of dims.
- **NoPE** (NeurIPS 2023, confirmed 2026) — *no* position encoding beats RoPE/ALiBi/APE on length generalization for decoder-only; represents both absolute & relative under SGD.
- **RoPE-ID** (2026) — apply high-freq RoPE to a *subset of channels* only; generalizes to longer inputs by preserving the sink-token cluster geometry.
- **ScoPE** (2026) — replace arithmetic PE with per-head exponentially-scaled look-back *scopes* (mask-based); 8× attention FLOP reduction, parameter-free, beats RoPE on extrapolation.
- **LeRoPE** (2026) — learnable scalar per frequency; emergent high-norm "LeRoPE band"; slowest bands carry *semantic* not positional info.
- **RoPE provably fails at long context** (2026) — loses locality bias AND token-relevance consistency; failure → 0.5 (random). Base tuning trades position-distinguish vs token-distinguish, can't keep both.

### What-if ideation

- **[FLIP] NoPE + ScoPE hybrid — arithmetic-free positioning.** NoPE works but has no explicit long-context handle. ScoPE gives exponential order-awareness via masks, no arithmetic. What if we use **NoPE for the semantic bands + ScoPE masks for the positional bands**? Drop RoPE entirely; let masks carry order, let attention carry semantics. Eliminates RoPE's provable long-context failure AND the rotation compute. The slow LeRoPE bands (semantic) become the NoPE bands; the fast bands become ScoPE scopes.
  *Test when free:* ablate RoPE off the ported model, add ScoPE-style per-head look-back masks, measure LongBench-style retrieval at 4x train length.

- **[PUSH] Per-layer RoPE base (not per-model).** YaRN/LongRoPE pick one base. LeRoPE learns per-frequency. What if **each layer has its own base** — early layers (local patterns) high base (slow rotation), deep layers (long-range) low base? Matches the depth-dependent specialization PolyGLU found in activations. The 2026 "RoPE fails at long context" result is *aggregate*; per-layer bases may sidestep it by letting each layer operate in its reliable regime.
  *Test when free:* sweep per-layer base on tiny_model, measure per-layer attention-entropy vs position.

- **[TRY] RoPE on the V projection, not Q/K.** RoPE is on Q/K so the *score* is position-relative. But the *output* (V-weighted sum) has no explicit position signal — position enters only through which tokens get attended. What if we also rotate V by a *coarse* position so the written value carries an order tag? Combined with NoPE-style implicit positioning, this gives the residual stream a positional memory without Q/K rotation. Risk: breaks the "relative only" property.
  *Test when free:* add a low-frequency rotation to V, measure length-generalization on synthetic recall tasks.

- **[FLIP] Inverse RoPE — rotate by 1/position.** Standard RoPE rotates more with more position (angle = pos·θ). At long context this frays. What if high-frequency bands rotate by *1/pos* (decay) so distant tokens rotate *less* and stay distinguishable, while near tokens rotate more (fine local resolution)? Inverts the locality assumption: precise near, stable far. Pairs with RoPE-ID's "subset of channels" to split bands into near-precise / far-stable.
  *Test when free:* implement inverse-rotation on half the bands, measure Needle-in-a-Haystack at 128K.

**ForgeAI fit:** composes YaRN key, PartialRoPE key, longrope2.py. NoPE+ScoPE is lossy (arch change); per-layer base is a config change (lossless at init).

---

## Component 3 — Q/K/V Projections & Attention

### Mechanics
`Q = X·W_q`, `K = X·W_k`, `V = X·W_v`. Scores `S = Q·Kᵀ/√d`, weights `A = softmax(S)`, output `O = A·V`, then `Y = O·W_o`. Multi-head: split into h heads, each does this independently, concat, project. GQA shares K/V across query-head groups. MLA compresses K/V to a latent.

### Optimizations that touch it
- **MLA / GQA / MQA** — KV compression (ForgeAI has MLA working, cos=0.9999).
- **QK-Norm** (ForgeAI) — RMSNorm Q/K, absorb into projections (lossless).
- **WQ Elimination** (ForgeAI) — replace W_q with identity, save 25% attn params.
- **FlashAttention 2/3, FlexAttention** — kernels.
- **Sparse: NSA, MoBA, SSA, SnapKV, StreamingLLM, Quest** — select subset of K/V.
- **MISA** (2026) — MoE on the *head axis* of sparse-attention indexers.
- **Softmax replacements:** sigmoid (Apple, +17% kernel), polynomial (Frobenius-reg view), ASEntmax (sparse, 1000× extrapolation), Softpick (kills sinks, helps quant), SSA (fixes ICL shift).
- **Poly-attention** (2026) — higher-order (triples) for compositional tasks self-attn can't do.
- **Linear/hybrid attention** (ForgeAI hybrid_linear) — elu+1 feature map, O(T·d).

### What-if ideation

- **[PUSH] FoldedHeadMoE (from ideation doc) — head-axis MoE at zero cost.** MISA sparsifies heads but adds a router. Fold the router into the Q projection (QK-Norm absorption pattern). *Push further:* what if the head-selection *is* the QK-Norm scale — heads whose QK-Norm scale is below threshold are skipped? No separate router at all; the norm scale you already compute becomes the gate. Lossless at init (all scales = 1 = all heads active).
  *Test when free:* per-head QK-Norm scale histogram on ported model; check if a natural bimodal split exists (active/inactive heads).

- **[FLIP] Softpick + sink-free training → no StreamingLLM sink needed.** Softpick (2026) eliminates attention sinks. StreamingLLM/LearnedSink exist *because* of sinks. What if adopting Softpick lets us **delete the sink-token machinery entirely** — no special first-token handling, no learned sink, simpler long-context? Softpick also helps quantization, so this pairs with the Lossless Quant Chain. The sink was a workaround for softmax; remove softmax's sum-to-one constraint, remove the workaround.
  *Test when free:* swap softmax→softpick in tiny_model, train 1k steps, measure sink-rate (should be ~0) + PPL.

- **[TRY] Polynomial attention + linear-KV hybrid.** Polynomial attention (2026) replaces softmax with a polynomial that gives the same Frobenius regularization. Linear attention uses `φ(Q)·φ(K)ᵀ` with a feature map to get O(T·d). What if the polynomial *is* the feature map? A polynomial attention score `p(Q·Kᵀ)` can be expanded via kernel trick into `φ(Q)·φ(K)ᵀ` for low-degree polys — giving **linear-complexity polynomial attention**. Best of both: softmax-quality regularization, linear cost.
  *Test when free:* derive φ for degree-2 polynomial, implement linear-attn form, compare PPL + speed vs softmax on tiny_model.

- **[PUSH] ASEntmax as a *runtime* sparsity knob.** ASEntmax interpolates sparse↔dense via a learnable temperature. What if the temperature is **per-prompt and set by confidence** (ORCA-style conformal)? High-confidence prompts → sparse (fast, pattern-focused); low-confidence → dense (softmax-like, careful). The same attention layer changes sparsity per input. Pairs with ConfSpec (ideation Idea 5): the conformal threshold controls *both* spec-decode verify AND attention sparsity.
  *Test when free:* measure attention entropy per prompt on a reasoning set; check if entropy bimodally splits (sparse-able vs not).

- **[FLIP] V-only attention — drop Q and K entirely.** WQ Elimination (ForgeAI) drops W_q. Push to the limit: what if attention is just `A = softmax(V·Vᵀ)`, output `A·V`? No Q, no K — V self-attends. Loses query-driven selection but V already encodes content. Combined with a position mask (ScoPE) for order. Extreme parameter saving (kills W_q, W_k, W_o). Almost certainly needs fine-tuning but the question is *how much* — WQ Elim already showed 25% param save is recoverable.
  *Test when free:* V-only attention on tiny_model from scratch, 5k steps, compare PPL to baseline at same param budget.

- **[TRY] Poly-attention at *one* layer for compositional tasks.** Self-attention can't detect triples of correlated tokens (proven). Poly-attention can but is superquadratic. What if **only one mid-deep layer** uses poly-attention (3rd-order) and the rest stay standard? Cost: one expensive layer; gain: compositional reasoning the rest can't do. Pairs with MoD — that one layer only runs on hard tokens (top-k routed).
  *Test when free:* synthetic Match3 task; standard vs +1-poly-layer model, measure triple-detection accuracy.

**ForgeAI fit:** composes MLA, QK-Norm MLA, WQ Elim, Hybrid Linear, Softpick (new), MISA. V-only and poly-layer are lossy/research.

---

## Component 4 — FFN / MLP (SwiGLU)

### Mechanics
`gate = silu(X·W_gate)`, `up = X·W_up`, `inter = gate ⊙ up`, `out = inter·W_down`. SwiGLU. The FFN is where factual knowledge is stored (Meng et al. ROME). intermediate_size >> hidden (e.g., 8960 vs 1536).

### Optimizations that touch it
- **SwiGLU / GeGLU / ReGLU** (ForgeAI Activation Transmute swaps among these, near-lossless via per-channel α,β).
- **MoE** (ForgeAI) — split FFN into routed experts.
- **BitNet** (ForgeAI) — ternary weights.
- **Fact Injection / Context Patch** (ForgeAI) — rank-1 MLP updates store facts/ICL.
- **PolyGLU** (2026) — per-neuron routing among K=4 activations; depth specialization (early=GELU, deep=Tanh).
- **SwiGLU Clamp** (ForgeAI) — GPT-OSS clamped SwiGLU.
- **SliceGPT** — SVD prune the FFN.

### What-if ideation

- **[PUSH] PolyGLU via Activation Transmute (lossless init).** ForgeAI's Activation Transmute already swaps SwiGLU→ReGLU/GeGLU with per-channel α,β via grid search. PolyGLU routes *per neuron* among activations. What if we **transmute each neuron to its best-fitting activation individually** (the transmute already does per-channel)? That's PolyGLU *without* the Gumbel router — a closed-form, training-free version. The grid search picks the activation + α,β per neuron from calibration data. Lossless-ish (transmute is 0.7% error). The router is replaced by a static per-neuron choice.
  *Test when free:* run Activation Transmute per-neuron (not per-layer), measure error vs per-layer.

- **[FLIP] Shrink the FFN, grow the experts.** Conventional: big intermediate_size for capacity. What if the *base* FFN is tiny (intermediate = 2× hidden, minimal) and *all* capacity lives in MoE experts? The base handles routing/common cases; experts hold specialized knowledge. Pairs with ExpertGenesis (ideation Idea 3): new knowledge = new expert, base never grows. The FFN becomes a routing scaffold, not a knowledge store. Fact Injection targets experts, not base.
  *Test when free:* ablate base FFN size down on the MoE-converted model, measure cos vs original.

- **[TRY] FFN as a key-value memory, literally.** ROME showed FFN ≈ key-value memory (W_gate=keys, W_down=values). Knowledge Pack injects *attention* KV. What if we inject **FFN KV** — treat each expert as a KV store where W_gate rows are keys and W_down cols are values, and retrieval is the matmul itself? Then Fact Injection is literally "insert a key-value pair into the memory." Unifies FFN and attention KV mechanisms. Context Patch (rank-1 MLP) is already half this; make it explicit.
  *Test when free:* extract W_gate rows as keys, W_down cols as values, check if nearest-neighbor lookup on keys recovers the matmul output on test inputs.

- **[PUSH] Clamp → ternary-friendly SwiGLU.** SwiGLU Clamp (ForgeAI) clamps values. BitNet ternarizes weights. Clamping *before* ternarization removes the outliers that hurt ternary accuracy. What if the clamp threshold is **set to the BitNet absmean** so the clamped-then-ternarized weights are exactly the ternary set? Clamp becomes the quantizer. Lossless clamp + lossy ternary in one op, with the clamp making the ternary less lossy.
  *Test when free:* clamp gate/up to ±absmean, then ternarize, measure cos vs plain BitNet.

**ForgeAI fit:** composes Activation Transmute, MoE, BitNet, Fact Injection, Context Patch, SwiGLU Clamp. PolyGLU-via-transmute is near-lossless.

---

## Component 5 — Normalization (RMSNorm)

### Mechanics
`y = x / rms(x) · γ`. RMSNorm scales by root-mean-square (no mean subtraction). γ is a per-channel learned scale. Placed pre-sublayer (Pre-Norm) for stability.

### Optimizations that touch it
- **RMSNorm / LayerNorm / DeepNorm** — variants.
- **QK-Norm** (ForgeAI) — RMSNorm on Q/K.
- **SandwichNorm** (ForgeAI) — post-sublayer norm.
- **Norm Folding** (ForgeAI) — fold γ into adjacent Linear, lossless, kills 113 norms/forward.
- **Dual-Norm** (Gemma) — pre + post.

### What-if ideation

- **[PUSH] Fold QK-Norm into MLA latent.** Norm Folding folds γ into adjacent Linears. QK-Norm is on Q/K. MLA compresses K/V to a latent. What if we fold QK-Norm **into the MLA down-projection** (kv_down_proj) so the latent is *pre-normalized*? The norm happens once at compression, not per-head at expansion. Fewer ops, smaller latent (normalized values are bounded → better quantize). Lossless if the fold math holds.
  *Test when free:* derive the fold for MLA's kv_down_proj, verify cos=1.0 vs unfolded on ported model.

- **[FLIP] No norms at all — rely on init + clamping.** Norm Folding removes norms from the *forward pass* but they're still in the weights. What if we go further: bake *all* normalization into init (scale-aware init) + Logit Cap (ForgeAI) + SwiGLU Clamp as the *only* stability mechanisms? No RMSNorm anywhere. The clamps bound extremes; init prevents drift. Extreme version of Norm Folding. Almost certainly needs fine-tuning but the question is whether clamps alone suffice.
  *Test when free:* fold all norms, remove the residual norm ops, run tiny_model 1k steps, watch for NaN/divergence.

- **[TRY] Norm as a Knowledge Pack injection point.** Norms have a γ scale per channel. Context Patch (2026) showed ICL effects = rank-1 patch to MLP + **RMSNorm scale**. What if Knowledge Packs inject via the *norm scale* (a per-channel γ delta) instead of KV? γ is tiny (d params) vs a KV pack (T·d). A domain pack becomes a d-vector. Inject dozens of domains as γ-vectors, summed. Norm Folding would then fold these back into adjacent weights — *permanent* domain knowledge at zero runtime.
  *Test when free:* extract γ deltas from a fine-tuned-on-domain model vs base, inject as additive γ, measure domain-task accuracy.

**ForgeAI fit:** composes Norm Folding, QK-Norm MLA, SandwichNorm, Logit Cap, Context Patch. The γ-injection idea is lossless and composes with Knowledge Pack.

---

## Component 6 — Residual Stream & Depth

### Mechanics
`x_{l+1} = x_l + sublayer_l(x_l)`. The residual stream is the model's "memory bus" — every layer reads from and writes to it. Depth = number of layers. Information dilution: deep layers see a compressed history.

### Optimizations that touch it
- **Pre-Norm** (stability), **SandwichNorm** (post).
- **DenseFormer** (ForgeAI) — depth-weighted averaging (DWA), cross-layer dense connections, identity-init lossless.
- **mHC / xHC / go-mHC** (2026) — N parallel residual streams, learned mixing. xHC scales beyond N=4 via sparse update (k=4 of N=16).
- **MoD** (2024/2026) — top-k routing: token runs the block OR skips via residual. Dynamic per-token depth.
- **LayerSkip / Early Exit** — exit early.
- **mHC ablation** (2026) — streams encode distinct info, asymmetric utilization, not just redundancy.

### What-if ideation

- **[PUSH] DenseFormer + MoD = dynamic dense connections.** DenseFormer adds cross-layer dense connections (static weights). MoD skips layers per-token. What if the DenseFormer averaging weights are **gated by the MoD router** — a token that skips layer l doesn't get l's contribution in the DWA? Dynamic DenseFormer: the dense connection weight is 0 for skipped layers. Combines depth-history preservation (DenseFormer) with compute savings (MoD) without the information dilution either alone suffers.
  *Test when free:* implement DWA with per-token skip mask, measure PPL vs FLOPs on tiny_model.

- **[FLIP] Shrinking residual stream (narrow with depth).** Conventional: constant d throughout. xHC grows streams. What if the stream **narrows** with depth — early layers wide (rich input processing), deep layers narrow (compressed abstractions)? Like an encoder funnel. Saves params in deep layers (where forgetting localizes, per mechanistic-forgetting 2026). Pairs with BudgetGuard: more expert budget early (wide), less deep (narrow). Inverts the "deep needs more capacity" assumption — maybe deep needs *less* because it's compressed.
  *Test when free:* linearly taper d from 1536→768 across layers on tiny_model, measure PPL + param count.

- **[TRY] xHC sparse-update + AirMoE = stream-level offload.** xHC updates only k=4 of N=16 streams. What if the *non-active streams live on disk* (AirMoE-style) and are loaded only when a layer writes to them? N=16 streams at d=1536 is 16× the residual memory; offloading 12 of them keeps VRAM at ~4-stream cost while retaining 16-stream capacity. The sparse update pattern means most streams are cold most of the time — perfect for offload.
  *Test when free:* profile stream write-frequency per layer on ported model; check if a few streams dominate writes.

- **[PUSH] MoD routing from the residual norm.** MoD uses a learned router. What if the skip decision is just `‖x_l‖` — if the residual hasn't changed much (small delta norm), skip? No router params; the residual's own magnitude is the signal. Pairs with Norm Folding (norm is already computed). A "quiet" token that the stream isn't updating skips; an "active" token runs. Mechanistic-forgetting says early attn heads drift entropically — those are the *opposite* of quiet, so they'd never skip (good).
  *Test when free:* log per-token residual-delta-norm per layer, check if a threshold cleanly separates "needs compute" from "skip".

**ForgeAI fit:** composes DenseFormer, MoD key, SandwichNorm. Norm-gated MoD is lossless at init (threshold=0 = no skip).

---

## Component 7 — MoE Router

### Mechanics
`g = router(x)`, top-k experts selected, `out = Σ softmax(g_i)·expert_i(x)`. Router is a small Linear `d→n_experts`. Load balancing via auxiliary loss or bias.

### Optimizations that touch it
- **MoE split** (ForgeAI) — dense→experts, exact.
- **Expert Consolidation** (ForgeAI) — merge similar experts.
- **AirMoE** (ForgeAI) — hot-load experts.
- **ϕ-balancing / DUAL** (2026) — better load balancing.
- **Alloc-MoE** (2026) — non-uniform per-layer budget.
- **LightMoE** (2026) — redirect-to-similar resident expert.
- **MoE router key** (ForgeAI) — centroid init.

### What-if ideation

- **[FLIP] No router — hash the token.** Routers need training (ForgeAI MoE router starts uniform). What if expert selection is a **deterministic hash of the token id + layer** (like a hash table)? Zero router params, zero training, perfectly balanced (uniform hash). Loses content-aware routing but Expert Consolidation's merged experts are broad enough that any expert is "okay." The hash is the router. Combined with LightMoE redirect, a hash-miss redirects to the nearest resident expert. Trades routing quality for zero-overhead perfect balance.
  *Test when free:* replace router with `hash(token,layer) % n_experts`, measure PPL drop vs trained router.

- **[PUSH] Router as a conformal gate.** ORCA gives conformal confidence. What if the router outputs a *conformal prediction set* — "experts {3,7,11} are 90%-coverage-correct for this token" — and we run *all* of them only when coverage is uncertain, top-1 when confident? Adaptive top-k via conformal guarantee. High-confidence tokens: 1 expert (fast). Uncertain: 3 experts (safe). The correctness guarantee carries to expert selection.
  *Test when free:* calibrate router logits as conformal scores, measure coverage vs top-k on a held-out set.

- **[TRY] Router lives in the embedding (token-typed experts).** What if expert assignment is **determined at embedding time** — each token's embedding row includes a soft expert-preference vector? The router is the embedding lookup itself (zero extra compute). Tokens of the same type (code keywords, math symbols) route consistently. Pairs with the "embedding as KV" idea (Component 1). The embedding becomes a routing table + representation simultaneously.
  *Test when free:* add an n_experts-dim head to the embedding, use as router logits, measure routing consistency per token-type.

**ForgeAI fit:** composes MoE, Expert Consolidation, AirMoE, MoE router key. Hash-router is lossy/research; conformal-router is lossless at init.

---

## Component 8 — KV Cache

### Mechanics
During decoding, store past K/V per layer per head. Grows as O(T·h·d_head). The dominant memory cost at long context. Paged (vLLM) or compressed.

### Optimizations that touch it
- **MLA** (ForgeAI) — compress to latent.
- **RotorQuant** (ForgeAI) — Givens rotation, 3.88×.
- **4-bit KV** (ForgeAI) — group quant + scale absorption.
- **SnapKV / PyramidKV / H2O / Quest / StreamingLLM** — eviction/selection.
- **Knowledge Pack** (ForgeAI) — inject precomputed KV.
- **CoT Knowledge Pack** (ForgeAI) — reasoning-trace KV.

### What-if ideation

- **[PUSH] CompressedPack (ideation Idea 10) — RotorQuant + 4-bit + QK-Norm absorption.** Already in the ideation doc. *Push further:* what if the rotation is **per-domain** (different Givens rotation per Knowledge Pack domain) so each pack is independently compressed? Domains don't share a rotation → no cross-domain quantization interference. A library of domain-rotated 4-bit packs.
  *Test when free:* fit per-domain rotation on 3 domain packs, measure per-pack error vs global rotation.

- **[FLIP] KV cache as the *only* memory — no weights.** Extreme: what if the "model" at inference is just a KV cache (precomputed from training) and generation is pure attention over it, with minimal weight params? The Knowledge Pack idea taken to the limit — the packs *are* the model. Weight params shrink to a tiny projection; all knowledge is KV. AirMoE for KV: load packs from disk on demand. This is essentially a retrieval-augmented extreme; the question is the quality floor.
  *Test when free:* precompute KV for the whole training set, run generation as attention-only over retrieved KV, measure PPL.

- **[TRY] KV cache eviction via conformal coverage.** SnapKV/H2O evict by attention score. What if eviction guarantees **conformal coverage** — "the retained KV covers 90% of likely future queries"? ORCA-style. Eviction becomes a coverage problem, not a score problem. Retains the *minimal* KV that provably covers future attention. Pairs with Quest (page-aware).
  *Test when free:* compute conformal coverage of retained-KV subsets, compare retention size vs score-based at same coverage.

- **[PUSH] KV cache as continual-learning replay (from research gaps).** Self-generated replay (2026) re-generates old data to prevent forgetting. A Knowledge Pack *is* compressed old context. Replace replay *tokens* with KV-pack re-injection → **replay at zero tokens**. The pack is the replay. Directly addresses FOREVER's model-time replay but at zero sequence cost.
  *Test when free:* during continual fine-tune, inject base-model KV pack every N steps instead of replaying data, measure forgetting vs replay baseline.

**ForgeAI fit:** composes RotorQuant, 4-bit KV, Knowledge Pack, CoT Pack, QK-Norm. Per-domain rotation is lossy but bounded.

---

## Component 9 — Output / LM Head / Logits

### Mechanics
`logits = x_final · W_out` (tied to E or separate). `probs = softmax(logits / T)`. Sample (greedy/top-k/top-p/temperature).

### Optimizations that touch it
- **Tied/untied embeddings.**
- **Logit Cap** (ForgeAI) — clamp to ±30.
- **Z-Loss** — penalize large logits.
- **MTP heads** (ForgeAI) — predict t+1, t+2.
- **Matryoshka logits** (this doc, Component 1 ideation).

### What-if ideation

- **[PUSH] MTP-EAGLE-DSpark (ideation Idea 12) — native MTP as EAGLE draft.** *Push further:* what if the MTP heads predict a **whole token *distribution*** (not just argmax) that becomes the draft distribution for speculative sampling? EAGLE-3 uses feature-level fusion; MTP heads already have the features. The MTP head's softmax *is* the draft distribution — no separate draft model softmax. One fused op.
  *Test when free:* compare MTP-head softmax to a small draft model's softmax on a held-out set, measure KL.

- **[FLIP] Logit Cap as a *quantization* aid, not just stability.** Logit Cap clamps to ±30. What if the cap is **set to the int4 range** so logits are directly quantizable to int4 without scaling? The cap *is* the quantizer. Logits become int4-native. Pairs with Softpick (kills massive activations → tighter logit range → smaller cap → better int4). Stability + quantization in one clamp.
  *Test when free:* measure logit distribution post-Softpick, find the 99.9% range, set cap there, int4-quantize logits, measure sampling-quality delta.

- **[TRY] Conformal early-stopping at the head.** ORCA stops *sampling* via conformal. What if the LM head itself has a **conformal exit** — after computing logits, a conformal test decides "this token is confident enough, emit it" vs "re-run with more context/depth"? The head becomes the early-exit gate (pairs with MoD). High-confidence logits exit at the head; uncertain ones trigger deeper compute. Unifies early-exit (depth) with conformal sampling (head).
  *Test when free:* calibrate logit-max as conformal score, measure early-exit rate vs accuracy.

**ForgeAI fit:** composes Logit Cap, MTP, DSpark, EAGLE. Logit-cap-as-quantizer is lossy but bounded.

---

## Component 10 — Loss / Training Objective

### Mechanics
Pretraining: next-token CE. Distillation: KL(student||teacher) + CE. DPO/RLHF: preference loss. The objective shapes what the weights learn.

### Optimizations that touch it
- **Top-K KL** (ForgeAI) — 1500× faster distillation.
- **Chunked CE** (ForgeAI) — VRAM savings.
- **MiniLLM** — reverse KLD (better calibration).
- **SpecKD** — KD only on accepted spec tokens.
- **TEMPO** (2026) — EM framing for TTT.
- **PoT** (2026) — GRPO on transient LoRA.
- **DiSCTT** (2026) — uncertainty-routed SFT vs RL.

### What-if ideation

- **[PUSH] TEMPO-EM self-play (from research gaps).** ForgeAI self-play has no critic recalibration. TEMPO says that's "incomplete EM." *Push:* add a labeled recalibration set to the self-play loop — every K self-play rounds, recalibrate the critic (the confidence scorer) on verified solutions. The self-play loop becomes proper EM. Direct application of 2026 theory to existing ForgeAI code.
  *Test when free:* add a 100-example verified recalibration set to recursive_self_play, run 5 EM rounds, measure solution quality vs no-recalibration.

- **[FLIP] Lossless loss — no gradient at all.** ForgeAI's closed-form keys (Fact Injection, Context Patch, GRAIL) produce weights *without* a loss function. What if the *entire* training is closed-form — no gradient descent, no loss? Self-play generates (Q,A) pairs → Fact Injection stores them → GRAIL heals → repeat. The "loss" is the test-passed gate (Test-Gated Injection). Pure data→weight, zero gradient. This is the ForgeAI philosophy pushed to its limit. The question is the quality ceiling vs gradient training.
  *Test when free:* run N rounds of self-play→fact-injection on tiny_model, measure benchmark accuracy vs same-FLOPs gradient training.

- **[TRY] Conformal loss weighting.** ORCA gives per-prompt conformal confidence. What if the loss is **weighted by conformal confidence** — high-confidence (in-distribution) examples get full weight, low-confidence (OOD) get down-weighted? The model focuses capacity on what it can calibrate. Pairs with DiSCTT's uncertainty routing but at the *loss* level, not the data-routing level.
  *Test when free:* weight CE by conformal score on a mixed-distribution train set, measure OOD generalization.

**ForgeAI fit:** composes Self-Play, Fact Injection, GRAIL, Test-Gated Injection, recursive_self_play. The lossless-loss idea is the ForgeAI thesis itself.

---

## Cross-Component What-Ifs (the best ideas span components)

These combine ideation across components — the richest territory.

- **NormGatedMoD** (Components 5+6): skip layers based on residual-delta norm (no router). The norm you'd fold is the gate. Lossless at init.
- **SoftpickSinkFree** (Components 3+8): Softpick kills sinks → delete StreamingLLM sink machinery → simpler long-context + better KV quant.
- **PolyGLUTransmute** (Components 4+5): per-neuron activation choice via closed-form transmute, no Gumbel router. Near-lossless.
- **ConformalEverything** (Components 3+7+8+9+10): one conformal threshold controls attention sparsity (ASEntmax), expert count (router), KV retention, sampling stop, and loss weight. A single calibrated knob governs the whole inference compute budget per prompt.
- **LosslessTraining** (Components 4+10): Self-play → Fact Injection → GRAIL, no gradient. The ForgeAI endgame.
- **KVasReplay** (Components 8+10): Knowledge Pack replaces replay tokens → zero-token continual learning.

---

## Test-When-Free Priority

Ordered by (losslessness × composability × ease of test on tiny_model):

1. **NormGatedMoD** — lossless at init, log residual-delta-norm (1-line instrument).
2. **PolyGLUTransmute** — near-lossless, reuse Activation Transmute per-neuron.
3. **FoldedHeadMoE** — lossless, check QK-Norm scale bimodality.
4. **SoftpickSinkFree** — lossy but easy swap, train 1k steps.
5. **TEMPO-EM self-play** — lossless, add recalibration set to existing loop.
6. **Per-domain KV rotation** — lossy bounded, fit on 3 packs.
7. **Asymmetric tying** — lossless at init, SVD decompose embed+head.
8. **KVasReplay** — lossless, inject pack during continual FT.
9. **Matryoshka logits** — lossless at full dim, test truncation curve.
10. **V-only attention** — lossy/research, from-scratch 5k steps.

---

## Key Mapping (see `KEY_MAPPING_MASTER.md`)

Every component what-if → a ForgeAI key. Training avoided wherever possible.

| Comp | What-If | Key | Class | Train? |
|---|---|---|---|---|
| 1 | Asymmetric tying | `asymmetric_tying_key.py` | **FULL** | None |
| 1 | Matryoshka logits | `matryoshka_logit_key.py` | TRIVIAL | None |
| 1 | No-embedding hash | — | — | **Full (NOT A KEY)** |
| 2 | NoPE+ScoPE | `nope_scope_key.py` | TRIVIAL | None |
| 2 | Per-layer RoPE base | `per_layer_rope_key.py` | TRIVIAL | None |
| 2 | RoPE on V | `rope_v_key.py` | PARTIAL | Minimal |
| 2 | Inverse RoPE | `inverse_rope_key.py` | PARTIAL | Minimal |
| 3 | FoldedHeadMoE | `folded_head_moe_key.py` | **FULL** | None |
| 3 | Softpick | `softpick_key.py` | TRIVIAL | None⁴ |
| 3 | Poly-linear attn | `poly_linear_attn_key.py` | PARTIAL | None⁵ |
| 3 | ASEntmax knob | `asentmax_key.py` | TRIVIAL | None |
| 3 | V-only attention | — | — | **Full (NOT A KEY)** |
| 3 | Poly-attn one layer | `poly_attn_layer_key.py` | PARTIAL | Minimal |
| 4 | PolyGLU transmute | `polyglu_transmute_key.py` | PARTIAL | None |
| 4 | Shrink base FFN | `shrink_ffn_key.py` | PARTIAL | None |
| 4 | Clamp→ternary | `clamp_ternary_key.py` | PARTIAL | None |
| 5 | Fold QK-Norm into MLA | `fold_qknorm_mla_key.py` | **FULL** | None |
| 5 | No norms | — | — | **Full (NOT A KEY)** |
| 5 | Norm γ injection | `norm_gamma_pack_key.py` | PARTIAL | None |
| 6 | DenseFormer+MoD | `dynamic_dwa_key.py` | TRIVIAL | None |
| 6 | Shrinking stream | — | — | **Full (NOT A KEY)** |
| 6 | xHC+AirMoE offload | `xhc_offload_key.py` | TRIVIAL | None |
| 6 | MoD from residual norm | `norm_gated_mod_key.py` | TRIVIAL | None |
| 7 | Hash router | `hash_router_key.py` | TRIVIAL | None |
| 7 | Conformal router | `conformal_router_key.py` | TRIVIAL | None |
| 7 | Router in embedding | `embed_router_key.py` | PARTIAL | Minimal |
| 8 | Per-domain rotation | `per_domain_rotor_key.py` | PARTIAL | None |
| 8 | KV-only memory | — | — | **Full (NOT A KEY)** |
| 8 | Conformal KV evict | `conformal_kv_evict_key.py` | TRIVIAL | None |
| 8 | KV pack as replay | `kv_replay_key.py` | TRIVIAL | None |
| 9 | MTP-head as draft | `mtp_draft_key.py` | TRIVIAL | None |
| 9 | Logit Cap as quantizer | `logit_cap_quant_key.py` | TRIVIAL | None |
| 9 | Conformal early-exit | `conformal_exit_key.py` | TRIVIAL | None |
| 10 | TEMPO-EM self-play | `tempo_em_key.py` | PARTIAL | None⁶ |
| 10 | Lossless loss | (existing SelfPlay+FactInject+GRAIL) | PARTIAL | None |
| 10 | Conformal loss weight | `conformal_loss_key.py` | TRIVIAL | None |

⁴ Softpick: drop-in swap, optional fine-tune. ⁵ Kernel-trick expansion is closed-form; quality may need fine-tune. ⁶ Recalibration on labeled set, no model gradient.

**NOT A KEY (5):** no-embedding hash, V-only attention, no-norms, shrinking stream, KV-only-memory → architecture redesigns for `config.py`, not `keys/`.

**Full forward/reverse sketches + registration plan:** `docs/research/KEY_MAPPING_MASTER.md`

---

*Compiled 2026-08-06. Companion to `LLM_TECHNIQUES_SURVEY.md` (Parts 1-6) and `NOVEL_SOLUTION_IDEATION.md`. All ideas are theory-only pending compute availability. Raw research notes in `.devin/scratchpad.md`.*
