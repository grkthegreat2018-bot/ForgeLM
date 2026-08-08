# Deep Subjects Ideation — New Keys from Fundamental LLM Mechanics

> Third-tier ideation applying the generative rule ("find the uniform/global/binary assumption, make it structured/local/spectral") to **deeper structural subjects** not covered by `FIRST_PRINCIPLES_IDEATION.md` (F1-F12), `NOVEL_SOLUTION_IDEATION.md` (Ideas 1-12), or `KEY_NOVELTY_AUDIT_PART2.md`. Each idea is grounded in how LLM improvements actually work, targets a specific ForgeAI key composition, and is lossless or near-lossless at init where possible.
>
> Compiled 2026-08-07. Extends the existing 77 mapped ideas (38 no-train + 9 minimal-train + 7 NOT-A-KEY + 23 research concepts).

---

## The Deeper Subjects (not yet covered)

The existing ideation covers: attention scan, memory hierarchy, weight spectrum, defactoring, temperature, time, distillation, shared basis, working memory, precision, generation direction, kernel shape, MoE serving, TTT, expert growth, head routing, spec decode, expert merging, budget, expert compression, patch stacking, KV compression, uncertainty, MTP-EAGLE, tokenizer, sampling, batching, prompt, compute, SAE, cross-attention, model merging.

The **deeper subjects** below are structural assumptions that are EVEN MORE embedded — they're about the algebra, geometry, information flow, and duality of the transformer itself.

| # | Deep Subject | The hidden assumption |
|---|---|---|
| D1 | The algebra of the residual | Residual is additive only (commutative monoid) |
| D2 | The geometry of attention output | Attention output is trapped in V's convex hull |
| D3 | The duality of attention and FFN | They're separate, sequential operations |
| D4 | Information conservation across depth | All layers carry equal information |
| D5 | The temporal redundancy of KV cache | Each token's K/V is independent of neighbors |
| D6 | The coupling of routing and computation | Router and expert are separate modules |
| D7 | The symmetry of Q and K | Q and K play symmetric roles in dot product |
| D8 | The dimensionality of "concept space" | All layers operate in the same d-dimensional space |
| D9 | The granularity of weight updates | Weight updates are per-matrix (full rank) |
| D10 | The linearity of the forward pass | Each token sees each layer exactly once |

---

## D1 — The Algebra of the Residual Stream

### The hidden assumption
The residual connection is `x_{l+1} = x_l + f_l(x_l)`. This is an **Euler integration step** of the ODE `dx/dl = f(x, l)`. The algebra is purely additive — a commutative monoid. Every layer CONTRIBUTES to the stream; no layer can GATE, SUPPRESS, or OVERWRITE.

### Why it's wasteful
Addition is the weakest algebraic operation. It means:
- A layer can only ADD information, never REMOVE noise. Errors accumulate.
- The residual grows monotonically in norm (or needs norms to counteract growth).
- A layer that wants to "correct" a previous layer's error can only add a counter-signal, not zero out the error.
- Multiplicative interactions (where one layer's output GATES another's) are impossible in the residual.

Real computation uses richer algebras: multiplicative gating (LSTMs, GRUs), attention itself (weighted averaging), and write-overwrite (memory systems). The transformer's residual is the simplest possible.

### Derivation — three algebraic generalizations

**D1a. MultiplicativeResidual**: `x_{l+1} = x_l ⊙ (1 + γ_l * g_l(x_l))`
- γ_l is a per-layer scalar, init γ_l = 0 → `x_{l+1} = x_l` (identity, lossless).
- Fine-tune γ_l > 0 → the layer GATES the residual (multiplicative). A layer can AMPLIFY or SUPPRESS specific dimensions.
- This is different from GateSkip (which gates the ADDITIVE contribution). Multiplicative gating changes the algebra — the layer can zero out dimensions, not just add to them.
- **Composes with:** NormGatedMoD (skip + gate), SandwichNorm (stability), DenseFormer (DWA in multiplicative space).
- **Class:** TRIVIAL (one scalar per layer, identity init). **File:** `mult_residual_key.py`

**D1b. ResidualOverwrite**: `x_{l+1} = (1 - m_l) ⊙ x_l + m_l ⊙ f_l(x_l)`
- m_l is a per-layer, per-dimension GATE (sigmoid of a learned vector), init m_l = 0 → `x_{l+1} = x_l` (lossless).
- Fine-tune m_l → the layer can OVERWRITE specific dimensions of the residual instead of adding to them.
- This is the transformer analog of LSTM's forget gate + input gate. The residual becomes a MEMORY CELL that layers can selectively write to.
- **Composes with:** Knowledge Pack (overwrite = inject knowledge into specific dims), Fact Injection (fact = overwrite a fact dimension).
- **Class:** PARTIAL (per-layer-dim gate, identity init, needs fine-tune to learn gates). **File:** `residual_overwrite_key.py`

**D1c. HigherOrderResidual**: `x_{l+1} = x_l + f_l(x_l) + 0.5 * f_l'(x_l) * f_l(x_l)`
- A midpoint (Runge-Kutta 2nd order) correction to the Euler step.
- Init the correction term to 0 → standard Euler (lossless).
- The correction improves the "integration accuracy" of the depth ODE — the model approximates a continuous-depth neural ODE more accurately with the same number of layers.
- **Published:** Neural ODEs (Chen et al. 2018), but the RK2 correction to standard transformer residuals is underexplored.
- **Class:** PARTIAL (needs fine-tune to learn the correction). **File:** `rk2_residual_key.py`

### Residual problem
Multiplicative/overwrite residuals change the training dynamics — the gradient flow is different. Need fine-tuning to learn the gates. The RK2 correction doubles the per-layer compute (two function evaluations).

---

## D2 — The Geometry of Attention Output

### The hidden assumption
Attention output = `softmax(QK^T) @ V`. Softmax produces non-negative weights summing to 1. The output is always a **convex combination** of V rows — it lives inside the convex hull of V. Attention can only INTERPOLATE, never EXTRAPOLATE.

### Why it's wasteful
The convex hull is a strict subset of the vector space. If the model needs to produce an output OUTSIDE the V manifold (e.g., a "negation" of all values, or an extrapolation beyond the range), attention literally cannot do it. The residual connection partially fixes this (adds x back), but the attention CONTRIBUTION is still interpolation-only. The model wastes capacity learning to work around this geometric constraint.

### Derivation

**D2a. SignedAttention**: `A = softmax(scores) ⊙ tanh(scores)`
- tanh can be negative → the output can be OUTSIDE the V convex hull.
- At init, if we use standard softmax (no tanh), it's lossless. The signed version is a runtime toggle.
- The tanh factor is bounded [-1, 1], so the output is bounded by ±max(|V|) — stable.
- **Different from Softpick** (softmax * sigmoid): sigmoid is always positive → softpick is still non-negative (sparse, but still interpolation). Signed attention allows NEGATIVE weights → extrapolation.
- **Composes with:** PerQueryTemp (temperature + signed = richer geometry), QK-Norm (normalize before signing).
- **Class:** TRIVIAL (runtime formula change, lossless if disabled). **File:** `signed_attn_key.py`

**D2b. AffineAttention**: `output = softmax(QK^T) @ V + β_h`
- Add a per-head bias β_h to the attention output. This shifts the output OUTSIDE the V convex hull by a constant.
- Init β_h = 0 → lossless. Fine-tune to learn the shift.
- This is the attention analog of the bias term in Linear layers. Attention currently has no bias — every other operation (Q, K, V, O projections) has biases. Attention's output is bias-free, which is a geometric restriction.
- **Composes with:** MLA (bias in latent space), Norm Folding (fold β into output projection).
- **Class:** TRIVIAL (per-head vector, identity init). **File:** `attn_bias_key.py`

### Residual problem
Signed attention can produce large-magnitude outputs if V has large values and the signed weights align. Need clamping (Logit Cap style) for stability.

---

## D3 — The Duality of Attention and FFN

### The hidden assumption
Attention and FFN are separate, sequential operations. Attention routes information between positions; FFN transforms information at each position. They use different mechanisms.

### Why it's wasteful
They're actually the SAME operation at different granularities:
- **Attention** = `softmax(Q @ K^T) @ V` — routes between POSITIONS. The "keys" are other positions' representations.
- **FFN** = `W_down(silu(W_gate @ x) ⊙ (W_up @ x))` — routes between FEATURES. The "keys" are the gate's learned weights, the "values" are the up projection's features. `silu(W_gate @ x)` IS the attention weight (a per-feature gate), and `W_up @ x` IS the value.

This means: every FFN optimization applies to attention, and vice versa. MoE (routing the FFN) = MoA (routing attention heads). Sparse FFN = sparse attention. Linear FFN = linear attention. The field hasn't fully exploited this duality.

### Derivation

**D3a. FFNAsAttention**: reformulate the FFN as an explicit attention operation.
- `FFN(x) = W_down @ (silu(W_gate @ x) ⊙ (W_up @ x))`
- = `W_down @ diag(silu(W_gate @ x)) @ (W_up @ x)`
- = "attention" where Q = x, K = W_gate (static keys), V = W_up @ x, attention_weight = silu(Q @ K^T) (per-feature gate).
- **Implication:** the FFN can be SPARSIFIED the same way as attention. Instead of computing all d_ff features, compute only the top-k activated features (by silu gate value). This is MoE with k=1 per feature group — but derived from the attention duality, not from the MoE literature.
- **Composes with:** MoE (unified routing), BitNet (ternary gates = binary attention), Wanda (prune = sparsify FFN-attention).
- **Class:** TRIVIAL (reformulation, no weight change). **File:** `ffn_as_attn_key.py`

**D3b. UnifiedRouting**: share the routing mechanism between MoE and MoA.
- Currently: MoE routes the FFN, MoA routes attention heads. Two separate routers.
- Unified: one router decides BOTH which expert AND which heads to activate. The routing decision is shared because the operations are dual.
- This saves router parameters and ensures the routing is consistent (if you route to the "code" expert, you also route to the "code" attention heads).
- **Composes with:** FoldedHeadMoE (fold the unified router into Q projection), MoE router.
- **Class:** PARTIAL (needs fine-tune to learn unified routing). **File:** `unified_router_key.py`

### Residual problem
The FFN-as-attention reformulation doesn't change computation — it's a theoretical lens. The practical benefit (sparsification) requires changing the forward pass, which is lossy without calibration.

---

## D4 — Information Conservation Across Depth

### The hidden assumption
All layers carry equal information. The model is a uniform stack of identical-capacity blocks. Depth = more transformation steps, each equally important.

### Why it's wasteful
Information theory says: `I(x_l; x_0)` (mutual information between input and layer l's output) typically DECREASES with depth (data processing inequality), while `I(x_l; weights)` (information from the model's stored knowledge) INCREASES with depth. Early layers are "feature extractors" (high input info, low weight info). Late layers are "knowledge retrievers" (low input info, high weight info). Middle layers are "reasoning" (both). But the architecture treats all layers identically — same precision, same capacity, same quantization.

### Derivation

**D4a. InformationProfileProfiling**: measure `I(x_l; x_0)` and `I(x_l; weights)` per layer.
- Use the ported model + calibration data.
- `I(x_l; x_0)`: compute mutual information between the input embeddings and each layer's output. High → "computation layer" (quantize aggressively — it's robust to noise).
- `I(x_l; weights)`: compute how much the output changes when weights are perturbed. High → "knowledge layer" (keep high precision — it's lossy to quantize).
- **Use the profile to:**
  - Assign per-layer precision (Task Precision, but informed by MI not heuristics).
  - Target Fact Injection to knowledge layers (high weight-info).
  - Target Context Patch to computation layers (high input-info).
  - Skip low-information layers (NormGatedMoD, but informed by MI not residual norm).
- **Class:** TRIVIAL (profiling, no weight change). **File:** `info_profile_key.py`

**D4b. DepthStratifiedQuant**: different quantization precision per depth stratum.
- Early layers (0-8): high precision (feature extraction, sensitive to noise).
- Middle layers (9-19): low precision (reasoning, robust to noise — this is where int4 works best).
- Late layers (20-27): high precision (prediction, sensitive to noise).
- This is a STRUCTURED version of Task Precision, based on the information profile, not the task.
- **Composes with:** Lossless Quant Chain (per-layer rotation + int4), Task Precision.
- **Class:** PARTIAL (per-layer quant config, near-lossless if profile is correct). **File:** `depth_quant_key.py`

### Residual problem
MI estimation is noisy on small models. The profile is task-dependent (a code model's profile differs from a math model's).

---

## D5 — The Temporal Redundancy of KV Cache

### The hidden assumption
Each token's K/V vector is stored independently in the KV cache. Token t's K/V has no relationship to token t-1's K/V.

### Why it's wasteful
In natural language, adjacent tokens are highly correlated. Their K/V vectors are often similar (especially in long sequences with repetitive structure — code, documentation, boilerplate). Storing full independent vectors for each token is redundant. The KV cache grows linearly with sequence length, but the INFORMATION grows sublinearly (because of redundancy).

### Derivation

**D5a. KVDeltaEncoding**: store `ΔK[t] = K[t] - K[t-1]` instead of `K[t]`.
- For tokens where `‖ΔK[t]‖` is small (similar to previous), store only the delta (quantized, sparse).
- For tokens where `‖ΔK[t]‖` is large (topic shift), store the full vector.
- A per-token flag indicates: delta or full.
- **Compression:** for repetitive content (code, lists, boilerplate), delta is near-zero → ~10-50x compression on those tokens. For diverse content, delta is large → no savings (but no loss).
- **Reconstruction:** `K[t] = K[t-1] + ΔK[t]` — cumulative sum, O(T) reconstruction.
- **Different from RotorQuant** (rotates K/V for uniform compression) and **4-bit KV** (quantizes uniformly). Delta encoding exploits TEMPORAL redundancy, which neither addresses.
- **Composes with:** RotorQuant (rotate first, then delta-encode), 4-bit KV (quantize the delta), Knowledge Pack (compress packs via delta).
- **Class:** PARTIAL (compression, near-lossless if delta is quantized well). **File:** `kv_delta_key.py`

**D5b. KVPredictiveCoding**: predict `K[t]` from `K[t-1], K[t-2], ...` and store only the residual.
- A tiny linear predictor `K[t] ≈ Σ a_i * K[t-i]` (AR model, fit on calibration data).
- Store `residual[t] = K[t] - prediction[t]` (quantized).
- The residual is smaller in magnitude than the original → better quantization SNR.
- This is predictive coding (signal processing) applied to KV cache.
- **Composes with:** KVDeltaEncoding (AR-1 is the simplest predictor), RotorQuant.
- **Class:** PARTIAL (needs calibration for predictor, near-lossless). **File:** `kv_predictive_key.py`

### Residual problem
Delta/predictive encoding adds a sequential dependency to KV reconstruction (can't random-access K[t] without reconstructing K[0..t-1]). Need checkpointing (store full K every C tokens) for random access.

---

## D6 — The Coupling of Routing and Computation

### The hidden assumption
The MoE router and the experts are SEPARATE modules. The router computes a distribution over experts; the experts compute independently; the outputs are weighted by the router's distribution.

### Why it's wasteful
The routing decision and the computation are COUPLED in reality — the "best" expert depends on the input, and the input's needs depend on the expert's capabilities. But the architecture treats them as independent: the router doesn't see the expert outputs, and the experts don't see the routing weights. This is like a navigation system that picks a route without seeing traffic.

### Derivation

**D6a. ExpertAwareRouting**: the router sees a SUMMARY of each expert's recent outputs.
- Maintain an exponential moving average of each expert's output: `EMA_e = α * output_e + (1-α) * EMA_e`.
- The router input includes the EMAs: `router(x, EMA_1, ..., EMA_N)`.
- The router can learn to AVOID experts that have been producing low-quality outputs recently, or PREFER experts that are "warmed up."
- **At init:** EMA contribution to router = 0 → standard routing (lossless).
- **Composes with:** MoE router, Expert Consolidation (merge experts with similar EMAs), AirMoE (offload experts with cold EMAs).
- **Class:** PARTIAL (needs fine-tune to learn EMA-aware routing). **File:** `expert_aware_router_key.py`

**D6b. ComputeAwareRouting**: the router considers the COMPUTE COST of each expert.
- Each expert has a "cost" (FLOPs, memory, latency). The router's objective: maximize quality subject to a compute budget.
- `route = argmax_i (quality_i - λ * cost_i)` where λ is a compute budget knob.
- At runtime, adjusting λ trades quality for speed. High λ → cheap experts (fast). Low λ → best experts (slow).
- **At init:** λ = 0 → standard quality-maximizing routing (lossless).
- **Composes with:** AirMoE (cost = disk load time), PocketExpert (cost = decompression time), BudgetGuard (budget = forgetting profile).
- **Class:** TRIVIAL (runtime knob, lossless at λ=0). **File:** `compute_aware_router_key.py`

### Residual problem
Expert-aware routing adds a sequential dependency (router depends on previous outputs). Compute-aware routing needs accurate cost estimates per expert.

---

## D7 — The Symmetry of Q and K

### The hidden assumption
Attention is `softmax(Q @ K^T)`. Q and K play symmetric roles in the dot product — `Q @ K^T = K @ Q^T` (transposed). The field treats them as two projections of the same input, learned independently.

### Why it's wasteful
The symmetry means: if we transpose the attention matrix, Q and K swap roles. But the model doesn't exploit this — Q and K are learned independently, and there's no constraint that they should be related. For many attention patterns (self-attention where Q and K come from the same input), the Q and K projections are learning RELATED functions, but without any mechanism to share or constrain this relationship.

### Derivation

**D7a. QKTying**: tie Q and K projections via a shared subspace.
- Decompose: `W_q = U_q @ S_qk`, `W_k = U_k @ S_qk` where `S_qk` is a SHARED low-rank matrix.
- Q and K share the "what to attend to" subspace (S_qk) but have different "how to project" matrices (U_q, U_k).
- **At init:** `S_qk = I`, `U_q = W_q`, `U_k = W_k` → lossless (standard Q/K).
- Fine-tune to learn the shared subspace → fewer parameters, better generalization.
- **Composes with:** MLA (MLA already compresses K/V; this compresses the Q-K relationship), QK-Norm (normalize the shared subspace).
- **Class:** FULL (reversible via SVD, lossless at init). **File:** `qk_tying_key.py`

**D7b. AsymmetricQK**: break the symmetry deliberately — Q and K have DIFFERENT dimensionalities.
- Q projects to d_q dimensions, K projects to d_k dimensions, with d_q ≠ d_k.
- The dot product becomes: `Q @ W_shared @ K^T` where W_shared (d_q × d_k) bridges the asymmetry.
- **Why:** Q needs to represent "what I'm looking for" (rich, high-dim). K needs to represent "what I have" (compact, low-dim). Forcing them into the same dimensionality wastes capacity.
- **At init:** d_q = d_k = d_head, W_shared = I → lossless. Fine-tune to learn the asymmetry.
- **Composes with:** MLA (K is already compressed; this extends to Q), GQA (different Q/K group sizes).
- **Class:** PARTIAL (needs fine-tune to learn W_shared). **File:** `asymmetric_qk_key.py`

### Residual problem
QK tying reduces expressiveness (Q and K can't learn fully independent functions). Asymmetric QK adds a bridging matrix (extra params + compute).

---

## D8 — The Dimensionality of "Concept Space"

### The hidden assumption
All layers operate in the same d-dimensional space. The residual stream is R^d at every layer. Layer 0 and layer 27 both work in R^1536.

### Why it's wasteful
"Concept space" is not uniform. Early layers process "token-level" features (low-dimensional: syntax, morphology). Late layers process "concept-level" features (high-dimensional: semantics, reasoning). Forcing all layers into the same d means:
- Early layers are OVER-DIMENSIONALIZED (they only need ~256 dims for syntax, but get 1536).
- Late layers are UNDER-DIMENSIONALIZED (reasoning might want 4096 dims, but only get 1536).
- The residual stream carries "dead dimensions" through layers that don't use them.

### Derivation

**D8a. VariableWidthResidual**: the residual stream changes dimensionality across depth.
- Layers 0-8: d=512 (token features).
- Layers 9-19: d=1536 (reasoning).
- Layers 20-27: d=3072 (prediction).
- Projection matrices at layer boundaries: `x_wide = W_expand @ x_narrow`, `x_narrow = W_compress @ x_wide`.
- **At init:** all layers d=1536, projection matrices = I → lossless. Fine-tune to learn the width profile.
- **Composes with:** SliceGPT (which prunes dimensions — this makes width DEPTH-DEPENDENT), Shared Basis (share basis across same-width layers).
- **Class:** PARTIAL (needs fine-tune to learn width profile). **File:** `variable_width_key.py`

**D8b. SpectralResidual**: decompose the residual into frequency bands, route bands to layers by depth.
- DCT of the residual: `x = Σ x_freq_k` where x_freq_k is the k-th frequency band.
- Early layers receive high-frequency bands (syntactic, local patterns).
- Late layers receive low-frequency bands (semantic, global patterns).
- Middle layers receive all bands (reasoning needs both).
- **At init:** all bands passed to all layers → lossless. After calibration, route selectively.
- **Composes with:** NormGatedMoD (skip layers that don't need certain bands), VariableWidth (frequency = a form of width).
- **Class:** PARTIAL (needs calibration, lossless at init). **File:** `spectral_residual_key.py`

### Residual problem
Variable width changes the architecture (projection matrices at boundaries). Spectral routing adds DCT/iDCT overhead.

---

## D9 — The Granularity of Weight Updates

### The hidden assumption
Weight updates are per-matrix (full rank). When we update a weight (training, Fact Injection, Context Patch), we update the entire matrix or a rank-1 slice. There's no notion of updating a SUBMATRIX or a BLOCK.

### Why it's wasteful
Weights have STRUCTURE — they're not random matrices. Blocks of a weight matrix correspond to specific feature groups. Updating the whole matrix when only a block needs changing is wasteful (and risks interfering with other blocks). Fact Injection's rank-1 update touches the whole matrix; Context Patch's rank-1 update touches one direction. Neither can target a SPECIFIC BLOCK.

### Derivation

**D9a. BlockTargetedPatch**: decompose weight matrices into blocks, patch only the relevant block.
- `W = [[W_11, W_12], [W_21, W_22]]` (block decomposition).
- A fact about "coding" might only need to update W_11 (the coding-feature block).
- A fact about "math" might only need to update W_22 (the math-feature block).
- **How to identify blocks:** cluster the weight matrix's rows/columns by activation patterns (which inputs activate which rows). Blocks = clusters.
- **Patch:** `W_block += rank_1_update` — only the relevant block changes.
- **Benefit:** patches DON'T INTERFERE (a coding patch doesn't touch the math block). This solves Context Patch's residual problem (patch interference).
- **Composes with:** Fact Injection (block-targeted), Context Patch (block-targeted), DoRAPatch (direction-only + block-targeted), GRAIL (heal per-block).
- **Class:** PARTIAL (needs block identification, lossless if blocks are correct). **File:** `block_patch_key.py`

**D9b. SparseWeightUpdate**: represent weight updates as SPARSE matrices, not rank-1.
- Instead of `ΔW = a @ b^T` (rank-1, dense), use `ΔW = Σ_i a_i @ e_i^T` (sparse, only specific columns).
- The sparse update touches only the COLUMNS (feature dimensions) that are relevant to the fact.
- **Benefit:** multiple sparse updates can stack without interference (they touch different columns). Rank-1 updates all interfere (they all span the full space).
- **Composes with:** Fact Injection (sparse instead of rank-1), GRAIL (heal sparse updates).
- **Class:** PARTIAL (needs column selection, lossless if columns are correct). **File:** `sparse_patch_key.py`

### Residual problem
Block/sparse identification needs calibration data. Wrong blocks → patches go to the wrong place → quality loss.

---

## D10 — The Linearity of the Forward Pass

### The hidden assumption
Each token sees each layer exactly once, in order, left to right. The forward pass is a single linear chain: `x → L0 → L1 → ... → L27 → output`. No iteration, no revisiting, no skipping back.

### Why it's wasteful
Some tokens are "hard" (key reasoning steps, rare words) and might benefit from being processed MULTIPLE TIMES through the same layers. Some tokens are "easy" (filler, common words) and only need a few layers. The fixed single-pass forward gives every token the same compute, regardless of difficulty.

### Derivation

**D10a. IterativeRefinement**: allow a token to RE-ENTER earlier layers for refinement.
- After the forward pass, a "confidence" signal (from the output distribution's entropy) decides: is this token's representation good enough?
- If not, feed the token's final hidden state BACK INTO layer L (where L is chosen by the confidence signal), and re-run layers L to 27.
- This is a form of "internal chain-of-thought" — the model iterates on a token's representation before committing to an output.
- **At init:** confidence threshold = 0 → never re-enter → standard single pass (lossless).
- **Composes with:** MTP (multi-token prediction as the confidence signal), DSpark (speculative decoding with iterative refinement), NormGatedMoD (skip layers during re-entry).
- **Class:** TRIVIAL (runtime loop, lossless at init). **File:** `iterative_refine_key.py`

**D10b. CrossTokenRevisiting**: when generating token t, allow it to RE-ATTEND to tokens 0..t-1 with UPDATED attention patterns.
- Standard generation: token t attends to tokens 0..t-1 once, produces output, moves on.
- Cross-token revisiting: after generating token t, if the model is uncertain, re-compute token t's attention with the FULL context (including tokens t-1, t-2 that were generated after t's initial computation).
- This is a form of "backward pass at inference" — the model revisits earlier decisions with new information.
- **At init:** never revisit → standard (lossless).
- **Composes with:** DSpark (speculative decoding with backward verification), Knowledge Pack (inject context during revisit).
- **Class:** TRIVIAL (runtime, lossless at init). **File:** `cross_token_revisit_key.py`

### Residual problem
Iterative refinement adds compute (re-running layers). Cross-token revisiting requires re-computing attention (KV cache update). Both add latency for uncertain tokens.

---

## D11 — The Attention Pattern as Transferable Knowledge (bonus)

### The hidden assumption
Attention patterns are computed fresh for each input. The model "learns" to attend correctly during training, and at inference it re-derives the pattern from scratch.

### Why it's wasteful
For specific TASK TYPES (code completion, math reasoning, translation), the OPTIMAL attention pattern is predictable. The model spends compute learning to attend correctly when the pattern could be PRE-COMPUTED and INJECTED. This is the attention-level analog of Knowledge Pack (which injects KV values, not attention patterns).

### Derivation

**D11a. AttentionPatternPack**: extract attention patterns from a fine-tuned model, store as packs, inject at inference.
- **Extract:** run a fine-tuned model on task-specific data, record the attention matrices `A_l` per layer per task.
- **Compress:** the attention matrix is sparse (most mass on a few tokens). Store only the top-k positions and their weights.
- **Inject:** at inference, if the task type is known, override the computed attention with the pre-computed pattern. The model doesn't need to "learn" the pattern — it's injected.
- **Different from Knowledge Pack:** Knowledge Pack injects KV VALUES (what the keys/values are). AttentionPatternPack injects the attention WEIGHTS (which positions to attend to). Different level of abstraction.
- **Composes with:** Knowledge Pack (inject both values and patterns), Softpick (sparse patterns + sparse weights), Indexed Attention (F1 — the pattern pack IS the index).
- **Class:** PARTIAL (needs extraction + compression, lossless if pattern matches). **File:** `attn_pattern_pack_key.py`

**D11b. AttentionDistillation**: distill the teacher's attention pattern into the student.
- Standard distillation matches OUTPUTS (logits). Attention distillation matches PATTERNS.
- Loss: `L = L_CE + α * ||A_student - A_teacher||^2`
- The student learns to attend WHERE the teacher attends, not just produce the same output.
- This is F7 (Mechanistic Distillation) but specifically for attention patterns.
- **Composes with:** Mech Distill (F7), Self-Play (generate teacher patterns from self-play solutions).
- **Class:** PARTIAL (training-time key, no weight transform). **File:** `attn_distill_key.py`

### Residual problem
Attention patterns are input-dependent — a pattern pack only works for inputs similar to the extraction data. Pattern injection overrides the model's learned attention, which may hurt on novel inputs.

---

## Summary — New Keys Proposed

| # | Deep Subject | Key Name | Class | Train? | Lossless? | File |
|---|---|---|---|---|---|---|
| D1a | Residual algebra | `mult_residual_key.py` | TRIVIAL | None (fine-tune optional) | Yes (γ=0 init) | MultiplicativeResidual |
| D1b | Residual algebra | `residual_overwrite_key.py` | PARTIAL | Minimal | Yes (m=0 init) | ResidualOverwrite |
| D1c | Residual algebra | `rk2_residual_key.py` | PARTIAL | Minimal | Yes (correction=0 init) | HigherOrderResidual |
| D2a | Attention geometry | `signed_attn_key.py` | TRIVIAL | None | Yes (disabled init) | SignedAttention |
| D2b | Attention geometry | `attn_bias_key.py` | TRIVIAL | None (fine-tune optional) | Yes (β=0 init) | AffineAttention |
| D3a | Attn/FFN duality | `ffn_as_attn_key.py` | TRIVIAL | None | Yes (reformulation) | FFNAsAttention |
| D3b | Attn/FFN duality | `unified_router_key.py` | PARTIAL | Minimal | Yes (shared init) | UnifiedRouting |
| D4a | Information flow | `info_profile_key.py` | TRIVIAL | None | Yes (profiling) | InformationProfile |
| D4b | Information flow | `depth_quant_key.py` | PARTIAL | None | Near-lossless | DepthStratifiedQuant |
| D5a | KV temporal redundancy | `kv_delta_key.py` | PARTIAL | None | Near-lossless | KVDeltaEncoding |
| D5b | KV temporal redundancy | `kv_predictive_key.py` | PARTIAL | None (calibration) | Near-lossless | KVPredictiveCoding |
| D6a | Routing/compute coupling | `expert_aware_router_key.py` | PARTIAL | Minimal | Yes (EMA=0 init) | ExpertAwareRouting |
| D6b | Routing/compute coupling | `compute_aware_router_key.py` | TRIVIAL | None | Yes (λ=0 init) | ComputeAwareRouting |
| D7a | Q/K symmetry | `qk_tying_key.py` | FULL | None | Yes (S=I init) | QKTying |
| D7b | Q/K symmetry | `asymmetric_qk_key.py` | PARTIAL | Minimal | Yes (d_q=d_k init) | AsymmetricQK |
| D8a | Concept space dimensionality | `variable_width_key.py` | PARTIAL | Minimal | Yes (uniform init) | VariableWidthResidual |
| D8b | Concept space dimensionality | `spectral_residual_key.py` | PARTIAL | None (calibration) | Yes (all bands init) | SpectralResidual |
| D9a | Weight update granularity | `block_patch_key.py` | PARTIAL | None (calibration) | Yes (correct blocks) | BlockTargetedPatch |
| D9b | Weight update granularity | `sparse_patch_key.py` | PARTIAL | None (calibration) | Yes (correct columns) | SparseWeightUpdate |
| D10a | Forward pass linearity | `iterative_refine_key.py` | TRIVIAL | None | Yes (threshold=0 init) | IterativeRefinement |
| D10b | Forward pass linearity | `cross_token_revisit_key.py` | TRIVIAL | None | Yes (never revisit init) | CrossTokenRevisiting |
| D11a | Attention as knowledge | `attn_pattern_pack_key.py` | PARTIAL | None (extraction) | Yes (if pattern matches) | AttentionPatternPack |
| D11b | Attention as knowledge | `attn_distill_key.py` | PARTIAL | None (training-time) | Yes (extra loss) | AttentionDistillation |

**Total: 23 new key candidates.**
- **TRIVIAL (runtime, lossless at init):** 9 (D1a, D2a, D2b, D3a, D4a, D6b, D10a, D10b, + D4a profiling)
- **FULL (reversible, lossless):** 1 (D7a QKTying)
- **PARTIAL (needs calibration or fine-tune):** 13

---

## Implementation Priority (by impact × losslessness × ForgeAI fit)

### Tier 1 — LOSSLESS, implement first (zero quality risk)

| Priority | Key | Why |
|---|---|---|
| 1 | **D7a QKTying** | FULL lossless, reduces Q/K params, composes with MLA. Only FULL key in this set. |
| 2 | **D2a SignedAttention** | TRIVIAL runtime, enables extrapolation beyond V hull. Pairs with Softpick. |
| 3 | **D2b AffineAttention** | TRIVIAL, adds attention bias (every other op has one). Pairs with Norm Folding. |
| 4 | **D1a MultiplicativeResidual** | TRIVIAL, changes residual algebra. Pairs with NormGatedMoD. |
| 5 | **D6b ComputeAwareRouting** | TRIVIAL runtime knob for speed/quality trade-off. Pairs with AirMoE. |
| 6 | **D10a IterativeRefinement** | TRIVIAL runtime loop for quality. Pairs with MTP + DSpark. |
| 7 | **D4a InformationProfile** | TRIVIAL profiling that informs ALL other keys (which layers to quantize, skip, inject into). |

### Tier 2 — NEAR-LOSSLESS, implement second (small risk, needs calibration)

| Priority | Key | Why |
|---|---|---|
| 8 | **D5a KVDeltaEncoding** | KV compression via temporal redundancy. Pairs with RotorQuant + 4-bit KV. |
| 9 | **D4b DepthStratifiedQuant** | Per-depth precision. Pairs with Lossless Quant Chain. |
| 10 | **D9a BlockTargetedPatch** | Solves Context Patch interference. Pairs with Fact Injection. |
| 11 | **D11a AttentionPatternPack** | Attention-level Knowledge Pack. Pairs with Knowledge Pack + Indexed Attention. |
| 12 | **D8b SpectralResidual** | Frequency-band routing. Pairs with NormGatedMoD. |

### Tier 3 — NEEDS FINE-TUNE, implement last (research-heavy)

| Priority | Key | Why |
|---|---|---|
| 13 | **D1b ResidualOverwrite** | LSTM-style forget gate for residual. Needs fine-tune. |
| 14 | **D3b UnifiedRouting** | Shared MoE + MoA router. Needs fine-tune. |
| 15 | **D7b AsymmetricQK** | Different Q/K dims. Needs fine-tune. |
| 16 | **D8a VariableWidth** | Depth-dependent width. Architecture change. |
| 17 | **D6a ExpertAwareRouting** | Router sees expert EMAs. Needs fine-tune. |
| 18 | **D1c RK2Residual** | Higher-order integration. Needs fine-tune. |

---

## The Meta-Pattern (how to generate more)

The 11 deep subjects above were found by applying the generative rule to STRUCTURAL ASSUMPTIONS:

| Assumption Category | What's uniform/global/binary | What it could be (structured/local/spectral) |
|---|---|---|
| **Algebra** (D1) | Additive residual | Multiplicative, overwrite, higher-order |
| **Geometry** (D2) | Convex-only output | Signed (extrapolation), affine (shift) |
| **Duality** (D3) | Separate attn/FFN | Unified (same operation, different granularity) |
| **Information** (D4) | Equal info per layer | Stratified by depth (MI profile) |
| **Temporal** (D5) | Independent KV per token | Delta/predictive (temporal redundancy) |
| **Coupling** (D6) | Separate router/expert | Aware (router sees expert state), compute-aware |
| **Symmetry** (D7) | Symmetric Q/K | Tied (shared subspace), asymmetric (different dims) |
| **Dimensionality** (D8) | Uniform d_model | Variable (depth-dependent), spectral (frequency bands) |
| **Granularity** (D9) | Full-matrix updates | Block-targeted, sparse-column |
| **Linearity** (D10) | Single forward pass | Iterative (re-enter layers), cross-token (revisit) |
| **Transferability** (D11) | Fresh attention per input | Pattern packs (extract + inject) |

**To generate more:** pick a component, identify what it treats as uniform/global/binary, and ask "what if it were structured/local/spectral?" The answer is a new key candidate. Verify: (1) lossless at init? (2) composes with existing keys? (3) has a clear mechanism? If yes to all, implement.

---

## Key Compositions (the most powerful combinations)

These new keys compose with existing ForgeAI keys to create compound effects:

### Composition 1: Full Spectral Compression
```
KVDeltaEncoding (D5a) → RotorQuant → 4-bit KV → CompressedPack
```
Delta-encode KV (temporal redundancy) → rotate (outlier smoothing) → quantize (4-bit) → compress into pack. Each stage exploits a different redundancy: temporal, rotational, precision, spatial. Combined: ~20-50x KV compression.

### Composition 2: Lossless Quality Stack
```
QKTying (D7a) → FoldQKNormMLA → NormFolding → SignedAttention (D2a) → AffineAttention (D2b)
```
Tie Q/K (shared subspace) → fold QK-Norm → fold all norms → enable signed attention (extrapolation) → add attention bias. All lossless at init. Quality gains from fine-tuning the tied subspace + signed geometry.

### Composition 3: Information-Driven Adaptation
```
InformationProfile (D4a) → DepthStratifiedQuant (D4b) → NormGatedMoD → ComputeAwareRouting (D6b)
```
Profile each layer's information content → assign precision by depth → skip low-info layers → route by compute cost. The profile INFORMS all other adaptations. Result: per-layer, per-token adaptive compute with information-theoretic backing.

### Composition 4: Non-Linear Forward Pass
```
IterativeRefinement (D10a) → CrossTokenRevisiting (D10b) → MTP → DSpark
```
Iterate on uncertain tokens → revisit with full context → use MTP for confidence signal → DSpark for speculative verification. The forward pass becomes ADAPTIVE: easy tokens get one pass, hard tokens get multiple. This is test-time compute scaling, but INTERNAL to the model.

### Composition 5: Patch Without Interference
```
BlockTargetedPatch (D9a) → SparseWeightUpdate (D9b) → DoRAPatch → GRAIL
```
Identify blocks → patch only relevant columns → apply to DoRA direction → heal with GRAIL. Multiple patches stack without interference because they touch different blocks/columns. This solves the Context Patch interference problem.

---

## Relationship to Existing Ideation

| New Key | Extends | Relationship |
|---|---|---|
| D1a MultiplicativeResidual | GateSkip, NormGatedMoD | Changes residual ALGEBRA (multiplicative vs additive gating) |
| D2a SignedAttention | Softpick | Allows NEGATIVE weights (extrapolation), not just sparsity |
| D3a FFNAsAttention | MoE, MoA | Theoretical unification — FFN optimizations apply to attention |
| D4a InformationProfile | TaskPrecision, NormGatedMoD | Information-theoretic PROFILING to inform other keys |
| D5a KVDeltaEncoding | RotorQuant, 4-bit KV | Exploits TEMPORAL redundancy (neither rotation nor quant does) |
| D6b ComputeAwareRouting | AirMoE, BudgetGuard | Runtime compute budget knob (λ=0 lossless) |
| D7a QKTying | MLA, QK-Norm | FULL lossless — shared Q/K subspace |
| D8b SpectralResidual | F2 MemoryHierarchy | Frequency-band decomposition of residual |
| D9a BlockTargetedPatch | ContextPatch, FactInjection | Solves patch INTERFERENCE (block-level targeting) |
| D10a IterativeRefinement | MTP, DSpark, F10 TaskPrecision | Internal test-time compute (re-enter layers) |
| D11a AttentionPatternPack | KnowledgePack, F1 IndexedAttn | Attention-level pack (inject patterns, not values) |

---

## ForgeAI V2 Application (which new keys to apply)

For V2 specifically (lossless priority):

1. **D7a QKTying** — FULL lossless, reduces Q/K params. Apply first.
2. **D2a SignedAttention** — TRIVIAL runtime toggle. Apply for quality (extrapolation).
3. **D2b AffineAttention** — TRIVIAL, adds attention bias. Apply for quality.
4. **D1a MultiplicativeResidual** — TRIVIAL, γ=0 init. Apply for potential quality (fine-tune γ).
5. **D4a InformationProfile** — TRIVIAL profiling. Run to inform D4b, NormGatedMoD, TaskPrecision.
6. **D6b ComputeAwareRouting** — TRIVIAL runtime knob. Apply for speed control.
7. **D10a IterativeRefinement** — TRIVIAL runtime loop. Apply for quality on uncertain tokens.

**All 7 are lossless at init.** They add capabilities without changing the model's output until fine-tuning or runtime toggles are enabled. This is the ForgeAI philosophy: lossless at init, quality from fine-tuning, speed from runtime.

---

*Compiled 2026-08-07. Extends `FIRST_PRINCIPLES_IDEATION.md` (F1-F12), `NOVEL_SOLUTION_IDEATION.md` (Ideas 1-12), and `KEY_NOVELTY_AUDIT_PART2.md` (8 novel keys). Total ideas across all docs: 100 (77 existing + 23 new). Use the Meta-Pattern to generate more.*
