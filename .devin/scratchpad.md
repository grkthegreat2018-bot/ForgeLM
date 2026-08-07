# Scratchpad — LLM Research Notes (2026-08-06)

## 2026 Techniques Found (gaps in existing survey)

### Sparse Attention (MoE-ification of attention)
- **MISA** (Mixture of Indexer Sparse Attention, arxiv 2605.07363): treats DSA indexer's H^I heads as MoE pool; lightweight router picks h<<H^I active heads per query. O(H^I L) -> O(hL + H^I M). 8x fewer indexer heads, matches dense DSA on LongBench. MISA+ hierarchical re-ranks. 3.82x speedup on H200. KEY INSIGHT: MoE on the HEAD axis of attention, not just FFN.
- **Sparse Frontier** (ACL 2026 findings): largest training-free sparse attn eval. Findings: (1) larger sparse > smaller dense at equal cost (Pareto); (2) prefill fine-grained per-query estimation impractical -> choose global-to-token vs block-to-block by task; (3) decoding token-to-page selection feasible; (4) longer seq tolerates higher sparsity -> fixed-budget suboptimal.
- **Alloc-MoE** (ACL 2026): budget-aware expert activation. Alloc-L (layer-level, sensitivity profiling + DP) + Alloc-T (token-level, dynamic redistribute by routing scores). 1.15x prefill / 1.34x decode at HALF budget on DeepSeek-V2-Lite. KEY: non-uniform expert budget across layers.
- **SpecMoE** (arxiv 2604.10152): self-assisted speculative decoding for MoE offload. No extra training. Up to 4.30x throughput, reduced bandwidth. Spec decode hides expert load latency.
- **LightMoE** (ACL 2026 findings): task-aware expert availability. frequency-aware resident core experts + similarity-based redirection (no I/O for missing experts) + coarse task-level replacement. KEY: resident experts + redirect-to-similar instead of load.

### Test-Time Training (TTT) for reasoning
- **TEMPO** (arxiv 2604.19295): scaling TTT via EM. Interleave policy refinement (unlabeled) with periodic critic recalibration (labeled). Prior methods = incomplete EM (missing E/M recalibration). OLMO3-7B AIME 33->51%, Qwen3-14B 42->66%. Maintains diversity.
- **Policy of Thoughts (PoT)** (arxiv 2601.20379): per-instance TTT. MCTS generates candidates -> GRPO updates TRANSIENT LoRA adapter using execution feedback -> discard adapter after. 4B model 49.71% LiveCodeBench, beats GPT-4o/DeepSeek-V3, 50x smaller. KEY: throwaway per-instance LoRA.
- **ORCA** (arxiv 2604.01170): conformal prediction + TTT for calibration. Meta-learned calibration module updated per input. Valid confidence under distribution shift. Qwen2.5-32B 47.5% savings in-dist, MATH-500 24.8->67% OOD.
- **DiSCTT** (arxiv 2603.05357): difficulty-aware consensus-guided self-curriculum. High consensus -> SFT with majority pseudo-labels; low consensus -> RL with consensus-regularized objective. Adaptive strategy by epistemic uncertainty.
- **Decision-theoretic TTT** (arxiv 2606.15569): TTT = implicit Bayesian inference in kernel regime. Spectrally match updates to prompt SNR, align to query-relevant eigen-directions. PAC-Bayes guarantee on step selection. Bayes-optimal update subspace -> scoring rule for which Transformer blocks/heads to adapt.

### Model Merging (2026 theory)
- **LoRA Soups / CAT** (COLING 2025): concatenation of LoRAs with optimal weighting beats model/data merging. Math+code: +43% over model-merge, +12% over data-merge. Skill composition via CAT.
- **SVD+CUR LoRA merging** (ACL 2026 findings): SVD captures shared structure, CUR preserves task-specific/localized. Geometrically misaligned, complementary. Training-free combine -> consistent gains.
- **Task vectors = gradients** (PMLR UniReps 2026): task vector from 1 epoch finetune = -lr * grad(loss). Multi-epoch approx holds with bounded 2nd-order term. First-epoch gradient dominates. KEY: single-epoch finetune merges ~ as good as converged. Merging = approx multitask learning.
- **Pico** (arxiv 2604.16826): LoRA merge interference from B matrix (output-side), not A. B reuses shared directions -> overemphasizes them. Data-free: downscale B on shared dirs, rescale after. +3.4-8.3 points over Task Arithmetic/TIES/TSV-M. KEY: treat A and B separately.
- **CT-Merging** (arxiv 2607.20561): consensus directions from avg task subspace projectors + task-level RMS coefficient scales. +2.56 over DC-Merge.

### Continual Learning (2026)
- **FOREVER** (ACL 2026): Ebbinghaus forgetting curve for LLM CL. Model time = magnitude of optimizer updates (not raw steps). Forgetting-curve replay scheduler + intensity-aware regularization. 0.6B-13B.
- **Self-generated replay** (arxiv 2605.26097): LLM samples own training distribution as replay -> nearly eliminates forgetting. BUT persists when capacity saturated (pretrained close to saturation). Replay breaks low-lr/many-steps tradeoff. KEY: capacity is the real constraint, not just replay.
- **MSSR** (arxiv 2603.09892): Memory-Inspired Sampler + Scheduler Replay. sample-level memory strength + adaptive rehearsal intervals.
- **OAKS benchmark** (ACL 2026): online adaptation to continual knowledge streams. Facts evolve over time. 14 models + agentic memory systems FAIL. delays in state-tracking, distraction. KEY: even SOTA + RAG/memory agents fail at streaming fact updates.
- **Mechanistic forgetting** (arxiv 2601.18699): early-layer attention heads = entropic dispersion; mid-deep FFN/MoE expert blocks = localized representation collapse. CKA + routing gate drift. KEY: forgetting is layer-localized, not uniform.

## Novel Combination Patterns (the AirMoE template)
Template: Technique A has problem P_A; Technique B has problem P_B; A+B solves both.
- AirLLM (problem: not needed for small) + MoE (problem: experts in VRAM) = AirMoE (tiny base + hot-load experts like LoRA).

## 2026 Round 2 — Per-Component Research

### Position Encoding (RoPE)
- **RoPE provably fails at long context** (arxiv 2605.15514): as length grows, RoPE loses locality bias AND token-relevance consistency; failure prob -> 0.5 (random). Adjusting base trades position-distinguish vs token-distinguish, can't keep both.
- **Frayed RoPE / RoPE-ID** (arxiv 2603.18017): unified geometric view. Q/K form tight clusters, sink token near origin. RoPE on long inputs damages cluster separation -> kills sink functionality. Fix: apply RoPE high-freq to a SUBSET of channels only (RoPE-ID). Generalizes to longer inputs out-of-box.
- **NoPE beats RoPE** (NeurIPS 2023, confirmed 2026): NoPE outperforms ALiBi/RoPE/APE on length generalization. NoPE can represent both absolute AND relative PE; under SGD resembles T5 relative. No extra compute. Decoder-only.
- **ScoPE** (ACL 2026): Scoped Position Encoding. Replaces arithmetic PE with exponentially-scaled look-back SCOPES per head -> hierarchical processor, order-awareness horizon grows exponentially with depth. Parameter-free, 8x attention FLOP reduction via masking. Beats RoPE on native length extrapolation + retrieval.
- **LeRoPE** (arxiv 2607.10134): learnable scalar per RoPE frequency (frequencies as params not hyperparams). Emergent high-norm "LeRoPE band". Beats RoPE + partial-RoPE at all scales (52M-2.5B); RoPE needs 3.4% more compute to match. Slowest bands carry SEMANTIC not positional info.

### Attention Activation (softmax replacements)
- **Polynomial attention** (arxiv 2410.18613, v3 2026): softmax's power = implicit Frobenius-norm regularization, NOT the probability distribution. Polynomials achieve same regularization without positivity/normalization/sparsity. Drop-in.
- **Sigmoid attention** (Apple): universal function approximator, better regularity than softmax. FLASHSIGMOID = 17% kernel speedup over FA2 on H100. Drop-in if early-training norm stabilized. Matches softmax across language/vision/speech.
- **ASEntmax** (ICLR 2026): adaptive-scalable entmax. Interpolates sparse (pattern-focused) <-> dense (softmax-like) via learnable temp. 1000x length extrapolation on synthetic, better PPL + retrieval at 8x train length.
- **Softpick** (ACL 2026 findings): rectified, non-sum-to-one softmax replacement. ELIMINATES attention sink + massive activations. 0% sink rate. Lower kurtosis hidden states, sparse attention maps. Quantized models BEAT softmax, esp at low bits. Opens: quantization, low-precision training, pruning, interpretability.
- **SSA** (ACL 2026): Scaled Signed Averaging. Softmax causes ICL distribution-shift failures. SSA fixes them, beats softmax on NLP + linguistic probing (decoder + encoder).
- **Poly-attention** (arxiv 2602.02422): general higher-order attention (triples, compositional). Self-attn can't detect triples of correlated tokens. Poly-attention generalizes; tight complexity/expressiveness tradeoffs.

### Embedding / Weight Tying
- **Weight tying biases to OUTPUT space** (ACL 2026 findings): tied matrix aligns with unembedding not input embedding; output gradients dominate early training. Hurts early-layer computations (tuned lens). Scaling input gradients reduces bias. Explains why tying hurts at scale + for small LLMs (embed is big param fraction).
- **MEL (Matryoshka Embedding Learning)** (arxiv 2605.15081, 3D-ML): factorized low-rank embedding matrix, nested-trained. Reduces TOTAL params (not just trainable like LoRA). Embedding is 1/4 of params for multilingual Qwen3-0.6B. 9/17 MTEB records.
- **MIPIC** (ACL 2026): MRL via self-distilled intra-relational + progressive info chaining. Aligns full vs truncated reps via top-k CKA. Depth-wise semantic consolidation (deep->early layer transfer).
- **Random truncation vs MRL** (OpenReview): random truncation of NON-MRL models is competitive with MRL unless >=80% reduction. MRL extra training cost only pays off at extreme truncation.

### FFN / Activation
- **PolyGLU** (arxiv 2603.13347): per-neuron dynamic routing among K=4 activations (GELU/Tanh/etc) via Gumbel-Softmax. Emergent depth specialization: early layers -> GELU, deep -> Tanh. 0.23% param overhead. Near-deterministic routing (entropy 0.03% of max). Biological neurotransmitter analogy.
- **Mamba-3** (arxiv 2603.15569): improved SSM. Selective state space, hardware-aware.
- **Rethinking nonlinearity as gating** (arxiv 2607.03148): unify activations as gating ops.

### Residual Stream / Depth
- **mHC** (Manifold-Constrained Hyper-Connections): N parallel residual streams, learned routing matrices. DeepSeek-V4. First open mHC model (780M) analyzed: streams encode distinct info, asymmetric utilization, not just redundancy.
- **xHC** (arxiv 2607.14530): expands HC beyond N=4 (mHC stops there - diminishing returns + cubic mixing cost). xHC: temporal feature augmentation + sparse residual (update only k=4 of N=16 streams, dense access to full state). +4.0 points over mHC on 18B MoE. xHC-Flash cuts memory traffic 73.5C->40C. Vanilla needs 1.50x compute, mHC 1.19x to match xHC.
- **go-mHC** (arxiv 2604.02309): direct parameterization via generalized orthostochastic matrices. O(d^3), single hyperparam s interpolates efficient-boundary <-> full Birkhoff polytope.
- **MoD (Mixture-of-Depths)** (arxiv 2404.02258): top-k routing per layer -> token either runs the block OR skips via residual. Static compute graph, dynamic per-token depth. Applies to BOTH attention AND MLP.
- **MoD Attention** (arxiv 2603.15619): depth-scaling via attention-style dynamic mixing across layers (not just residual). Solves info dilution that vanilla residual leaves unresolved.
