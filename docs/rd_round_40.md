# Round 40 — Exotic & Next-Gen Architecture Research Sweep

## Overview
Six parallel research subagents surveyed the frontier across six domains
to identify novel techniques applicable to ForgeEngine on RTX 5070 12GB.
This round is **research-only** (no code changes); findings inform R41+
implementation priorities.

## Research Domains & Top Picks

### R40-1: Sequence-Mixing Alternatives (Mamba/Attention replacements)
**Top candidates ranked by ForgeEngine fit:**

| Architecture | Complexity | Recall | Best use in ForgeEngine |
|---|---|---|---|
| **Gated DeltaNet-2** | O(L) | Best recurrent recall (RULER) | Replace Mamba layers in hybrid |
| **RWKV-7 "Goose"** | O(L) | Excellent state tracking | Tiny state, ideal for 12GB |
| **Mamba-2 (SSD)** | O(L) | Better than Mamba-1 | Drop-in Mamba upgrade |
| **Based** | O(L)+small window | Best subquadratic recall | Sliding-window + linear hybrid |
| **GSA** | O(L) | Strong T2R conversion | Finetune Transformer→RNN |
| **GLA** | O(L) | Moderate | Long prefill replacement |

**Key insight**: Zoology paper shows 82% of perplexity gap vs Transformers
is **associative recall** — pure SSM/linear attention fails at recall because
of fixed-size state. Hybrid designs (Based, Gated DeltaNet + sliding window)
close 97%+ of the gap.

**Recommendation for R41**: Replace Mamba-1 with **Mamba-2 SSD** (drop-in,
2-8x faster, larger state). Prototype **Gated DeltaNet-2** as the recurrent
layer in a hybrid with sliding-window attention for recall.

**Honorable mentions**: TTT (test-time training layers), Titans (2M context
neural memory), Longhorn (closed-form SSM from online recall).

---

### R40-2: Decoding & Inference Acceleration
**Top candidates for hybrid Mamba/Attention on 12GB:**

| Priority | Technique | Speedup | Integration cost |
|---|---|---|---|
| 1 | **MatryoshkaKV** | 60% KV compression | Low — attention-only, Mamba unaffected |
| 2 | **Medusa** | 2.2-2.8x | Low — add heads on final hidden state |
| 3 | **EAGLE-2/3** | 3-6.5x | High — draft head + tree attention |
| 4 | **ReDrafter (Mamba draft)** | 2.5-3.5x | Medium — Mamba O(1) state = cheap drafts |
| 5 | **TIDE / Dr.LLM** | 5-20% | Low — post-training dynamic depth |

**Mamba synergy**: Mamba's O(1) recurrent state makes draft generation
extremely cheap (no KV cache for draft model). Target verification uses
chunk-wise SSM scan. **ReDrafter with a Mamba draft model is the most
Mamba-native speculative design.**

**Recommendation for R41**: Implement **MatryoshkaKV** (biggest 12GB win —
KV cache is the memory wall). Add **Medusa heads** (lowest integration cost
speculative decoding). Both are already partially supported by engine
infrastructure.

---

### R40-3: Product Key Memory (PKM)
**What**: Learnable key-value memory layer. Two sub-key codebooks
(C¹ × C² = n_keys² total keys). Top-k retrieval from value table.
O(√|K|) search cost. Capacity grows with n_keys² × dim, compute with √n_keys.

**Key results**:
- 12-layer Transformer + 1 PKM layer **outperforms 24-layer dense baseline**
  at 2x faster inference (Lample 2019)
- Memory Layers at Scale (Meta 2024): 128B memory params, matches dense
  models trained with **4x more compute**
- FwPKM (SakanaAI 2026): online chunk-level updates → 128K context recall

**ForgeEngine integration path**:
- Add `use_pkm` config flag in `research/config.py`
- Implement `PKMResidual` class (augment, don't replace FFN — Kim & Jung 2020)
- Zero-init value table → identity at start (matches `zero_init_residual`)
- Sparse gradients via `EmbeddingBag(sparse=True)`, higher LR for values

**12GB VRAM budget** (d_model=2048, bf16):
- num_keys=128 → 64 MB/layer → 1.0 GB for 16 layers
- num_keys=256 → 256 MB/layer → 4.1 GB for 16 layers
- **Practical**: num_keys=128, topk=8, PKM on 4-8 attention layers only,
  quantize value table with IRI-FP4/NVFP4

**Recommendation for R42**: Prototype residual PKM augmentation on 4
attention layers with num_keys=128. This is novel for hybrid Mamba/Attention
models — no existing work combines PKM with SSM layers.

---

### R40-4: KAN (Kolmogorov-Arnold Networks) as FFN
**What**: Replace fixed activations on nodes with learnable B-spline functions
on edges. Each "weight" is a 1D spline with G+k coefficients.

**Verdict**: **Novel but unproven for LLMs.** Literature shows no consistent
quality/speed win over SwiGLU:
- KAN is 1.36-100x slower than MLP
- Under matched params/FLOPs, MLP generally outperforms KAN
- KAN's real advantage is interpretability, not throughput

**Existing hybrid work**: Spectra (339M Mamba+KAN), Mamba_KAN, KANama
(Llama3+KAN). None show clean Pareto win on both quality and throughput.

**If pursued**: Use **rational/grouped KAN (GR-KAN)** — best latency/quality
trade-off. Add `ffn_type="kan_rational"` in config, implement `KANFFN`
beside `SwiGLUFFN`, train small model (d_model=448, 16 layers, seq=512)
for controlled ablation.

**Recommendation**: **Low priority.** Park as R43+ experiment only if
interpretability/edge-pruning becomes a goal. KAN's spline overhead on 12GB
is risky.

---

### R40-5: Training Objectives & Data Strategies
**Top picks for single-GPU ForgeEngine:**

| Technique | Status in codebase | Priority |
|---|---|---|
| **Multi-Token Prediction (MTP)** | Already wired (`mtp.py`, config) | **Use now** |
| **Evol-Instruct / Phi-style data** | Chat ratings → SFT export exists | **Use now** |
| **DPO** | Implemented (`dpo_align.py`) | **Use now** |
| **KTO** | Small patch on DPO | High — binary labels, no pairs |
| **RPO / RLVR** | Implemented (`rpo_train.py`, `rlvr_train.py`) | High — cheap reasoning RL |
| **Curriculum / InfoDensity** | `curriculum_sft.py` exists | High — free compute |
| **SPIN (self-play)** | Pieces exist (DPO + self-gen) | Medium |
| **SePO (sparse token rewards)** | `grpo_trainer.py` has `use_gvpo` | Medium |

**Mamba/Attention pairing**: MTP, InfoDensity, and process rewards benefit
most from hybrid — SSM compresses long low-entropy traces, attention handles
high-uncertainty "pivotal" tokens. `ForgeHybrid` already routes by entropy.

**Recommendation for R41**: Activate MTP (`--mtp-weight 0.3 --mtp-n-heads 2-4`)
in next training run. Implement KTO as binary-label variant of `dpo_align.py`.
Add InfoDensity reward shaping to `rlvr_train.py`.

---

### R40-6: Extreme Quantization (beyond INT4)
**Landscape**:

| Method | Bits/weight | Architecture | RTX 5070 ready? |
|---|---|---|---|
| **NVFP4 + MR-GPTQ** | 4 bpw | Transformer | Yes (SM120 native) |
| **AQLM + PV-Tuning** | 1-2 bpw | Transformer | Needs kernel compile |
| **QuIP#** | 2-4 bpw | Transformer | Needs SM120 build |
| **BitNet b1.58** | 1.58 bpw | Transformer (train from scratch) | `bitnet.cpp` only |
| **BTC-LLM** | 0.7-1.11 bpw | Transformer | Research stage |
| **Quamba2** | W4A8 | Mamba/SSM | Best SSM PTQ |
| **Bi-Mamba** | 1 bpw | Mamba (train from scratch) | Research |

**Critical rule for hybrid models**: Use **different quantizers per block
type**. Attention → AQLM/QuIP#/NVFP4. Mamba/SSM → Quamba2/SSDi8. Naive
application of attention-optimized 1-2b PTQ to SSM blocks causes
catastrophic error (PPL ~13M).

**ForgeEngine already supports**: GPTQ INT4, Hadamard INT4 KV, RotorQuant KV,
NVFP4 (via fallback chain), FP8, QuaRot.

**Recommendation for R41**: Wire **Quamba2 W4A8** for Mamba blocks
alongside existing NVFP4 for attention blocks. This gives 4x weight
compression on both block types with SSM-appropriate activation handling.

---

### R40-7: Mamba-3 & Latest SSM Frontier (2025-2026)

**Mamba-3** (arXiv:2603.15569, March 2026, Lahoti/Li/Chen/Wang/Bick/Kolter/Dao/Gu):
Inference-first SSM with three core improvements over Mamba-2:

1. **Exponential-trapezoidal discretization**: More expressive recurrence than
   Mamba-2's exponential-Euler. Second-order accurate (O(Δ²) vs O(Δ²) first-order).
   Recurrence becomes: `h_t = α_t h_{t-1} + β_t B_{t-1} x_{t-1} + γ_t B_t x_t`
   (3-term vs Mamba-2's 2-term). Data-dependent `trap` mixing coefficient.
2. **Complex-valued SSM state**: `A_log` shape `(d_inner, d_state, 2)` — last dim
   is [real, imag]. Enables richer state tracking (parity, modular arithmetic).
   Imaginary part zero-init from Mamba-2 = lossless warm start.
3. **MIMO (Multi-Input Multi-Output)**: Multiple parallel SSMs with rank-R
   outer-product state updates. Boosts accuracy +1.2 points at 1.5B scale
   without increasing decode latency. `mimo_rank=4` default.

**Key results** (1.5B scale):
- Mamba-3 SISO: +0.6 avg downstream accuracy over Gated DeltaNet
- Mamba-3 MIMO: +1.8 total gain over Gated DeltaNet
- Comparable perplexity to Mamba-2 with **half the state size**
- Beats Llama-3.2-1B Transformer on prefill+decode latency at all seq lengths

**ForgeAI status**: `Mamba3Key` already exists at
`research/keys/architecture/mamba3_key.py` — implements lossless Mamba-2→Mamba-3
complex state conversion (imag=0 zero-init). Config references at
`research/config.py:423-424, 880-881, 925-926`. The key handles:
- `A_log` (d_inner, d_state) → (d_inner, d_state, 2) with imag=0
- `x_proj` gains extra rows for imaginary B/C (zero-init)
- Norms gain trailing dim 2 (complex RMSNorm, imag=0)
- Round-trip Mamba-2→Mamba-3→Mamba-2 is identity (verified)

**What's missing**: The actual `Mamba3Layer` forward pass (complex SSM scan with
exponential-trapezoidal discretization + MIMO). The Key converts weights but
there's no Mamba-3 module to load them into. Need to port `mamba3_mimo_combined`
kernel from `state-spaces/mamba/modules/mamba3.py`.

**Other SSM advances 2025-2026**:

| Advance | What it does | Relevance |
|---|---|---|
| **ReplaySSM** (Dao Lab, 2026) | Cache SSM inputs (not state) in ring buffer for cheap speculative decode rollback. 1.48x AR speedup, 1.87-1.96x spec decode. | **Critical for ForgeEngine** — enables efficient speculative decoding with Mamba layers |
| **Fused SSD kernel** (PyTorch blog) | Fuses 5 Mamba-2 SSD kernels into 1 Triton kernel. 1.5-2.5x prefill speedup. | Drop-in for Mamba-2 prefill |
| **SSDi8** (arXiv:2608.21952) | INT8 PTQ for SSD. W4A8/W8A8, 1.4x speedup, FP16 accuracy. | SSM-specific quant (already in R41 plan) |
| **Granite 4.0** (IBM, Oct 2025) | Hybrid Mamba-2/Transformer + MoE. 9:1 Mamba:attention ratio. 70% lower memory, 2x faster. 32B/9B active, 7B/1B active, 3B dense. Apache 2.0. | Validates hybrid ratio; 1B/3B models fit 12GB |
| **Zamba2-VL** (arXiv:2606.00390) | Hybrid Mamba-2 + shared transformer blocks. 1.2B/2.7B/7B. 10x lower TTFT than Transformer VLMs. | Shared-attention design for 12GB |
| **Bamba v2** (IBM Research) | Experimental hybrid predecessor to Granite 4.0. | Architecture reference |
| **Snakes & Ladders** (NeurIPS 2024 workshop) | SSM speculative decoding via "Joint Attainment" and "Separate Attainment" methods. | Theoretical basis for ReplaySSM |

**Hybrid architecture ratios** (from systematic analysis):

| Model | SSM:Attention ratio | Notes |
|---|---|---|
| Jamba | 7:1 (or 3:1 small) | MoE every other layer |
| Granite 4.0 | 9:1 | IBM's production ratio |
| Griffin | 2:1 (local attn) | RG-LRU + local attention |
| Samba | 1:1 (SWA) | Mamba + sliding window attention |
| Zamba2 | ~7:1 (shared) | All route to 1 shared attention block |

**Recommendation for R41**: Port the Mamba-3 forward pass (complex SSM + MIMO)
into ForgeEngine using the existing `Mamba3Key` for weight conversion. This is
the highest-value SSM upgrade — +1.8 accuracy, half state size, inference-first.
Pair with **ReplaySSM** for speculative decoding compatibility.

---

## Implementation Priority Matrix (R41-R43)

### R41 — Immediate wins (low effort, high return)

| Priority | Technique | Source | Effort | Benefit |
|---|---|---|---|---|
| 1 | MTP activation in training | MTP (Meta ICML'24) | Low | +12% HumanEval, 3x decode speedup |
| 2 | MatryoshkaKV for attention KV | MatryoshkaKV (ICLR'25) | Medium | 60% KV compression, >90% quality |
| 3 | Mamba-3 forward pass (complex SSM + MIMO) | Mamba-3 (arXiv'26) | Medium | +1.8 accuracy, half state size, inference-first |
| 4 | Medusa speculative heads | Medusa (ICML'24) | Medium | 2.2-2.8x decode, no draft model |
| 5 | KTO alignment (binary labels) | KTO (ICML'24) | Low | Easier preference data than DPO |
| 6 | Quamba2 W4A8 for SSM blocks | Quamba2 (arXiv'25) | Medium | 4x SSM weight compression |
| 7 | SIGReg hidden-state regularizer | LeWM (LeCun) | Low | Training stability, prevents collapse |
| 8 | AdaLN-zero conditioning | LeWM (LeCun) | Low | Lossless conditioning injection |
| 9 | ReplaySSM for speculative decode | ReplaySSM (Dao Lab'26) | Medium | 1.48x AR, 1.87-1.96x spec decode for SSM |

### R42 — Architecture expansion (medium-high effort, novel gains)

| Priority | Technique | Source | Effort | Benefit |
|---|---|---|---|---|
| 10 | Expert streaming (fixed-stride + LFU cache) | Swiftlet / LLM-in-a-Flash | Medium | Run 35-80B MoE on 12GB (Qwen3-35B at 6-15 tok/s) |
| 11 | WriteableMemory + BAEE budget cap | SynapNet | Medium | Long context without KV bloat (71% retention at 90% eviction) |
| 12 | CAJQ mixed-precision quant | SynapNet | Medium | 4.4x compression for SSM+attn (13.8 eff bits) |
| 13 | Gated DeltaNet-2 hybrid layer | Gated DeltaNet (ICLR'25) | High | Best recurrent recall (RULER) |
| 14 | PKM residual augmentation | PKM (Lample NeurIPS'19) | High | Capacity without compute (12-layer+PKM > 24-layer dense) |
| 15 | InfoDensity reward shaping | InfoDensity (arXiv'26) | Low | Concise CoT via entropy penalties |
| 16 | Synapse graph memory | Synapse (ACL'26) | Medium | Agent memory, 95% token reduction (814 vs 16910 tok/query) |

### R43 — Advanced / experimental

| Priority | Technique | Source | Effort | Benefit |
|---|---|---|---|---|
| 17 | STRIDE reasoning training | STRIDE (arXiv:2605.18851) | High | Generator+verifier co-training, 6.8% CSR on zero-pass problems |
| 18 | ReDrafter (Mamba draft model) | ReDrafter (Apple) | High | 2.5-3.5x speculative, Mamba-native |
| 19 | TIDE dynamic depth | TIDE (arXiv'26) | Low | 5-20% compute reduction |
| 20 | RWKV-7 layer option | RWKV-7 (arXiv'25) | High | Tiny state alternative |
| 21 | KAN FFN ablation | KAN (ICLR'25) | Medium | Interpretability R&D only |
| 22 | Fused SSD Triton kernel | PyTorch blog (2025) | Medium | 1.5-2.5x Mamba-2 prefill speedup |
| 23 | Granite 4.0 H-Tiny port | IBM Granite 4.0 | Medium | 7B/1B active hybrid MoE, Apache 2.0 |

### Parked / Not recommended
- **BTC-LLM** (sub-1-bit): research stage, no SM120 kernel
- **Soft MoE**: only relevant if model is already MoE
- **Token merging (ToMe/SLERP)**: risky for SSM recurrence
- **BitNet PTQ on Mamba**: catastrophic (PPL ~13M), must train from scratch

### Critical rules
1. **Never apply attention-optimized extreme PTQ (AQLM/QuIP#/BitNet) to
   Mamba/SSM blocks.** Use Quamba2/SSDi8 for SSM, AQLM/NVFP4 for attention.
2. **PKM must augment, not replace, the FFN** (Kim & Jung 2020 — replacement
   causes catastrophic drift; addition with zero-init residual is stable).
3. **Every new architecture key needs a port-first lossless load test**
   before any training experiment (AGENTS.md rule A).
4. **State the VRAM budget** for each feature. 12GB ceiling is hard; mixed
   CPU/GPU fallback is mandatory for anything that pushes past it.

## Key References
- Gated DeltaNet: arXiv:2412.06464, `NVlabs/GatedDeltaNet`
- RWKV-7: arXiv:2503.14456, `RWKV-LM/RWKV-v7`
- Mamba-2: arXiv:2405.21060, `mamba_ssm/modules/mamba2.py`
- Based: arXiv:2402.18668, `HazyResearch/based`
- MatryoshkaKV: arXiv:2410.14731 (ICLR 2025)
- Medusa: arXiv:2401.10774, `FasterDecoding/Medusa`
- PKM: arXiv:1907.05242, Memory Layers at Scale arXiv:2412.19437
- MTP: arXiv:2404.19737 (ICML 2024)
- KTO: arXiv:2402.01306 (ICML 2024)
- AQLM: arXiv:2401.06118, QuIP#: arXiv:2402.04396
- Quamba2: arXiv:2503.22879
- Bi-Mamba: arXiv:2411.11843
- Zoology (recall benchmark): arXiv:2312.04927
- Swiftlet: Apple Silicon MoE expert streaming
- SynapNet: episodic memory + BAEE budget + CAJQ quant
- LeWM (LeCun): JEPA world model, SIGReg, AdaLN-zero
- STRIDE: reasoning training framework
- Synapse: graph-based agent memory
- Mamba-3: arXiv:2603.15569, `state-spaces/mamba/modules/mamba3.py`
- ReplaySSM: `Johnny-Liou/ReplaySSM`, dao-lab.ai/blog/2026/replayssm
- Fused SSD: pytorch.org/blog/accelerating-mamba2-with-kernel-fusion
- SSDi8: arXiv:2608.21952
- Granite 4.0: ibm.com/granite/docs/models/granite4-0, `ibm-granite/granite-4.0-language-models`
- Zamba2-VL: arXiv:2606.00390
- LLM-in-a-Flash: arXiv:2312.11514 (expert streaming origin)
- MoE-Infinity: arXiv:2401.14361, `EfficientMoE/MoE-Infinity`
- LLM-JEPA: arXiv:2509.14252 (JEPA for LLMs)
- AdaLN-zero: arXiv:2212.09748 (DiT origin)
