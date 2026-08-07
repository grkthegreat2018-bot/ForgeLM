# LLM Techniques Survey — Weights, Runtime, KV Cache, Architecture

> Exhaustive reference compiled 2026-07-30 from parallel research across weight technology, runtime/inference, KV cache, and architecture innovations. Each entry: **What** / **Key Idea** / **Numbers** / **Paper** / **Limitations**.

## Table of Contents

- [Part 1 — Weight Technology](#part-1--weight-technology)
  - [1.1 Quantization](#11-quantization)
  - [1.2 Pruning & Sparsity](#12-pruning--sparsity)
  - [1.3 LoRA / PEFT Family](#13-lora--peft-family)
  - [1.4 Weight Transforms / Cross-Arch Porting](#14-weight-transforms--cross-arch-porting)
  - [1.5 Distillation & Compression](#15-distillation--compression)
- [Part 2 — Runtime / Inference Technology](#part-2--runtime--inference-technology)
  - [2.1 Speculative Decoding](#21-speculative-decoding)
  - [2.2 Attention Kernels & Variants](#22-attention-kernels--variants)
  - [2.3 Inference Engines & Runtime](#23-inference-engines--runtime)
  - [2.4 Parallelism & Serving](#24-parallelism--serving)
  - [2.5 Prefill / Decode Optimization](#25-prefill--decode-optimization)
- [Part 3 — KV Cache & Architecture](#part-3--kv-cache--architecture)
  - [3.1 KV Cache Compression](#31-kv-cache-compression)
  - [3.2 KV Cache Quantization](#32-kv-cache-quantization)
  - [3.3 Paged / Disaggregated KV](#33-paged--disaggregated-kv)
  - [3.4 Long Context](#34-long-context)
  - [3.5 Architecture Innovations](#35-architecture-innovations)
- [Part 4 — Summary Tables & Cross-Cutting Insights](#part-4--summary-tables--cross-cutting-insights)
- [Part 5 — ForgeAI Relevance Map](#part-5--forgeai-relevance-map)
- [Part 6 — 2026 Addendum](#part-6--2026-addendum-compiled-2026-08-06)
  - [6.1 Sparse Attention: MoE-ification of the Head Axis](#61-sparse-attention-moe-ification-of-the-head-axis)
  - [6.2 Test-Time Training (TTT) for Reasoning](#62-test-time-training-ttt-for-reasoning)
  - [6.3 Model Merging: Theory & LoRA-Specific Methods](#63-model-merging-theory--lora-specific-methods)
  - [6.4 Continual Learning: Forgetting Mechanics & Replay](#64-continual-learning-forgetting-mechanics--replay)

---

# Part 1 — Weight Technology

## 1.1 Quantization

### Basic Integer Quantization

- **INT8** — 8-bit integer quantization (symmetric/affine). Scale FP32/FP16 to [-127,127] with per-tensor/per-channel granularity. 4× memory reduction, 2-4× speedup with INT8 kernels. *Limit:* activation outliers cause accuracy loss at W8A8.

- **INT4** — 4-bit weight quantization (W4A16 regime). Quantize weights to 4-bit, keep activations FP16. 8× memory reduction. GPTQ/AWQ implementations. *Limit:* requires calibration, accuracy degradation vs INT8.

- **FP8 (E5M2)** — 8-bit floating point, 5 exponent / 2 mantissa bits. Wider dynamic range than INT8, better for outliers. Native H100+ support. 4× memory reduction. *Limit:* hardware-dependent.

- **FP8 (E4M3)** — 8-bit floating point, 4 exponent / 3 mantissa bits. Higher precision than E5M2, narrower range. Better for training (forward pass). Blackwell MXFP8 default. *Limit:* overflow risk for large values.

- **FP4 (E2M1)** — 4-bit floating point, 2 exponent / 1 mantissa bit. Minimal FP format, range [-6, 6]. Used in MXFP4. *Limit:* very limited range, no NaN/inf.

- **NF4 (Normal Float 4)** — Information-theoretically optimal 4-bit format for normally distributed weights. Quantile-based binning matching normal distribution. QLoRA default. Better than FP4 for LLM weights. *Limit:* assumes normal distribution.

### Advanced Post-Training Quantization (PTQ)

- **GPTQ** — Post-training quantization using Hessian-aware approximation. Approximate Hessian inverse to minimize quantization error layer-by-layer. W4A16, ~4 GPU-hours for 175B models. *Paper:* Frantar et al. 2022. *Limit:* slow calibration, weight-only.

- **AWQ (Activation-aware Weight Quantization)** — Protects salient weights via scaling. Identify 1% salient weights, scale them to preserve information. Fast calibration (128 samples). W4A16. *Paper:* Lin et al. 2023. *Limit:* weight-only, not FP8-capable.

- **SmoothQuant** — Migrates quantization difficulty from activations to weights via smoothing. Transform W = W/s, X = X*s with smoothing factor α. W8A8/FP8. Up to 1.56× speedup, 2× memory reduction. *Paper:* Xiao et al. 2022. *Limit:* requires calibration data.

- **LLM.int8()** — Mixed-precision decomposition with outlier handling. Vector-wise INT8 for most features, FP16 for outliers. 50% memory reduction, no performance degradation. *Paper:* Dettmers et al. 2022. *Limit:* FP16 compute overhead, complex implementation.

- **QLoRA** — 4-bit quantization + LoRA for training. NF4 quantization, double quantization of constants, paged optimizers. Fine-tune 65B on single 48GB GPU. *Paper:* Dettmers et al. 2023. *Limit:* training-only, requires bitsandbytes.

- **bitsandbytes** — Library implementing LLM.int8() and QLoRA. CUDA kernels for 4/8-bit operations, block-wise quantization. 75% memory reduction (4-bit), 50% (8-bit). *Limit:* NVIDIA GPU focused.

- **ZeroQuant** — Dynamic per-token activation quantization + group-wise weight quantization. Runtime statistics for activations, grouped weights. W8A8 achievable. *Paper:* Yao et al. 2022. *Limit:* dynamic overhead.

- **ZeroQuant-V2** — Improved ZeroQuant with better calibration. Enhanced scaling strategies, mixed precision. Better accuracy at low bits. *Limit:* still dynamic quantization overhead.

- **OmniQuant** — Omnidirectionally calibrated quantization. Learnable equivalent transformation, weight-activation optimization. W2A16 to W4A4. Better than GPTQ/AWQ/SmoothQuant. *Paper:* Chen et al. 2023 (ICLR spotlight). *Limit:* calibration required.

- **AffineQuant** — Direct optimization using equivalent affine transformations. Extend optimization scope beyond scaling, ensure invertibility. W4A4: C4 PPL 15.76 vs OmniQuant 18.02. *Paper:* 2024. *Limit:* requires training for transformation matrix.

- **SqueezeLLM** — Dense-and-sparse quantization. Split weights into dense (heavily quantized) + sparse (preserve outliers) components. 3-4 bit with 0.05-0.45% sparsity. *Paper:* 2023. *Limit:* sparse format overhead.

- **QServe** — W4A8KV4 system co-design. Progressive quantization, SmoothAttention, compute-aware reordering. 1.2-3.5× throughput vs TensorRT-LLM. *Paper:* MIT Han Lab 2024. *Limit:* GPU-focused, system integration required.

- **QuaRot** — Outlier-free 4-bit via rotations. Randomized Hadamard transformations remove outliers from hidden states. W4A4KV4, 99% zero-shot performance. *Paper:* Ashkboos et al. 2024. *Limit:* requires rotation pre-processing.

- **SpinQuant** — Learned rotations for quantization. Optimize rotation matrix via Cayley SGD to minimize quantization loss. Better than QuaRot (random rotations). 2.9 points gap vs full precision on LLaMA-2 7B. *Paper:* Facebook Research 2024. *Limit:* training overhead for rotation optimization.

- **PrefixQuant** — Eliminate token-wise outliers via prefixed tokens. Prefix outlier tokens in KV cache, blockwise training compensation. +3.08 points over SpinQuant on W4A4KV4. *Paper:* 2024. *Limit:* prefix overhead.

- **MergeQuant** — Per-channel static quantization with QSM (Quantization Step Migration). Eliminates dequant overhead. 1.77× speedup decoding. *Paper:* 2025. *Limit:* static quantization limitations.

- **SASQ** — Static activation scaling for QAT. Learn quantization factors during training, not weights. Address dynamic quantization overhead. *Paper:* 2024. *Limit:* QAT required.

### Microscaling Formats

- **MXFP8** — Microscaling FP8 with block-wise scaling. One E8M0 scale per 32 elements (powers of 2). Finer granularity than standard FP8. Blackwell native. Near-lossless performance. *Limit:* requires Blackwell hardware.

- **MXFP6** — 6-bit microscaling (E3M2/E2M3 variants). 6-bit elements with 8-bit E8M0 scale per 32 elements. Experimental. *Limit:* limited hardware support.

- **MXFP4** — 4-bit microscaling (E2M1). 4-bit elements with 8-bit scale per 32 elements. 136 bits per 32 elements. Significant accuracy degradation vs MXFP8. *Limit:* challenging accuracy, limited hardware.

- **MXINT8** — Integer microscaling. INT8 elements with E8M0 scale (powers of 2). Better than standard INT8 for some cases. *Limit:* hardware support limited.

- **NVFP4** — NVIDIA's 4-bit floating point for KV. E4M1/E5M2 formats with block scaling, optimized for Blackwell; dequantized to FP8 before attention. 50% reduction vs FP8, doubles context/batch, <1% accuracy loss. *Limit:* Blackwell-only.

### GGUF / GGML Formats (llama.cpp)

- **GGUF** — Self-contained binary format for llama.cpp. Single file with model + metadata + quantized weights. CPU/Apple Silicon optimized. mmap loading. *Limit:* CPU-focused, not GPU-optimized.

- **K-quants** — Hierarchical block quantization. Super-blocks (256) with sub-blocks, multiple scales per block. Better accuracy at same bit-width. Q4_K_M recommended default (4.5 bits/weight). *Limit:* requires calibration for I-quants.

- **Q2_K** — 2.5 bits/weight, 256-element super-blocks. Extreme compression. ~30% of F16 size. Noticeable quality loss. *Limit:* quality cliff below 70B models.

- **Q3_K** — 3.4-3.875 bits/weight. Balance of compression/quality. *Limit:* still significant quality loss vs 4-bit.

- **Q4_K** — 4.5 bits/weight, preferred 4-bit format. Good quality, 45% of F16. Recommended default. *Limit:* not as efficient as EXL2 for GPU.

- **Q5_K** — 5.5 bits/weight. Very good quality, 55% of F16. *Limit:* diminishing returns vs Q4_K.

- **Q6_K** — 6.5625 bits/weight. Near-lossless for most models. *Limit:* close to INT8 size.

- **I-quants (Importance-aware)** — Non-uniform quantization grids using importance matrix. Allocate precision based on weight importance (sensitivity). IQ1_S (1.56 bits) to IQ3_M (3.6 bits). Better perplexity than K-quants at same bit-width. *Limit:* requires importance matrix calibration.

### EXL2 Format

- **EXL2** — ExLlamaV2 mixed-precision format. Variable bitrate per group (2-8 bits) to hit target average bpw. Per-column error map for precision allocation. Better quality than uniform quantization. GPU-optimized. *Limit:* NVIDIA GPU only, requires ExLlamaV2.

- **Marlin Format** — 4-bit quantization format optimized for inference. Efficient 4-bit weight representation with fast dequantization kernels. *Limit:* 4-bit quality loss, format-specific.

### Extreme Low-Bit Quantization

- **BitNet (1.58b ternary)** — Native 1.58-bit weights {-1, 0, +1}. Train from scratch with ternary weights using absmean quantization. W1.58A8. 10× memory savings, 2.65× faster CPU inference. *Paper:* Microsoft 2023-2024. *Limit:* requires training from scratch, specialized kernels.

- **BitNet b1.58** — 2.4B parameter native 1-bit model. Trained on 4T tokens with ternary weights. Competitive with full-precision similar size. *Paper:* Microsoft 2024. *Limit:* smaller scale than current SOTA.

- **BitDistill** — Distillation pipeline to ternarize off-the-shelf LLMs. SubLN module, multi-head attention distillation, continual pre-training warm-up. 10× memory savings, 2.65× faster CPU. *Paper:* Microsoft 2024. *Limit:* task-specific, requires distillation.

- **Ternary weights** — {-1, 0, +1} quantization. Extreme compression via 3-level quantization. Used in BitNet. *Limit:* severe accuracy loss if not trained natively.

### Quantization Granularities

- **Per-tensor** — Single scale for entire tensor/layer. Simplest, least accurate. *Limit:* poor for heterogeneous distributions.

- **Per-channel** — Different scale per output channel. Better accuracy for weights. Standard in modern PTQ. *Limit:* increased metadata.

- **Per-token** — Different scale per input token (row). Better for activations with outliers. Used in dynamic quantization. *Limit:* runtime overhead.

- **Block-wise** — Scale per block of elements (e.g., 32). Balance between per-tensor and per-channel. Used in MX formats, K-quants. *Limit:* block size tuning.

- **Group quant** — Group of channels share scale. Coarser than per-channel, finer than per-tensor. Used in GPTQ/AWQ (g128). *Limit:* accuracy trade-off.

- **Dynamic vs Static** — Dynamic computes scales at runtime; static pre-computes. Dynamic more accurate, static faster. Static preferred for deployment. *Limit:* dynamic has overhead.

### KV Cache Quantization (Weight-side)

- **KV quant** — Quantize keys/values in attention cache. Major memory bottleneck for long context. 4-8 bit KV cache. *Limit:* accuracy sensitive, requires smooth attention.

- **SmoothAttention** — Mitigate KV quantization accuracy loss. Smooth key-value distributions before quantization. Used in QServe. *Limit:* additional computation.

---

## 1.2 Pruning & Sparsity

### Unstructured Pruning

- **Magnitude pruning** — Remove smallest-magnitude weights. Simple threshold-based pruning. Scalable but poor accuracy for LLMs. *Limit:* no hardware speedup without sparse kernels, severe accuracy loss.

- **SparseGPT** — One-shot unstructured pruning for massive GPT models. Hessian-based update to maintain accuracy. 50-60% sparsity with minimal perplexity increase. OPT-175B/BLOOM-176B in <4.5 hours. *Paper:* Frantar et al. 2023 (ICML). *Limit:* unstructured (no hardware speedup), calibration cost.

- **Wanda** — Prune by product of weight magnitude and input activation norm. Per-output importance metric. Better than magnitude/SparseGPT at same sparsity. 50% unstructured: PPL 6.42 vs 6.51 (SparseGPT) on LLaMA2-7B. *Paper:* 2023. *Limit:* unstructured.

### Structured Pruning

- **Structured pruning** — Remove entire structures (rows, columns, heads, layers). Hardware-friendly speedup. Better acceleration but harder accuracy. *Limit:* more aggressive accuracy loss.

- **SliceGPT** — SVD-based structured pruning via computational invariance. Orthogonal transformations then slice rows/columns. Reduce embedding dimension. Up to 25% parameters removed, 99% zero-shot performance. 64-66% compute on 24GB/40GB GPUs. *Paper:* Microsoft 2024 (ICLR). *Limit:* requires transformation, embedding dimension change.

- **SVD-LLM** — Truncation-aware SVD for compression. Data whitening, sequential low-rank approximation with updates. Better than SliceGPT at high compression. *Paper:* 2024. *Limit:* requires calibration.

- **ShortGPT** — Layer drop based on Block Influence metric. Measure layer importance by input-output similarity. Prune redundant layers. Better than complex methods. *Paper:* ACL 2025 Findings. *Limit:* layer-wise only.

- **LoRAPrune** — LoRA-guided structured pruning. Use LoRA weights/gradients for importance, not base model gradients. 50% compression: 4.81 PPL reduction vs LLM-Pruner on WikiText2. Memory-efficient. *Paper:* ACL 2024 Findings. *Limit:* requires LoRA fine-tuning first.

- **Compresso** — Structured pruning with collaborative prompting + L0 regularization. Learnable binary masks for heads/FFN/hidden dims, LoRA for updates. Instruction-tuning data as alternative to training data. *Paper:* 2023. *Limit:* requires training with prompts.

- **Linear layer pruning** — Remove entire input/output channels. Reduce weight matrix dimensions. Significant speedup. *Limit:* severe accuracy loss if not careful.

- **Head pruning** — Remove attention heads. Some heads are redundant. Based on importance metrics. *Limit:* may lose specialized functionality.

### Semi-Structured Sparsity (N:M)

- **N:M sparsity** — Exactly N non-zeros per M consecutive weights. Hardware-accelerated pattern (e.g., 2:4). NVIDIA Ampere+ support. 2:4 standard. *Limit:* constrained pattern, accuracy loss.

- **2:4 sparsity** — 2 non-zeros per 4 weights. 50% sparsity with hardware acceleration. Supported on NVIDIA RTX 30xx/40xx, A100+. *Limit:* 50% max with current hardware.

- **4:8 sparsity** — 4 non-zeros per 8 weights. 50% sparsity alternative pattern. Sometimes better accuracy than 2:4. *Limit:* less common hardware support.

- **MaskLLM** — Learnable N:M sparsity via Gumbel Softmax. Model mask distribution as learnable, end-to-end training. 2:4: PPL 6.72 vs 10+ (SOTA) on Wikitext. *Paper:* 2024. *Limit:* training overhead.

- **Cannistraci-Hebb Training (CHT)** — Brain-inspired sparse training with N:M. Integrate CHT with 2:4 semi-structured sparsity. Better than SR-STE. *Paper:* 2024. *Limit:* training complexity.

### Movement & Gradient-Based Pruning

- **Movement pruning** — Retain weights moving away from zero during fine-tuning. First-order method adaptive to transfer learning. 95% BERT performance with 5% weights (with distillation). *Paper:* Sanh et al. 2020. *Limit:* requires fine-tuning data.

- **Soft pruning** — Continuous relaxation of hard pruning. Learnable mask values via sigmoid/L0 regularization. Better gradients. *Limit:* not true sparsity at inference.

- **L0 regularization** — Learn sparse masks via concrete relaxation. Continuous relaxation of discrete sparsity. Used in Compresso. *Paper:* Louizos et al. 2018. *Limit:* training instability, hyperparameter sensitivity.

### Lottery Ticket Hypothesis for LLMs

- **Lottery Ticket Hypothesis** — Dense networks contain winning subnetworks. Prune + rewind to early initialization. Frankle & Carbin 2019. For BERT: "all tickets are winning" - many subnetworks work. *Paper:* 2020. *Limit:* expensive to find, transferability questions.

- **Lottery Ticket Adaptation (LoTA)** — Sparse adaptation for multi-task LLMs. Identify sparse subnetwork per task, avoid destructive interference. Better than LoRA/full fine-tuning for multi-task. *Paper:* 2024. *Limit:* requires task-specific identification.

- **Lottery Rank-Pruning Adaptation (LoRPA)** — Prune LoRA ranks based on magnitude. Train large rank, prune to smaller. Better than various LoRA ranks. *Paper:* 2024. *Limit:* LoRA-specific.

### Safety Pruning

- **Pruning Unsafe Tickets** — Remove unsafe subnetworks. Identify "unsafe tickets" via gradient-free attribution. Reduce unsafe generations. *Paper:* 2024. *Limit:* safety-specific, may reduce capability.

---

## 1.3 LoRA / PEFT Family

### Core LoRA Variants

- **LoRA (Low-Rank Adaptation)** — Add low-rank matrices BA to frozen weights. W = W₀ + BA, r ≪ min(m,n). 0.1-1% trainable parameters. No inference overhead (merge at deployment). *Paper:* Hu et al. 2021. *Limit:* suboptimal for large width, sensitive to hyperparameters.

- **QLoRA** — 4-bit quantized base + LoRA. NF4 quantization, double quantization, paged optimizers. Fine-tune 65B on single 48GB GPU. 99.3% ChatGPT performance (Vicuna). *Paper:* Dettmers et al. 2023. *Limit:* bitsandbytes dependency, training-only.

- **DoRA (Weight-Decomposed LoRA)** — Decompose weight into magnitude + direction. LoRA for direction, learn magnitude separately. Better learning capacity, closer to full fine-tuning. *Paper:* Liu et al. 2024. *Limit:* additional compute, slight VRAM overhead.

- **rsLoRA (Rank-Stabilized LoRA)** — Stabilize high-rank LoRA training. Scale alpha with rank, adjust learning rate. Better stability at high ranks. Strictly better than standard LoRA. *Limit:* hyperparameter tuning.

- **LoRA+** — Different learning rates for A and B matrices. B should have higher LR than A (scaling argument). 2× speedup, 1-2% performance improvement. *Paper:* Hayou et al. 2024. *Limit:* requires LR tuning.

- **PiSSA (Principal Singular values and Singular vectors Adaptation)** — Initialize LoRA with SVD components. A,B from principal components, residual frozen. Faster convergence, better performance. Gemma-7B: 77.7% vs 74.53% (LoRA) on GSM8K. *Paper:* 2024 (NeurIPS). *Limit:* requires SVD computation.

- **AdaLoRA (Adaptive LoRA)** — Dynamic rank allocation per layer. Allocate budget to important layers. Rank importance scoring. *Limit:* complex training, not mergeable.

- **LongLoRA** — Efficient long-context fine-tuning. Sparse local attention during training (S²-Attn), trainable embeddings/norms. Extend LLaMA2 7B to 100k context. *Paper:* Chen et al. 2023 (ICLR Oral). *Limit:* context-specific.

- **VeRA (Vector-based Random Matrix Adaptation)** — Shared random matrices across layers, learn scaling vectors. Freeze LoRA matrices, learn per-layer scalars. Even fewer parameters. *Paper:* Kopiczko et al. 2023. *Limit:* depends on hidden dimension.

- **LoRA-XS** — Extremely small parameter LoRA. Further reduce parameters via shared projections. *Paper:* Bałazy et al. 2024. *Limit:* may sacrifice capacity.

- **LoRA-FA (Memory-Efficient)** — Optimized for memory. Factorized architecture for gradient efficiency. *Paper:* 2023. *Limit:* specific optimizations.

- **OLoRA (Orthogonal LoRA)** — Orthogonal constraints on adapters. Preserve geometry of weight space. *Limit:* training complexity.

- **GaLore (Gradient Low-Rank Projection)** — Low-rank gradient projection for full-parameter learning. Project gradients to low-rank subspace, update full weights. 65.5% optimizer memory reduction. Pre-train 7B on RTX 4090 (24GB). *Paper:* Zhao et al. 2024. *Limit:* gradient projection overhead.

- **ReLoRA** — High-rank training through low-rank updates. Iterative LoRA reset/retrain for effective high-rank. *Paper:* NeurIPS Workshop 2023. *Limit:* multi-stage training.

- **Norm-LoRA** — Norm-bounded LoRA with explicit singular value control. Learnable scaling vector, bound singular values. Better stability. *Paper:* 2025. *Limit:* additional regularization.

- **LoftQ (LoRA-Fine-Tuning-Aware Quantization)** — Joint quantization + LoRA initialization. Find quantized backbone Q and LoRA A,B simultaneously. Better than quantize-then-LoRA. Especially effective at 2-bit. *Paper:* 2023. *Limit:* requires pre-computed initialization.

- **LoRA-Pro** — Improved LoRA training dynamics. Enhanced optimization strategies. *Limit:* less widely adopted.

### MoE + LoRA Combinations

- **MoLE (Mixture of LoRA Experts)** — Treat LoRA layers as experts with routing. Hierarchical gating for LoRA composition. 9% accuracy improvement in multi-task. *Paper:* Wu et al. 2024. *Limit:* routing overhead.

- **MixLoRA** — LoRA-based MoE for multi-task. Multiple LoRA experts in FFN, independent attention LoRAs. 40% memory reduction, 30% latency reduction. *Paper:* 2024. *Limit:* MoE complexity.

- **LoRAMoE** — LoRA + MoE architecture. Sparse activation of LoRA adapters. *Limit:* system complexity.

- **LD-MoLE** — Learnable dynamic routing for MoLE. Differentiable routing, closed-form solution, adaptive expert count. *Paper:* 2024. *Limit:* training complexity.

### Other PEFT Methods

- **IA³ (Infused Adapter by Inhibiting/Amplifying Inner Activations)** — Learn rescaling vectors for K,V,FFN. Multiply activations by learned vectors. Very few parameters. Better than adapters. *Paper:* Liu et al. 2022. *Limit:* less expressive than LoRA.

- **Prefix Tuning** — Learn continuous prefix embeddings at input. Prepend learnable tokens to input. 0.1% parameters. *Paper:* Li & Liang 2021. *Limit:* context length consumption, slow convergence.

- **Prompt Tuning** — Learn soft prompts at input layer only. Only input embeddings learnable. Simpler than prefix tuning. *Paper:* Lester et al. 2021. *Limit:* less expressive, needs T5-style pre-training.

- **P-tuning / P-tuning v2** — Prompt tuning with LSTM/MLP encoders. Learn prompt encoders for better representation. *Paper:* 2021-2022. *Limit:* additional parameters.

- **AdapterFusion** — Non-destructive task composition for adapters. Two-stage: learn adapters, then fuse with attention. Combine multiple tasks without catastrophic forgetting. *Paper:* Pfeiffer et al. 2021 (EACL). *Limit:* inference overhead with many adapters.

- **BitFit** — Train only bias terms. Freeze all weights, train biases + classifier. <0.1% parameters. Competitive with full fine-tuning on small data. *Paper:* Ben Zaken et al. 2021. *Limit:* limited capacity, not for large data.

- **MAM Adapter** — Multi-head attention adapter with parallel design. Efficient adapter architecture. Integrated in adapter-transformers. *Paper:* He et al. 2022. *Limit:* adapter-specific.

---

## 1.4 Weight Transforms / Cross-Arch Porting

### SVD-Based Methods

- **SliceGPT** — SVD + orthogonal transforms for structured pruning. Project signals to principal components, slice off dimensions. Computational invariance via RMSNorm. 25% parameter reduction, 99% performance. *Paper:* Microsoft 2024. *Limit:* changes embedding dimension.

- **SVD-LLM** — Truncation-aware SVD for compression. Data whitening, sequential low-rank approximation with updates. Better than SliceGPT at high compression. *Paper:* 2024. *Limit:* requires calibration.

- **SlimLlama** — Low-rank feature distillation via SVD. SVD initialization, joint teacher-student activation loss. Compress Mixtral-8x7B by 10B params in minutes. 95% performance retained. *Paper:* 2024. *Limit:* calibration data needed.

### Knowledge Distillation Across Architectures

- **Cross-architecture distillation** — Distill from one architecture to another. Match intermediate representations despite structural differences. Common in model compression. *Limit:* architectural mismatch challenges.

- **Layer-wise distillation** — Match hidden states per layer. Align student layers to teacher layers. Used in TinyBERT. *Limit:* layer mapping complexity.

### Weight Interpolation (Model Soups)

- **Model Soups** — Average weights of multiple fine-tuned models. Models lie in same low-loss basin, averaging improves accuracy/robustness. ViT-G 90.94% top-1 on ImageNet (SOTA 2022). *Paper:* Wortsman et al. 2022. *Limit:* requires same base model, hyperparameter sweep.

### Model Merging

- **Task Arithmetic** — Combine task vectors (θ_finetuned - θ_base). Add/subtract task vectors algebraically. Add for combining, negate for forgetting. *Paper:* Ilharco et al. 2022. *Limit:* interference when tasks conflict.

- **TIES (Trim, Elect Sign, Disjoint Merge)** — Sparsify task vectors + sign consensus. Trim small changes, resolve sign conflicts, merge disjoint parameters. Better than naive task arithmetic. *Paper:* Yadav et al. 2023. *Limit:* requires base model, density parameter.

- **DARE (Deterministic And Rank-reduce Ensemble)** — Random pruning + rescaling of task vectors. Drop most weight updates (99% don't matter), rescale remaining. Better than TIES for instruction-following. *Paper:* Yu et al. 2023. *Limit:* random pruning variance.

- **SLERP (Spherical Linear Interpolation)** — Interpolate on hypersphere surface. Geometric interpolation between weight vectors. Better than linear for 2 models. *Limit:* only 2 models, no interference handling.

- **Linear merging** — Simple weighted average of weights. θ_merged = Σ w_i θ_i. Simplest method. *Limit:* poor for conflicting updates.

- **Model stitching** — Combine different parts of different models. Stitch layers from different checkpoints. Experimental. *Limit:* architectural compatibility.

- **DELLA** — Distribution-aware merging. Consider parameter distributions when merging. *Limit:* complex.

- **Model Breadcrumbs** — Track parameter changes during training. Use breadcrumbs for merging. *Limit:* requires training-time tracking.

- **RAM (Rank-Aware Merging)** — Rank-aware model merging. *Limit:* complexity.

- **SCE (Source Consensus Ensemble)** — Consensus-based merging. *Limit:* limited adoption.

### Weight Tying

- **Weight tying** — Share parameters across different parts of model. Reduce parameters by reusing weights (e.g., embedding/output layers). Standard in some architectures. *Limit:* may limit expressivity.

### Matryoshka Embeddings

- **Matryoshka Representation Learning (MRL)** — Learn nested representations at multiple scales. Optimize loss at multiple dimensions (768, 512, 256, 128, 64). Frontload important information. Truncate at inference for speed. *Paper:* Kusupati et al. 2022 (NeurIPS). *Limit:* training overhead, specific to embeddings.

### BitNet Conversion

- **Ternarization via absmean** — Convert to {-1, 0, +1} using absolute mean. w_quant = sign(w) if |w| > mean(|w|) else 0. Used in BitNet. *Limit:* accuracy loss if not trained natively.

### MLA Conversion from GQA

- **TransMLA** — Convert GQA models to MLA (Multi-head Latent Attention). Decouple RoPE, absorb up-projections, enable DeepSeek compatibility. 93% KV cache compression, 10.6× speedup. *Paper:* 2025 (NeurIPS). *Limit:* requires fine-tuning (6B tokens).

- **TransGQLA** — Convert GQA to GQLA (Group-Query Latent Attention). Expose two decoding paths (MQA-absorb + GQA) for hardware adaptability. Single weights for H100 + H20. *Paper:* 2025. *Limit:* conversion complexity.

- **CARE** — Covariance-aware MLA conversion. Activation-aware SVD, adjusted-rank allocation. 215× PPL reduction over baselines. *Paper:* 2025. *Limit:* requires activation statistics.

- **GQA uptraining** — Convert MHA to GQA via continued pre-training. Mean-pool K/V heads, 5% compute for adaptation. *Paper:* Ainslie et al. 2023. *Limit:* requires additional training.

- **X-EcoMLA** — Upcycle pre-trained attention to MLA. Convert existing GQA/MQA models to MLA without retraining from scratch. *Paper:* 2025. *Limit:* conversion may lose some performance.

### MoE Splitting from Dense

- **Upcycling** — Convert dense model to MoE. Duplicate FFN weights as experts, add router, continued training. Better than continued dense training. *Paper:* Komatsuzaki et al. 2022. *Limit:* requires training.

- **Innovator** — Fine-grained MoE upcycling for scientific LLMs. 4-stage upcycle (expert induction, splitting, routing warmup, integration). 25% improvement on 30 scientific tasks, 99% general performance. *Paper:* 2024. *Limit:* domain-specific.

- **LLaMA-MoE** — Build MoE from LLaMA via continual pre-training. Expert construction from FFN parameters, 200B token training. Outperforms dense at similar activation. *Paper:* EMNLP 2024. *Limit:* training cost.

- **SPRI** — SVD-partitioned residual initialization for MoE upcycling. Distribute SVD residuals across experts for controlled diversity. +3.39 BLEU over baseline. *Paper:* 2025. *Limit:* data-constrained scenarios.

### Continuous Pretraining / Uptraining

- **Continuous pretraining** — Extend pretraining on new data. Add knowledge without catastrophic forgetting. Standard practice. *Limit:* forgetting risk, compute cost.

- **Uptraining** — Shorter continued pretraining for adaptation. Smaller compute budget than full continuous pretraining. Used in GQA conversion. *Limit:* may not fully adapt.

---

## 1.5 Distillation & Compression

### Knowledge Distillation Types

- **Logits KD** — Match teacher output distribution. KL divergence between student and teacher logits. Standard KD. *Paper:* Hinton et al. 2015. *Limit:* requires teacher forward pass, may overfit to teacher errors.

- **Hidden state KD** — Match intermediate layer representations. Align student hidden states with teacher. Better feature transfer. *Limit:* layer mapping needed.

- **Sequence-level KD** — Match sequence-level distributions. Minimize sequence-level KL, not just token-level. Eliminates need for beam search. 10× faster, +4.2 BLEU. *Paper:* Kim & Rush 2016. *Limit:* more complex loss.

- **MiniLLM** — Reverse KLD for generative LLM distillation. Reverse KL to prevent overestimating low-probability regions, on-policy optimization. Better precision, calibration, long-text. *Paper:* Gu et al. 2023. *Limit:* on-policy sampling overhead.

- **GKD** — General framework for large-scale PLM distillation. Support 100B+ scale, 25 methods, dynamic hook mechanism. 8 A100 40GB for 100B models. *Paper:* 2023. *Limit:* system complexity.

- **DistilBERT-style** — Layer reduction + distillation. Remove every other layer, distill from remaining. 40% smaller, 97% performance. *Paper:* Sanh et al. 2019. *Limit:* BERT-specific, may not generalize to LLMs.

- **Online distillation** — Continuously update student during serving. Use spare compute for real-time distillation. Speculative distillation. *Limit:* system complexity.

- **Self-distillation** — Model distills to itself (smaller version). Iterative compression. *Limit:* quality degradation.

- **Speculative distillation** — Distillation for speculative decoding alignment. Align draft model with target via KD. 10-45% speedup over standard SD. *Paper:* 2023. *Limit:* draft model requirement.

- **SpecKD** — Speculative knowledge distillation with token gating. Apply KD loss only to "accepted" tokens (teacher confident). Filter noise. SOTA results. *Paper:* 2024. *Limit:* verification overhead.

- **OSD (Online Speculative Decoding)** — Online distillation of draft models. Use spare FLOPs to update draft on query distribution. 1.22-3.06× latency reduction. *Paper:* 2023. *Limit:* serving system integration.

### Layer Reduction

- **Layer dropping** — Remove entire layers. Some layers are redundant. Used in ShortGPT, pruning. *Limit:* may lose depth-dependent capabilities.

- **Structured layer reduction** — SVD-based compression across layers. SlimLlama approach. *Limit:* requires calibration.

### Speculative Distillation

- **DistillSpec** — Improve speculative decoding via distillation. On-policy data generation, tailored divergence function. 10-45% speedup. *Paper:* 2023. *Limit:* draft model training.

- **SKD (Speculative Knowledge Distillation)** — Interleaved sampling for KD. Student proposes, teacher replaces poor tokens. High-quality on-the-fly data. *Paper:* 2024. *Limit:* teacher availability.

---

# Part 2 — Runtime / Inference Technology

## 2.1 Speculative Decoding

### Classic Speculative Decoding

- **Classic Speculative Decoding** — Small draft model generates multiple tokens, large target verifies in parallel via rejection sampling. Draft proposes K tokens, target verifies all in one forward pass. 2-4× typical, bounded by token acceptance rate (60-80%). *Paper:* Leviathan et al. 2023, Chen et al. 2023. *Limit:* requires separate draft model, speedup limited by acceptance rate.

- **Medusa** — Multiple decoding heads added to target model predict future tokens in parallel. Add K extra LM heads on top of frozen backbone, each predicts token at different future position, tree-based verification accepts longest correct prefix. 2.2-2.8× (Medusa-1: 2.2× lossless; Medusa-2: 2.3-2.8× with joint training). *Paper:* Cai et al. 2024. *Limit:* requires training/fine-tuning, tree attention complexity.

- **Medusa+Tree** — Tree-structured candidate generation in Medusa for higher effective acceptance. Each Medusa head generates Top-K candidates forming a tree, sparse attention mask enables parallel verification. 10-30% improvement over linear Medusa. *Paper:* TensorRT-LLM 2024. *Limit:* increased memory for tree state.

- **EAGLE** — Lightweight head predicts target model's next hidden state instead of tokens. Draft model operates at feature level (second-to-top-layer), uses target's hidden states plus advanced token sequence to resolve feature uncertainty. 2.5-3× typical, ~10-15% extra KV cache overhead. *Paper:* Li et al. 2024. *Limit:* requires training draft head, feature-level prediction adds complexity.

- **EAGLE-2** — Dynamic draft tree structure based on draft model confidence scores. Context-aware dynamic tree sizing using draft model's confidence as proxy for acceptance rate. 3.05-4.26×, 20-40% faster than EAGLE-1. *Paper:* Li et al. EMNLP 2024. *Limit:* dynamic tree adds scheduling complexity.

- **EAGLE-3** — Abandons feature prediction for direct token prediction with multi-layer feature fusion. Removes feature prediction constraint, uses training-time testing to simulate feature process, fuses low/mid/high-level features. Up to 6.5×, 1.4× improvement over EAGLE-2. *Paper:* SafeAILab 2024. *Limit:* more complex training pipeline.

- **SpecInfer** — Tree-based speculation with parallel verification across multiple candidates. Draft a tree of candidate tokens instead of linear chain, verify branches in parallel using tree attention. 2-3× on long sequences. *Paper:* Miao et al. 2023. *Limit:* tree management overhead, requires custom kernels.

- **SpecBee** — Beehive-style parallel draft generation with multiple draft models. Multiple draft models generate candidates in parallel, combined verification maximizes acceptance rate. 2.5-3.5× by diversifying draft sources. *Paper:* 2024. *Limit:* multiple draft models increase VRAM.

- **Lookahead Decoding** — Target model itself drafts via n-gram extraction and Jacobi iteration, no separate draft model. Break sequential dependency by concurrently extracting and verifying n-grams directly from LLM using Jacobi iteration. 1.8× on MT-bench, 4× with multi-GPU scaling on code completion. *Paper:* Feng et al. 2024. *Limit:* higher per-step compute, requires specific n-gram patterns.

- **Jacobi Decoding** — Iterative refinement of draft tokens using Jacobi method within lookahead framework. Treat speculation as fixed-point iteration, refine draft candidates iteratively before verification. Component of Lookahead, contributes 10-20% of total speedup. *Limit:* iteration overhead, convergence not guaranteed.

- **Self-Speculative Decoding (LayerSkip)** — Early layers draft, later layers verify within same model. Train with layer dropout (low early, high later) and shared early exit loss, inference exits early for draft, remaining layers verify/correct. 1.34-2.16× (2.16× on summarization, 1.82× on coding). *Paper:* Facebook Research ACL 2024. *Limit:* requires specific training recipe.

- **Draft-Then-Verify** — General paradigm separating draft generation from verification. Two-stage process: draft stage generates candidates cheaply, verify stage checks correctness. Framework speedup 2-4× depending on draft quality. *Limit:* overhead of two stages.

- **BiLD (Big-Little Decoder)** — Big model and little model collaborate with specialized roles. Little model handles easy tokens, big model handles hard tokens, dynamic switching based on confidence. 2-3× by avoiding big model for predictable tokens. *Paper:* 2024. *Limit:* switching overhead, confidence threshold tuning.

- **REST (Retrieval-Based Speculative Decoding)** — Retrieval from datastore instead of parametric draft model. Build trie from retrieved document continuations matching current context, prune low-frequency branches, verify with tree attention. 1.62-2.36× on code/text generation, plug-and-play no training. *Paper:* He et al. NAACL 2024. *Limit:* requires datastore, retrieval latency.

- **Speculative Streaming** — Multi-stream attention integrates drafting within target model. Multiple parallel attention streams with interdependencies, non-autoregressive draft generation with minimal overhead, 1000× fewer parameters than Medusa. 2-3.5× across diverse tasks. *Paper:* Apple ML 2024. *Limit:* requires model modification.

- **Online Speculative Distillation (OSD)** — Continuously update draft model during serving using distillation. Exploit spare FLOPs in serving cluster to fine-tune draft model on target corrections. Acceptance rate increase 0.1-0.65, 1.42-2.17× latency reduction. *Paper:* Liu et al. ICML 2024. *Limit:* requires ongoing training compute.

- **Token-Level vs Sequence-Level Speculation** — Token-level speculates individual tokens (classic), sequence-level speculates entire phrases/segments. Token-level 2-4×, sequence-level potentially higher but harder to verify. *Limit:* sequence-level harder to verify, higher rejection penalty.

- **SpecDec** — Speculative decoding with cascade of draft models. Multiple draft models in cascade, each refines previous output, hierarchical verification. 3-5× with well-chosen cascade. *Paper:* Spector & Re 2023. *Limit:* multiple models increase complexity.

- **TriForce** — Hierarchical speculation for long sequence generation. Retrieval-based KV cache selection + hierarchical speculation with different drafting methods per phase, targets long-context bottlenecks. Significant on long sequences. *Paper:* Fu et al. 2024. *Limit:* complex multi-phase system.

- **Hydra Decoding** — Sequentially-dependent draft heads for Medusa. Draft heads are functions of both base hidden state AND previous draft tokens, captures sequential dependencies missing in independent heads. 10-20% improvement over standard Medusa heads. *Paper:* 2024. *Limit:* sequential dependency reduces parallelism.

- **Draft Model Training** — Training strategies for effective draft models. Knowledge distillation from target, matching vocabulary, architectural alignment, domain-specific fine-tuning. Well-trained draft can achieve 70-85% acceptance rate. *Limit:* training cost, draft-target gap.

- **Target Verification** — Verification mechanisms in speculative decoding. Rejection sampling, tree verification, parallel acceptance checking, ensures output distribution matches target exactly. Verification overhead typically 5-15% of total time. *Limit:* verification cost limits speculation budget.

- **Rejection Sampling** — Core verification mechanism preserving exact distribution. For each draft token, sample from target distribution conditioned on draft, accept with probability ratio, resample if rejected. Enables lossless acceleration. *Paper:* Stern et al. 2018, Leviathan 2023. *Limit:* sequential acceptance checking, resampling cost.

- **Lossless Guarantee** — Mathematical property that speculative decoding preserves exact output distribution. Rejection sampling ensures accepted tokens follow target distribution exactly, no quality degradation. *Paper:* proven in Leviathan et al. 2023. *Limit:* only applies with correct rejection sampling implementation.

---

## 2.2 Attention Kernels & Variants

### FlashAttention Family

- **FlashAttention 1** — IO-aware exact attention with tiling and recomputation. Split input into blocks (tiling), incrementally compute softmax without full matrix, recompute attention in backward pass instead of storing. 2-4× speedup, 10-20× memory reduction vs standard attention. *Paper:* Dao et al. NeurIPS 2022. *Limit:* increased FLOPs from recomputation, requires CUDA implementation.

- **FlashAttention 2** — Improved parallelism and work partitioning for Hopper/Ampere. Better threadblock/warp-level parallelism, optimized for H100/A100, reduced shared memory usage. 2× faster than FA1, up to 7.6× total vs baseline. *Paper:* Dao et al. 2023. *Limit:* hardware-specific optimizations, Ampere+ only.

- **FlashAttention 3** — Exploits Hopper (H100) features: asynchrony, TMA, FP8. Warp-specialization overlapping compute/data movement, block-wise matmul/softmax interleaving, FP8 tensor cores with block quantization. 1.5-2× over FA2 (BF16), 1.3 PFLOPs/s (FP8), 85% H100 utilization. *Paper:* Dao et al. NeurIPS 2024. *Limit:* Hopper (SM90) only, FP8 numerical precision concerns.

- **FlashAttention on Hopper/Blackwell** — Hardware-specific optimizations for latest NVIDIA GPUs. TMA (Tensor Memory Accelerator) for async data movement, hardware-accelerated FP8, warp-specialized kernels. Near-peak TFLOPs utilization on H100/Blackwell. *Limit:* latest hardware only.

### Other Attention Kernels

- **xFormers** — Memory-efficient attention with various implementations. Includes MEA (Memory Efficient Attention), sparse attention, backward-compatible PyTorch interface. 1.5-2× vs standard attention. *Paper:* Facebook Research 2022. *Limit:* less optimized than FlashAttention for some cases.

- **SDPA (Scaled Dot Product Attention)** — PyTorch native attention with automatic kernel selection. `F.scaled_dot_product_attention` automatically selects fastest kernel (FlashAttention, memory-efficient, math) based on input/hardware. Automatic optimization, typically matches FlashAttention when available. *Paper:* PyTorch 2.0+. *Limit:* limited to supported kernels.

- **Triton Kernels** — Python-based GPU kernel writing for attention. Write GPU kernels in Triton language, easier than CUDA, auto-tuning, flexible attention patterns. Competitive with hand-written CUDA for many patterns. *Paper:* OpenAI Triton 2021+. *Limit:* slightly lower peak performance than hand-tuned CUDA.

- **FlexAttention** — PyTorch API for custom attention patterns with FlashAttention performance. Define attention mask/modification as Python function, `torch.compile` lowers to fused FlashAttention kernel, supports sliding window, alibi, document masking. Competitive with hand-written kernels for custom patterns. *Paper:* PyTorch 2024. *Limit:* requires `torch.compile`.

- **FlashInfer Backend** — High-performance attention kernel library. Optimized attention kernels for various patterns, competitive with FlashAttention. *Limit:* ecosystem maturity vs FlashAttention.

- **Liger-Kernel** — Optimized kernels for training efficiency. Fused kernels for common operations, reduce overhead. *Limit:* ecosystem maturity, coverage.

### Paged / Distributed Attention

- **PagedAttention (vLLM)** — KV cache management inspired by OS virtual memory pages. Store KV cache in non-contiguous fixed-size pages, eliminates fragmentation, enables efficient memory allocation and sharing. 2-4× throughput improvement via better batching, near-zero fragmentation waste. *Paper:* Kwon et al. SOSP 2023. *Limit:* page management overhead, requires vLLM or compatible system.

- **RingAttention** — Distributed attention for sequences longer than single GPU memory. Sequence split across GPUs in ring, each GPU computes local attention and passes blocks to neighbors, enables 1M+ token context. Linear scaling for sequence length. *Paper:* Liu et al. 2023. *Limit:* communication overhead, requires multi-GPU.

- **Context Parallelism** — Split sequence dimension across devices for long contexts. Each device handles portion of sequence, attention computation coordinated via ring or other communication patterns. Enables 8K+ token sequences on single model instance. *Paper:* DeepSpeed-Ulysses 2023, RingAttention 2023. *Limit:* communication cost.

- **Sequence Parallelism** — Split sequence activations within tensor-parallel group. Shard sequence dimension in LayerNorm/Dropout, reduces activation memory when combined with tensor parallelism. 1/N activation memory reduction. *Paper:* Megatron-LM 2023. *Limit:* requires tensor parallelism.

- **DeepSpeed Ulysses** — Sequence parallelism via attention head partitioning. Partition sequence dimension, all-to-all communication to get full sequence per head subset, compute attention in parallel, gather results. Enables 15M+ token sequences, constant communication volume with proportional scaling. *Paper:* DeepSpeed 2023. *Limit:* limited by number of attention heads.

- **USP (Unified Sequence Parallelism)** — Combine DeepSpeed-Ulysses and Ring-Attention approaches. Overcome limitations of both, robust to model architecture and network topology, 47% MFU on 208K sequences. *Paper:* 2024. *Limit:* system complexity.

### Attention Variants (Quality)

- **Grouped-Query Attention (GQA)** — Intermediate between MHA and MQA: groups of query heads share KV heads. H query heads, G KV groups (1 < G < H), each group shares K/V projections. 4× KV cache reduction vs MHA (for G=H/4), within 0.5 points quality of MHA. *Paper:* Ainslie et al. 2023. *Limit:* quality degradation vs full MHA, requires training or uptraining.

- **Multi-Query Attention (MQA)** — Extreme KV sharing: single K/V head for all query heads. All query heads share one K/V pair, maximum cache compression. 32× KV cache reduction for 32-head model. *Paper:* Shazeer 2019. *Limit:* significant quality degradation, training instability.

- **Multi-Head Latent Attention (MLA)** — Compress Q/K/V into low-rank latent representations. Project queries/keys/values to compressed latent space, store only latent in KV cache, up-project for attention, DeepSeek-V2 innovation. KV cache from O(H·d) to O(d_c), extreme compression with multi-head diversity. *Paper:* DeepSeek-AI 2024. *Limit:* compression artifacts, requires model architecture support.

- **GroupedMLA** — Grouped variant of MLA. Group latent representations instead of full sharing, balances MLA and GQA. Reduced memory duplication, better for distributed. *Paper:* Grouped Latent Attention 2025. *Limit:* more complex than pure MLA.

- **Sliding Window Attention** — Each token attends only to recent W tokens. Fixed window of recent tokens, O(N·W) complexity instead of O(N²), local context focus. Linear in sequence length for fixed W. Used in Mistral, Longformer. *Limit:* loses long-range dependencies, effective receptive field limited by window × layers.

- **Local+Global (Longformer, BigBird)** — Combine sliding window with global tokens attending to everything. Most tokens use local window, special global tokens attend to all positions and are attended by all. O(N·(g+w)) where g=global tokens, w=window. *Paper:* Longformer 2020, BigBird 2020. *Limit:* requires global token selection, sparse kernels needed.

- **Dilated Attention** — Attend to tokens at regular intervals (strided pattern). Skip connections with stride, create diagonal patterns, different dilation rates per head capture multi-scale dependencies. O(N·N/s) where s=stride. *Paper:* LongNet 2023, Sparse Transformer 2019. *Limit:* may miss intermediate patterns.

- **Sparse Attention** — Compute only subset of attention matrix entries. Structured sparsity patterns (block-sparse, strided, random), reduce O(N²) to O(N·k) for k attended positions per row. Sub-quadratic complexity. *Paper:* Sparse Transformer 2019, BigBird 2020, Longformer 2020. *Limit:* quality degradation vs full attention, requires custom kernels.

- **Block-Sparse Attention** — Block-level sparsity patterns for GPU efficiency. 128×128 block structure, BSR sparse matrix operations, maps cleanly to GPU tiles. *Limit:* block size selection, pattern constraints.

- **Attention with Bias** — Add learnable or positional biases to attention scores. ALiBi (Attention with Linear Biases), RoPE (Rotary Positional Embeddings), relative position biases. Quality improvement, not speed. *Paper:* ALiBi 2021, RoPE 2021. *Limit:* adds computation, may not be compatible with all kernels.

### Linear / Sub-Quadratic Attention

- **Linear Attention (Linformer)** — Approximate attention with low-rank projection. Project keys to low-rank dimension, attention becomes linear in sequence length, O(N·d²) instead of O(N²·d). Linear complexity. *Paper:* Wang et al. 2020. *Limit:* approximation error, quality degradation.

- **Performer** — Kernel-based linear attention with random features. FAVOR+ mechanism approximates softmax/gaussian kernels with random feature maps, provable unbiased estimation. O(N) time and space. *Paper:* Choromanski et al. 2020. *Limit:* approximation variance.

- **RWKV** — Linear attention via time-decay weighting in RNN formulation. WKV operator with exponential time decay, parallelizable during training via chunk-wise computation, RNN-like at inference. O(N) training, O(1) per-step inference, 5× inference speedup vs Transformer. *Paper:* Peng et al. 2023+. *Limit:* exponential decay limits very-long recall.

- **Mamba/SSM** — Selective state space model with input-dependent gating. Selective scan with data-dependent parameters, hardware-efficient, parallel training via scan, recurrent inference. O(N) training, O(1) state inference, matches Transformer perplexity at 1.4B with 5× speedup. *Paper:* Gu & Dao 2024. *Limit:* no explicit pairwise token interaction, struggles on precise retrieval tasks.

- **RetNet** — Retention mechanism with exponential decay, dual recurrent/parallel modes. Replace softmax with fixed exponential decay, supports parallel (training), recurrent (inference), and chunkwise modes. O(N) training, O(1) inference, 8.4× inference speedup vs Transformer. *Paper:* Sun et al. 2023. *Limit:* fixed decay rate limits flexibility.

- **Mamba2** — Generalized SSM framework unifying linear attention variants. Structured State Space Duality (SSD), shows Mamba, RetNet, RWKV, GLA as special cases, improved training efficiency. Better training efficiency than Mamba1. *Paper:* Gu & Dao 2024. *Limit:* complexity of unified framework.

- **GLA (Gated Linear Attention)** — Linear attention with data-dependent gates and hardware-efficient kernels. FlashLinearAttention algorithm trades memory movement for parallelizability, gated variant adds data-dependent gates. Faster than FlashAttention-2 even on short sequences (1K). *Paper:* Yang et al. 2024. *Limit:* linear attention quality limitations.

- **Lightning Attention** — Hardware-optimized linear attention kernels. Triton-based kernels for linear attention variants, optimized for modern GPU architectures. Competitive with FlashAttention for linear patterns. *Paper:* FlashLinearAttention work 2024. *Limit:* linear attention limitations, hardware-specific.

- **FlashLinearAttention** — Hardware-efficient algorithm for linear attention. Trade memory movement against parallelizability, faster than FlashAttention-2 even on short sequences. *Paper:* GLA paper 2024. *Limit:* linear attention quality limitations.

- **Hyena** — Long convolution-based attention alternative. Very long convolution kernels with implicit parameterization, O(N log N) complexity via FFT. Efficient for very long-range patterns. *Paper:* Poli et al. 2023. *Limit:* O(N log N) not truly linear.

### Hybrid SSM + Attention

- **Hybrid SSM+Attention (Jamba, Zamba)** — Interleave SSM and attention layers for best of both. Attention handles global context/retrieval, SSM handles sequential patterns cheaply, layer schedule optimized for task. 25-50% SSM layers reduces compute vs pure attention, maintains quality. *Paper:* Jamba (AI21 2024), Zamba (Zyphra 2024). *Limit:* hybrid complexity, optimal schedule unclear.

- **Jamba** — Transformer-Mamba hybrid with MoE. Interleave Transformer and Mamba layers, add MoE in some layers, 7B-based with 12B active/52B total. 256k context, fits in 80GB GPU. *Paper:* AI21 2024. *Limit:* complex architecture, three innovations combined.

- **Zamba** — Mamba backbone with shared attention. Mamba backbone + single shared attention module, minimal parameter cost for attention benefits. 7B, 1T tokens, best non-transformer at this scale. *Paper:* 2024. *Limit:* single attention may be bottleneck.

- **Hawk** — Pure RNN with gated linear recurrences, attention-less. RG-LRU (Real-Gated Linear Recurrence Unit), interleaved with MLPs, exceeds Mamba on downstream tasks. RNN efficiency, matches Mamba quality at 1.4B. *Paper:* De et al. 2024. *Limit:* pure recurrence limits some tasks.

- **Griffin** — Hybrid mixing gated linear recurrences with local attention. Interleaves RG-LRU blocks with local attention windows, matches Llama-2 quality with 6× fewer training tokens. Lower latency and higher throughput than Transformer, extrapolates to longer sequences. *Paper:* De et al. 2024. *Limit:* local attention adds back some quadratic cost.

- **Conv-Hybrids (LFM2, Liquid)** — Convolution-based hybrid models. Use gated convolutions instead of some attention layers, more CPU-friendly. LFM2: 2× faster on CPU, 350M-8.3B params, 32k context. *Paper:* Liquid AI 2025. *Limit:* convolution may miss long-range dependencies.

### Advanced Sparse / Block Attention

- **Native Sparse Attention (DeepSeek NSA)** — Hardware-aligned, natively trainable sparse attention. Dynamic hierarchical sparse strategy: coarse-grained token compression + fine-grained token selection, arithmetic intensity-balanced design. Substantial speedups on 64K sequences across decode/forward/backward, maintains model performance. *Paper:* DeepSeek-AI 2025. *Limit:* complex algorithm, requires training support.

- **MoBA (Mixture of Block Attention)** — MoE applied to attention blocks. Divide context into blocks, route query to top-K blocks via affinity scores, "less structure" principle. 80% sparsity with block size 512/top-3, deployed in Kimi. *Paper:* Moonshot AI 2025. *Limit:* block size/top-K tuning, routing overhead.

- **PowerAttention** — Exponentially scaling receptive fields for sparse attention. Exponential receptive field growth (2^d tokens in d layers), complete context extension, 3.0× faster on 128K. *Paper:* 2025. *Limit:* static pattern, may not suit all tasks.

- **LongNet** — Dilated attention for billion-token sequences. Dilated attention with exponential allocation, linear complexity, logarithmic dependency between tokens. *Paper:* 2023. *Limit:* quality vs full attention, dilated pattern limitations.

- **Compressed Sparse Attention (CSA)** — Low-compression pool with overlapping windows in DeepSeek-V4. Pool with compression rate m=4, overlapping windows, lightning indexer for selection. *Paper:* DeepSeek-V4 2025. *Limit:* DeepSeek-specific, compression artifacts.

- **Heavily Compressed Attention (HCA)** — High-compression pool with non-overlapping windows in DeepSeek-V4. Pool with compression rate m'=128, non-overlapping windows, no indexer. *Paper:* DeepSeek-V4 2025. *Limit:* DeepSeek-specific, high compression quality loss.

- **Lightning Indexer (DeepSeek V4)** — Indexer for compressed sparse attention scoring. Score queries against compressed pool, gather top-k blocks before core attention. *Paper:* DeepSeek-V4 2025. *Limit:* DeepSeek-specific, indexing overhead.

---

## 2.3 Inference Engines & Runtime

### Major Engines

- **vLLM** — High-throughput LLM serving with PagedAttention and continuous batching. PagedAttention for KV cache management, continuous batching (iteration-level scheduling), wide model support. 2-4× vs prior state-of-art, 1000-2000 tok/s on Llama-70B (A100). *Paper:* Kwon et al. SOSP 2023, UC Berkeley. *Limit:* Python overhead (partially addressed in V1).

- **TGI (Text Generation Inference)** — Hugging Face's production inference server (archived 2026). Production-ready serving with quantization, batching, safety, HF ecosystem integration. 800-1500 tok/s on Llama-70B, moved to maintenance mode in favor of vLLM/SGLang. *Paper:* Hugging Face 2022-2025. *Limit:* archived by HF in 2026.

- **TensorRT-LLM** — NVIDIA's optimized inference engine with TensorRT integration. Compile models to TensorRT engines, FP8/INT8 quantization, fused kernels, NVIDIA-only optimization. 2500-4000+ tok/s on Llama-70B (H100), highest raw throughput on NVIDIA. *Paper:* NVIDIA 2023+. *Limit:* NVIDIA hardware only, complex build process, vendor lock-in.

- **llama.cpp** — Lightweight C++ inference for CPU/GPU with GGUF quantization. Pure C++ with minimal dependencies, GGUF quantization format, runs on everything (CPU, GPU, mobile, web). 80-100 tok/s on edge devices, extreme portability. *Paper:* Georgi Gerganov 2023+. *Limit:* lower throughput than GPU-optimized engines.

- **ExLlamaV2** — High-performance CUDA inference for consumer GPUs. Optimized for consumer NVIDIA GPUs, EXL2 quantization format, GPTQ/AWQ support, TabbyAPI serving. Best-in-class for consumer GPUs (4090, 5090). *Paper:* turboderp 2023+. *Limit:* NVIDIA only, consumer-focused.

- **mlx** — Apple Silicon-native ML framework. Designed for Apple Silicon, Metal performance shaders, PyTorch-like API, unified memory architecture utilization. 35-60 tok/s on Apple Silicon, native performance. *Paper:* Hugging Face 2023+. *Limit:* Apple Silicon only.

- **sglang** — High-performance serving with RadixAttention and structured output. RadixAttention (prefix caching in trie), prefill-decode disaggregation, structured output, tool use optimization. 16,200 tok/s on Llama-3.1 8B (H100), 29% faster than vLLM, best for multi-turn/prefix-heavy. *Paper:* LMSYS 2024+. *Limit:* newer than vLLM, smaller community.

- **LMDeploy** — InternLM team's TurboMind C++ engine. Pure C++ engine removes Python overhead, Int4-first quantization, online int8/int4 KV cache. 16,100 tok/s on Llama-3.1 8B (H100), Int4 path 2.4× faster than FP16. *Paper:* InternLM/MMRazor 2023+. *Limit:* smaller community, less model support.

- **candle** — Minimalist Rust ML framework for serverless inference. Rust-based for no Python overhead, small binaries, serverless deployment, FlashAttention integration. Fast for serverless, minimal cold start, cross-platform (CUDA, Metal, CPU, WASM). *Paper:* Hugging Face 2023+. *Limit:* smaller ecosystem, Rust learning curve.

- **candle + accelerate** — Candle with Apple Accelerate framework integration. Use Apple's Accelerate framework for optimized CPU operations on Mac, MKL on x86. Optimized CPU backend for Mac/x86. *Limit:* CPU-focused, GPU requires separate CUDA backend.

- **optimum** — Hugging Face optimization toolkit for various hardware. Unified interface for ONNX Runtime, Intel Neural Compressor, Habana Gaudi, Graphcore IPU optimizations. Hardware-specific optimizations via single API. *Paper:* Hugging Face 2022+. *Limit:* abstraction layer, depends on backend quality.

- **ONNX Runtime** — Cross-platform inference engine for ONNX models. ONNX format execution with multiple execution providers (CPU, CUDA, OpenVINO, TensorRT). Portable optimization, hardware-specific EPs. *Paper:* Microsoft 2018+. *Limit:* ONNX conversion overhead, less LLM-specific.

- **OpenVINO** — Intel's optimization toolkit for Intel hardware. CPU/GPU/NPU optimization for Intel hardware, quantization, model compression. Optimized for Intel CPUs/GPUs, INT8 acceleration. *Paper:* Intel 2018+. *Limit:* Intel hardware only.

- **AITemplate** — Meta's framework transforming models to CUDA/HIP C++ code. Python frontend, C++ GPU backend, close to TensorCore/MatrixCore performance, extensive operator fusion. Up to 12× on NVIDIA, 4× on AMD vs PyTorch eager. *Paper:* Meta AI 2022+. *Limit:* limited dynamic shape support.

- **FasterTransformer** — NVIDIA's optimized transformer inference library. Highly optimized CUDA kernels for transformer layers, tensor/pipeline parallelism, integration with TensorRT. Foundation for many NVIDIA optimizations. *Paper:* NVIDIA 2019+. *Limit:* largely superseded by TensorRT-LLM.

- **PowerInfer** — CPU/GPU hybrid inference exploiting activation locality. Hot neurons on GPU, cold neurons on CPU, power-law activation distribution, adaptive predictors. Up to 11.69× faster than llama.cpp, 82% of A100 performance on RTX 4090. *Paper:* SJTU 2023. *Limit:* requires ReLU-sparse models, CPU-GPU transfer overhead.

- **PowerInfer-2** — Smartphone-optimized inference with neuron clusters. Neuron cluster abstraction, NPU for dense clusters, CPU for sparse, I/O-computation pipelining, segmented cache. Up to 27.8× vs state-of-art, 11.68 tok/s for Mixtral 47B on smartphone. *Paper:* SJTU 2024. *Limit:* smartphone-specific.

- **llamafile** — Single-file executable LLM distribution combining llama.cpp + Cosmopolitan Libc. Collapse model + runtime into one executable, cross-platform (Windows/Linux/macOS/ARM), no installation. Convenience-focused, performance similar to llama.cpp. *Paper:* Mozilla 2023+. *Limit:* 4GB file size limit on Windows.

### Batching & Scheduling

- **Continuous Batching** — Iteration-level scheduling adding/removing requests between steps. After each forward pass, evict finished requests and add waiting ones, keeps GPU occupied, eliminates idle time. 2-4× throughput vs static batching, standard in all modern engines. *Paper:* Orca OSDI 2022. *Limit:* scheduler complexity, fragmentation management.

- **In-Flight Batching** — Same as continuous batching, alternative naming. Requests join/leave batch during generation, dynamic batch composition. *Limit:* same as continuous batching.

- **Disaggregated Prefill/Decode** — Separate prefill and decode phases onto different GPU pools. Prefill (compute-bound) and decode (memory-bound) have different bottlenecks, separate them for independent optimization and scaling. Eliminates prefill-decode interference, enables phase-specific parallelism, up to 7.4× more requests or 12.6× tighter SLO. *Paper:* DistServe 2024, Splitwise 2024. *Limit:* KV cache transfer overhead, system complexity.

- **Chunked Prefill** — Split long prompts into chunks interleaved with decode. When prompt exceeds token budget, process partial prompt, interleave with decode steps, reduces TTFT for other requests. Better TTFT under load, higher throughput for mixed workloads. *Paper:* Hugging Face TGI, vLLM, TensorRT-LLM. *Limit:* increased total prompt processing time.

- **Prefix Caching** — Cache KV for shared prefixes across requests. Store KV cache for common system prompts/prefixes, reuse across requests with same prefix, radix tree structure (RadixAttention). 2-10× for repeated prompts, critical for multi-turn chat/RAG. *Limit:* memory overhead for cache, cache invalidation complexity.

- **RadixAttention (sglang)** — Prefix caching using radix tree data structure. Organize cached KV in radix tree by token sequence, efficient prefix matching and sharing, LCP-based cache invalidation. 3.2-4.8× kernel speedup for shared prefixes, best for multi-turn conversations. *Paper:* SGLang 2024. *Limit:* tree management overhead.

- **Speculative Serving** — Server-side integration of speculative decoding. Engine manages draft model, verification, batching of speculative requests, efficient KV cache handling. 2-4× on top of continuous batching. *Limit:* draft model management, acceptance rate monitoring.

- **DistServe** — Disaggregated prefill/decode with co-optimized resource allocation. Separate prefill/decode instances, co-optimize resource allocation and parallelism for TTFT/TPOT SLOs, placement based on bandwidth. 7.4× more requests or 12.6× tighter SLO vs state-of-art. *Paper:* Zhong et al. MLSys 2024. *Limit:* system complexity, requires multi-GPU cluster.

- **Splitwise** — Phase splitting for heterogeneous hardware optimization. Prefill on compute-optimized hardware, decode on memory-optimized hardware, phase-specific resource management. 1.4× higher throughput at 20% lower cost, or 2.35× more throughput at same power/cost. *Paper:* Microsoft Research ISCA 2024. *Limit:* heterogeneous cluster required, state transfer overhead.

- **Cascade Inference** — Model cascades for adaptive accuracy/latency tradeoffs. Chain models of increasing size/quality, early exit if sufficient confidence, adaptive switching based on load/SLO. Latency reduction for easy queries, throughput adaptation to load. *Paper:* CascadeServe 2024. *Limit:* multiple models in memory, switching overhead.

- **PD Disaggregation** — Prefill-Decode disaggregation (same as disaggregated prefill/decode). Separate compute-intensive prefill from memory-intensive decode. *Paper:* AWS Neuron, various 2024. *Limit:* same as disaggregated prefill/decode.

- **Dynamic SplitFuse** — Dynamic splitting and fusion of prefill/decode work. Adaptively split requests between prefill and decode phases, fuse when beneficial. *Paper:* DeepSpeed-MII 2024. *Limit:* dynamic scheduling complexity.

- **Stream2LLM** — Overlap context streaming and prefill for reduced TTFT. Two-phase scheduling, priority-based ordering, LCP-based cache invalidation for streaming inputs. *Paper:* MLSys 2026. *Limit:* streaming-specific, cache invalidation complexity.

---

## 2.4 Parallelism & Serving

### Core Parallelism

- **Tensor Parallel (TP)** — Split individual model layers across GPUs. Each GPU holds portion of weight matrices for each layer, all-reduce communication for partial results, reduces per-GPU memory. Enables models larger than single GPU memory, near-linear scaling for large layers. *Paper:* Megatron-LM 2019. *Limit:* high communication bandwidth requirement.

- **Pipeline Parallel (PP)** — Split model layers across GPUs in pipeline stages. Sequential layer stages, micro-batching to fill pipeline, reduces per-GPU memory for deep models. Enables very deep models, pipeline bubbles reduce efficiency. *Paper:* GPipe/PipeDream 2019. *Limit:* pipeline bubbles, micro-batch complexity.

- **Sequence Parallel (SP)** — Split sequence dimension across devices. Shard sequence activations, reduce memory for long sequences, often combined with TP. Enables longer sequences, reduces activation memory. *Paper:* Megatron-LM 2023. *Limit:* communication overhead, requires TP for best results.

- **Expert Parallel (EP)** — Distribute MoE experts across devices. Each GPU holds subset of experts, all-to-all dispatch/combine for token routing, enables large MoE models. Enables models with hundreds of experts, essential for DeepSeek-V3 class models. *Paper:* Megatron-MoE 2023+. *Limit:* all-to-all communication bottleneck.

- **Data Parallel (DP)** — Replicate model across devices, split batch. Each device has full model copy, processes different batch subset, gradient all-reduce. Linear scaling for batch size, simplest parallelism. *Limit:* memory inefficient for large models, gradient communication overhead.

- **FSDP (Fully Sharded Data Parallel)** — Shard model states (params, gradients, optimizer) across data parallel ranks. ZeRO-3 style sharding, reduce memory per device, all-gather for computation. Enables larger models in DP setup, memory-efficient. *Paper:* FairScale/PyTorch 2021. *Limit:* communication overhead.

- **ZeRO 1/2/3** — Memory optimization stages for data parallelism. ZeRO-1: shard optimizer states, ZeRO-2: shard gradients+optimizer, ZeRO-3: shard params+gradients+optimizer. Progressive memory reduction, ZeRO-3 enables 100B+ parameter models. *Paper:* Microsoft DeepSpeed 2020. *Limit:* communication increases with ZeRO stage.

- **Megatron** — Framework for large-scale model training with TP/PP. Tensor+pipeline parallelism, efficient kernels, optimized for NVIDIA GPUs. Foundation for training largest models (GPT-3, etc.). *Paper:* NVIDIA 2019+. *Limit:* NVIDIA-focused, complex setup, training-focused.

- **Context Parallel (CP)** — Parallelize attention computation for long sequences. Ring attention or other patterns to split sequence across devices, enables million-token contexts. Linear scaling for sequence length. *Paper:* RingAttention 2023, DeepSpeed-Ulysses 2023. *Limit:* communication overhead.

- **3D Parallelism** — Combine data + tensor + pipeline parallelism. Simultaneous use of DP, TP, PP for maximum scale, each dimension handles different aspect. Enables training largest models (trillion+ parameters). *Paper:* Megatron-DeepSpeed 2021. *Limit:* extreme complexity, communication patterns.

### MoE Serving

- **Expert Routing** — Mechanism to select which experts process each token in MoE. Router/gating network scores experts, top-k selection, dispatch tokens to selected experts. Enables MoE efficiency, only k experts active per token. *Paper:* Shazeer 2017 (Sparse Transformer). *Limit:* load imbalance, routing quality, communication overhead.

- **Load Balancing Loss** — Auxiliary loss encouraging even token distribution across experts, typically λ·CV(frequency, scores). Prevents expert collapse, improves utilization. *Paper:* Switch Transformer 2021. *Limit:* may inhibit expert specialization.

- **MoE Serving** — Specialized serving considerations for MoE models. Expert parallelism, load balancing, capacity factors, token dropping, routing optimization. Enables efficient serving of huge models (DeepSeek-V3 671B). *Limit:* routing overhead, load imbalance, stragglers.

- **Expert Offloading** — Move less-used experts to slower storage/compute. Keep hot experts in fast memory, cold experts on CPU/disk, dynamic loading based on usage. Reduces memory for MoE models with many experts. *Limit:* loading latency, prediction of hot/cold experts.

- **EPLB (Expert Placement with Load Balancing)** — Redundant placement of hot experts for load balancing. Replicate frequently-used experts across GPUs, improve load balance at inference. *Limit:* memory overhead for replicas.

- **Capacity Factor** — Limit tokens per expert to prevent overload. Maximum tokens per expert, excess tokens dropped or spilled to other experts. *Paper:* Switch Transformer 2021. *Limit:* token dropping hurts quality.

- **Token Dropping** — Drop excess tokens when expert capacity exceeded. When capacity factor exceeded, drop lowest-importance tokens. *Limit:* quality degradation.

- **Token Choice Routing** — Standard routing: tokens pick top-k experts. Each token selects top-k experts, standard in production (Mixtral, DeepSeek). *Limit:* load imbalance requires auxiliary loss/capacity factor.

- **Expert Choice Routing** — Each expert picks top-C tokens instead of tokens picking experts. Perfect load balance by construction, but broken at batch=1 inference. *Limit:* train-inference mismatch, not suitable for autoregressive serving.

- **Null Experts** — Shared expert always active in MoE for stability. Add shared expert to all tokens, improves training stability and load balance. *Paper:* Meta AI 2024. *Limit:* adds compute overhead.

- **Global Batch Load Balancing** — Calculate load balancing loss over global batch instead of micro-batch. Encourages load balance at corpus level rather than sequence level, improves expert specialization. *Paper:* ACL 2025. *Limit:* extra communication step.

- **Replicate-and-Quantize (R&Q)** — Training-free load balancing for sparse MoE at inference. Replicate heavy-hitter experts for more capacity, quantize less important experts, stay within memory budget. Near-lossless workload balancing without retraining. *Paper:* 2025. *Limit:* quantization quality, replication overhead.

### Offload / Heterogeneous

- **CPU-Offload Inference (PowerInfer, DeepSpeed-MII)** — Offload part of model/computation to CPU. Keep hot layers/neurons on GPU, cold on CPU, overlap computation and transfer. Enables larger models on limited GPU memory, PowerInfer 11.69× vs llama.cpp. *Paper:* PowerInfer 2023, DeepSpeed-MII 2022. *Limit:* PCIe transfer bottleneck.

- **Heterogeneous Inference** — Serve across mixed hardware (H100, A100, CPU, etc.). Capability-weighted scheduling, phase-aware partition, adaptive quantization per device. Cost reduction vs homogeneous high-end cluster, 2.13-2.88× throughput improvement. *Paper:* LLM-PQ 2024, HeteroServe 2024, Hetis 2025. *Limit:* scheduling complexity, load imbalance.

- **LLM-PQ** — Phase-aware partition and adaptive quantization for heterogeneous clusters. Mixed-precision quantization combined with phase-aware model partition, micro-batch sizing for heterogeneous GPUs. 2.88× throughput improvement (2.26× average) on 11 different clusters. *Paper:* 2024. *Limit:* heterogeneous cluster required.

- **HeteroServe** — Capability-weighted batch scheduling for heterogeneous GPU clusters. Hardware capability scoring + queue-depth feedback + length-binned admission, route requests to suitable devices. 2.13× throughput improvement, 68.8% SLO compliance vs 26.4% baseline. *Paper:* 2024. *Limit:* requires capability profiling.

- **Hetis** — Fine-grained dynamic parallelism for heterogeneous clusters. Selectively parallelize compute-intensive ops, distribute attention to low-end GPUs at head granularity, online load dispatching. 2.25× throughput improvement, 1.49× latency reduction. *Paper:* 2025. *Limit:* complex fine-grained parallelism.

- **Elastic Serving** — Dynamically scale resources based on load. Add/remove instances, adjust parallelism, migrate requests, autoscaling based on SLOs. Cost efficiency, meets SLOs under varying load. *Limit:* migration overhead, state management.

- **CascadeServe** — Model cascades with adaptive switching based on SLOs. Multiple model cascades, switch based on system load and latency requirements, joint optimization with placement. Adaptive throughput/latency tradeoffs. *Paper:* 2024. *Limit:* multiple models in memory.

### Multi-LoRA Serving

- **LoRAX (Multi-LoRA Serving)** — Serve thousands of LoRA adapters on single base model. Base model shared, adapters dynamically loaded, heterogeneous continuous batching, SGMV kernel for mixed-adapter batches. 1000s of adapters on single GPU, 12× vs prior multi-tenant systems. *Paper:* Predibase 2024. *Limit:* adapter loading overhead.

- **SGMV (Segmented Gather Matrix-Vector)** — Kernel for efficient multi-LoRA batched computation. Group requests by adapter, fuse heterogeneous low-rank deltas into one batched operation, 12× throughput. *Paper:* Punica 2023. *Limit:* LoRA-specific, kernel complexity.

- **Unified Paging** — Unified memory management for variable-rank adapters and KV cache. Single pooled allocator for adapter weights and KV cache, fight fragmentation. *Paper:* S-LoRA 2024. *Limit:* unified allocator complexity.

### Other Runtime Optimizations

- **CUDA Graphs** — Capture kernel launches as single graph for reduced launch overhead. Sequence of kernels captured as CUDA graph, reduced CPU-GPU synchronization. *Paper:* NVIDIA technique, adopted by vLLM/candle-vllm. *Limit:* graph capture overhead, less flexible than dynamic execution.

- **Operator Fusion** — Combine multiple operators into single kernel. Reduce memory reads/writes by fusing consecutive ops, common in compilers. *Limit:* fusion opportunities limited by graph structure.

- **Data Layout Management** — Optimize tensor layouts for hardware (NHWC vs NCHW, etc.). *Limit:* layout conversion overhead, hardware-specific.

- **Activation Checkpoint Offload** — Offload activations to CPU during training. Store checkpoints on CPU, recompute during backward, reduce GPU memory. *Paper:* DeepSpeed, FSDP. *Limit:* recomputation overhead, CPU-GPU transfer.

- **TiledMLP** — Tiled computation for MLP layers. Tile MLP computation for better memory access patterns. *Paper:* Arctic Long Sequence Training. *Limit:* tiling overhead.

- **Graph Compiler Optimizations** — Compiler-level optimizations for inference graphs. Operator fusion, layout management, pipelining, memory management. *Paper:* XLA, TorchScript, TVM. *Limit:* compiler limitations, debuggability.

- **Lazy Mode Execution (SynapseAI)** — Lazy graph execution in Habana SynapseAI. Accumulate operations in graph, trigger execution lazily, graph-level optimizations. *Paper:* Habana Gaudi. *Limit:* Habana hardware only.

- **Eager Mode Execution (SynapseAI)** — Eager execution in Habana SynapseAI. Execute operations one at a time like standard PyTorch. *Paper:* Habana Gaudi. *Limit:* less optimization than lazy mode.

- **Multi-Process Tensor Parallelism** — Tensor parallelism across processes instead of threads. Better isolation, memory separation, more scalable for multi-GPU. *Paper:* candle-vllm 2024. *Limit:* process overhead vs threads.

- **Multi-Threaded Tensor Parallelism** — Tensor parallelism within process using threads. Lower overhead than multi-process, shared memory. *Paper:* candle-vllm 2024. *Limit:* thread synchronization.

- **TCP-based Multi-Node Inference** — Multi-node coordination via TCP instead of specialized interconnect. Use standard TCP for multi-node communication, no InfiniBand required. *Paper:* candle-vllm 2024. *Limit:* higher latency than InfiniBand.

- **Metal SDPA** — Scaled dot product attention for Apple Metal. Vector kernel for q_seqlen=1 (decode), full tiled kernel for long sequences. *Paper:* Candle Metal backend. *Limit:* Apple Silicon only.

- **Steel Attention** — Tiled attention implementation for Metal. Tiled "Steel Attention" for longer sequences on Metal. *Paper:* Candle Metal backend. *Limit:* Apple Silicon only.

- **In-Situ Quantization** — Quantize model in-place during loading. Quantize weights as they're loaded, avoid storing multiple copies. *Paper:* candle-vllm 2024. *Limit:* quantization overhead during load.

- **TurboQuant** — Aggressive KV cache quantization (2-4 bit). Quantize KV cache to 2-4 bits with minimal quality loss, extends context 4.7×. *Paper:* candle-vllm 2024. *Limit:* quantization quality loss.

- **Block-wise FP8 Models** — FP8 quantization at block granularity. Quantize model weights in blocks for better precision distribution, SM90+ support. *Paper:* Qwen3 series, SM90+ hardware. *Limit:* SM90+ only.

- **FP8 KV Cache** — FP8 quantization for KV cache. Store KV cache in FP8, reduce memory bandwidth and storage. *Paper:* candle-vllm, various engines. *Limit:* FP8 precision, kernel support required.

- **TurboSparse** — Activation-sparse models for efficient inference. Train models with predictable activation sparsity, optimize for sparse computation. PowerInfer/PowerInfer-2 leverage this for 11-27× speedups. *Paper:* PowerInfer 2023-2024. *Limit:* requires training sparse models.

- **MCP (Model Context Protocol)** — Standard protocol for tool calling and context management. Unified interface for model-tool interactions, context passing. *Paper:* Anthropic 2024. *Limit:* protocol overhead, ecosystem adoption.

---

## 2.5 Prefill / Decode Optimization

- **Attention Sinks** — First few tokens receive disproportionate attention regardless of content. Softmax must sum to 1, early tokens become default "parking spot" for attention mass, critical for streaming. *Paper:* StreamingLLM observation 2023. *Limit:* phenomenon, not optimization.

- **Streaming LLM** — Enable infinite-length generation with fixed cache via attention sinks. Keep ~4 initial sink tokens + sliding window, pin sinks non-evictable, stable generation to 4M+ tokens. 22.2× vs sliding window recomputation baseline. *Paper:* Xiao et al. ICLR 2024. *Limit:* loses mid-context tokens.

- **Chunked Attention** — Process attention in chunks for memory efficiency. Split sequence into chunks, process attention chunk-wise, reduce peak memory. Enables longer sequences on limited memory. *Limit:* chunk boundary effects.

- **Multi-Token Prediction (MTP)** — Train model to predict multiple future tokens at each position. Add auxiliary heads predicting tokens at positions t+2, t+3, etc., improves data efficiency and can accelerate inference. Up to 3× faster inference using auxiliary heads, 12% better HumanEval, 17% better MBPP. *Paper:* DeepSeek-V3 2024. *Limit:* training complexity.

- **DeepSeek MTP** — DeepSeek-V3's implementation of multi-token prediction. MTP modules with shared embedding, projection, transformer block, output head per depth, auxiliary loss scaling. Part of DeepSeek-V3's efficiency. *Paper:* DeepSeek-V3 Technical Report 2024. *Limit:* DeepSeek-specific.

- **Medusa Heads** — Multiple decoding heads for parallel token prediction. K heads predict tokens at different future positions, tree-based verification, trained via fine-tuning. 2.2-2.8× acceleration. *Paper:* Cai et al. 2024. *Limit:* requires training, adds parameters.

- **Decode-Time Batching** — Batch decode operations across requests. Combine single-token decode steps from multiple requests into one batch, improve GPU utilization. Standard in continuous batching. *Paper:* Orca OSDI 2022. *Limit:* variable sequence lengths cause padding/fragmentation.

- **Prefix Sharing** — Share KV cache for common prefixes across requests. System prompts, RAG contexts, conversation history shared, store once, reference multiple times. 2-10× for repeated prompts. *Limit:* cache invalidation, memory overhead.

- **Prompt Caching** — Cache computed KV for input prompts. Store KV cache after prefill, reuse for identical or similar prompts, reduces repeated computation. Near-instant for cached prompts. *Limit:* cache memory, invalidation.

- **KV Cache Reuse** — Reuse KV cache across related requests. Prefix caching, session caching, RAG caching, share computation for overlapping contexts. Major gains for workloads with shared context. *Limit:* cache management, invalidation.

- **Session Caching** — Maintain KV cache across multi-turn conversations. Keep conversation history in cache, append new tokens each turn, avoid recomputing full history. Essential for chat applications. *Limit:* memory grows with conversation length.

- **RAG Caching** — Cache KV for retrieved documents in RAG systems. Documents often reused across queries, cache their KV, share across requests with same retrieval. Significant for RAG workloads. *Limit:* cache invalidation when documents update.

- **H2O (Heavy-Hitter Oracle)** — KV cache eviction policy retaining heavy-hitter tokens. Small subset of tokens contribute most attention value (heavy hitters), dynamic retention of recent + heavy hitters. 29× throughput improvement vs baseline at 20% cache budget, 1.9× latency reduction. *Paper:* NeurIPS 2023. *Limit:* score computation overhead.

- **SnapKV** — KV cache compression via token importance scoring. Score tokens by attention importance, keep important tokens, compress cache dynamically. Significant memory reduction with minimal quality loss. *Paper:* 2024. *Limit:* scoring overhead.

- **Quest** — KV cache eviction with learned policy. Learn which tokens to keep via reinforcement learning or other methods, adaptive eviction. Better quality than simple policies for same budget. *Paper:* 2024. *Limit:* training overhead, policy generalization.

- **KeyDiff** — Evict tokens with redundant keys (cosine similarity). Minimize pairwise cosine similarity among retained keys, maximize diversity in cache. Better preservation of diverse information. *Paper:* 2024. *Limit:* similarity computation overhead.

---

# Part 3 — KV Cache & Architecture

## 3.1 KV Cache Compression

### Eviction / Selection Policies

- **H2O (Heavy Hitter Oracle)** — Dynamic KV cache eviction based on accumulated attention scores. Maintains "heavy hitter" tokens with highest cumulative attention across all layers, evicting low-attention tokens. 5× memory reduction, maintains quality at 20% cache retention. *Paper:* Zhang et al., 2023. *Limit:* requires attention score tracking, may miss tokens important for future queries.

- **StreamingLLM** — Fixed-size sliding window with attention sink tokens. Retains initial "sink" tokens (first few) + recent sliding window, enabling infinite streaming without recompute. 5× compression, 70B model with 18k context. *Paper:* Xiao et al., 2024. *Limit:* fixed window size may lose important distant tokens.

- **SnapKV** — Observation-window based KV selection across heads. Uses attention scores from initial observation window to select important tokens globally, clusters similar tokens. 8.2× compression, 35B model with 26k context. *Paper:* Li et al., 2024. *Limit:* static observation window may miss late-emerging important tokens.

- **PyramidKV** — Layer-wise pyramidal cache budget allocation. Allocates more cache budget to lower layers (less sparse attention) and less to higher layers (more sparse), matching attention sparsity patterns. 8.3× compression, works at 0.7% cache retention, outperforms H2O/SnapKV. *Paper:* Cai et al., 2024. *Limit:* requires layer-specific budget tuning.

- **PyramidInfer** — Pyramid-shaped cache budget with head-aware selection. Extends PyramidKV with head-specific budget allocation and importance-aware eviction. Similar to PyramidKV with improved head-level granularity. *Paper:* Yang et al., 2024. *Limit:* increased complexity from head-level management.

- **FastGen** — Head-adaptive KV cache management. Different retention strategies per attention head type (local vs global vs special-token focused). Significant memory savings with head-aware policies. *Paper:* Ge et al., 2024. *Limit:* requires head classification.

- **Scissorhands** — Persistence of importance hypothesis-based eviction. Tokens important at one step remain important; maintains fixed budget via pivotal token selection. 5× memory reduction without quality loss. *Paper:* Liu et al., 2023 (NeurIPS). *Limit:* persistence assumption may fail.

- **SubGen** — Sublinear complexity KV caching via clustering. Online clustering on key tokens + ℓ2 sampling on values, streaming attention data structure with error bounds. Sublinear time and memory, provable error bounds. *Paper:* 2024. *Limit:* clustering overhead.

- **LazyLLM** — Dynamic token pruning for prefill acceleration. Selectively computes KV for important tokens only, both in prefill and decoding stages; different subsets per generation step. 2.34× prefill acceleration on multi-document QA. *Paper:* 2024. *Limit:* may miss tokens important for later steps.

- **ThinK** — Query-driven key channel pruning. Prunes channel dimension of key cache based on query importance, targets low-rank structure in attention weights. 20%+ KV reduction, 2.8× peak memory with KIVI, 5× batch size increase. *Paper:* Xu et al., 2024. *Limit:* channel pruning affects all queries.

- **Quest** — Query-aware sparsity with page-level metadata. Tracks min/max key values in KV pages, estimates page criticality using query vectors, loads only top-K critical pages. 7.03× self-attention speedup, 2.23× latency reduction. *Paper:* 2024. *Limit:* page-level granularity may miss intra-page important tokens.

- **KVCompress** — Paged KV cache compression with variable rates per head. Block-level eviction compatible with paged attention, variable compression per layer/head, squared past attention for eviction decisions. Improved over Ada-SnapKV. *Paper:* 2024. *Limit:* requires paged attention infrastructure.

- **KV-Compress** — Paged KV compression with variable rates. Block-level eviction compatible with paged attention, variable compression per layer/head. *Paper:* 2024. *Limit:* requires paged attention.

- **KVzip** — Query-agnostic KV compression with reconstruction. Reconstruction-based scoring, rank KVs by contribution to context reconstruction fidelity, reusable compressed cache. Good for multi-query scenarios. *Paper:* 2025. *Limit:* higher upfront computation.

- **CriticalKV** — Identify critical KV from output perturbation. Measure output sensitivity to KV entries to identify critical ones. Better importance estimation. *Paper:* 2025. *Limit:* perturbation computation overhead.

- **Head-Level Compression** — Compress at head granularity. Different compression strategies per attention head. Better head-specific optimization. *Limit:* head management complexity.

- **HeadKV** — Head-aware retrieval/reasoning cache. Different cache strategies per attention head based on their role. Better head-level utilization. *Limit:* head classification complexity.

- **AdaKV** — Adaptive KV cache eviction. Head-wise adaptive budget allocation, theoretical loss bound guidance. Significant quality improvement over uniform allocation. *Paper:* NeurIPS 2025. *Limit:* head-level budget management complexity.

- **SqueezeAttention** — Layer-level KV eviction. Evict KV entries at layer granularity rather than token/head. Simpler than fine-grained eviction. *Paper:* 2024. *Limit:* less flexible.

- **RazorAttention** — Attention-based KV pruning. Use attention patterns to guide KV pruning. Effective pruning. *Paper:* 2024. *Limit:* attention may not predict future importance.

- **CORM** — Cache optimization method. Various cache optimization techniques. Cache improvements. *Paper:* 2024. *Limit:* method-specific limitations.

- **TOVA** — Last-attention based eviction. Use most recent attention scores for eviction decisions. Better than accumulated for some tasks. *Paper:* Oren et al., 2024. *Limit:* last attention may not predict future.

- **VATP** — Variance-aware token pruning. Consider variance in attention patterns across queries for token importance. Improved selection stability. *Paper:* Guo et al., 2024. *Limit:* variance tracking overhead.

- **BUZZ** — Buzz-based importance. Use attention activity ("buzz") as importance metric. Competitive with attention-based. *Paper:* Zhao et al., 2024. *Limit:* buzz may not correlate with all tasks.

- **L2KV** — L2-norm based selection. Select tokens with highest L2 norm in key/value space. Simple, effective baseline. *Limit:* norm doesn't always correlate with attention importance.

- **NACL** — Proxy-token score reduction with eviction. Use proxy tokens to represent groups, reduce scores for eviction candidates. Effective for grouped eviction. *Paper:* Chen et al., 2024. *Limit:* proxy representation may lose granularity.

- **CAKE** — Cascading adaptive eviction with layer preferences. Cascades eviction decisions across layers with layer-specific preferences. Improved over uniform eviction. *Paper:* Qin et al., 2025. *Limit:* complex cascading logic.

- **D2O** — Dynamic cache size based on attention density. Adjust cache size per layer based on current layer's attention density. Adaptive to layer-specific patterns. *Paper:* Wan et al., 2025. *Limit:* density estimation overhead.

- **SepLLM** — Separated KV cache management. Separate caches for different purposes (e.g., prefix vs suffix). Better cache utilization. *Paper:* Chen et al., 2025. *Limit:* management complexity.

- **LaCache** — Ladder-shaped cross-layer KV caching. Stores KV pairs sequentially within layers AND across layers (shallow to deep), extended span for long-range dependencies; iterative compaction for space. Enhanced long-range capabilities under fixed budget. *Paper:* ICML 2025. *Limit:* cross-layer coordination complexity.

- **KVCompose** — Composite token-based structured compression. Aggregates attention scores for importance, head-specific selection aligned into composite tokens respecting uniform cache structure. Outperforms structured/semi-structured methods. *Paper:* 2025. *Limit:* composite token alignment complexity.

- **DiffKV** — Differential KV cache management. Store only differences between similar KV states. High compression for similar contexts. *Paper:* Zhang et al., 2025. *Limit:* requires similarity detection.

- **EvolKV** — Evolutionary search for layer-wise budget allocation. Multi-objective optimization via evolutionary algorithms to configure layer budgets maximizing task performance. Surpasses baselines by 7% on GSM8K, 1.5% budget beats full cache on code completion. *Paper:* 2025. *Limit:* evolutionary search overhead.

- **DynamicKV** — Dynamic KV cache allocation. Adjust cache allocation dynamically based on current needs/importance. Adaptive to varying requirements. *Paper:* Zhou et al., 2025. *Limit:* dynamic allocation overhead.

- **Keyformer** — Key-based token selection. Select tokens based on key vector properties rather than attention scores. Alternative to attention-based selection. *Paper:* Adnan et al., 2024. *Limit:* key properties may not predict output importance.

- **MiniCache** — Cross-layer compression with SLERP. Adjacent layers share KV via SLERP interpolation, magnitude restore, selective token retention. Significant cross-layer compression. *Paper:* Liu et al., 2024. *Limit:* SLERP approximation error.

- **CAM (Cache Merging)** — Merge KV caches with attention-informed aggregation. Merge similar KV entries using attention-weighted aggregation. Reduces cache size. *Limit:* merging quality degradation.

### General / Eviction Infrastructure

- **KV Cache Pruning (General)** — Remove less important KV entries. Various importance metrics (attention scores, norms, gradients) to identify prunable entries. 2-10× compression possible. *Limit:* quality degradation if importance metric is wrong.

- **Eviction Policies** — Strategies for selecting KV entries to evict. LRU, LFU, attention-based, importance-based policies for cache management. Policy-dependent improvements. *Limit:* no single policy optimal for all workloads.

- **Sliding Window Cache** — Fixed-size recent token window. Only keep most recent N tokens, discard older ones. Linear memory growth capped, simple implementation. *Limit:* loses long-range dependencies.

- **Rolling Cache** — Rolling buffer with position tracking. Similar to sliding window but with position-aware rolling mechanism for better locality. *Limit:* still loses distant context.

- **Sink Tokens** — Special initial tokens always retained. First few tokens serve as "attention sinks" to stabilize attention scores when evicting other tokens. 4-8 sink tokens typical. *Paper:* StreamingLLM. *Limit:* sink assumption doesn't always hold.

- **CacheFlush** — Aggressive cache eviction policy. Flushes cache aggressively based on workload patterns, prioritizes fresh computation over stale cache. Memory savings at cost of recomputation. *Limit:* high recomputation cost.

---

## 3.2 KV Cache Quantization

- **Per-Token Quantization** — Quantize each token's KV independently. Separate scale per token, captures per-token distribution differences. Better than per-tensor for values. *Limit:* high scale storage overhead.

- **Per-Channel Quantization** — Quantize each channel independently. Separate scale per channel, handles channel-wise outliers better. Essential for keys (KIVI insight). *Paper:* KIVI, KVQuant. *Limit:* scale overhead, less effective for values.

- **INT4 KV** — 4-bit integer quantization of KV cache. Reduce precision from FP16 to INT4, 4× memory reduction. 4× compression, some quality loss without careful calibration. *Limit:* quality degradation without outlier handling.

- **INT2 KV** — 2-bit integer quantization (extreme). Aggressive 2-bit quantization for maximum compression. 8× compression, challenging to maintain quality. *Paper:* KIVI (2-bit). *Limit:* significant quality loss without sophisticated techniques.

- **FP8 KV** — 8-bit floating point quantization. Use FP8 (E4M3/E5M2) instead of FP16, 2× compression with better dynamic range than INT8. 2× compression, native support on H100+. *Paper:* NVIDIA FP8 standard. *Limit:* requires FP8 hardware.

- **Mixed Precision KV** — Different precisions for different parts. Use higher precision for important KV (outliers, recent tokens), lower for rest. Balance quality and compression. *Paper:* KVQuant, GEAR. *Limit:* complexity of managing multiple precisions.

- **KIVI (Channel-wise 2-bit)** — Asymmetric 2-bit with per-channel keys, per-token values. Keys: per-channel quantization (handles channel outliers), Values: per-token (streaming-friendly), asymmetric quantization. 2.6× memory reduction, 2.35-3.47× throughput, tuning-free. *Paper:* ICML 2024. *Limit:* 2-bit quality degradation on very long generations.

- **KVQuant** — Multi-technique low-precision quantization. Per-channel pre-RoPE keys, non-uniform quantization (NUQ), dense-and-sparse for outliers, calibration-based. Enables 1M context on single A100, 10M on 8-GPU. *Paper:* NeurIPS 2024. *Limit:* requires offline calibration.

- **Palu** — Low-rank projection + quantization. Decompose linear layers into low-rank, cache intermediate states, reconstruct on-fly; low-rank-aware quantization with Hadamard for outliers. 91.25% compression, 1.61× speedup, better perplexity than pure quantization. *Paper:* 2024. *Limit:* reconstruction overhead.

- **LightKV** — Lightweight KV quantization scheme. Simplified quantization for minimal overhead. Good compression with low complexity. *Limit:* may sacrifice quality for simplicity.

- **Q-Hitter** — Quantization-aware heavy hitter selection. Select tokens that are both important AND quantization-compatible; lossless 4-bit on complex tasks, infinite length via position rolling. Lossless 4-bit on summarization/multi-doc QA, 4M token length. *Paper:* MLSys 2024. *Limit:* compatibility constraint may exclude some important tokens.

- **Group Quant for KV** — Quantize groups of tokens/channels together. Balance between per-token (high overhead) and per-tensor (poor quality) by grouping. Good quality/overhead tradeoff. *Limit:* group size tuning.

- **Outlier-Aware KV Quant** — Special handling for outlier values. Identify outliers (channels/tokens with extreme values), handle separately (keep in FP16, sparse representation, etc.). Critical for low-bit quantization. *Paper:* KVQuant, GEAR. *Limit:* outlier detection overhead.

- **KV in NF4** — Normal Float 4-bit data type. Use NF4 (normal distribution-based 4-bit) instead of uniform INT4, better represents KV distribution. Better quality than uniform INT4. *Limit:* requires NF4 support.

- **MX KV (Mixed Precision)** — MX-format KV quantization. Use MX formats (MXFP4, MXINT4) with block scaling for better quality. NVFP4: 50% memory reduction vs FP8, <1% accuracy loss. *Paper:* NVIDIA. *Limit:* requires hardware support.

- **NQKV** — Normal distribution-based KV quantization. Quantize based on normal distribution characteristics, use storage data types (NF4) aligned with distribution. Better than uniform quantization. *Paper:* 2025. *Limit:* assumes normal distribution.

- **Asymmetric NF4** — NF4 with per-channel mean subtraction. Subtract per-channel mean before NF4 quantization, add back after; handles DC offset in keys. Rescues collapse on high-ratio-GQA, 6.3×→7.9× compression improvement. *Paper:* TurboQuant 2024. *Limit:* requires mean computation/storage.

- **AWQ-KV** — Activation-aware weight quantization extended to KV cache. Leverages AWQ's activation-aware quantization principles for KV cache, focusing on channel-wise importance. 4-bit KV with minimal quality loss. *Limit:* requires calibration data.

- **SmoothQuant-KV** — SmoothQuant applied to KV cache activations. Smooths activation outliers by migrating quantization difficulty from activations to weights via mathematically equivalent transformation. 8-bit KV with minimal loss. *Limit:* originally designed for weights.

- **DuoQuant** — Dual quantization strategy for keys and values. Different quantization schemes for K vs V based on their distinct distributions and sensitivity. Effective 4-bit quantization. *Limit:* complexity of managing two quantization paths.

- **ZCache** — Zero-shot KV cache compression. Training-free compression using importance metrics without calibration. Competitive with calibrated methods. *Limit:* generally lower quality than calibrated methods.

- **RoCo** — Robust compression. Robust quantization/compression techniques resilient to outliers and distribution shifts. Better quality at same compression. *Paper:* Ren & Zhu, 2024. *Limit:* robustness techniques add overhead.

- **GEAR** — Error recovery framework for quantization. Augments quantization with low-rank matrix for error approximation + sparse matrix for outlier correction; near-lossless at high compression. 2.39× peak memory reduction, 2.1-5.07× throughput, 24.42% improvement over baselines at 2-bit. *Paper:* Kang et al., 2024. *Limit:* three-component complexity.

---

## 3.3 Paged / Disaggregated KV

- **PagedAttention (vLLM)** — OS-style virtual memory for KV cache. Split KV into fixed-size blocks, non-contiguous allocation, block tables per request, enables sharing and efficient reuse. 2-4× throughput improvement, near-zero waste, up to 24× on shared workloads. *Paper:* Kwon et al., 2023. *Limit:* requires custom CUDA kernels.

- **Block-Level Paging** — KV cache organized in blocks. Fixed-size blocks (typically 16 tokens) as allocation unit, enables flexible memory management. *Limit:* block size choice affects efficiency.

- **Virtual Memory for KV** — Apply OS virtual memory concepts to KV. Page tables, virtual-to-physical mapping, demand paging, copy-on-write for sharing. Enables prefix caching, COW for beam search. *Paper:* vLLM. *Limit:* page table overhead.

- **RadixAttention (SGLang)** — Radix tree for prefix sharing. Organize KV cache as radix tree (trie) where common prefixes are shared nodes; automatic detection and reuse. Significant memory savings for shared prefixes, 3-10× TTFT speedup. *Paper:* SGLang. *Limit:* tree management complexity.

- **Prefix Tree KV Reuse** — Trie-based sharing of KV states. Store KV in trie structure, shared prefixes stored once, reference counting. Memory savings proportional to prefix overlap. *Limit:* tree traversal overhead.

- **Mooncake (KV-centric arch)** — KVCache-centric disaggregated serving. Separate prefill/decode clusters, disaggregated KV cache pool using CPU/DRAM/SSD/NIC resources, KVCache-centric global scheduler. 59-498% effective capacity increase, 115%/107% more requests on A800/H800, 100B tokens/day production. *Paper:* FAST 2025. *Limit:* complex architecture.

- **DistServe** — Distributed serving with disaggregation. Separate prefill and decode onto different worker pools, KV cache transfer between them. Improved utilization for long-context workloads. *Limit:* transfer latency critical.

- **Splitwise** — Phase splitting (prefill vs decode). Separate machine pools for prompt processing (prefill) and token generation (decode), third mixed pool, KV cache transfer. Better latency at low rate, avoids fragmentation at high rate. *Paper:* ISCA 2024. *Limit:* transfer overhead.

- **Cascade Attention** — Multi-level cascade inference. KV cache stored in multi-level hierarchy (GPU/CPU/SSD), cascade attention across levels, merge results. Memory-efficient multi-level inference. *Paper:* FlashInfer cascade. *Limit:* merge overhead.

- **Cache-Aware Routing** — Route requests based on KV cache locality. Scheduler considers which nodes have relevant KV cache cached, routes to maximize cache hits. Significant TTFT improvement for cache hits. *Paper:* llm-d-router. *Limit:* requires global KV index.

- **KV Cache Offload to CPU/Disk** — Move KV cache from GPU to cheaper storage. Offload cold KV to CPU RAM or SSD, keep hot in GPU, load on demand. Extends effective capacity beyond GPU HBM. *Paper:* vLLM offloading, LMCache. *Limit:* load latency.

- **AttentionStore** — Persistent KV cache storage. Store KV cache persistently for reuse across sessions/queries. Enables cross-session reuse. *Limit:* storage management, staleness.

- **Cached KV Transfer Across Nodes** — Transfer KV cache between compute nodes. Move precomputed KV from one node to another instead of recomputing. Critical for disaggregated serving. *Paper:* Mooncake, LMCache. *Limit:* network bandwidth bottleneck.

- **KV Migration** — Move KV cache between storage tiers/nodes. Promote/demote KV based on access patterns, migrate for load balancing. Improves cache hit rates. *Paper:* Mooncake. *Limit:* migration overhead.

- **Redis-Style KV Store** — KV cache as key-value store. Treat KV cache entries as key-value pairs in a distributed store, use Redis-like semantics. Familiar abstraction, easy integration. *Limit:* may not optimize for LLM-specific patterns.

- **Distributed KV Pool** — Pool KV cache across multiple nodes. Aggregate KV cache from many nodes into single logical pool, distributed access. Unified KV pooling: 4.1× TTFT reduction, 23.2× I/O reduction. *Paper:* 2025. *Limit:* coordination overhead.

- **P-D Disaggregation** — Prefill-Decode disaggregation. Separate prefill (compute-bound) and decode (bandwidth-bound) into different resource pools. Better resource utilization, specialized hardware per phase. *Paper:* Mooncake, Splitwise. *Limit:* KV transfer critical path.

- **LMCache** — Efficient KV cache layer for enterprise. Extract KV from vLLM/SGLang, store outside GPU, share across engines/queries; supports offloading and PD disaggregation. 15× throughput improvement on multi-round QA/document analysis. *Paper:* 2025. *Limit:* requires connector integration.

- **Unified KV Pooling** — Aggregate multiple storage tiers into single pool. Pool CPU memory + SSDs into single logical KV pool, distribute based on bandwidth; KV-passthrough bypasses kernel filesystem. 4.1× TTFT reduction under 10s, 23.2× blocked I/O reduction. *Paper:* 2025. *Limit:* requires SPDK.

- **CacheGen** — KV cache compression and streaming. Compresses KV cache for efficient streaming between prefill and decode stages. Significant transfer speedup. *Paper:* 2024. *Limit:* requires streaming infrastructure.

- **CacheBlend** — Non-prefix KV cache fusion for RAG. Fuses multiple pre-computed KV caches (not just prefixes) via selective KV recompute on small fraction of tokens; enables pipelining with fetching. Higher quality than full reuse with minimal extra compute. *Paper:* 2024. *Limit:* requires identifying reusable chunks.

- **Blocked KV Caching** — Cache KV in blocks for efficient memory access. Store KV cache in fixed-size blocks, improve memory locality and management. *Paper:* DeepSpeed-MII, various systems. *Limit:* block size selection.

---

## 3.4 Long Context

### Position Encoding

- **RoPE (Rotary Position Embedding)** — Rotary position encoding via complex rotation. Rotate query/key vectors by angles proportional to position, relative position encoded in dot product. Dominant position encoding in modern LLMs. *Paper:* Su et al., 2021. *Limit:* fails to extrapolate beyond training length.

- **YaRN (Yet another RoPE extensioN)** — Efficient RoPE extension method. NTK-by-parts + attention temperature; partitions dimensions by wavelength for interpolation/extrapolation. 10× less tokens, 2.5× less steps than previous methods, extends to 128k. *Paper:* Peng et al., 2023. *Limit:* requires fine-tuning for best results.

- **NTK-Aware Scaling** — RoPE base scaling for extrapolation. Scale RoPE base (10000 → 10000·s^(d/(d-2))) to interpolate low frequencies heavily, high frequencies barely. No fine-tuning needed, used in Code Llama, Qwen. *Paper:* bloc97, 2023. *Limit:* under-scales lowest frequencies.

- **Position Interpolation (PI)** — Scale positions instead of extrapolating. Divide position by scale factor s, compress positions into trained range. Works but compresses high-freq dims (short-range quality drops). *Paper:* Chen et al., 2023. *Limit:* degrades short-range performance.

- **LongRoPE** — Long-context RoPE extension. Extended version of RoPE scaling for very long contexts. Supports very long contexts. *Limit:* complexity increases with length.

- **DualChunkAttention (DCA)** — Training-free long-context via chunk decomposition. Decompose attention into intra-chunk and inter-chunk modules, capture relative positions within/across chunks. Llama2 70B to 100k tokens without training. *Paper:* 2024. *Limit:* chunk size parameter.

- **3D-RoPE** — 3D rotary position encoding. Rotary encoding on 3D sphere instead of 2D circle, controllable long-term decay, improved resolution. Better than RoPE on long-context NLU. *Paper:* AAAI 2025. *Limit:* more complex computation.

- **Rotary on K Only** — Apply RoPE only to keys. Only rotate keys, queries unrotated or differently encoded. Some efficiency gains. *Limit:* may hurt performance, non-standard.

- **ALiBi (Attention with Linear Biases)** — Linear bias for position. Add linear penalty based on distance instead of explicit position encoding. Extrapolates better than absolute PE. *Paper:* Press et al. 2022. *Limit:* not as good as RoPE for most tasks.

- **NoPE (No Positional Encoding)** — Train without explicit position encoding. Let model learn position implicitly from token patterns. Outperforms ALiBi/RoPE/APE on length generalization in some studies. *Paper:* McGill-NLP 2023. *Limit:* may not work for all tasks.

- **Dynamic YaRN** — Dynamic scaling based on position. Adjust YaRN parameters dynamically based on position in sequence. Better extrapolation. *Paper:* Dynamic NTK variants. *Limit:* dynamic parameter overhead.

### Long-Context Architectures

- **LongLoRA (S2-Attn)** — Efficient long-context fine-tuning. Shifted sparse attention (S²-Attn) during training (sparse local), dense global during inference; LoRA with trainable embedding/norm. Llama2 7B from 4k to 100k, 70B to 32k on 8×A100, 16× computation savings. *Paper:* 2023. *Limit:* sparse attention during training only.

- **RingAttention** — Distributed attention with ring communication. Blockwise computation, distribute sequence across devices, overlap KV block communication with blockwise attention computation. Device count × longer sequences, millions of tokens possible. *Paper:* ICLR 2024. *Limit:* communication overhead.

- **Context Parallelism** — Parallelize across sequence dimension. Split sequence across devices, each device processes part, communicate for attention. Enables very long sequences. *Paper:* DeepSpeed-Ulysses, RingAttention, USP. *Limit:* communication bottleneck.

- **Landmark Attention** — Random-access infinite context via landmarks. Special landmark token per block represents block, attention to landmark selects relevant blocks, enables random access. Comparable to Transformer-XL with fewer retrieved tokens, extends LLaMA 7B to 32k. *Paper:* NeurIPS 2023. *Limit:* landmark selection critical.

- **InfLLM** — Infinite context via efficient attention. Various techniques for very long context (often landmark/sparse-based). Supports very long contexts. *Limit:* quality degradation at extreme lengths.

- **Anchor-based Attention** — Use anchor tokens for long-range. Select anchor tokens to represent distant context, attend to anchors instead of all tokens. Reduces quadratic complexity. *Limit:* anchor selection critical.

- **RMT (Recurrent Memory Transformer)** — Recurrent memory for long sequences. Special memory tokens passed between segments, enable recurrence for long context. Extends context without quadratic cost. *Limit:* recurrence adds complexity.

- **Block-Recurrent** — Block-level recurrence. Process in blocks with recurrent connections between blocks. Linear complexity with sequence length. *Limit:* recurrence overhead.

- **Longformer** — Sliding window + global attention. Local sliding window attention + few global tokens attending to all, linear complexity. Handles 4096 tokens vs 512 for BERT. *Paper:* 2020. *Limit:* global tokens limited.

- **BigBird** — Sparse attention (local + random + global). Mix local sliding window, random attention, and global tokens; theoretically universal approximator. 8× longer sequences than similar hardware. *Paper:* Zaheer et al., 2020. *Limit:* random attention may not be task-optimal.

- **Sparse Attention for Long Context** — Various sparse patterns. Only compute subset of attention pairs (fixed patterns, learned patterns, data-dependent). Reduces O(n²) to near O(n). *Limit:* may miss important pairs.

- **StreamingLLM (for long context)** — Streaming with attention sinks. Fixed-size cache with initial sink tokens + sliding window, enables infinite streaming. 5× compression, maintains quality. *Paper:* 2024. *Limit:* fixed window.

- **Sliding Window + Sink** — Combine sliding window with sink tokens. Keep initial sink tokens + recent sliding window, balance local and global. Standard approach for streaming. *Limit:* window size tradeoff.

- **Llama-3.1 128k** — Native 128k context in Llama 3.1. Trained with long context from scratch, no extension needed. 128k native context. *Paper:* Meta 2024. *Limit:* training cost, still quadratic attention.

- **Gemma 2M** — Gemma with 2M context. Extended context techniques for Gemma. 2M token context. *Paper:* Google. *Limit:* quality at extreme lengths.

- **Nemo Long Context** — NVIDIA's long context models. Various techniques for long context in Nemo models. *Paper:* NVIDIA. *Limit:* NVIDIA-specific.

- **Parallel Context Windows (PCW)** — Multiple parallel context windows. Process multiple context windows in parallel for efficiency. Speedup for certain workloads. *Limit:* memory overhead.

- **RAG vs Long Context Tradeoffs** — Compare retrieval-augmented generation vs long context. RAG: cheaper but may miss context; Long context: complete but expensive. RAG: O(k) where k is retrieved; Long: O(n) where n is full context. *Limit:* neither is universally better.

---

## 3.5 Architecture Innovations

### MoE (Mixture of Experts)

- **MoE (Mixture of Experts)** — Sparse activation of expert networks. Each token routed to subset of experts (top-k), only those experts activated, large capacity with constant compute. Mixtral: 47B params, 13B active; DeepSeek-V3: 671B total. *Paper:* Shazeer 2017, Mixtral 2023, DeepSeek 2024. *Limit:* routing complexity, load imbalance, training instability.

- **Mixtral** — 8×7B sparse MoE from Mistral. 8 experts per layer, top-2 routing, 47B total/13B active, matches Llama2 70B/GPT-3.5. 32k context, top-2 routing, 8 experts. *Paper:* Mistral 2024. *Limit:* load imbalance without auxiliary loss.

- **DeepSeek MoE** — Fine-grained MoE architecture. Finely segment experts into many small ones, isolate shared experts for common knowledge, auxiliary-loss-free training. 2B matches GShard 2.9B with 1.5× fewer params, 40% computation of dense. *Paper:* 2024. *Limit:* many experts increase routing overhead.

- **Sparse MoE** — Sparse activation of experts. Only activate subset of experts per token, not all. Standard in modern MoE. *Limit:* load balancing, routing quality.

- **Fine-Grained Experts** — Many small experts instead of few large. More experts with smaller capacity per expert for finer-grained specialization. DeepSeek uses 64+ experts per layer. *Paper:* DeepSeek. *Limit:* routing overhead.

- **Shared Experts** — Experts shared across all tokens. Some experts always activated (shared) for common knowledge, others routed for specialized knowledge. DeepSeek uses shared experts. *Paper:* DeepSeek. *Limit:* shared experts become bottleneck if not sized properly.

- **DeepSeek-V2 MoE** — MLA + fine-grained MoE. Combines Multi-head Latent Attention with fine-grained MoE architecture. 236B total, 21B active. *Paper:* 2024. *Limit:* complex architecture.

- **Expert Routing** — Mechanism to select experts per token. Router network outputs logits, top-k selection, various routing strategies. Critical for MoE performance. *Limit:* routing quality, load balance, collapse.

- **Top-k Gating** — Select top-k experts per token. Router outputs scores, select k highest, renormalize weights. Mixtral uses top-2, typical k=1-4. *Limit:* load imbalance, k choice tradeoff.

- **Softmax vs Sigmoid Gating** — Different gating functions. Softmax: normalized probabilities; Sigmoid: independent expert selection. Softmax standard, sigmoid for independent selection. *Limit:* softmax encourages competition, sigmoid allows overlap.

- **Expert Balancing** — Ensure even expert utilization. Auxiliary loss (load balancing), bias-based methods, capacity factors. Critical for training stability. *Paper:* Switch Transformer, ϕ-balancing, DUAL. *Limit:* conflicts with primary loss.

- **DeepSeek-V3 MoE** — 671B parameter MoE. 256 experts per layer, top-8 routing, auxiliary-loss-free with capacity limits. 671B total, 37B active, competitive with GPT-4. *Paper:* 2024. *Limit:* massive scale, routing complexity.

- **Qwen MoE** — Alibaba's MoE models. Various MoE architectures in Qwen family. *Paper:* Alibaba. *Limit:* proprietary details limited.

- **OLMoE** — Open MoE model from AllenAI. Fully open-source MoE with various configurations. 1B+ parameters. *Paper:* AllenAI 2024. *Limit:* smaller scale.

- **Expert Balancing (ϕ-balancing, DUAL)** — Advanced load balancing for MoE. ϕ-balancing: principled population-level balance via convex optimization; DUAL: Lagrange dual-based bias updates with sparsemax. ϕ-balancing outperforms Switch baselines; DUAL prevents gating thrash. *Paper:* ICML 2026 (ϕ), various (DUAL). *Limit:* more complex than simple auxiliary loss.

- **MoE-Dense Hybrid** — Mix MoE and dense layers. Some layers MoE, some dense, balance capacity and efficiency. Used in many modern MoE models. *Limit:* architecture complexity.

### Attention Variants (Architecture)

- **MLA (Multi-head Latent Attention)** — Low-rank latent KV compression. Project Q/K/V to low-dimensional latent, cache latent instead of full K/V, reconstruct on-demand. 28.44× fewer values than MHA, 4× more compute but higher throughput. *Paper:* DeepSeek-V2 2024. *Limit:* reconstruction overhead, requires custom kernels.

- **GQA (Grouped-Query Attention)** — Share K/V across query head groups. Group query heads, each group shares one K/V pair, reduces KV cache by factor of groups. Llama2-70B uses GQA, 4-8× KV reduction. *Paper:* Ainslie et al., 2023. *Limit:* some quality loss vs MHA.

- **MQA (Multi-Query Attention)** — Single K/V shared across all heads. All query heads share one K/V pair, extreme KV reduction. KV cache 1/h of MHA. *Paper:* Shazeer 2019. *Limit:* quality degradation, training instability.

- **MTP (Multi-Token Prediction)** — Predict multiple future tokens. Add heads to predict tokens at t+1, t+2, etc., used for speculative decoding. 2-3× speedup in self-speculative decoding. *Limit:* training complexity.

### Normalization

- **RMSNorm** — Root Mean Square Normalization. Simplified LayerNorm without mean subtraction, only RMS scaling. 50% fewer parameters than LayerNorm, standard in modern LLMs. *Paper:* Zhang & Sennrich 2019. *Limit:* slightly less expressive.

- **LayerNorm** — Standard layer normalization. Normalize by mean and variance per token. Original Transformer norm. *Paper:* Vaswani 2017. *Limit:* more parameters than RMSNorm.

- **DeepNorm** — Deep network normalization. Modified normalization for very deep networks, improves stability. Enables deeper networks. *Paper:* 2022. *Limit:* more complex.

- **QK-Norm** — Normalize Q and K before attention. Apply RMSNorm to Q and K inside attention, stabilizes training. Used in Gemma 2/3, OLMo 2. *Limit:* extra computation.

- **Normalization Placement (Pre/Post/Parallel)** — Where to place normalization. Pre-norm: before sublayer (standard); Post-norm: after; Parallel: norm applied to parallel branches. Pre-norm standard for stability. *Limit:* post-norm unstable deep.

- **Dual-Norm** — Dual normalization layers. Two normalization layers (e.g., pre and post, or different types). Used in Gemma 2/3 for stability. *Paper:* Gemma team. *Limit:* extra computation.

### Activations & FFN

- **SwiGLU** — Gated activation function. Swish gating with GLU structure: Swish(xW) ⊙ (xV). Standard in Llama, Qwen, DeepSeek. *Paper:* Shazeer 2020. *Limit:* more parameters than standard FFN (3× vs 2×).

- **GeGLU** — GELU-based GLU. GELU gating instead of Swish: GELU(xW) ⊙ (xV). Used in T5, PaLM, Gemma. *Limit:* similar parameter overhead.

- **Gating** — Gated linear units. Element-wise multiplication of gated and non-gated paths for expressive computation. Standard in modern FFNs. *Limit:* parameter overhead.

### Speculation Heads

- **Medusa Heads** — Multiple decoding heads for speculation. Add extra heads to predict multiple future tokens, tree-based attention, verify candidates. 2.2× speedup (Medusa-1), 2.3-2.8× (Medusa-2). *Paper:* 2024. *Limit:* requires training heads.

- **MTP Heads** — Multi-token prediction heads. Heads predict tokens at t+1, t+2, etc., used for self-speculative decoding. DeepSeek-V3 uses MTP, 2-3× speedup. *Limit:* training complexity.

### Embedding / Output

- **Tied Embeddings** — Share input and output embeddings. Use same embedding matrix for token input and output projection. Reduces parameters by vocab size. *Limit:* may hurt performance.

- **Untied Head** — Separate embeddings per head. Different embedding/output projections per attention head. More expressive, used in some models. *Limit:* parameter overhead.

- **Logit Capping** — Cap logits before softmax. Clamp logits to prevent extreme values, improves numerical stability. Used in Gemma 2/3. *Limit:* may affect distribution.

- **Z-Loss** — Penalize large logit magnitudes. Add term proportional to log(sum-exp(logits)) to loss, keeps logits bounded. Critical for MoE stability, coefficient ~0.0001 typical. *Paper:* PaLM, DeepSeek. *Limit:* hyperparameter tuning.

### Block / Layer Design

- **Parallel Attention (GPT-J style)** — Attention and FFN in parallel. Compute attention and FFN in parallel then add, instead of serial. ~15% throughput improvement in GPT-J. *Paper:* GPT-J 2021. *Limit:* more memory.

- **Layer Skip / Early Exit** — Exit early from network. Layer dropout during training (low early, high later), early exit loss, exit at early layers during inference. 2.16× speedup on summarization, 1.82× on coding. *Paper:* Meta LayerSkip 2024. *Limit:* requires special training.

- **Depth-Wise Stacking** — Depth-specific architectures. Different layer types or configurations at different depths. Used in some hybrid models. *Limit:* architecture complexity.

- **Sliding Window Attention (Gemma, Mistral)** — Local attention window. Each token attends only to local window (e.g., ±4096 tokens), reduces complexity. Gemma/Mistral use sliding window in some layers. *Limit:* loses long-range dependencies.

- **Manifold-Constrained Hyper-Connections (mHC)** — Replace residual connections in DeepSeek-V4. Parallel residual streams with manifold constraints, improved optimization. *Paper:* DeepSeek-V4 2025. *Limit:* DeepSeek-specific.

---

# Part 4 — Summary Tables & Cross-Cutting Insights

## 4.1 Quantization Format Comparison

| Format | Bits | Memory Reduction | Speedup | Hardware Support | Best Use Case |
|--------|------|------------------|---------|-----------------|--------------|
| FP16 | 16 | 1× | 1× | Universal | Baseline |
| INT8 | 8 | 2× | 2-4× | Universal | General deployment |
| INT4 | 4 | 4× | 2-3× | Most GPUs | Memory-constrained |
| NF4 | 4 | 4× | 2-3× | NVIDIA | QLoRA training |
| FP8 E4M3 | 8 | 2× | 2-4× | H100+ | Training |
| FP8 E5M2 | 8 | 2× | 2-4× | H100+ | Inference |
| MXFP8 | 8 | 2× | 4×+ | Blackwell | Next-gen deployment |
| MXFP4 | 4 | 4× | 4×+ | Blackwell | Extreme compression |
| NVFP4 | 4 | 4× | 4×+ | Blackwell | KV cache |
| GGUF Q4_K_M | 4.5 | 4.5× | CPU fast | CPU/Apple | Local inference |
| EXL2 | Variable | Target bpw | GPU fast | NVIDIA | GPU serving |
| BitNet 1.58 | 1.58 | 10× | 2.65× CPU | Custom kernels | Edge / extreme |
| Ternary | 1.58 | 10× | 2.65× CPU | Custom kernels | Native-trained models |

## 4.2 Pruning Method Comparison

| Method | Sparsity | Type | Accuracy | Speedup | Best For |
|--------|----------|------|----------|---------|----------|
| Magnitude | 50% | Unstructured | Poor | None (no kernel) | Simple baseline |
| SparseGPT | 50% | Unstructured | Good | None (no kernel) | Research |
| Wanda | 50% | Unstructured | Better | None (no kernel) | Accuracy-focused |
| SliceGPT | 25% | Structured | 99% | 34-36% | Production |
| SVD-LLM | Variable | Structured | Good | Linear | High compression |
| ShortGPT | Layer-wise | Structured | Good | Linear | Long-context |
| LoRAPrune | 50% | Structured | Good | Linear | LoRA models |
| 2:4 Semi-structured | 50% | Structured | Moderate | 2× | NVIDIA GPUs |
| MaskLLM | 50% (2:4) | Semi-structured | Good | 2× | Learned masks |

## 4.3 LoRA Variant Comparison

| Variant | Parameters | Memory | Speed | Accuracy | Best For |
|---------|------------|--------|-------|----------|----------|
| LoRA | 0.1-1% | Low | Same | Good | General |
| QLoRA | 0.1-1% | Very Low | Same | Good | Memory-constrained training |
| DoRA | 0.1-1% | Low | Same | Better | High accuracy |
| rsLoRA | 0.1-1% | Low | Same | More stable | High ranks |
| LoRA+ | 0.1-1% | Low | 2× faster | Better | Fast training |
| PiSSA | 0.1-1% | Low | Faster convergence | Better | Fast convergence |
| AdaLoRA | Variable | Low | Same | Good | Dynamic allocation |
| GaLore | Full | 65% less optimizer | Same | Full-rank | Pre-training |
| VeRA | <0.1% | Very Low | Same | Good | Extreme param reduction |
| LoftQ | 0.1-1% | Very Low | Same | Better at 2-bit | Quantized base |

## 4.4 Model Merging Comparison

| Method | Models | Interference Handling | Complexity | Best For |
|--------|--------|----------------------|------------|----------|
| Linear | 2+ | None | Low | Simple blending |
| SLERP | 2 | None | Low | 2-model interpolation |
| Task Arithmetic | 2+ | Poor | Medium | Compatible tasks |
| TIES | 2+ | Sign consensus | Medium | Multi-task conflicts |
| DARE | 2+ | Random pruning | Medium-High | Instruction-following |
| Model Soups | 2+ | Averaging | Low | Same-base fine-tunes |

## 4.5 Speculative Decoding Comparison

| Method | Speedup | Training Required | Draft Model | Best For |
|--------|---------|-------------------|-------------|----------|
| Classic SD | 2-4× | Yes (draft) | Separate | General |
| Medusa | 2.2-2.8× | Yes (heads) | None (heads) | Single-model |
| EAGLE | 2.5-3× | Yes (head) | None (head) | Feature-level |
| EAGLE-2 | 3-4.3× | Yes (head) | None (head) | Dynamic trees |
| EAGLE-3 | Up to 6.5× | Yes (head) | None (head) | Multi-layer fusion |
| Lookahead | 1.8-4× | No | None | Training-free |
| LayerSkip | 1.34-2.16× | Yes (recipe) | Same model | Self-speculative |
| REST | 1.6-2.4× | No | Datastore | Code/text retrieval |
| OSD | 1.42-2.17× | Online | Separate | Distribution shift |

## 4.6 Attention Variant Comparison

| Variant | KV Cache | Quality | Compute | Best For |
|---------|----------|---------|---------|----------|
| MHA | O(H·d) | Best | O(N²·H·d) | Quality-critical |
| MQA | O(d) | Worst | O(N²·d) | Extreme cache cut |
| GQA | O(G·d) | Near-MHA | O(N²·G·d) | Production balance |
| MLA | O(d_c) | Good | O(N²·d_c) + recon | DeepSeek-style |
| Sliding Window | O(W) | Local only | O(N·W) | Streaming |
| Linear Attn | O(d²) | Lower | O(N·d²) | Very long seq |
| Mamba/SSM | O(d) (state) | Varies | O(N·d) | Hybrid models |

## 4.7 KV Cache Compression Comparison

| Method | Compression | Quality | Training | Best For |
|--------|-------------|---------|----------|----------|
| H2O | 5× | Good | No | General eviction |
| StreamingLLM | 5× | Good (streaming) | No | Infinite streaming |
| SnapKV | 8.2× | Good | No | Observation-based |
| PyramidKV | 8.3× | Better | No | Layer-aware |
| KIVI (2-bit) | 2.6× | Near-lossless | No | Quant + eviction |
| KVQuant | 10×+ | Good | Yes (calib) | Extreme long ctx |
| GEAR | 2.4× | Near-lossless | No | 2-bit recovery |
| ThinK | 20%+ | Good | No | Channel pruning |
| Quest | 7× | Good | No | Page-aware |
| AdaKV | Variable | Best | No | Head-adaptive |

## 4.8 Inference Engine Comparison

| Engine | Throughput (Llama-70B) | Hardware | Best For |
|--------|------------------------|----------|----------|
| vLLM | 1000-2000 tok/s | NVIDIA | General serving |
| TensorRT-LLM | 2500-4000+ tok/s | NVIDIA only | Max NVIDIA throughput |
| sglang | 16200 tok/s (8B) | NVIDIA | Multi-turn, prefix-heavy |
| LMDeploy | 16100 tok/s (8B) | NVIDIA | C++ engine, Int4 |
| ExLlamaV2 | High | NVIDIA consumer | Consumer GPUs |
| llama.cpp | 80-100 tok/s (edge) | All | Portability, edge |
| mlx | 35-60 tok/s | Apple Silicon | Mac native |
| PowerInfer | 11.69× vs llama.cpp | CPU+GPU | Hybrid offload |
| TGI | 800-1500 tok/s | NVIDIA | (Archived 2026) |

## 4.9 Cross-Cutting Insights

1. **Quantization hierarchy**: FP16 → INT8 → INT4 → ternary is the progression, but accuracy drops sharply below 4-bit without native training (BitNet).

2. **Hardware matters**: MX formats (Blackwell), 2:4 sparsity (Ampere+), FP8 (H100+) require specific hardware. CPU inference favors GGUF.

3. **PTQ vs QAT**: PTQ (GPTQ, AWQ, SmoothQuant) is faster but QAT (training-aware) achieves better accuracy at extreme compression.

4. **LoRA dominance**: LoRA and variants are the de facto standard for PEFT due to no inference overhead and strong performance.

5. **Merging is powerful**: Model soups, TIES, DARE enable combining capabilities without retraining - a paradigm shift.

6. **Cross-arch porting**: TransMLA, GQA upcycling enable migrating to efficient architectures without full retraining.

7. **Distillation evolution**: From logits → hidden states → sequence-level → speculative → online, continuously improving efficiency.

8. **Sparsity reality**: Unstructured pruning has limited speedup without hardware support. Semi-structured (2:4) is the practical path.

9. **KV cache critical**: For long context, KV quantization (QServe, QuaRot) is as important as weight quantization.

10. **System co-design**: Best results (QServe, EXL2, vLLM) come from algorithm + system co-design, not just algorithms.

11. **Speculative decoding convergence**: EAGLE-3 (6.5×) shows the field is converging on feature-level, multi-layer, dynamic-tree approaches.

12. **Disaggregation is the future**: Mooncake, DistServe, Splitwise show prefill/decode disaggregation as the path to scale.

13. **Hybrid architectures win**: Jamba, Griffin, LFM2 show SSM/conv + attention hybrids match quality with linear complexity.

14. **PagedAttention revolutionized serving**: vLLM's OS-style paging is now table stakes for any modern engine.

15. **MLA is the new GQA**: DeepSeek's MLA gives 28× KV compression with quality retention, becoming the new standard.

---

# Part 5 — ForgeAI Relevance Map

Mapping techniques to the ForgeAI project (Qwen2.5-Coder-1.5B → 360M MLA research model). See `AGENTS.md` for current implementation status.

## 5.1 Already Implemented in ForgeAI

| Technique | ForgeAI Module | Status |
|-----------|---------------|--------|
| MLA (Multi-head Latent Attention) | `research/convert_key_svd.py` | WORKING (cos=0.9999) |
| MoE (dense → routed experts) | `research/convert_key_svd.py` | WORKING (cos=1.0) |
| BitNet (ternary {-1,0,+1}) | `research/convert_keys.py` | WORKING (26% smaller) |
| SVD resize (large → small) | `research/convert_key_svd.py` | WORKING |
| Liquid (LFM2 conv+attn hybrid) | `research/liquid.py` | IMPLEMENTED |
| DSpark (speculative decoding) | `research/dspark.py` | IMPLEMENTED (60-85% speedup) |
| RotorQuant (Givens KV compression) | `research/rotorquant.py` | IMPLEMENTED (3.88×, 0.94% error) |
| DoRA fine-tuning | `research/dora.py` | IMPLEMENTED |
| QLoRA-style 8-bit AdamW | `research/train.py`, `research/sft_align.py` | Default (bnb) |
| EMA weight averaging | training scripts | IMPLEMENTED (15% boost) |
| YaRN 4x context extension | `research/train.py` | IMPLEMENTED (--yarn-factor) |
| GaLore optimizer | `research/train.py` | IMPLEMENTED (--optimizer galore) |
| RMSNorm | `research/train.py` | IMPLEMENTED (--norm-type rmsnorm) |
| GateSkip + MTP + MoE + BitNet combo | `research/train.py` | IMPLEMENTED |
| Speculative decoding (draft model) | `research/speculative_decode.py` | IMPLEMENTED |
| Online speculative distillation | `research/speculative_decode.py --online` | IMPLEMENTED |
| Multi-teacher distillation | `research/distill_multi.py` | IMPLEMENTED |
| Sequence-level distillation | `research/distill_synthetic.py` | IMPLEMENTED |
| Online learning + replay | `research.online_learn.py` | IMPLEMENTED (24.3% ppl reduction) |
| Hidden-state distillation loss | `research/distill.py` | IMPLEMENTED (default path) |
| Top-K KL distillation | `research/distill.py --top-k-kl` | IMPLEMENTED (1500× faster) |
| Chunked CE | `research/chunked_ce.py` | IMPLEMENTED (saves 1.86 GB VRAM) |
| Activation checkpointing | `research/train.py` | IMPLEMENTED |
| Safetensors checkpoint format | `research/checkpoint_io.py` | IMPLEMENTED |
| Curriculum learning | `research/quality_score.py` | IMPLEMENTED |
| Quality scoring (6-dim) | `research/quality_score.py` | IMPLEMENTED |
| Chat template reformatting | `research/reformat_chat.py` | IMPLEMENTED |
| Crash recovery + safeguards | `research/training_utils.py` | IMPLEMENTED (all scripts) |
| VRAM watchdog | training scripts | IMPLEMENTED (--vram-limit-gb) |
| NaN detection | training scripts | IMPLEMENTED |
| Teacher compile | `research/distill.py --teacher-compile` | IMPLEMENTED (~1.3× teacher fwd) |
| Weight tying (embed/lm_head) | model architecture | IMPLEMENTED |
| GQA (Qwen native) | ported from Qwen | Native |
| SwiGLU (Qwen native) | ported from Qwen | Native |
| RoPE (Qwen native) | ported from Qwen | Native |

## 5.2 High-Value Candidates to Add Next

### Weight Tech
- **GPTQ / AWQ PTQ** — Drop-in 4-bit quant for the 360M model for deployment. Easy win.
- **EXL2 mixed-precision** — Variable bitrate for consumer GPU serving.
- **SpinQuant / QuaRot** — Rotation-based W4A4KV4, would pair with existing BitNet path.
- **SliceGPT** — SVD structured pruning to drop 360M → 270M with 99% quality.
- **PiSSA LoRA init** — Better convergence than vanilla LoRA for SFT.
- **Model merging (TIES/DARE)** — Combine multi-teacher distilled checkpoints.

### Runtime
- **EAGLE-3** — 6.5× speedup, feature-level + multi-layer fusion. Highest-impact single addition.
- **Medusa heads** — Simpler than EAGLE, no separate draft model.
- **FlashAttention 3** — If Blackwell available, 1.5-2× over FA2.
- **FlexAttention** — Custom masks (sliding window, document masking) at FA speed.
- **PagedAttention-style KV management** — For batched serving of the 360M model.
- **Continuous batching** — If serving multiple requests.
- **Prefix caching / RadixAttention** — Huge for multi-turn chat workloads.
- **Chunked prefill** — Better TTFT under load.

### KV Cache
- **SnapKV / PyramidKV** — 8× KV compression, training-free, drop-in for long context.
- **KIVI (2-bit KV)** — 2.6× memory, pairs with existing RotorQuant.
- **H2O eviction** — Simple baseline for streaming workloads.
- **StreamingLLM (sink + window)** — Infinite context for chat.
- **Quest** — Page-aware, 7× speedup for self-attention.

### Architecture
- **Native Sparse Attention (DeepSeek NSA)** — Hardware-aligned sparse, trainable.
- **MoBA** — Block attention routing, 80% sparsity.
- **Mamba2 / GLA hybrid layers** — Replace ~50% of attention with linear complexity.
- **3D-RoPE** — Better long-context NLU than 2D RoPE.
- **QK-Norm** — Training stability, used in Gemma 2/3.
- **Logit capping + Z-Loss** — Stability for MoE training.
- **ϕ-balancing / DUAL** — Better MoE load balancing than auxiliary loss.

### Distillation
- **MiniLLM (reverse KLD)** — Better calibration than forward KL.
- **SpecKD (token gating)** — Apply KD only to accepted tokens.
- **Layer-wise hidden state KD** — Better feature transfer.

## 5.3 Research Frontier (2025-2026)

- **DeepSeek-V4 architecture** — mHC residual, CSA + HCA attention pools, lightning indexer.
- **Mooncake KV-centric serving** — 100B tokens/day production, disaggregated KV pool.
- **EAGLE-3 + MTP combination** — Speculation from native MTP heads.
- **TransMLA / CARE** — Convert existing GQA checkpoints to MLA (ForgeAI already does this manually).
- **AdaKV / EvolKV** — Head-adaptive / evolutionary KV budget allocation.
- **Unified KV Pooling** — 4.1× TTFT, 23.2× I/O reduction across tiers.
- **PowerAttention** — Exponential receptive fields, 3× on 128K.
- **Hetis heterogeneous parallelism** — 2.25× throughput on mixed GPU clusters.

---

# Part 6 — 2026 Addendum (compiled 2026-08-06)

> Recent techniques not in Parts 1-5, from ACL 2026 / arXiv 2026. Each entry: **What** / **Key Idea** / **Numbers** / **Limitations** / **Novel-combination potential**.

## 6.1 Sparse Attention: MoE-ification of the Head Axis

- **MISA (Mixture of Indexer Sparse Attention)** — Treats DeepSeek Sparse Attention's `H^I` indexer heads as a MoE pool; a lightweight router picks `h << H^I` active heads per query using cheap block-level stats. Cost `O(H^I·L)` → `O(h·L + H^I·M)`, `M=⌈L/B⌉`. *Numbers:* 8× fewer indexer heads, matches dense DSA on LongBench, 3.82× speedup on H200, 92% token-overlap with full indexer. *Limit:* needs router training/calibration. *Novel:* MoE on the **attention head axis**, orthogonal to token-axis hierarchies — composable with ForgeAI's existing MoE-FFN.

- **Sparse Frontier (ACL 2026 findings)** — Largest training-free sparse-attn eval (6 methods, ≤128K, sparsity ≤0.95, 9 tasks). *Findings:* (1) larger-sparse beats smaller-dense at equal cost (Pareto shift); (2) prefill fine-grained per-query estimation is impractical → choose global-to-token vs block-to-block per task; (3) decoding token-to-page selection is feasible and generalizes; (4) longer sequences tolerate higher sparsity → **fixed-budget methods are suboptimal**. *Novel:* motivates adaptive/length-aware sparsity budgets.

- **Alloc-MoE (ACL 2026)** — Budget-aware expert activation. Alloc-L (layer-level: sensitivity profiling + dynamic programming for optimal per-layer budget) + Alloc-T (token-level: dynamic redistribution by routing scores). *Numbers:* 1.15× prefill / 1.34× decode at **half** the activation budget on DeepSeek-V2-Lite. *Limit:* sensitivity profiling offline. *Novel:* non-uniform expert budget across layers — pairs with ForgeAI Expert Consolidation.

- **SpecMoE** — Self-assisted speculative decoding for CPU-offloaded MoE. No extra training. *Numbers:* up to 4.30× throughput, reduced memory/interconnect bandwidth. *Novel:* spec-decode **hides expert load latency** — directly complementary to AirMoE's disk-transfer bottleneck.

- **LightMoE (ACL 2026 findings)** — Task-aware expert availability for edge MoE. Frequency-aware resident core experts + **similarity-based redirection** (serve a similar resident expert instead of loading the missing one, no I/O) + coarse task-level replacement. *Novel:* redirect-to-similar eliminates the load entirely for near-matches — pairs with Expert Consolidation (merged experts become the redirect targets).

## 6.2 Test-Time Training (TTT) for Reasoning

- **TEMPO** — Scaling TTT via Expectation-Maximization. Interleave policy refinement on unlabeled questions with periodic critic recalibration on a labeled set; prior methods are "incomplete EM" missing the recalibration step. *Numbers:* OLMO3-7B AIME 33.0→51.1%, Qwen3-14B 42.3→65.8%, preserves diversity. *Limit:* needs a labeled recalibration set. *Novel:* EM framing gives a principled stopping/recalibration criterion for self-play loops.

- **Policy of Thoughts (PoT)** — Per-instance TTT. MCTS generates candidates → GRPO updates a **transient LoRA adapter** using execution feedback → adapter discarded after solving (base untouched). *Numbers:* 4B model 49.71% LiveCodeBench, beats GPT-4o/DeepSeek-V3 at 50× smaller. *Novel:* throwaway per-instance LoRA = "compute-to-quality" without permanent weight change — composable with GRAIL compensation to fold successful adapters back in.

- **ORCA** — Conformal prediction + TTT for calibration. Meta-learned calibration module updated per input; valid confidence under distribution shift. *Numbers:* Qwen2.5-32B 47.5% in-dist savings, MATH-500 24.8→67.0% OOD savings at δ=0.1. *Novel:* conformal guarantees on when to stop sampling — pairs with self-play confidence filtering.

- **DiSCTT** — Difficulty-aware consensus-guided self-curriculum. High-consensus inputs → SFT with majority pseudo-labels; low-consensus → RL with consensus-regularized diversity objective. *Novel:* routes by epistemic uncertainty — directly applicable to ForgeAI self-play (`recursive_self_play.py`).

- **Decision-Theoretic TTT** — TTT = implicit Bayesian inference in the kernel regime. Spectrally match updates to the prompt's SNR; align to query-relevant eigen-directions. PAC-Bayes guarantee on step selection; Bayes-optimal update subspace yields a scoring rule for **which Transformer blocks/heads to adapt**. *Novel:* principled block/head selection for TTT — tells you *where* in the model to update, not just how much.

## 6.3 Model Merging: Theory & LoRA-Specific Methods

- **Task Vectors = Gradients (PMLR 2026)** — A task vector from one epoch of finetuning is *exactly* `-lr · ∇loss`; multi-epoch holds approximately with a bounded 2nd-order term. First-epoch gradient dominates the trajectory. *Implication:* **single-epoch finetune merges ≈ converged merges**; merging = approximate multitask learning. *Novel:* justifies cheap 1-epoch expert/LoRA training for merging pipelines.

- **Pico** — LoRA merge interference comes from the **B (output-side) matrix**, not A. B reuses a small set of shared directions across tasks → merged adapter overemphasizes them, losing task-specific info. Data-free: downscale B on shared directions, rescale after. *Numbers:* +3.4–8.3 points over Task Arithmetic/TIES/TSV-M across 8 benchmarks. *Novel:* **treat A and B separately** when merging — applies to ForgeAI DoRA (which decomposes magnitude/direction).

- **SVD+CUR LoRA Merging (ACL 2026)** — SVD captures shared structure, CUR preserves task-specific/localized updates; geometrically misaligned and complementary. Training-free combine. *Novel:* two-decomposition fusion beats single-decomposition merging.

- **LoRA Soups / CAT (COLING 2025)** — Concatenation of LoRAs with optimal weighting beats model- and data-merging for skill composition. Math+code: +43% over model-merge, +12% over data-merge. *Novel:* CAT (concatenation) > averaging for compositional tasks.

- **CT-Merging** — Consensus directions from average task subspace projectors + task-level RMS coefficient scales. +2.56 over DC-Merge. *Novel:* coefficient rescaling after basis construction.

## 6.4 Continual Learning: Forgetting Mechanics & Replay

- **FOREVER (ACL 2026)** — Ebbinghaus forgetting curve for LLM CL. **Model time = magnitude of optimizer updates** (not raw steps); forgetting-curve replay scheduler + intensity-aware regularization. 0.6B–13B. *Novel:* model-centric time aligns replay to actual parameter drift.

- **Self-Generated Replay (arxiv 2605.26097)** — LLM samples its own training distribution as replay → nearly eliminates forgetting **when capacity is not saturated**. Forgetting persists when pretrained close to saturation. Replay breaks the low-lr/many-steps tradeoff. *Novel:* capacity is the real constraint → **MoE/free-dimension methods (Fact Injection, Context Patch) reduce forgetting by adding capacity, not just replaying**.

- **MSSR** — Sample-level memory strength + adaptive rehearsal intervals for continual fine-tuning. *Novel:* per-sample, not per-task, replay scheduling.

- **OAKS Benchmark (ACL 2026)** — Online adaptation to continually evolving facts. 14 models + agentic memory systems **fail** at streaming fact updates (delays, distraction). *Novel:* even RAG/memory-agents fail → motivates closed-form fact injection (ForgeAI `fact_injection_key.py`) as the fix.

- **Mechanistic Forgetting (arxiv 2601.18699)** — Early-layer attention heads = entropic dispersion; mid-deep FFN/MoE expert blocks = localized representation collapse (CKA + routing-gate drift). *Novel:* **forgetting is layer-localized** → protect/merge only the susceptible layers, freeze the rest during continual updates.

---

## Document Stats

- **Total techniques documented**: 320+ across 6 parts
- **Weight tech**: 90+ (quantization, pruning, LoRA, transforms, distillation)
- **Runtime tech**: 100+ (speculative decoding, attention kernels, engines, parallelism, prefill/decode)
- **KV cache + architecture**: 110+ (compression, quantization, paging, long context, arch innovations)
- **Sources**: Web research across 2020-2026 papers, production systems, and emerging techniques

*Compiled 2026-07-30 by parallel research subagents. See `AGENTS.md` for ForgeAI implementation status and `docs/keys/KEY_DEFINITION.md` for the weight extraction theory.*
