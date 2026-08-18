# FFN Research Report — 2026-08-18

## Summary

Deep dive into Feed-Forward Network (FFN) optimization for LLM inference.
Our FFN-SkipLLM implementation was **fundamentally wrong** — used the wrong
metric and skipped the wrong layers. This report corrects that.

---

## 1. FFN-SkipLLM (EMNLP 2024) — The Actual Method

**Paper**: "FFN-SkipLLM: A Hidden Gem for Autoregressive Decoding with
Adaptive Feed Forward Skipping" (Jaiswal et al., UT Austin)

### Key Insight: Cosine Similarity, NOT RMS Norm

The paper uses **cosine similarity between FFN input and FFN output** as the
skip metric — NOT the RMS norm of the FFN input (which is what we implemented).

- **High cosine similarity** = FFN barely changes the representation = redundant
- **Low cosine similarity** = FFN significantly transforms the input = important

### Cold vs Non-Cold Regions

The paper identifies two regions in the model:
- **Cold regions** (first few + last few layers): Low cosine similarity.
  FFN is important. **Never skip these.**
- **Non-cold region** (middle layers): High cosine similarity.
  FFN is redundant. **Safe to skip here.**

This is the **opposite** of what we did — we skipped early layers (cold region)!

### Warm-Up Tokens

First 5-10% of generated tokens use the full model (no skip) to stabilize
the KV cache via the attention sink phenomenon. After warm-up, FFN skipping
begins.

### Input-Adaptive Algorithm

```
for each token:
    if token_index <= warm_up_index:
        run full model (no skip)
    else:
        for layers 0..cold_s:        # cold start region
            run attention + FFN
        for layers cold_s..cold_e:   # non-cold (skippable) region
            run attention
            if not already skipping:
                compute FFN output
                sim = cosine(FFN_input, FFN_output)
                if sim >= threshold:
                    start skipping (greedy k layers)
            else:
                skip FFN
        for layers cold_e..N:        # cold end region
            run attention + FFN
```

### Results (LLaMa-2 7B/13B)

| Skip Ratio | Factoid-QA | Multi-turn | Summarization |
|------------|-----------|------------|---------------|
| Full model | 79.02%    | 7.61       | 8.15          |
| 5% skip    | 80.05%    | -          | -             |
| 15% skip   | 78.42%    | -          | -             |
| 25% skip   | 78.09%    | 7.55       | 8.11          |
| 35% skip   | 75.61%    | -          | -             |

At 25% skip: only 0.93% drop on Factoid-QA. At 5% skip: **improves** performance!

### Why It Works

- FFN blocks hold ~2/3 of layer parameters (Table 1: 135M FFN vs 67M attn in LLaMa-7B)
- Skipping FFN doesn't touch KV cache (unlike layer-skip/early-exit)
- No hallucination or token collapse (the main issues with layer-skipping)
- Monotonically increasing cosine similarity in middle layers = increasingly redundant

---

## 2. Layerwise FFN Importance (COLM 2025)

**Paper**: "Layerwise Importance Analysis of Feed-Forward Networks in
Transformer-based Language Models" (Ikeda et al., Tohoku University)

### Key Finding: Middle Layers Matter Most

Tested 285M, 570M, and **1.2B** parameter models (same scale as ForgeLM V3!):

- **Most important**: Middle 70% of layers (concentrate FFN here)
- **Least important**: First few + last few layers (FFN can be removed)
- **Best config**: `middle_70` — FFNs in 70% of consecutive middle layers
  outperforms uniform FFN distribution across all tasks

### 1.2B Model Results (40 layers)

| Config | Avg Improvement |
|--------|----------------|
| middle_70 | +1.29% |
| first_70  | +0.69% |
| final_70  | +0.52% |

`middle_70` wins consistently. FFNs in first/last layers are redundant.

### Why Middle Layers?

- Mid-to-final FFN layers do the most significant information processing
- First layers: shallow pattern matching (attention does most work)
- Last layers: overspecialize toward pretraining objective
- Middle layers: knowledge storage and semantic processing

### Apparent Contradiction with FFN-SkipLLM

- FFN-SkipLLM skips **middle** layers (high cosine similarity = redundant)
- Layerwise paper says middle layers are **most important**

**Resolution**: They measure different things:
- FFN-SkipLLM: per-token input-adaptive skip (some tokens skip, others don't)
- Layerwise: structural removal during pretraining (FFN gone entirely)

Per-token skipping is safe because the FFN is still there for tokens that
need it. Structural removal forces the model to adapt during training.

---

## 3. FinerCut (2024) — Attention vs FFN Pruning

**Paper**: "FinerCut: Finer-grained Interpretable Layer Pruning for LLMs"

### Key Finding: Attention Layers Are More Prunable Than FFN

- Treats attention and FFN layers as **separate** pruning candidates
- Llama3-70B: 42% of self-attention layers removed, 99% performance retained
- **Preference for pruning attention layers**, especially at deeper layers
- FFN layers are more critical than attention layers for most tasks

### Implication for ForgeLM V3

If we want to skip computation, **attention layers at the top** are safer
to skip than FFN layers. This aligns with the "Attend First, Consolidate
Later" paper below.

---

## 4. Attend First, Consolidate Later (2024)

**Paper**: "Attend First, Consolidate Later: On the Importance of Attention
in Different LLM Layers" (Ben-Artzy & Schwartz, Hebrew University)

### Two-Phase Processing Model

1. **Phase 1 (bottom 50-70%)**: Information gathering
   - Attention is critical — gathers info from previous tokens
   - Any manipulation of hidden states → severe performance degradation
   - FFN processes and stores knowledge

2. **Phase 2 (top 30-50%)**: Internal consolidation
   - Attention is less important — info already gathered
   - Can freeze/skip attention with minimal impact
   - FFN continues processing internally
   - **Exception**: Math/reasoning tasks need attention all the way through

### Experimental Evidence

- Replacing hidden states with random vectors in top 30% → minimal impact
- Freezing hidden states after 50% of layers → no loss in some cases
- Switching "Italy" → "France" in top 1/3 → model still answers "Rome"
  (info already consolidated, ignores the switch)
- Skipping attention in top layers → matches baseline on capitals/SQuAD
  but **fails on math** (every token matters for reasoning)

### Implication

For non-math tasks: skip attention in top 30-50% of layers.
For math/reasoning: keep attention throughout, skip FFN in middle layers instead.

---

## 5. FFN as Key-Value Memory (Geva 2021, 2024)

**Paper**: "Transformer Feed-Forward Layers Are Key-Value Memories"

### FFN = Neural Memory

- First matrix (W_gate/W_up) = **keys** (input pattern matchers)
- Second matrix (W_down) = **values** (output distribution shifts)
- Each neuron in the FFN stores a "memory" of a textual pattern

### Layer Specialization

- **Lower layers**: Shallow patterns (n-grams, syntax)
- **Upper layers**: Semantic patterns (facts, concepts)
- Knowledge is stored primarily in **middle-to-upper** FFN layers

### InfoSteer (2025)

Post-training can encourage better use of FFN stored memory:
- Simple tokens (commas, "and") use fewer FFN resources
- Semantic tokens activate more FFN "memories"
- Steering during SFT improves knowledge utilization

---

## 6. Corrected Implementation for ForgeLM V3

### What We Did Wrong

1. **Wrong metric**: Used RMS norm of FFN input instead of cosine similarity
   between FFN input and output
2. **Wrong layers**: Skipped early layers (cold region) instead of middle
   layers (non-cold region)
3. **No warm-up**: Didn't preserve first few tokens for KV cache stability
4. **Static skip**: Used a fixed skip set instead of input-adaptive per-token

### Correct Approach

```
1. Calibrate: measure cosine similarity per layer on a calibration set
2. Identify cold regions: first 2-3 layers + last 2-3 layers (low cos sim)
3. Non-cold region: middle layers (high cos sim) — candidates for skipping
4. Warm-up: first 5-10% of tokens use full model
5. Per-token: compute cos(input, output) in non-cold region
   - If sim >= threshold: skip next k FFN blocks (greedy)
   - If sim < threshold: keep FFN
```

### For Our 16-Layer Model

Based on calibration data + research findings:
- **Cold start**: Layers 0-1 (never skip)
- **Non-cold**: Layers 2-13 (skip candidates)
- **Cold end**: Layers 14-15 (never skip)
- **Warm-up**: First 5-10 tokens of generation
- **Skip ratio target**: 25% (4 out of 16 FFN blocks per token)

### Expected Results

- 25% FFN skip → ~20-25% compute reduction
- Quality: <1% drop on knowledge tasks (per FFN-SkipLLM paper)
- No KV cache issues (FFN skip doesn't touch KV state)
- No hallucination or token collapse (unlike layer-skipping)

### Alternative: Skip Attention Instead

Per FinerCut + Attend-First-Consolidate-Later:
- Skip attention in top 30-50% layers for non-math tasks
- Keep all FFNs (they store knowledge)
- This is safer but saves less compute (attention is ~1/3 of layer params)

---

## 7. Recommended Path for ForgeLM V3

### Calibration Results (2026-08-18)

**Cosine similarity measured on ForgeLM V3 (1.2B, 16 layers):**

| Layer | Type      | Cos Sim | Status |
|-------|-----------|---------|--------|
| 0     | conv      | -0.6622 | COLD (important) |
| 1     | conv      | -0.3155 | COLD (important) |
| 2     | attention | -0.6288 | COLD (important) |
| 3     | conv      | -0.4322 | COLD (important) |
| 4     | conv      | -0.1024 | COLD (important) |
| 5     | attention | -0.3315 | COLD (important) |
| 6     | conv      | -0.2263 | COLD (important) |
| 7     | conv      | +0.0410 | COLD (important) |
| 8     | attention | -0.1596 | COLD (important) |
| 9     | conv      | +0.0897 | COLD (important) |
| 10    | attention | -0.1312 | COLD (important) |
| 11    | conv      | +0.0365 | COLD (important) |
| 12    | attention | +0.0396 | COLD (important) |
| 13    | conv      | +0.1022 | COLD (important) |
| 14    | attention | -0.1260 | COLD (important) |
| 15    | conv      | -0.1272 | COLD (important) |

**ALL layers have low cosine similarity (max +0.10, most negative).**

### Why FFN-SkipLLM Doesn't Apply to V3

1. **No saturation region**: The paper found monotonically increasing cosine
   similarity in LLaMa-2 7B/13B (32+ layers), reaching 0.95+ in middle layers.
   Our 16-layer model has no such region — FFN actively transforms in all layers.

2. **Hybrid architecture**: LFM2.5 has 10 conv + 6 attention layers, not the
   uniform attention+FFN blocks of LLaMa. Conv layers have different FFN dynamics.

3. **Scale**: FFN-SkipLLM targets 7B+ models with 32+ layers. At 1.2B/16 layers,
   every FFN is critical — there's no redundancy to exploit.

4. **Negative cosine similarities**: The FFN is not just "passing through" —
   it's substantially changing the representation direction. Skipping any FFN
   would destroy information.

### Conclusion

**FFN-SkipLLM is NOT applicable to ForgeLM V3.** The infrastructure is kept
in the codebase (`ffn_skip_threshold` config field, skip logic in ModularBlock)
for future larger models (32+ layers) where saturation regions emerge.

### What DOES Work for V3

Based on the research, the following techniques ARE applicable:

1. **Attention skipping in top layers** (Attend-First-Consolidate-Later):
   - Skip attention in layers 12-15 for non-math tasks
   - Keep all FFNs (they're all important)
   - Saves ~20% of attention compute

2. **middle_70 architecture** (for next model version):
   - Concentrate FFN parameters in middle 70% of layers
   - Remove FFN from first/last layers, expand middle FFNs
   - Per layerwise paper: +1.29% improvement at 1.2B scale

3. **LayerSkip** (after SFT):
   - Self-speculative decoding with early exit
   - Requires layer dropout during training
   - 1.5-2x speedup on appropriate tasks

4. **Speculative decoding** (already implemented):
   - EAGLE-3, DSpark, MTP — all available in research/decoding/
   - No architecture changes needed

### Phase 1 (Now): Document and disable
- FFN-SkipLLM disabled (threshold=0.0) — calibration proves no saturation
- Infrastructure kept for future 32+ layer models
- This research doc serves as reference

### Phase 2 (Future): Attention skipping
- Implement attention skip in top 30% of layers for non-math tasks
- Based on "Attend First, Consolidate Later" findings
- Keep all FFNs (they're all important in V3)

### Phase 3 (Future): middle_70 architecture
- For ForgeLM V4: concentrate FFN in middle 70% of layers
- Remove FFN from first/last layers, expand middle FFNs
- Per layerwise paper: +1.29% improvement at 1.2B scale
