# Round 38 — Advanced Fine-Tuning

## Overview
6 advanced fine-tuning adapter methods, including a novel entropy-guided
dynamic adapter fusion (ForgeAdapter).

## Adapters

### R38-1: DLoRA — Decoupled LoRA with Dynamic Rank
- **File**: `research/training/dlora.py`
- **What**: LoRA with growable rank. Start at rank-4, grow as needed based
  on gradient norm. Growth is a no-op at growth time (new B cols zero).
- **Scale**: Fixed at alpha/initial_rank (not alpha/current_rank) to
  ensure growth is a no-op.
- **Tests**: 12

### R38-2: DoRA — Weight-Decomposed Low-Rank Adaptation
- **File**: `research/training/dora.py`
- **What**: Decompose W into magnitude m + direction V. Apply LoRA to
  direction only, fine-tune magnitude separately. Lossless when delta=0.
- **Tests**: 12

### R38-3: PiSSA — Principal Singular Values Adaptation
- **File**: `research/training/adapter_variants.py`
- **What**: Initialize LoRA with top-rank singular values of pre-trained
  weight (SVD init). Faster convergence than random init.
- **Tests**: 7

### R38-4: AdaLoRA — Adaptive Budget Allocation
- **File**: `research/training/adapter_variants.py`
- **What**: Adaptive rank budget across layers. Prune unimportant singular
  values, grow important ones. Reallocate based on gradient norms.
- **Tests**: 11

### R38-5: rsLoRA — Scale-Free LoRA
- **File**: `research/training/adapter_variants.py`
- **What**: Scale by 1/sqrt(r) instead of 1/r. Enables stable training
  at high ranks without LR tuning.
- **Tests**: 9

### R38-6: ForgeAdapter — Entropy-Guided Dynamic Adapter Fusion (NOVEL)
- **File**: `research/training/forge_adapter.py`
- **What**: Cross-domain combination of AdaLoRA + entropy monitoring +
  mixture-of-adapters. Multiple LoRA adapters at different ranks. At
  inference, dynamically fuse based on per-token entropy (already computed
  for sampling — no router model needed). Low-entropy tokens use low-rank
  (fast), high-entropy tokens use high-rank (accurate).
- **Tests**: 18

## Test Results
- R38-1 (DLoRA): 12 tests
- R38-2 (DoRA): 12 tests
- R38-3 (PiSSA): 7 tests
- R38-4 (AdaLoRA): 11 tests
- R38-5 (rsLoRA): 9 tests
- R38-6 (ForgeAdapter): 18 tests
- **Total R38: 69 tests, 0 failures**

## Files Created
- `research/training/dlora.py`
- `research/training/dora.py`
- `research/training/adapter_variants.py`
- `research/training/forge_adapter.py`
- `tests/unit/test_r38_dlora_dora.py`
- `tests/unit/test_r38_adapter_variants.py`
- `tests/unit/test_r38_forge_adapter.py`
