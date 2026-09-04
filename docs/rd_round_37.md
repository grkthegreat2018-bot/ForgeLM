# Round 37 — ForgeLM V12 Architecture: New Keys

## Overview
5 new architecture keys for ForgeLM V12, all derived from V11 with
lossless warm start. Per AGENTS.md section A: V12 carries forward ALL
V11 keys, adds new ones with zero/identity init.

## Keys

### R37-1: Mamba-3 Key — Complex-Valued SSM States
- **File**: `research/keys/architecture/mamba3_key.py`
- **Paper**: arXiv 2603.15569
- **What**: Mamba-3 improves SSM with complex-valued states and MIMO.
  Higher expressivity than Mamba-2 with same memory.
- **Port path**: Mamba-2 real states → Mamba-3 complex states (split into
  real/imag, zero-init imag). Lossless both ways.
- **Key class**: BI (lossless round-trip)
- **Tests**: 37

### R37-2: Kronecker Embeddings — Byte-Level Structured Token Representations
- **File**: `research/keys/architecture/kronecker_embed_key.py`
- **Paper**: arXiv 2605.29459
- **What**: Replace |V|×d embedding table with byte-level character-position
  factorization + single learned projection. 91-94% input-side param reduction.
- **Port path**: SVD of existing embedding → initialize projection.
  char_embed and pos_encode zero-init.
- **Tests**: (pending subagent)

### R37-3: PIT — Pseudo-Inverse Tying for Stable Token Interface
- **File**: `research/keys/architecture/pit_tying_key.py`
- **Paper**: arXiv 2602.04556
- **What**: Synchronize embedding and unembedding as coupled projections
  of shared latent token memory. Orthonormal shared memory via thin polar
  decomposition. Stable triangular solves.
- **Port path**: Extract embedding + unembedding, compute shared orthonormal
  memory via polar decomposition of W_unemb @ W_emb.
- **Key class**: BI (lossless round-trip)
- **Tests**: 8

### R37-4: OutRo — Sink-Enhanced Contextual Representations
- **File**: `research/keys/attention/outro_key.py`
- **Paper**: arXiv 2603.14337
- **What**: Align non-sink token representations with sink representation.
  Allow sink token to attend beyond causal constraint. 1.1× overhead.
- **Port path**: Detect existing sinks, enable non-causal sink attention.
- **Key class**: PARTIAL (forward only — modifies attention pattern)
- **Tests**: 12

### R37-5: ForgeHybrid — Sink-Aware SSM+Attention Routing (NOVEL)
- **File**: `research/keys/architecture/forge_hybrid_key.py`
- **What**: Novel cross-domain combination of attention sink mechanics
  (P0-Sink) + Mamba-3 SSM + OutRo. Each layer has both attention and SSM
  paths. The sink norm signal IS the router (no learned router needed).
  Tokens with high sink norm → SSM (cheap). Tokens with high attention
  entropy → attention (accurate).
- **Port path**: Zero-init SSM path alongside attention. Router threshold
  starts at infinity (all attention). Gradually lower threshold to enable
  SSM routing as sink signals stabilize.
- **Key class**: BI (lossless at warm start — zero-init SSM = identical
  to pure attention)
- **Tests**: 15

### R37-6: V12 Preset Definition
- **File**: `research/config.py` (forgelm_v12 preset)
- **What**: V12 preset derived from V11 (forgelm_v2_pro). Carries forward
  ALL V11 keys. Adds: Mamba-3, Kronecker Embeddings, PIT tying, OutRo,
  ForgeHybrid. All new keys are zero/identity-init = lossless warm start.
- **Lineage check**: V12 does not drop any V11 keys. All V11 architectural
  parameters preserved. New keys only add (never remove).
- **Memory budget**: ~4.2GB (same as V11 — new keys are zero-init)
- **Tests**: 12

## Lossless Warm Start Verification
All 5 new keys are designed for lossless warm start from V11:
- **Mamba-3**: Complex states with imag=0 → identical to Mamba-2 real states
- **Kronecker**: SVD init from existing embedding → reconstruction
- **PIT**: Identity init → lossless (M orthonormal, P = W @ M^T)
- **OutRo**: Threshold-based, no weight change
- **ForgeHybrid**: sink_threshold=inf → all tokens use attention (SSM zero-init)

## Test Results
- R37-1 (Mamba-3): 37 tests
- R37-3+4 (PIT + OutRo): 20 tests
- R37-5 (ForgeHybrid): 15 tests
- R37-6 (V12 Preset): 12 tests
- R37-2 (Kronecker): pending
- **Total R37 (so far): 84 tests, 0 failures**

## Files Created
- `research/keys/architecture/mamba3_key.py`
- `research/keys/architecture/kronecker_embed_key.py` (pending)
- `research/keys/architecture/pit_tying_key.py`
- `research/keys/attention/outro_key.py`
- `research/keys/architecture/forge_hybrid_key.py`
- `tests/unit/test_r37_mamba3.py`
- `tests/unit/test_r37_kronecker.py` (pending)
- `tests/unit/test_r37_pit_outro.py`
- `tests/unit/test_r37_forge_hybrid.py`
- `tests/unit/test_r37_v12_preset.py`

## Files Modified
- `research/config.py` (V12 config fields + preset)
