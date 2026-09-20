# R49 Implementation Plan — Frontier Architecture Sweep: Uno, KDA, MoVA, DSA

Status: **IN PROGRESS (Phase 0 + R49-1 implemented 2026-09-08)**. Derived from
the two-round SOTA research sweep (full findings in `.devin/scratchpad.md`,
sections "SOTA Research Sweep" / "Round 2" / "Round 3"). Builds on R48
(extreme low-bit quant) and the V12 preset lineage per rule A. Sources: K2
Horizon (IFM), Uno arXiv:2609.04010, Qwen3-Next/Qwen3.5, Kimi Linear
arXiv:2510.26692, DeepSeek-V3.2 DSA, Nemotron 3, MiniMax M2/M2.5, Muon
ecosystem (QK-Clip/Dion/NorMuon), MoDA arXiv:2603.15619.

## Implementation status (2026-09-08)

| Item | Status | Files |
|---|---|---|
| Phase 0-1 sigmoid MoE gating | DONE | `forge/moe/moe.py` (`gating="sigmoid"`, MiniMax-M2 renormalized scores; softmax default bit-exact) |
| Phase 0-2 Dion2 row-sampling | DONE | `forge/training/optim/muon_sf_blockwise.py` (`rank_fraction`, vendored NS fallback, optional muon/schedulefree imports) |
| Phase 0-3 QK-Clip | DONE | `forge/training/optim/qk_clip.py` (new), `model_loader.py` observe hook, `sft_train.py` `--qk-clip-tau` |
| CISPO + ETR | DONE | `forge/self_play/grpo_trainer.py` (`rl_algorithm="cispo"`, `use_etr_reward`) + fixed pre-existing `kl` UnboundLocalError in GTPO path |
| R49-1 Uno decoding | DONE | `forge/decoding/uno.py` (new: NgramProposer + UnoDecoding Psi-Spec), `engine/decoding.py` factory, `forge_engine._activate_decoding`, GUI activation catalog |
| QK-Clip GUI | DONE | `forge_gui/pages/finetune.py` (QK-Clip tau spinbox → `--qk-clip-tau`) |
| R49-2 KDA key | DONE (Phase 2) | `forge/keys/attention/kda_key.py` (KDALayer + KDAKey BI port, gate=0 bit-exact; config `use_kda`; recurrent+conv state wired into reset/prefill-snapshot/prefix-cache paths; naive fp32 scan — chunked kernel is a follow-up) |
| R49-3 MoVA key | TODO (Phase 3) | `forge/keys/attention/mova_key.py` (planned) |
| R49-4 DSA indexer | TODO (Phase 4) | `forge/engine/attention/lightning_indexer.py` (planned) |
| zc-RMSNorm / attn output gate flags | TODO (Phase 0 remainder) | model_loader + config |
| Uno diffusion-distillation trainer | TODO (follow-up) | forge/training side of Uno |

Tests: `tests/unit/test_r49_uno.py` (7) + `tests/unit/test_r49_phase0.py` (18)
— all passing; 2515 passed / 15 skipped across the suite (excluding the
pre-existing test_r36_empty_states→test_r36_gui native crash, see scratchpad).

## Scope decision

**R49 core = 4 lossless-at-init features, one per subsystem:**

| ID | Feature | Path touched | Why first |
|---|---|---|---|
| R49-1 | Uno diffusion-augmented decoding | `forge/decoding/` | No arch change; up to 3x lossless speedup; beats EAGLE-3 at every batch size; ships as adapter |
| R49-2 | KDA (Kimi Delta Attention) key | `forge/keys/attention/` | −75% KV, ≥ full-attention quality at 3:1 hybrid; validated at 48B scale |
| R49-3 | MoVA (Mixture-of-Value attention) key | `forge/keys/attention/` | New sparsity axis inside attention; zero-init router = lossless |
| R49-4 | DSA lightning indexer | `forge/engine/attention/` | Trainable top-k sparse attention; lossless at k=∞ |

All four are CPU-testable, follow the port-first rule (lossless identity path at
init, bit-exact forward check before any training), and state VRAM budgets per
directive D. Phase 0 carries the cheap config-level wins so the round ships
value even if a key needs another iteration.

---

## Phase 0 — Quick wins (config/optimizer level, no new keys)

1. **Gated Attention + zero-centered RMSNorm** (Qwen3-Next/3.5): new flags
   `use_attn_output_gate`, `use_zc_rmsnorm` in `ActivationConfig` +
   `feature_registry`. Output gate is identity at gate=1; zc-RMSNorm is identity
   until weight decay on gamma is enabled. Wired in `model_loader.py` attention
   path. Tests: identity-at-init, gate gradient flows, flag wiring.
2. **Dion2 row-sampled orthogonalization** in `forge/training/optim/muon_sf_blockwise.py`:
   add `rank_fraction` (default 1.0 = current behavior, bit-identical). At 0.25,
   subsample momentum rows before Newton-Schulz (Microsoft Dion2; 1B/100B tok:
   loss 2.635 vs Muon 2.623, up to 6x cheaper step via Dion3 Gram-NS).
3. **QK-Clip post-step hook** in `forge/training/runners/sft_train.py` +
   `training_utils.configure_optimizer`: per-head max|logit| tracking; if
   S_max > tau (default 100, off by default): rescale Wq/Wk by sqrt(gamma)
   (MLA variant: W_uq/W_uk sqrt(gamma), shared rotary gamma). Kimi K2: 15.5T
   tokens, zero spikes.
4. **Router variants** in `forge/moe/moe.py`: `gating={softmax,sigmoid}` (MiniMax
   M2 style: sigmoid + e_score_correction_bias, renormalize top-k) and
   `balancing={aux_loss,alf_bias}` (DeepSeek ALF-LB: b_k += -u(load_k - target)).
   Default unchanged (softmax + aux) = lossless config.
5. **CISPO loss mode** in `forge/self_play/grpo_trainer.py`: clipped importance
   weight detached (`sg`) so every token keeps gradient; flag `--rl-loss {grpo,cispo}`.
   Matches DAPO in ~half the time; better under high off-policy reuse.
6. **MTP multi-step heads**: extend `forge/decoding/mtp.py` + `mtp_key.py` to
   shared-weight multi-step prediction (Nemotron 3: 2 shared MTP layers;
   MiniMax M2.5: 3 modules). Backward-compatible: steps=1 = current.

---

## Phase 1 — R49-1: Uno diffusion-augmented decoding

**New file**: `forge/decoding/uno.py` (decoding dir has eagle/medusa/mtp/suffix —
no diffusion path exists; Uno is adapter-based, unlike DFlash which needs a
separate draft model).

**Baseline (community)**: Uno / Psi-Spec (arXiv:2609.04010, code
github.com/ifm-ai/uno). Two weight sets: AR weights (frozen, define the
distribution) + lightweight diffusion weights trained via Diffusion Distillation
to emit token blocks in parallel; Psi-Spec samplers draw blocks from the AR
distribution -> lossless (verify against AR model, accept/reject per token).
K2 Horizon ships this as conditional-LoRA adapters (392 tensors, ~3x speedup).

**Training path (self-distillation, no external teacher)**: diffusion weights
trained against the base model's own AR distribution on our SFT corpus —
teacher logits come from the model itself (fits the 0.5B time-to-model recipe;
DistilLLM-2-style cost, hours on the 5070).

**Novel variations (implement alongside baseline):**
- **N1 — Vegas-shared verification**: the Uno verify pass doubles as the Vegas
  verification-guided KV selection pass (`forge/engine/kv/vegas_kv.py`) — one
  forward serves both accept/reject and KV importance scoring.
- **N2 — Concurrency-adaptive dispatch**: extend `faser.py` phase manager:
  batch>=threshold -> Uno block decoding; batch small -> EAGLE/n-gram
  (`adaptive_speculative.py`). Faser already owns spec-phase switching.
- **N3 — Entropy-bounded block stop** (from DiffusionGemma): stop denoising a
  block when avg entropy < eps and two consecutive predictions match; cheap
  adaptive early-exit for the sampler.

**Design constraints:**
- Lossless: sampler with diffusion weights disabled = pure AR path, bit-exact
  (preset lineage check, rule A).
- VRAM budget: adapter ~2-5% of base params (1.2B -> 25-60M -> ~50-120MB bf16)
  + block KV reuse. **< 200 MB total**; verify with
  `torch.cuda.max_memory_allocated()` in the test.
- Integration: `_activate_decoding` (`forge_engine.py:1570`), registry entry
  `use_uno` in `feature_registry.py`; composes with `use_faser` + `use_matryoshka_kv`.
- Tests (CPU, `tests/unit/test_r49_uno.py`): bit-exact fallback when disabled,
  block sampler accepts/rejects per AR distribution on a tiny model, adapter
  shape/dtype, registry wiring, memory ceiling assert.

---

## Phase 2 — R49-2: KDA key (Kimi Delta Attention)

**New file**: `forge/keys/attention/kda_key.py` (attention keys dir; GLA/GTA/LiSA
precedent). Follows the ForgeHybrid zero-init pattern (`forge_hybrid_key.py`).

**Baseline (community)**: KDA recurrence (arXiv:2510.26692; FLA `fla/ops/kda`):
`S_t = Diag(alpha_t) S_{t-1} + beta_t k_t (v_t - (Diag(alpha_t) S_{t-1})^T k_t)^T`,
per-key-dim decay alpha in [0,1] (lower-bounded), beta = write strength,
q/k L2-normalized. Naive recurrent PyTorch for CPU tests; chunked path for speed.

**Port-first**: hybrid gate zero-init (ForgeHybrid pattern): KDA branch output
weighted by gate g=0 at init -> **bit-exact vs baseline forward**; GDN-style
parameterization for warm start (A_log = log(uniform(0.01,16)), dt_bias
softplus-inverse in [1e-3,0.1], conv kernel 4). Bit-exact forward-pass check
against the prior preset config on the BSP base before any training.

**Novel variations:**
- **N1 — LeRoPE-conditioned decay**: KDA per-channel decay rates driven by
  `lerope_key` learnable frequencies (decay as learned RoPE-like spectrum) —
  merges two existing keys, no prior work does this.
- **N2 — KDA + SpectralKV hybrid stack**: KDA layers carry compressed state;
  anchor full-attention layers use SpectralKV — dual compression axes.
- **N3 — beta>1 state-tracking mode** (negative eigenvalues, per GDN analysis):
  gated by config flag, default off (documented destabilization risk).

**Hybrid ratio**: default 3:1 (Qwen3-Next/Kimi Linear); ablate 1:1 and 5:1
(1B-scale ablations: 1:1 best quality, ~1:5 best quality/efficiency; anchors
mid-stack). Expose `full_attention_interval` in config.

**VRAM budget**: KDA state = heads x d_k x d_v (128x128/head, few MB total);
KDA layers hold NO KV cache -> -75% KV at 3:1. State kept bf16/fp32 (drift
when gate~1 in low precision).

**Tests**: bit-exact at gate=0; recurrent-vs-chunked parity; gate lower-bound +
q/k norm stability; preset lineage check vs V12 checkpoint.

---

## Phase 3 — R49-3: MoVA (Mixture-of-Value attention)

**New file**: `forge/keys/attention/mova_key.py`.

**Baseline (community)**: MoVA (K2-Horizon-MoVA-36B-A4B): routing moved into
attention — N value experts, top-m active per token (K2-Horizon: 64 experts / 4
active, 45/48 layers MoVA, GQA 32/8 heads, FlashAttention-compatible). 36B-A4B
~ dense-32B quality under identical recipe.

**Port-first**: zero-init router -> value experts average to the identity/shared
V at init (or identity expert + zero-init routed experts) -> bit-exact vs
baseline. Router logits zero-init.

**Novel variations:**
- **N1 — BitNet value experts**: routed experts in ternary/FP4 (pairs with
  `bitnet_residual_key` + R44-48 quant stack) — attention sparsity x quantization.
- **N2 — MoVA x GTA**: routed value experts with V=K tying at init
  (`gta_key.py` lineage) — halves KV bandwidth further.
- **N3 — AirMoE hotswap for value experts**: disk-backed value-expert library
  (`airmoe_hotswap.py`) — value-expert capacity beyond VRAM.

**VRAM budget** (1.2B scale): 16 experts x d_v(128) x d_model(2048) routed per
attn block ~= 4.2M params/layer bf16 if fully populated; default config uses
8 experts top-2 -> **< 60 MB incremental**; state the measured number in the
round results.

**Tests**: identity-at-init (router zero), top-m routing + load stats, FA/GQA
compat path, composition with `use_bitnet_residual`.

---

## Phase 4 — R49-4: DSA lightning indexer

**New file**: `forge/engine/attention/lightning_indexer.py`; hooks into the
QSA/CSA sparse-attention path (`forge/keys/attention/qsa_key.py`, `csa_key.py`).

**Baseline (community)**: DSA (DeepSeek-V3.2): per-layer lightweight low-head
scorer (index_n_heads x index_head_dim, own 1-head index-K cache via
`update_indexer`), top-k (2048) token selection -> additive mask into main
attention. Reference runs FP8 + Hadamard (orthogonal transform; bf16 scores
equivalent). Kernels: DeepGEMM indexer kernels, FlashMLA sparse (upstream refs).

**Port-first**: `index_topk = None` (or >= seq_len) -> all-zero mask -> bit-exact
vs full attention. Indexer warm start = **distill from base attention maps**
(novel: training-free — fit indexer scores to base attention entropy/mass
distribution on calibration batches; no extra training run needed).

**Novel variations:**
- **N1 — Shared indexer across layer groups** (one index-K cache per group, not
  per layer) — cuts index-K cache 4-8x.
- **N2 — Indexer x compact_attention**: fuse top-k indices with block-union KV
  selection for chunked prefill (`compact_attention.py`).
- **N3 — Indexer-guided eviction**: indexer scores feed SnapKV/paged eviction
  priority (cross-feed with R33 KV stack).

**VRAM budget**: index-K cache = 1 head x 128 dims x L ~ 256 B/token/layer
(~0.5 MB per 2K context per layer; shared-indexer variant /8). **< 10 MB** at
1.2B defaults.

**Tests**: bit-exact at k>=L; indexer distillation reduces KL vs random init;
mask correctness vs dense attention on tiny model; cache growth per token.

---

## Phase 4 backlog (R50+ candidates, ordered by impact x effort)

1. **LatentMoE** (Nemotron 3): compress tokens to latent dim before experts ->
   4x experts same cost; also cuts AirMoE disk I/O. Needs MoELayer changes.
2. **Mamba-3 MIMO upgrade** to `mamba3_key`: rank-R=4 B/C (state update as
   matmul), 3-term trapezoidal recurrence, data-dep RoPE on B/C (+1.2pp at 1.5B,
   half Mamba-2 state size). Extends existing `mamba3_key`.
3. **RWKV-7 key**: 4th linear-attention family (FLA kernels exist); lower
   priority — KDA strictly extends GDN validated at scale.
4. **NVFP4 pretraining path**: E2M1 + microblocks + RHT + stochastic rounding;
   SM120 caveat: official TE fused kernels FAIL (232KB smem, missing .rs);
   torch._scaled_mm via torchao dispatch works. High effort — schedule as own round.
5. **Hybrid ratio sweep** on ForgeLM 1.2B: 1:1 / 3:1 / 5:1 (Bae et al. datapoints),
   anchors mid-stack — feeds ForgeEvolve arch domain.
6. **DiffusionGemma-style canvas sampler** (entropy-bounded denoising, 256-token
   canvas) — research prototype only; Uno covers the production path.
7. **Prefix-tree merging** for self_play GRPO rollouts (trie + shared prefix KV).

## Counterpoint datapoints (record for ForgeEvolve scoring)
- MiniMax M2 (229B-A9.8B) deliberately kept FULL attention + GQA: hybrid rejected
  for RL-scale/eval/low-precision reasons — hybrid-vs-full must be scored, not assumed.
- K2-Horizon MoVA 36B-A4B ~= dense 32B (controlled same-recipe comparison) —
  attention sparsity is a real axis, not just FFN sparsity.
- Nemotron 3 Super pretrained in NVFP4 with BF16 islands (final 15% layers, QKV,
  embeddings, MTP) — quantization placement matters more than global bit-width.

## Rule A compliance (preset lineage)
- No new preset in R49. When V13 is created it MUST derive from V12 carrying all
  V12 keys + the new flags (`use_uno`, `use_kda`, `use_mova`, `use_lightning_index`,
  `use_attn_output_gate`, `use_zc_rmsnorm`), with bit-exact forward check at
  gate=0 / lambda=0 / k=inf against the V12 checkpoint on the BSP base.
- Dropped-key audit: none dropped; all V10->V12 keys carried forward.

## Verification (directive C — closed loop)
- Every Phase 0-4 item lands with a CPU-runnable test in `tests/unit/`.
- Bit-exactness checks: `torch.equal` / max-logit-diff == 0.0 at identity configs.
- GPU items (Uno adapter, KDA chunk path) additionally profiled with
  `torch.cuda.max_memory_allocated()` vs stated VRAM budgets.
