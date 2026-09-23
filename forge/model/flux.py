"""FLUX — sparse online associative-memory language model.

Not a transformer.  The "weights" are addressable memory cells that are
written locally and live: every observed token updates a handful of
cells (n-gram tables, a Hebbian topic row, episodic index entries, a
kNN context-fingerprint row) and nothing else — one change cannot
perturb unrelated regions.

Context format = a *sketch*, not a window:
  - rolling polynomial suffix-hashes at multiple orders (features over
    the whole history, order-agnostic),
  - an episodic ring buffer enabling longest-suffix-match retrieval of
    effectively unbounded order, PLUS archived entries that survive the
    ring horizon: each index entry carries its successor token and
    suffix-hash "certificates" at longer orders, so a position that has
    scrolled out of the ring still votes (graded confidence = deepest
    matching certificate).  Deep memory is therefore truly unbounded,
    not capped at ring_capacity,
  - an EMA topic hypervector,
  - a recency table.
No positional encoding, no maximum length, no training needed to handle
longer contexts — longer history just means more memory cells.

Prediction = geometric mixture over channels:
    logit(t) = log_uni(t) + sum_c  w_c * (log p_c(t) - log uni(t))
where each channel p_c is the empirical distribution of one feature
family and w_c adapts online by Hedge (multiplicative weights).

The "knn" channel is the approximate-match counterpart of the exact
episodic channel: every learned token writes one row
(normalize(context_EMA @ P_csem), successor) into a fixed-capacity
fingerprint bank; prediction is a top-k cosine retrieval whose
successors vote.  It generalizes deep recall to *similar* contexts —
the memorizing-transformer pattern, online and journaled.

``teach()`` is instant hot-training: a (context -> target) association
is written directly into the order tables (all orders), the csem
fingerprint of the context, and the knn bank — no gradients, no ring
requirement, revertible via ``revert_tag``.

Live learning is instant and journaled: every mutation is appended to
``FluxJournal`` with its source tag, so "what is stored in which
weights" is auditable and revertible (``revert_since``, ``revert_tag``,
``snapshot``/``load``).

Engine contract (forge/engine/decoding.py StandardDecoding):
    model(ids, past_key_values=None, use_cache=True) -> (logits, loss, None)
    logits: float32 [B, T, vocab]; past is always None (state is internal).
"""
from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

# Polynomial rolling hash over token ids, mod 2^64 (numpy uint64 wraparound).
_HASH_P = np.uint64(1099511628211)
_MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def _load_teacher_embedding(path) -> torch.Tensor:
    """Load a teacher's embedding table from a checkpoint file —
    safetensors (slice-read, no full load) or torch .pt.  The target is
    auto-detected: prefer a key containing 'embed'/'wte', else the
    largest first-dim 2-D tensor (vocab-sized rows)."""
    path = Path(path)
    if path.is_dir():
        cands = sorted(path.glob("*.safetensors"))
        if not cands:
            raise FileNotFoundError(f"no .safetensors in {path}")
        path = cands[0]
    if path.suffix == ".safetensors":
        from safetensors import safe_open
        with safe_open(str(path), framework="pt") as f:
            keys = [k for k in f.keys()
                    if "embed" in k.lower() or "wte" in k.lower()]
            if not keys:
                keys = [k for k in f.keys()
                        if len(f.get_slice(k).get_shape()) == 2]
            best = max(keys, key=lambda k: f.get_slice(k)
                       .get_shape()[0])
            return f.get_tensor(best).float()
    sd = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(sd, dict):
        raise ValueError(f"{path} is not a state dict")
    keys = [k for k in sd
            if isinstance(sd[k], torch.Tensor) and sd[k].dim() == 2
            and ("embed" in k.lower() or "wte" in k.lower())]
    if not keys:
        keys = [k for k in sd
                if isinstance(sd[k], torch.Tensor) and sd[k].dim() == 2]
    return sd[max(keys, key=lambda k: sd[k].shape[0])].float()


if HAS_TRITON:

    @triton.jit
    def _flux_deep_votes_kernel(
        pos_ptr,            # [ER] int64 — ctx_end per cell (-1 empty)
        succ_ptr,           # [ER] int32 — stored successor (-1 none)
        cert_ptr,           # [ER, LC] int64 — certificate suffix hashes
        hg_ptr,             # [cap] int64 — prefix-hash ring
        ring_ptr,           # [cap] int32 — token ring
        cur_ptr,            # [LC] int64 — current context cert hashes
        certw_ptr,          # [LC] int64 — cert order weights
        votes_ptr,          # [V] f32 — OUT: accumulated votes
        end_ptr, floor_ptr, ep_ptr, dk_ptr, pd_ptr,   # device scalars
        fe_ptr,             # [] bool — probe hit; a miss leaves se
                            # pointing at an unrelated row that must
                            # NOT be voted
        CAP: tl.constexpr, DO: tl.constexpr, MX: tl.constexpr,
        LC: tl.constexpr, ER: tl.constexpr,
    ):
        """One program per episodic cell: seed-hash check + backward
        extension for on-ring entries, certificate-graded weight for
        archived (off-ring) entries, atomic scatter of the vote.
        Replaces ~25 tiny torch kernels per predict."""
        j = tl.program_id(0)
        pos = tl.load(pos_ptr + j)
        succ = tl.load(succ_ptr + j).to(tl.int64)
        end = tl.load(end_ptr)
        floor = tl.load(floor_ptr)
        ep = tl.load(ep_ptr)
        dk = tl.load(dk_ptr)
        fnd = tl.load(fe_ptr)
        ok = fnd & (pos >= 0) & (pos < end) & (succ >= 0) \
            & (end - DO + 1 >= ep) & (end >= DO - 1) & (end >= 0)
        mlen = pos * 0                       # int64 scalar zero
        if ok:
            if pos >= floor:
                # on-ring: verify the seed-order suffix hash, then
                # extend backwards token-by-token while bytes match
                hp = tl.load(hg_ptr + pos % CAP)
                hb = tl.load(hg_ptr + tl.maximum(pos - DO, 0) % CAP)
                hv = hp - hb * tl.load(pd_ptr)
                if hv == dk:
                    mlen += DO
                    m = 1
                    while m <= MX:
                        pi = pos - m
                        ei = end - m
                        if (pi < floor) | (pi < 0) | (ei < ep) \
                                | (ei < 0):
                            m = MX + 1
                        else:
                            a = tl.load(ring_ptr + pi % CAP)
                            b = tl.load(ring_ptr + ei % CAP)
                            if a != b:
                                m = MX + 1
                            else:
                                mlen += 1
                                m += 1
            else:
                # archived: seed-order mass + deepest matching cert
                mlen += DO
                for l in tl.static_range(LC):
                    cert = tl.load(cert_ptr + j * LC + l)
                    cur = tl.load(cur_ptr + l)
                    w = tl.load(certw_ptr + l)
                    hit = ((cert != 0) & (cert == cur)
                           & (w > mlen)).to(tl.int64)
                    mlen += (w - mlen) * hit
            if mlen > 0:
                tl.atomic_add(votes_ptr + succ, mlen.to(tl.float32))

    @triton.jit
    def _flux_logit_tail_kernel(
        logits_ptr,         # [V] f32 in/out — already holds
                            # log_uni + order/backoff/deep/topic terms
        log_uni_ptr,        # [V] f32 — log unigram prior
        uni_pv_ptr,         # [V] f32 — smoothed unigram prob
        sims_ptr,           # [V] f32 — Ac @ proto-fingerprint cosine
        csims_ptr,          # [V] f32 — Ac @ csem-row cosine
        kv_ptr,             # [V] f32 — knn successor vote mass
        seen_ptr,           # [V] int64 — last-seen position
        wg_ptr,             # [C] f64 — hedge weights
        pvn_ptr, rn_ptr, cf_ptr, cok_ptr, kvt_ptr, end_ptr,
        V,
        SEMX: tl.constexpr, CSEMX: tl.constexpr, KNNX: tl.constexpr,
        RECX: tl.constexpr, FATX: tl.constexpr,
        USE_SEM: tl.constexpr, USE_CSEM: tl.constexpr,
        BETA_SEM: tl.constexpr, BETA_CSEM: tl.constexpr,
        ALPHA: tl.constexpr, KAPPA: tl.constexpr,
        LIFT: tl.constexpr,
        SPAN_R: tl.constexpr, LDEC_R: tl.constexpr,
        SPAN_F: tl.constexpr, LDEC_F: tl.constexpr,
        FAT_STR: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Fused post-topic logit tail: sem + csem + knn mixture terms
        and recency/fatigue adjustments in ONE V-map pass.  Channel
        gates arrive as device scalars — multiplicative gating keeps
        the kernel branchless and CUDA-graph safe (no host reads)."""
        pid = tl.program_id(0)
        off = pid * BLOCK + tl.arange(0, BLOCK)
        m = off < V
        lg = tl.load(logits_ptr + off, mask=m)

        pvn = tl.load(pvn_ptr)
        if USE_SEM:
            f = (pvn > 1e-8).to(tl.float32)
            w = tl.load(wg_ptr + SEMX).to(tl.float32)
            lg += f * w * BETA_SEM \
                * tl.load(sims_ptr + off, mask=m, other=0.0)

        if USE_CSEM:
            rn = tl.load(rn_ptr)
            okc = tl.load(cf_ptr) & tl.load(cok_ptr) & (rn > 1e-8)
            w = tl.load(wg_ptr + CSEMX).to(tl.float32)
            lg += okc.to(tl.float32) * w * BETA_CSEM \
                * tl.load(csims_ptr + off, mask=m, other=0.0)

        kvt = tl.load(kvt_ptr)
        f_k = (kvt > 0).to(tl.float32)
        wk = tl.load(wg_ptr + KNNX).to(tl.float32)
        # evidence lift — massive vote mass is self-evidencing
        if LIFT:
            wk = wk + kvt / (kvt + KAPPA)
        kv = tl.load(kv_ptr + off, mask=m, other=0.0)
        upv = tl.load(uni_pv_ptr + off, mask=m, other=1.0)
        lu = tl.load(log_uni_ptr + off, mask=m, other=0.0)
        pk = (kv + ALPHA * upv) / (kvt + ALPHA)
        lg += f_k * wk * (tl.log(tl.maximum(pk, 1e-12)) - lu)

        end = tl.load(end_ptr)
        seen = tl.load(seen_ptr + off, mask=m, other=-(1 << 40))
        age = end - seen
        is_r = (seen >= 0) & (age >= 0) & (age < SPAN_R)
        lg += tl.load(wg_ptr + RECX).to(tl.float32) * tl.where(
            is_r, tl.exp(age.to(tl.float32) * LDEC_R), 0.0)
        is_f = (seen >= 0) & (age >= 0) & (age < SPAN_F)
        lg -= tl.load(wg_ptr + FATX).to(tl.float32) * FAT_STR \
            * tl.where(is_f, tl.exp(age.to(tl.float32) * LDEC_F), 0.0)

        tl.store(logits_ptr + off, lg, mask=m)


@dataclass
class FluxConfig:
    """Hyperparameters for FluxLM.  All memory is bounded by construction."""

    name: str = "flux"
    vocab_size: int = 65536
    eos_token_id: int = 2

    # n-gram channel orders (suffix-hash table per order)
    orders: tuple = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
    # episodic channel: seed-match order, then extend backwards in the ring
    deep_order: int = 8
    deep_max_hits: int = 24        # seed positions examined per prediction
    deep_max_extend: int = 96      # extra tokens of match extension allowed
    epi_max_per_key: int = 512     # positions kept per seed key
    epi_evict_scan: int = 64       # oldest fifo candidates scored per evict
    epi_hit_w: float = 4.0         # one retrieval hit ~= hit_w recency units
    # archived entries (positions older than the ring) still vote: each
    # entry stores its successor + suffix-hash "certificates" at these
    # orders; the vote weight is the deepest certificate that matches
    # the current context, else the seed order.  0-length tuple disables.
    cert_orders: tuple = (24, 96)
    epi_total_cap: int = 3_000_000  # global FIFO bound on host entries

    ring_capacity: int = 1 << 20   # episodic ring (tokens), ~4 MB int32

    # topic channel (Hebbian hypervector memory)
    topic_dim: int = 192
    topic_decay: float = 0.995
    hebb_lr: float = 0.05
    topic_beta: float = 4.0        # sharpness of cosine -> pseudo-prob

    # semantic-spread channel (generativity): proto = EMA fingerprint of
    # "what kind of token comes next"; tokens distributionally similar
    # to memorized followers get lift -> recombination, not just recall.
    use_sem: bool = True
    sem_beta: float = 4.0
    proto_decay: float = 0.98

    # recency channel
    recency_span: int = 64
    recency_decay: float = 0.90

    # fatigue channel (anti-repeat): tokens emitted moments ago get
    # penalized — intrinsic degeneration pressure, hedged like the rest
    fatigue_span: int = 16
    fatigue_decay: float = 0.80
    fatigue_strength: float = 2.0

    # smoothing + meta-learning
    alpha: float = 0.4             # count smoothing (backoff to unigram)
    uni_alpha: float = 0.05
    hedge_lr: float = 0.5
    hedge_min: float = 1e-3
    hedge_max:  float = 50.0

    # capacity guards (hard memory bound)
    max_cells_per_order: int = 150_000   # decay-halving sweep beyond this
    journal_cap: int = 200_000           # journaled writes kept for undo
    vec_delta_cap: int = 65_536          # fp16 topic deltas kept for undo

    # speed: during bulk ingest, skip deep-match + topic reads inside the
    # Hedge reward (they fall back to the unigram prior). Channel weights
    # still adapt; full rewards apply in live/generation learning.
    fast_ingest: bool = True

    # gap tracking for selective distillation: positions where the model
    # was SURPRISED (1 - p_mix > gap_surprise) or the context was unseen
    # are recorded so a teacher can score just those contexts.
    track_gaps: bool = True
    gap_probe_order: int = 8
    gap_min_total: float = 3.0
    gap_surprise: float = 0.5
    gap_cap: int = 500_000

    # surprise-gated writes (Titans/MIRAS-style): every mutation scales
    # with 1 - p_mix(x) — well-predicted tokens barely write, surprising
    # ones commit hard.  Saves bounded-table capacity + collisions.
    surprise_gate: bool = True
    surprise_floor: float = 0.05     # minimum write magnitude

    # delta-rule topic writes (DeltaNet/GDN): the A-row update subtracts
    # the current readout along the context direction — error-corrected
    # associative memory, beats additive Hebbian on in-context recall.
    delta_rule: bool = True

    # backoff channel (infini-gram style): the longest order whose
    # context has >= backoff_min_tot evidence, confidence-discounted —
    # hedged alongside every other channel.
    use_backoff: bool = True
    backoff_min_tot: float = 2.0
    backoff_kappa: float = 8.0      # confidence = tot / (tot + kappa)

    # evidence lift (interpolated-Kneser-Ney semantics): at predict time
    # an order/backoff channel's effective weight is w + tot/(tot+kappa)
    # — hedge learns the reliability PRIOR, but a cell with real mass is
    # self-evidencing.  Without this, hedge collapse (weights -> floor)
    # makes injected/learned facts unreachable no matter their counts.
    evidence_lift: bool = True

    # AdaHedge-style adaptive lr: eta_t = ln(C) / mix_cum where mix_cum
    # accumulates the per-token mixability gap (max reward - p_mix).
    # Constant regret on easy streams, worst-case safe.
    adahedge: bool = True
    hedge_eta_max: float = 4.0

    seed: int = 0xF1A5             # deterministic token->hypervector table
    # "cuda" puts the dense readout on GPU (A/R/c + logits + sem matvec);
    # sparse tables/journal always stay in host RAM — the right split for
    # pointer-chasing hash lookups. Falls back to cpu if CUDA absent.
    device: str = "cpu"

    # ── CUDA-primary mode (device="cuda" + cuda_primary=True) ──────────
    # Sparse memory moves into open-addressed pair-key tables ON the GPU:
    # (context_key, follower) -> count.  Prediction = one V-wide probe per
    # order; ingest = vectorized chunks (closed-form EMAs, batch probes).
    cuda_primary: bool = False
    gpu_table_slots: int = 1 << 19   # context rows per order
    gpu_row_width: int = 16          # follower cells per context row
    gpu_probe_max: int = 64          # linear-probe iterations
    gpu_bulk_chunk: int = 8192       # tokens per vectorized ingest block
    epi_slots: int = 1 << 18         # GPU episodic index rows
    epi_row: int = 16                # positions per seed key

    # csem: per-CONTEXT semantic fingerprint (contextual recombination —
    # the per-key version of the global `sem` channel).  Own hash table
    # context_key -> fp16 fingerprint row; EMA'd toward follower A-rows.
    use_csem: bool = True
    csem_order: int = 4              # context key order for csem table
    csem_slots: int = 1 << 19        # 512k rows x csem_dim fp16
    csem_dim: int = 64
    csem_beta: float = 2.0
    csem_lr: float = 0.15

    # knn channel: approximate-match episodic memory.  Every learned
    # token writes (normalize(context_EMA @ P_csem), successor, pos)
    # into a fixed-capacity ring bank; prediction retrieves the top-k
    # fingerprint-similar past contexts and votes their successors —
    # "attention" over the entire stream, bounded memory, no ring
    # horizon.  This is the generative counterpart of `deep` (exact
    # match): knn fires on *similar* contexts, not just verbatim ones.
    use_knn: bool = True
    knn_capacity: int = 1 << 19      # fingerprint rows (~64 MB fp16)
    knn_k: int = 32                  # retrieved neighbours per predict
    knn_gamma: float = 2.0           # vote weight = relu(sim)^gamma
    knn_min_sim: float = 0.05        # cosine floor for a vote to count
    knn_scan_min: int = 4096         # live-slice granularity (bucket)

    # triton fused kernels (CUDA-primary only): single-launch deep-vote
    # eval + fused V-tail logit assembly.  Eager torch fallbacks cover
    # missing triton / non-CUDA devices; results are bit-comparable.
    use_triton: bool = True

    # teacher-embedding graft: path to a safetensors checkpoint whose
    # embedding table replaces the random bipolar `_R` feature matrix —
    # one transplant upgrades every semantic channel (topic `_A`, proto,
    # csem rows, knn fingerprints all flow from `_R[tok]`).  Graft
    # BEFORE any learning; snapshots persist the grafted matrix.
    teacher_embed: str = ""
    teacher_embed_method: str = "randproj"   # "randproj" | "pca" —
        # randproj (JL) preserves pairwise cosine structure; PCA's top
        # axes are frequency-dominated and collapse semantics (measured
        # on ForgeLM V2 embed.weight)


class FluxJournal:
    """Append-only provenance log for every weight mutation.

    Entry kinds (all tuples starting with a 1-char op):
      ("t", tok, tag)                 token marker (position in stream)
      ("b", ep_start, c_fp16)         episode boundary (soft_reset)
      ("u", tok)                      unigram count +1
      ("c", order, key, tok, delta)   table cell write
      ("e", key, pos)                 episodic index insert
      ("r", tok, prev_pos_or_-1)      recency last_seen write
      ("a", tok, vec_slot_or_-1)      topic row += delta (fp16 in vec ring)
      ("h", w_slot_or_-1)             hedge update — pre-update w vector
                                      snapshotted fp64 in w_ring (clip
                                      makes a multiplicative inverse
                                      lossy, so we store the vector)
    """

    __slots__ = ("entries", "cap", "dropped", "vec_ring", "vec_cursor",
                 "vec_cap", "w_ring", "w_cursor", "w_cap")

    def __init__(self, cap: int, vec_cap: int, topic_dim: int,
                 n_chan: int):
        self.entries: list[tuple] = []
        self.cap = cap
        self.dropped = 0
        self.vec_ring = np.zeros((vec_cap, topic_dim), dtype=np.float16)
        self.vec_cursor = 0
        self.vec_cap = vec_cap
        # +1 column: AdaHedge cumulative mix-gap rides alongside w so
        # undo restores the whole hedge state exactly
        self.w_ring = np.zeros((vec_cap, n_chan + 1), dtype=np.float64)
        self.w_cursor = 0
        self.w_cap = vec_cap

    def append(self, entry: tuple) -> None:
        self.entries.append(entry)
        if len(self.entries) > self.cap:
            # drop oldest chunk (amortized) — revert horizon shrinks
            del self.entries[: self.cap // 10]
            self.dropped += self.cap // 10

    def append_vec(self, tok: int, delta: np.ndarray) -> tuple:
        slot = -1
        if self.vec_cap:
            slot = self.vec_cursor
            self.vec_ring[slot] = delta.astype(np.float16)
            self.vec_cursor = (self.vec_cursor + 1) % self.vec_cap
        return ("a", tok, slot)

    def append_w(self, w: np.ndarray, mix: float = 0.0) -> tuple:
        slot = -1
        if self.w_cap:
            slot = self.w_cursor
            self.w_ring[slot, :w.size] = w
            self.w_ring[slot, w.size] = mix
            self.w_cursor = (self.w_cursor + 1) % self.w_cap
        return ("h", slot)

    def __len__(self) -> int:
        return len(self.entries)


# ── CUDA-primary sparse backend ────────────────────────────────────────
#
# Pair-key open-addressed tables: (context_key, follower) -> count live
# in flat int64/fp32 GPU tensors.  One V-wide probe per order replaces
# the per-follower dict walk, so prediction is ~30 kernel launches per
# step instead of hundreds of Python dict ops.  Sentinel 0 = empty slot;
# pair keys are forced nonzero.  Slots are never freed in place — decay
# zeroes small counts and `rehash` rebuilds a table dropping dead keys.

def _s64(v: int) -> int:
    """int64 bit pattern (signed) for a uint64 constant."""
    return v - (1 << 64) if v >= (1 << 63) else v


_MIX1 = _s64(0x9E3779B97F4A7C15)
_MIX2 = _s64(0xBF58476D1CE4E5B9)


def _mix64(x: torch.Tensor) -> torch.Tensor:
    """SplitMix64 finalizer on int64 (wraps mod 2^64 — fine, it's a
    hash).  Returns nonzero values."""
    x = (x + _MIX1) * _MIX2
    x = (x ^ (x >> 29)) * _MIX2
    x = x ^ (x >> 32)
    return torch.where(x == 0, torch.ones_like(x), x)


class _GPUCellTables:
    """orders -> open-addressed context-key -> follower-row tables,
    stacked: all per-order tensors live in [O, S] / [O, S, M] buffers
    so ONE probe loop serves every order (fewer kernel launches).

    Each occupied slot holds a fixed-width row of ``M`` (tok, count)
    pairs.  Rows that overflow keep their top-M followers (smallest
    count evicted), matching the heavy-tailed follower distribution.
    ``raw`` stores the pre-hash context key so rows export without a
    host reverse map.  Sentinel 0 = empty slot.
    """

    # extra shared-key-matrix rows: row O = csem, row O+1 = epi
    ROW_CSEM_SALT = _s64(0xC5E400C5E400C5E4 % (1 << 64))
    ROW_EPI_SALT = _s64(0xE91A55E91A55E91A % (1 << 64))

    def __init__(self, orders, slots: int, dev: torch.device,
                 probe_max: int, vocab: int, M: int = 16):
        self.orders = list(orders)
        self.kidx = {k: i for i, k in enumerate(orders)}
        self.O = len(self.orders)
        self.R = self.O + 2                 # + csem row + epi row
        self.row_csem = self.O
        self.row_epi = self.O + 1
        self.S = slots
        self.V = vocab
        self.M = M
        self.dev = dev
        self.probe_max = probe_max
        # ONE key matrix shared by order/csem/epi tables — a single
        # probe loop resolves every table's context per step
        self.tkeys = torch.zeros((self.R, slots), dtype=torch.int64,
                                 device=dev)
        self.traw = torch.zeros((self.R, slots), dtype=torch.int64,
                                device=dev)
        self.ttoks = torch.full((self.O, slots, M), -1,
                                dtype=torch.int32, device=dev)
        self.tcnt = torch.zeros((self.O, slots, M), dtype=torch.float32,
                                device=dev)
        self.ttot = torch.zeros((self.O, slots), dtype=torch.float32,
                                device=dev)
        # per-row load for 2-choice eviction: orders -> follower total,
        # epi -> position count, csem -> write count
        self.tocc = torch.zeros((self.R, slots), dtype=torch.float32,
                                device=dev)
        self.used = [0] * self.R
        self._ord_t = torch.arange(self.R, dtype=torch.int64,
                                   device=dev)
        self._ordx = torch.arange(self.O, dtype=torch.int64,
                                  device=dev)
        salts = [_s64((k * 0x9E3779B97F4A7C15) % (1 << 64))
                 for k in self.orders]
        salts += [self.ROW_CSEM_SALT, self.ROW_EPI_SALT]
        self._salt = torch.tensor(salts, dtype=torch.int64, device=dev)

    def ctx_hash(self, k: int, ctx_keys: torch.Tensor) -> torch.Tensor:
        return _mix64(ctx_keys * _MIX1 + _s64(
            (k * 0x9E3779B97F4A7C15) % (1 << 64)))

    def ctx_hash_all(self, ctx_keys: torch.Tensor) -> torch.Tensor:
        """pk for an [R] vector of context keys (orders+csem+epi)."""
        return _mix64(ctx_keys * _MIX1 + self._salt[:ctx_keys.numel()])

    def _slots2(self, pk: torch.Tensor) -> tuple[torch.Tensor,
                                                torch.Tensor]:
        """Two candidate slots per key (d-left / 2-choice hashing).
        Deterministic: writes and reads probe the same pair, so a key
        is always found at one of its two slots — no probe chain."""
        s1 = pk & (self.S - 1)
        s2 = (pk * _MIX2 + _MIX1) & (self.S - 1)
        return s1, s2

    def probe_fixed(self, pk: torch.Tensor,
                    iters: int = 0,
                    rows: torch.Tensor | None = None
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read probe over the shared [R,S] key matrix: each key
        checks its two candidate slots.  ~9 ops, zero host syncs —
        graph-safe.  ``rows`` overrides the default first-n rows so
        arbitrary (row, key) batches probe in one call."""
        n = pk.numel()
        if rows is None:
            rows = self._ord_t[:n]
        s1, s2 = self._slots2(pk)
        kr = self.tkeys
        f1 = kr[rows, s1] == pk
        f2 = kr[rows, s2] == pk
        found = f1 | f2
        slot = torch.where((~f1) & f2, s2, s1)
        return slot, found

    def probe_write(self, pk: torch.Tensor) -> tuple[
            torch.Tensor, torch.Tensor, torch.Tensor]:
        """Write probe over [R,S]: (slot, found, evict).  Insert
        prefers the empty candidate; when both are occupied the row
        with the smaller load (``tocc``) is evicted — bounded-memory
        LRU-ish forgetting."""
        n = pk.numel()
        rows = self._ord_t[:n]
        s1, s2 = self._slots2(pk)
        kr, oc = self.tkeys, self.tocc
        f1 = kr[rows, s1] == pk
        f2 = kr[rows, s2] == pk
        found = f1 | f2
        e1 = kr[rows, s1] == 0
        e2 = kr[rows, s2] == 0
        evict = ~found & ~e1 & ~e2
        pick2 = ((~f1) & f2) | ((~found) & (~e1) & e2) \
            | (evict & (oc[rows, s2] < oc[rows, s1]))
        return torch.where(pick2, s2, s1), found, evict

    def _probe(self, tab_keys: torch.Tensor,
               pk: torch.Tensor,
               check_every: int = 0) -> tuple[torch.Tensor,
                                              torch.Tensor]:
        """1-D read probe on a single table row (``tab_keys`` [S]).
        Two-choice lookup — no chain, no syncs."""
        s1, s2 = self._slots2(pk)
        f1 = tab_keys[s1] == pk
        f2 = tab_keys[s2] == pk
        return torch.where((~f1) & f2, s2, s1), f1 | f2

    def _probe_row_w(self, row: int, pk: torch.Tensor) -> tuple[
            torch.Tensor, torch.Tensor, torch.Tensor]:
        """1-D write probe on ``tkeys[row]`` — same policy as
        ``probe_write``."""
        s1, s2 = self._slots2(pk)
        kr, oc = self.tkeys[row], self.tocc[row]
        k1, k2 = kr[s1], kr[s2]
        f1, f2 = k1 == pk, k2 == pk
        found = f1 | f2
        e1, e2 = k1 == 0, k2 == 0
        evict = ~found & ~e1 & ~e2
        pick2 = ((~f1) & f2) | ((~found) & (~e1) & e2) \
            | (evict & (oc[s2] < oc[s1]))
        return torch.where(pick2, s2, s1), found, evict

    def read_row(self, k: int, ctx_key: int):
        """(toks[M], cnts[M], tot) for one context — predict path."""
        oi = self.kidx[k]
        pk = self.ctx_hash(k, torch.tensor(
            [_s64(ctx_key)], dtype=torch.int64, device=self.dev))
        slot, found = self._probe(self.tkeys[oi], pk)
        s = int(slot[0])
        if not bool(found[0]):
            z = torch.zeros(self.M, dtype=torch.float32,
                            device=self.dev)
            return torch.full((self.M,), -1, dtype=torch.int32,
                              device=self.dev), z, 0.0
        return (self.ttoks[oi][s], self.tcnt[oi][s],
                float(self.ttot[oi][s]))

    def read_tot(self, k: int, ctx_keys: torch.Tensor) -> torch.Tensor:
        oi = self.kidx[k]
        pk = self.ctx_hash(k, ctx_keys)
        slot, found = self._probe(self.tkeys[oi], pk)
        return torch.where(found, self.ttot[oi][slot],
                           torch.zeros((), device=self.dev))

    def add(self, k: int, ctx_keys: torch.Tensor, toks: torch.Tensor,
            d: torch.Tensor) -> torch.Tensor:
        """Batched follower-count writes into one order's rows.
        Fully sync-free — masked writes run unconditionally (empty
        index sets are legal); returns the device-side count of new
        EMPTY-slot inserts for deferred ``used`` bookkeeping."""
        oi = self.kidx[k]
        keys, raw, rt, rc, rt_tot = (self.tkeys[oi], self.traw[oi],
                                     self.ttoks[oi], self.tcnt[oi],
                                     self.ttot[oi])
        pk = self.ctx_hash(k, ctx_keys)
        slot, found, evict = self._probe_row_w(oi, pk)
        new = ~found
        keys[slot[new]] = pk[new]
        raw[slot[new]] = ctx_keys[new]
        ev = new & evict                        # victim rows: wipe
        rt[slot[ev]] = -1
        rc[slot[ev]] = 0.0
        rt_tot[slot[ev]] = 0.0
        rows_t = rt[slot]                          # [N, M]
        match = rows_t == toks[:, None].to(torch.int32)
        has = match.any(dim=1)
        empty = rows_t == -1
        has_empty = empty.any(dim=1)
        first_empty = empty.float().argmax(dim=1)
        rows_c = rc[slot]
        min_col = rows_c.argmin(dim=1)
        # per-slot rank among distinct tokens so same-row writes in
        # this batch land on distinct columns (not all on first_empty)
        ord_ = torch.argsort(slot * (self.V + 1) + toks)
        g, tk_s = slot[ord_], toks[ord_]
        is_new = ~((g[1:] == g[:-1]) & (tk_s[1:] == tk_s[:-1]))
        rank_g = torch.cat([torch.ones(1, dtype=torch.int64,
                                       device=self.dev),
                            is_new.long()]).cumsum(0) - 1
        row_first = torch.searchsorted(g, g)
        inv = torch.empty_like(ord_)
        inv[ord_] = torch.arange(ord_.numel(), device=self.dev)
        rank = (rank_g - rank_g[row_first])[inv]
        col = torch.where(
            has, match.float().argmax(dim=1),
            torch.where(has_empty,
                        (first_empty + rank).clamp(max=self.M - 1),
                        min_col))
        # non-existing followers need their token id written into the
        # chosen column (empty slot AND evict-min cases)
        write_tok = ~has
        vic = rows_c.gather(1, col[:, None]).squeeze(1)
        rt[slot[write_tok], col[write_tok]] = \
            toks[write_tok].to(torch.int32)
        ev_c = write_tok & ~has_empty
        rc[slot[ev_c], col[ev_c]] = 0.0
        flat = slot * self.M + col
        rc.view(-1).index_add_(0, flat, d)
        rt_tot.index_add_(0, slot[ev_c], -vic[ev_c])
        rt_tot.index_add_(0, slot, d)
        self.tocc[oi][slot] = rt_tot[slot]
        return (new & ~evict).sum()

    def add_multi(self, ctx_keys: torch.Tensor, toks: torch.Tensor,
                  d: torch.Tensor, valid: torch.Tensor
                  ) -> torch.Tensor:
        """All orders in ONE call: ``ctx_keys``/``toks``/``d``/``valid``
        are [O, N].  ~40 tensor ops total instead of ~20 per order.
        Same (row, slot) collisions inside the batch get distinct
        follower columns via intra-group rank, so results match
        sequential ``add`` exactly."""
        O, N = ctx_keys.shape
        dev = self.dev
        pk = _mix64(ctx_keys * _MIX1 + self._salt[:O, None])
        rf = self._ordx[:, None].expand(O, N).reshape(-1)
        pf = pk.reshape(-1)
        vf = valid.reshape(-1)
        tf = toks.reshape(-1)
        df = d.reshape(-1)
        s1, s2 = self._slots2(pf)
        k1 = self.tkeys[rf, s1]
        k2 = self.tkeys[rf, s2]
        f1 = k1 == pf
        f2 = k2 == pf
        found = f1 | f2
        e1, e2 = k1 == 0, k2 == 0
        evict = ~found & ~e1 & ~e2 & vf
        pick2 = ((~f1) & f2) | ((~found) & (~e1) & e2) \
            | (evict & (self.tocc[rf, s2] < self.tocc[rf, s1]))
        slot = torch.where(pick2, s2, s1)
        row_ix = rf * self.S + slot                       # [B]
        new = ~found & vf
        # capture victim rows BEFORE any write touches their slots
        # (exact undo needs the displaced key + followers + total)
        evr = row_ix[evict]                             # evict &= vf
        evict_snap = (
            evr,
            self.traw.view(-1).index_select(0, evr).clone(),
            self.ttoks.view(-1, self.M).index_select(0, evr).clone(),
            self.tcnt.view(-1, self.M).index_select(0, evr).clone(),
            self.ttot.view(-1).index_select(0, evr).clone())
        tk_v = self.tkeys.view(-1)
        tr_v = self.traw.view(-1)
        tk_v.index_copy_(0, row_ix[new], pf[new])
        tr_v.index_copy_(0, row_ix[new],
                         ctx_keys.reshape(-1)[new])
        self.ttoks.view(-1, self.M).index_fill_(0, evr, -1)
        self.tcnt.view(-1, self.M).index_fill_(0, evr, 0.0)
        self.ttot.view(-1).index_fill_(0, evr, 0.0)
        # follower columns — gather post-wipe rows
        rows_t = self.ttoks.view(-1, self.M)[row_ix]    # [B, M]
        rows_c = self.tcnt.view(-1, self.M)[row_ix]
        match = rows_t == tf[:, None].to(torch.int32)
        has = match.any(dim=1)
        empty = rows_t == -1
        has_empty = empty.any(dim=1)
        first_empty = empty.float().argmax(dim=1)
        min_col = rows_c.argmin(dim=1)
        # intra-(row,slot) rank over distinct tokens: positions sharing
        # a row with different followers must take different columns.
        # Rank must be PER-ROW — a global pair-rank offset makes
        # first_empty+rank overflow into col M-1 collisions.
        ord_ = torch.argsort(row_ix * (self.V + 1) + tf)
        g = row_ix[ord_]
        tk_s = tf[ord_]
        is_new_tok = ~((g[1:] == g[:-1]) & (tk_s[1:] == tk_s[:-1]))
        rank_g = torch.cat([torch.ones(1, dtype=torch.int64,
                                       device=dev),
                            is_new_tok.long()]).cumsum(0) - 1
        row_first = torch.searchsorted(g, g)
        rank_row = rank_g - rank_g[row_first]
        inv = torch.empty_like(ord_)
        inv[ord_] = torch.arange(ord_.numel(), device=dev)
        rank = rank_row[inv]
        col = torch.where(
            has, match.float().argmax(dim=1),
            torch.where(has_empty,
                        (first_empty + rank).clamp(max=self.M - 1),
                        min_col))
        write_tok = ~has & vf
        cell = row_ix * self.M + col
        # column-evict victims (full row -> min column displaced):
        # capture (cell, tok, cnt) BEFORE the write so undo restores
        # exactly; ttot must also drop the displaced mass
        rep = write_tok & ~has_empty
        ttfl = self.ttoks.view(-1)
        tcfl = self.tcnt.view(-1)
        col_snap = (cell[rep],
                    ttfl.index_select(0, cell[rep]).clone(),
                    tcfl.index_select(0, cell[rep]).clone())
        ttv = self.ttoks.view(-1)
        tcv = self.tcnt.view(-1)
        ttv.index_copy_(
            0, cell[write_tok], tf[write_tok].to(torch.int32))
        tcv[cell[write_tok & ~has_empty]] = 0.0
        zv = torch.zeros((), device=dev)
        tcv.index_add_(0, cell, torch.where(vf, df, zv))
        self.ttot.view(-1).index_add_(
            0, row_ix[rep], -col_snap[2])
        self.ttot.view(-1).index_add_(
            0, row_ix, torch.where(vf, df, zv))
        oc = self.tocc.view(-1)
        oc.index_copy_(
            0, row_ix,
            torch.where(vf,
                        self.ttot.view(-1).index_select(0, row_ix),
                        oc.index_select(0, row_ix)))
        # intra-batch slot collisions: two distinct new keys (or a new
        # key evicting a found row) can pick the same slot — the losing
        # key's follower landed in the winner's row.  Detect elements
        # whose key didn't stick, undo their misplaced cell exactly,
        # and re-insert them sequentially (rare: ~birthday collisions).
        stuck = vf & (tk_v.index_select(0, row_ix) != pf)
        extra_victims = []
        ins_list = []
        for j in torch.nonzero(stuck).squeeze(1).tolist():
            cell_j = cell[j]
            # undo the misplaced follower write
            tcv.index_add_(0, cell_j.reshape(1), -df[j].reshape(1))
            if float(tcv[cell_j]) <= 1e-9:
                ttv[cell_j] = -1
            row_j = row_ix[j]
            self.ttot.view(-1).index_add_(
                0, row_j.reshape(1), -df[j].reshape(1))
            # sequential re-insert at a fresh probe of the live table
            oi = int(rf[j]); pkj = pf[j:j + 1]
            s_j, f_j, e_j = self._probe_row_w(oi, pkj)
            sj = int(s_j[0]); fj = bool(f_j[0]); ej = bool(e_j[0])
            flat_j = oi * self.S + sj
            if not fj:
                if ej:   # evict victim at the new slot
                    extra_victims.append((
                        torch.tensor([flat_j], dtype=torch.int64,
                                     device=dev),
                        self.traw.view(-1)[flat_j].reshape(1).clone(),
                        self.ttoks.view(-1, self.M)[flat_j]
                            .reshape(1, -1).clone(),
                        self.tcnt.view(-1, self.M)[flat_j]
                            .reshape(1, -1).clone(),
                        self.ttot.view(-1)[flat_j].reshape(1).clone()))
                    self.ttoks[oi, sj] = -1
                    self.tcnt[oi, sj] = 0.0
                    self.ttot[oi, sj] = 0.0
                self.tkeys[oi, sj] = pkj[0]
                self.traw[oi, sj] = ctx_keys.reshape(-1)[j]
                if not ej:
                    ins_list.append(oi)
                row_t = self.ttoks[oi, sj]
                m_j = row_t == int(tf[j])
                if bool(m_j.any()):
                    cj = int(m_j.float().argmax())
                else:
                    em = row_t == -1
                    cj = (int(em.float().argmax()) if bool(em.any())
                          else int(self.tcnt[oi, sj].argmin()))
                    self.ttoks[oi, sj, cj] = int(tf[j])
            else:        # found after all — locate follower column
                row_t = self.ttoks[oi, sj]
                m_j = row_t == int(tf[j])
                em = row_t == -1
                cj = (int(m_j.float().argmax()) if bool(m_j.any())
                      else int(em.float().argmax()) if bool(em.any())
                      else int(self.tcnt[oi, sj].argmin()))
                if not bool(m_j.any()):
                    self.ttoks[oi, sj, cj] = int(tf[j])
            self.tcnt[oi, sj, cj] += float(df[j])
            self.ttot[oi, sj] += float(df[j])
            self.tocc[oi, sj] = float(self.ttot[oi, sj])
        if extra_victims:
            evict_snap = tuple(
                torch.cat([evict_snap[i]] +
                          [v[i] for v in extra_victims])
                for i in range(5))
        # per-row distinct inserts (for `used` bookkeeping) — one
        # bincount; caller does a single .tolist() D2H
        ins = torch.unique(row_ix[new & ~evict])
        if ins_list:
            ins = torch.unique(torch.cat(
                [ins, torch.tensor(ins_list, dtype=torch.int64,
                                   device=dev)]))
        return torch.bincount(ins // self.S, minlength=self.R), \
            evict_snap, col_snap

    def add_rows(self, ctx_keys: torch.Tensor, toks: torch.Tensor,
                 d: torch.Tensor,
                 valid: torch.Tensor) -> torch.Tensor:
        """Sync-free write across ALL orders at once: ctx_keys/toks/d
        are [O]-shaped (one write per order), ``valid`` masks dead
        orders.  Returns the [O] new-key mask for host bookkeeping."""
        pk = self.ctx_hash_all(ctx_keys)
        slot, found, evict = self.probe_write(pk)
        new = (~found) & valid
        self.tkeys[self._ordx, slot] = torch.where(
            new, pk, self.tkeys[self._ordx, slot])
        self.traw[self._ordx, slot] = torch.where(
            new, ctx_keys, self.traw[self._ordx, slot])
        wipe = (evict & valid)[:, None]           # victim row reset
        self.ttoks[self._ordx, slot] = torch.where(
            wipe, torch.full((), -1, dtype=torch.int32,
                             device=self.dev),
            self.ttoks[self._ordx, slot])
        self.tcnt[self._ordx, slot] = torch.where(
            wipe, torch.zeros((), device=self.dev),
            self.tcnt[self._ordx, slot])
        self.ttot[self._ordx, slot] = torch.where(
            evict & valid, torch.zeros((), device=self.dev),
            self.ttot[self._ordx, slot])
        rows_t = self.ttoks[self._ordx, slot]       # [O, M]
        rows_c = self.tcnt[self._ordx, slot]        # [O, M]
        match = rows_t == toks[:, None].to(torch.int32)
        has = match.any(dim=1)
        empty = rows_t == -1
        has_empty = empty.any(dim=1)
        first_empty = empty.float().argmax(dim=1)
        min_col = rows_c.argmin(dim=1)
        col = torch.where(has, match.float().argmax(dim=1),
                          torch.where(has_empty, first_empty, min_col))
        replace = (~has) & (~has_empty) & valid
        vic_c = rows_c.gather(1, col[:, None]).squeeze(1)
        rt_flat = self.ttoks.view(self.O, -1)
        rc_flat = self.tcnt.view(self.O, -1)
        flat = slot * self.M + col
        # evicted-min replacement: overwrite token, zero its count
        rep_tok = torch.where(replace, toks.to(torch.int32),
                              torch.zeros(self.O, dtype=torch.int32,
                                          device=self.dev))
        rt_flat[self._ordx, flat] = torch.where(
            replace, rep_tok,
            rt_flat[self._ordx, flat])
        rc_flat[self._ordx, flat] = torch.where(
            replace, torch.zeros((), device=self.dev),
            rc_flat[self._ordx, flat])
        rc_flat[self._ordx, flat] += torch.where(
            valid, d, torch.zeros((), device=self.dev))
        self.ttot[self._ordx, slot] += torch.where(
            replace, -vic_c, torch.zeros((), device=self.dev))
        self.ttot[self._ordx, slot] += torch.where(
            valid, d, torch.zeros((), device=self.dev))
        self.tocc[self._ordx, slot] = self.ttot[self._ordx, slot]
        return new & ~evict

    def read_pk(self, k: int, ctx_keys: torch.Tensor,
                toks: torch.Tensor) -> torch.Tensor:
        """Count of (key,tok) for a batch — reward/gap probes."""
        oi = self.kidx[k]
        pk = self.ctx_hash(k, ctx_keys)
        slot, found = self._probe(self.tkeys[oi], pk)
        rows_t = self.ttoks[oi][slot]
        match = rows_t == toks[:, None].to(torch.int32)
        c = torch.where(match, self.tcnt[oi][slot],
                        torch.zeros((), device=self.dev)).sum(dim=1)
        return torch.where(found, c, torch.zeros((), device=self.dev))

    def subtract(self, k: int, ctx_keys: torch.Tensor,
                 toks: torch.Tensor, d: torch.Tensor) -> int:
        """Decrement existing (key,tok) counts only — undo path; never
        inserts.  Returns #cells actually decremented."""
        oi = self.kidx[k]
        pk = self.ctx_hash(k, ctx_keys)
        slot, found = self._probe(self.tkeys[oi], pk)
        if not bool(found.any()):
            return 0
        rows_t = self.ttoks[oi][slot]
        match = rows_t == toks[:, None].to(torch.int32)
        msum = match.any(dim=1) & found
        if not bool(msum.any()):
            return 0
        cols = match.float().argmax(dim=1)
        flat = (slot * self.M + cols)[msum]
        self.tcnt[oi].view(-1).index_add_(0, flat, -d[msum])
        self.tcnt[oi].clamp_(min=0.0)
        self.ttot[oi].index_add_(0, slot[msum], -d[msum])
        self.ttot[oi].clamp_(min=0.0)
        # hygiene: columns emptied by the subtract revert to -1; a row
        # that loses all mass is cleared wholesale (CPU dicts pop empty
        # cells — the GPU table must match or undo leaves dead keys).
        # Thresholds are above fp32 index_add residue (~1e-7) but far
        # below the minimum real write (surprise_floor ~0.05).
        dead = self.tcnt[oi] <= 1e-6
        self.ttoks[oi][dead] = -1
        rows_s = slot[msum].unique()
        empt = rows_s[self.ttot[oi][rows_s] <= 1e-5]
        if empt.numel():
            self.tkeys[oi, empt] = 0
            self.traw[oi, empt] = 0
            self.ttoks[oi, empt] = -1
            self.tcnt[oi, empt] = 0.0
            self.ttot[oi, empt] = 0.0
            self.used[oi] -= int(empt.numel())
        self.tocc[oi].copy_(self.ttot[oi])
        return int(msum.sum())

    def _pick_slot(self, oi: int, pk: torch.Tensor) -> tuple[
            int, bool]:
        """Undo helper: slot for a re-insert of a displaced row.
        Prefers the slot still holding the key, then an empty slot,
        then the lower-load candidate (exact only if it's dead)."""
        s1, s2 = self._slots2(pk)
        s1i, s2i = int(s1[0]), int(s2[0])
        pkv = int(pk[0])
        k1 = int(self.tkeys[oi, s1i])
        k2 = int(self.tkeys[oi, s2i])
        if k1 == pkv or k1 == 0:
            return s1i, True
        if k2 == pkv or k2 == 0:
            return s2i, True
        t1 = float(self.tocc[oi, s1i])
        t2 = float(self.tocc[oi, s2i])
        s = s1i if t1 <= t2 else s2i
        return s, float(self.tocc[oi, s]) == 0.0

    def restore_row(self, flat: int, raw: int, toks: torch.Tensor,
                    cnts: torch.Tensor, tot: float) -> bool:
        """Re-insert an evicted follower row (undo path).  Exact when
        the victim's slot is free or holds a dead (subtracted) row."""
        oi = int(flat) // self.S
        dev = self.dev
        rt = torch.tensor([_s64(raw)], dtype=torch.int64, device=dev)
        pk = _mix64(rt * _MIX1 + self._salt[oi])
        s, exact = self._pick_slot(oi, pk)
        if int(self.tkeys[oi, s]) == 0:
            self.used[oi] += 1
        self.tkeys[oi, s] = pk[0]
        self.traw[oi, s] = _s64(raw)
        self.ttoks[oi, s] = toks
        self.tcnt[oi, s] = cnts
        self.ttot[oi, s] = tot
        self.tocc[oi, s] = tot
        return exact

    def decay(self, k: int, floor: float = 0.25) -> None:
        oi = self.kidx[k]
        self.tcnt[oi].mul_(0.5)
        self.tcnt[oi][self.tcnt[oi] < floor] = 0.0
        self.ttot[oi].mul_(0.5)
        self.tocc[oi].copy_(self.ttot[oi])

    def rehash(self, k: int) -> None:
        """Rebuild dropping rows whose counts all decayed to zero."""
        oi = self.kidx[k]
        keys = self.tkeys[oi]
        live = (keys != 0) & (self.ttot[oi] > 0)
        pk = keys[live].clone()
        rw = self.traw[oi][live].clone()
        rows_t = self.ttoks[oi][live].clone()
        rows_c = self.tcnt[oi][live].clone()
        tots = self.ttot[oi][live].clone()
        self.tkeys[oi].zero_()
        self.traw[oi].zero_()
        self.ttoks[oi].fill_(-1)
        self.tcnt[oi].zero_()
        self.ttot[oi].zero_()
        self.used[oi] = 0
        if pk.numel():
            slot, _, _ = self._probe_row_w(oi, pk)
            keys[slot] = pk
            self.traw[oi][slot] = rw
            self.ttoks[oi][slot] = rows_t
            self.tcnt[oi][slot] = rows_c
            self.ttot[oi][slot] = tots
            self.tocc[oi][slot] = tots
            self.used[oi] = int(pk.numel())

    def triplets(self) -> dict:
        """{k: (ctx_keys_u64, toks_i64, cnts_f32)} decoded rows."""
        out = {}
        for k in self.orders:
            oi = self.kidx[k]
            nz = (self.tkeys[oi] != 0) & (self.ttot[oi] > 0)
            rks = self.traw[oi][nz].cpu().numpy()
            rt = self.ttoks[oi][nz].cpu().numpy()
            rc = self.tcnt[oi][nz].cpu().numpy()
            kk, tt, cc = [], [], []
            for i, key in enumerate(rks):
                for j in range(self.M):
                    if rt[i, j] >= 0 and rc[i, j] != 0:
                        kk.append(np.uint64(int(key) % (1 << 64)))
                        tt.append(int(rt[i, j]))
                        cc.append(float(rc[i, j]))
            out[k] = (np.asarray(kk, dtype=np.uint64),
                      np.asarray(tt, dtype=np.int64),
                      np.asarray(cc, dtype=np.float32))
        return out

    def load_triplets(self, tri: dict) -> None:
        dev = self.dev
        for k in self.orders:
            ent = tri.get(k, tri.get(str(k)))
            if not ent:
                continue
            kk_np, tt_np, cc_np = ent
            if len(kk_np) == 0:
                continue
            kk = torch.from_numpy(kk_np.astype(np.int64)).to(dev)
            tt = torch.from_numpy(tt_np.astype(np.int64)).to(dev)
            dd = torch.from_numpy(cc_np.astype(np.float32)).to(dev)
            self.used[self.kidx[k]] += int(self.add(k, kk, tt, dd))


class FluxLM(nn.Module):
    """Sparse online associative-memory LM.  See module docstring."""

    # Engine capability flags (explicit-compat contract).
    unbounded_context = True     # engine must not truncate the prompt
    supports_kv_cache = False    # state is internal, no past_key_values

    def __init__(self, config: FluxConfig | None = None):
        super().__init__()
        self.config = config or FluxConfig()
        c = self.config
        self.vocab_size = c.vocab_size
        self.eos_token_id = c.eos_token_id

        V = c.vocab_size
        # ── channel weights (Hedge-adapted) ────────────────────────────
        self.channels = [f"o{k}" for k in c.orders] + ["deep", "topic",
                                                       "sem", "csem",
                                                       "knn",
                                                       "recency",
                                                       "fatigue", "uni",
                                                       "backoff"]
        self._chix = {ch: i for i, ch in enumerate(self.channels)}
        self._w = np.ones(len(self.channels), dtype=np.float64)
        self._w /= self._w.sum()
        self._hedge_mix = 0.0        # AdaHedge cumulative mix gap

        # ── sparse memory cells ────────────────────────────────────────
        self._tables: dict[int, dict[int, dict[int, float]]] = {
            k: {} for k in c.orders}
        # per-cell follower totals (cache — avoids O(followers) sums)
        self._totals: dict[int, dict[int, float]] = {
            k: {} for k in c.orders}
        self._uni = np.zeros(V, dtype=np.float64)
        self._total = 0

        # device: dense readout tensors live here; sparse memory stays host.
        self._dev = torch.device(
            c.device if (c.device == "cpu" or torch.cuda.is_available())
            else "cpu")
        self._log_uni_num = torch.full(
            (V,), math.log(c.uni_alpha), dtype=torch.float32,
            device=self._dev)

        # ── episodic ring + prefix-hash ring ───────────────────────────
        self._ring = np.zeros(c.ring_capacity, dtype=np.int32)
        self._H = np.zeros(c.ring_capacity, dtype=np.uint64)
        self._count = 0                     # absolute append position
        # key -> deque of (ctx_end, succ, certs) — succ stored at insert
        # time (deferred by one token) so the entry survives ring wrap;
        # certs = tuple of suffix hashes at config.cert_orders (0 = n/a)
        self._epi: dict[int, deque] = {}
        self._epi_fifo: deque = deque()     # (key, ctx_end) insert order
        self._epi_n = 0                     # live entries (global cap)
        self._epi_hits: dict[int, int] = {}  # key -> cells voted (utility)
        self._ep_start = 0                  # current episode start (abs pos)
        self._pows = np.empty(
            max(max(c.orders), c.deep_order, c.csem_order,
                *(c.cert_orders or (0,))) + 1,
            dtype=np.uint64)
        self._pows[0] = 1
        for i in range(1, len(self._pows)):
            self._pows[i] = (int(self._pows[i - 1]) * 1099511628211) \
                % (1 << 64)

        # ── topic channel (dense tensors on self._dev) ─────────────────
        g = torch.Generator().manual_seed(c.seed)
        self._R = ((torch.randint(0, 2, (V, c.topic_dim),
                                  generator=g, dtype=torch.int8) * 2 - 1)
                   .to(self._dev))
        self._A = torch.zeros((V, c.topic_dim), dtype=torch.float32,
                              device=self._dev)
        self._c = torch.zeros(c.topic_dim, dtype=torch.float32,
                              device=self._dev)
        # expected-next-token fingerprint (semantic-spread channel)
        self._proto = torch.zeros(c.topic_dim, dtype=torch.float32,
                                  device=self._dev)

        # ── recency channel ────────────────────────────────────────────
        self._last_seen: dict[int, int] = {}
        self._recent: deque = deque()       # (abs_pos, tok) for iteration
        # set by predict-only stream facades (forge/model/flux_stream):
        # gates the shared-memory writes in _learn_token while context
        # state (ring/hash/seen/hedge/uni/proto EMAs) still advances
        self._predict_only = False

        # ── csem: per-context semantic fingerprints ────────────────────
        # fixed projection A -> csem_dim for cheap per-context cosine
        g2 = torch.Generator().manual_seed(c.seed ^ 0xC5E4)
        self._Pcsem = (torch.randn(c.topic_dim, c.csem_dim,
                                   generator=g2)
                       / math.sqrt(c.topic_dim)).float().to(self._dev)
        self._csem_cpu: dict[int, np.ndarray] = {}   # cpu path rows fp32
        # GPU path: fp16 rows indexed by slot in the shared key matrix
        # (gt.tkeys[gt.row_csem] holds the probe keys)
        self._csem_rows: torch.Tensor | None = None

        # ── knn fingerprint bank (approximate-match episodic memory) ──
        # row i = unit-normalized fingerprint of one past context,
        # plus the token that followed it and the context-end position.
        # Rows are unit-normalized at write so a single matvec gives
        # cosines; _knn_pos < 0 marks an empty row.
        P = c.knn_capacity
        self._knn_fp = torch.zeros(
            (P, c.csem_dim),
            dtype=torch.float16 if self._dev.type == "cuda"
            else torch.float32, device=self._dev)
        self._knn_succ = torch.full((P,), -1, dtype=torch.int32,
                                    device=self._dev)
        self._knn_pos = torch.full((P,), -1, dtype=torch.int64,
                                   device=self._dev)
        self._knn_cur = 0                   # ring cursor (host mirror)

        # ── CUDA-primary backend ───────────────────────────────────────
        self._gpu = bool(c.cuda_primary
                         and self._dev.type == "cuda")
        # triton fusion gate — eager torch fallbacks cover missing
        # triton / non-CUDA devices (defined before _gpu so helpers can
        # read it unconditionally)
        self._use_triton = bool(
            self._gpu and c.use_triton and HAS_TRITON)
        self._graph_knn_b = -1
        self._gt: _GPUCellTables | None = None
        if self._gpu:
            self._gt = _GPUCellTables(c.orders, c.gpu_table_slots,
                                      self._dev, c.gpu_probe_max, V,
                                      M=c.gpu_row_width)
            # device mirrors of host-authoritative structures
            self._Hg = torch.zeros(c.ring_capacity, dtype=torch.int64,
                                   device=self._dev)
            self._ringg = torch.zeros(c.ring_capacity, dtype=torch.int32,
                                      device=self._dev)
            self._unig = torch.zeros(V, dtype=torch.float32,
                                     device=self._dev)
            self._seen_g = torch.full((V,), -10 ** 9,
                                      dtype=torch.int64,
                                      device=self._dev)
            self._pows_g = torch.tensor(
                [int(p) if int(p) < (1 << 63)
                 else int(p) - (1 << 64) for p in self._pows],
                dtype=torch.int64, device=self._dev)
            self._orders_g = torch.tensor(
                list(c.orders), dtype=torch.int64, device=self._dev)
            self._cert_g = torch.tensor(
                list(c.cert_orders) or [0], dtype=torch.int64,
                device=self._dev)
            # device scalars — graph-safe (no host reads inside a step)
            self._pos_g = torch.zeros((), dtype=torch.int64,
                                      device=self._dev)
            self._ep_g = torch.zeros((), dtype=torch.int64,
                                     device=self._dev)
            self._tot_g = torch.zeros((), dtype=torch.float64,
                                      device=self._dev)
            self._bx = torch.zeros((), dtype=torch.int64,
                                   device=self._dev)
            # GPU episodic index: seed key -> ring of context-end
            # positions.  Keys live in gt.tkeys[gt.row_epi] (shared
            # probe matrix); payloads here are indexed by that slot.
            # The host _epi dict stays authoritative for snapshots.
            S, ER = self._gt.S, c.epi_row
            self._epi_pos = torch.full((S, ER), -1,
                                       dtype=torch.int64,
                                       device=self._dev)
            self._epi_cnt = torch.zeros(S, dtype=torch.int64,
                                        device=self._dev)
            # deferred entries carry their successor + certificate
            # suffix hashes, so off-ring positions still vote
            self._epi_succ = torch.full((S, ER), -1,
                                        dtype=torch.int32,
                                        device=self._dev)
            self._epi_cert = torch.zeros((S, ER, len(c.cert_orders)),
                                         dtype=torch.int64,
                                         device=self._dev)
            self._csem_rows = torch.zeros((S, c.csem_dim),
                                          dtype=torch.float16,
                                          device=self._dev)
            # incremental projections/norms of A — updated by the same
            # rank-1 writes that touch A, so predict never rescans it
            self._Ac = torch.zeros((V, c.csem_dim), dtype=torch.float32,
                                   device=self._dev)
            self._Acn = torch.zeros(V, dtype=torch.float32,
                                    device=self._dev)
            self._An = torch.zeros(V, dtype=torch.float32,
                                   device=self._dev)
            # GPU journal rings — pre-update state for exact undo
            # without per-token D2H reads
            RC = c.vec_delta_cap
            # +1 column: AdaHedge mix-gap rides with w for exact undo
            self._w_ring_g = torch.zeros(
                (RC, len(self.channels) + 1), dtype=torch.float64,
                device=self._dev)
            self._avec_g = torch.zeros(
                (RC, c.topic_dim), dtype=torch.float32,
                device=self._dev)
            self._csem_prev = torch.zeros(
                (RC, c.csem_dim), dtype=torch.float16,
                device=self._dev)
            self._jcur = 0
            # device channel weights (authoritative when _gpu)
            self._wg = torch.from_numpy(self._w).to(self._dev)
            # AdaHedge cumulative mix-gap (device scalar, no host sync)
            self._hmg = torch.zeros((), dtype=torch.float64,
                                    device=self._dev)
            # step readback bundle (int64 bits for everything):
            # [O keys, epi key, epi slot, epi bits, csem slot, csem
            #  bits, order bits, gap, prevseen, certs(L), knn slot,
            #  knn prev pos, knn prev succ, w(C), surprise, mix]
            L = len(c.cert_orders)
            self._boff_cert = self._gt.O + 8
            self._boff_knn = self._boff_cert + L
            self._boff_w = self._boff_knn + 3
            # +2 tail slots: predict-time deep key (raw u64, -1 invalid)
            # and the retrieval-hit flag — feeds _epi_hits on the host
            self._boff_hit = self._boff_w + len(self.channels) + 2
            self._bout = torch.zeros(
                (self._boff_hit + 2,),
                dtype=torch.int64, device=self._dev)
            # knn ring: device cursor + prev-row ring for exact undo
            self._knncur_g = torch.zeros((), dtype=torch.int64,
                                         device=self._dev)
            self._kprev_g = torch.zeros(
                (c.vec_delta_cap, c.csem_dim), dtype=torch.float16,
                device=self._dev)
            self._blogits = torch.zeros(V, dtype=torch.float32,
                                        device=self._dev)
            self._jcur_t = torch.zeros((), dtype=torch.int64,
                                       device=self._dev)
            # precomputed device scalars for fused kernels / bit-pack
            self._pow_deep = self._pows_g[c.deep_order].clone()
            self._bitw = torch.pow(
                torch.full((self._gt.O,), 2, dtype=torch.int64,
                           device=self._dev),
                torch.arange(self._gt.O, dtype=torch.int64,
                             device=self._dev))
            self._graph = None
            self._warm = 0
            # clone workers run eager (_step_dev) — concurrent CUDA-graph
            # captures on one device invalidate each other
            self._use_cuda_graph = True
            # fp64 scalar constants — pre-baked so the graphed step
            # never needs an H2D scalar copy (capture poison)
            self._const_1e3 = torch.full((), 1e-3,
                                         dtype=torch.float64,
                                         device=self._dev)
            self._const_1 = torch.full((), 1.0, dtype=torch.float64,
                                       device=self._dev)
            self._const_lnC = torch.full(
                (), math.log(len(self.channels)),
                dtype=torch.float64, device=self._dev)
            self._const_eta_max = torch.full(
                (), c.hedge_eta_max, dtype=torch.float64,
                device=self._dev)
            self._const_hedge_lr = torch.full(
                (), c.hedge_lr, dtype=torch.float64,
                device=self._dev)
            self._boff_prevseen = torch.tensor(
                [self._gt.O + 7], dtype=torch.int64,
                device=self._dev)
            # deep-channel retrieval-hit flag: set by _logits_dev each
            # step, packed into bout for host-side utility accounting
            self._dhit_g = torch.zeros(1, dtype=torch.int64,
                                       device=self._dev)
            # eviction snapshot slots (one per key row): written
            # in-graph, read by host only when evict bits are set —
            # epi rows carry pos+succ+cert payloads per cell
            EVI = 3 + max(self._gt.M,
                          c.epi_row * (2 + len(c.cert_orders)))
            EVF = 1 + max(self._gt.M, c.csem_dim)
            self._bev_i = torch.zeros(
                (self._gt.R, EVI), dtype=torch.int64,
                device=self._dev)
            self._bev_f = torch.zeros(
                (self._gt.R, EVF), dtype=torch.float32,
                device=self._dev)
            self._brow_epi = torch.tensor(
                [self._gt.row_epi], dtype=torch.int64,
                device=self._dev)
            self._brow_csem = torch.tensor(
                [self._gt.row_csem], dtype=torch.int64,
                device=self._dev)

        # ── provenance ─────────────────────────────────────────────────
        self.journal = FluxJournal(c.journal_cap, c.vec_delta_cap,
                                   c.topic_dim, len(self.channels))
        self.learning = True
        self.tag = "boot"
        self._graft: dict | None = None   # teacher-embedding graft info
        if c.teacher_embed:
            self.graft_teacher(c.teacher_embed)
        # positions where prediction was thin (for selective distill)
        self._gaps: deque = deque(maxlen=c.gap_cap)

    # ── context sketch ──────────────────────────────────────────────────

    def _suffix_key(self, k: int, end: int) -> int | None:
        """Hash of the last ``k`` tokens ending at absolute pos ``end``,
        or None if fewer than ``k`` tokens exist in the current episode."""
        if end - k + 1 < self._ep_start or end < k - 1:
            return None
        cap = self.config.ring_capacity
        h = (int(self._H[end % cap])
             - int(self._H[(end - k) % cap]) * int(self._pows[k])) \
            % (1 << 64)
        return h

    def _keys_at(self, end: int) -> dict[int, int]:
        keys = {}
        for k in self.config.orders:
            key = self._suffix_key(k, end)
            if key is not None:
                keys[k] = key
        return keys

    def _raw_hash(self, k: int, end: int) -> int:
        """Suffix hash with no episode check — used to validate ring
        positions that may belong to older episodes."""
        cap = self.config.ring_capacity
        return (int(self._H[end % cap])
                - int(self._H[(end - k) % cap]) * int(self._pows[k])) \
            % (1 << 64)

    # ── live learning ───────────────────────────────────────────────────

    def _push(self, tok: int) -> None:
        """Append ``tok`` to the episodic stream (ring + hash + index)."""
        c = self.config
        pos = self._count
        slot = pos % c.ring_capacity
        self._ring[slot] = tok
        prev_h = int(self._H[(pos - 1) % c.ring_capacity]) if pos > 0 else 0
        self._H[slot] = (prev_h * 1099511628211 + tok + 1) % (1 << 64)
        if self._gpu:
            self._ringg[slot] = tok
            if pos == 0:
                self._Hg[0] = tok + 1
            else:
                self._Hg[slot] = (self._Hg[(pos - 1) % c.ring_capacity]
                                  * 1099511628211 + tok + 1)
        # deferred episodic index: token ``tok`` is the SUCCESSOR of the
        # context ending at pos-1 — inserting now means the entry
        # carries its successor + certificate hashes inline, so it still
        # votes after the ring wraps (archived deep memory).  Memory
        # write — gated on learning; skipped for observe-only pushes.
        if self.learning:
            key = self._suffix_key(c.deep_order, pos - 1)
            if key is not None:
                certs = tuple(
                    self._suffix_key(L, pos - 1) or 0
                    for L in c.cert_orders)
                dq = self._epi.get(key)
                if dq is None:
                    dq = self._epi[key] = deque(
                        maxlen=c.epi_max_per_key)
                elif len(dq) == c.epi_max_per_key:
                    # deque append will drop the oldest — journal it
                    # first so the write is exactly revertible
                    old = dq[0]
                    self.journal.append(
                        ("eE", key, old[0], old[1], old[2]))
                    self._epi_n -= 1
                dq.append((pos - 1, tok, certs))
                self._epi_fifo.append((key, pos - 1))
                self._epi_n += 1
                self._epi_evict()
                self.journal.append(("e", key, pos - 1))
            # knn bank write: fingerprint of the pre-``tok`` context
            # (self._c is still the context EMA — updated below)
            if c.use_knn:
                self._knn_write(tok, pos - 1)
                if self._gpu:
                    self._jcur = (self._jcur + 1) % c.vec_delta_cap
        # topic EMA: c <- lambda*c + (1-lambda)*r_tok
        self._c.mul_(c.topic_decay).add_(
            self._R[tok].float() * (1.0 - c.topic_decay))
        self._count += 1
        if self._gpu:
            self._pos_g += 1

    def _knn_fp_of(self, cvec: torch.Tensor) -> torch.Tensor:
        """Unit-normalized context fingerprint of a context vector."""
        fp = cvec.float() @ self._Pcsem
        return fp / (fp.norm() + 1e-8)

    def _knn_rows(self) -> int:
        """Bank rows worth scanning — the live ring prefix rounded up
        to a power-of-2 bucket (>= ``knn_scan_min``).  Bucketing keeps
        the slice shape static inside a captured CUDA graph between
        recapture boundaries; unwritten padding rows carry pos=-1 and
        are masked out by the caller as usual."""
        n = int(self._knn_cur)
        if n <= 0:
            return 0
        cap = self.config.knn_capacity
        if n >= cap:
            return cap
        b = max(1, self.config.knn_scan_min)
        while b < n:
            b <<= 1
        return min(b, cap)

    def _knn_write(self, succ: int, ctx_end: int,
                   fp: torch.Tensor | None = None,
                   journal: bool = True) -> int:
        """Write one fingerprint-bank row: (fp of context ending at
        ``ctx_end``, successor ``succ``).  Returns the slot used."""
        c = self.config
        slot = self._knn_cur % c.knn_capacity
        if fp is None:
            fp = self._knn_fp_of(self._c)
        prev_pos = int(self._knn_pos[slot])
        if journal:
            if self._gpu:
                prev_succ = int(self._knn_succ[slot])
                self._kprev_g[self._jcur] = self._knn_fp[slot]
                self.journal.append(
                    ("k", slot, prev_pos, prev_succ,
                     self._jcur if prev_pos >= 0 else -1))
            else:
                prev = (None if prev_pos < 0 else
                        (prev_pos, int(self._knn_succ[slot]),
                         self._knn_fp[slot].cpu().numpy().copy()))
                self.journal.append(("k", slot, prev))
        self._knn_fp[slot] = fp.to(self._knn_fp.dtype)
        self._knn_succ[slot] = succ
        self._knn_pos[slot] = ctx_end
        self._knn_cur += 1
        if self._gpu:
            self._knncur_g += 1       # keep device cursor in sync
        return slot

    def _epi_drop(self, key: int, pos: int) -> bool:
        """Drop one (key, ctx_end) entry — journaled (``eE``) so a
        revert restores it, and the GPU row is resynced so evicted
        cells stop voting device-side."""
        dq = self._epi.get(key)
        if not dq:
            return False
        for ent in dq:
            if ent[0] == pos:
                self.journal.append(
                    ("eE", key, ent[0], ent[1], ent[2]))
                dq.remove(ent)
                self._epi_n -= 1
                break
        else:
            return False
        if not dq:
            self._epi.pop(key, None)
            self._epi_hits.pop(key, None)
        self._epi_sync_row(key)
        return True

    def _epi_sync_row(self, key: int) -> None:
        """GPU mirror upkeep: rewrite the episodic row for ``key`` from
        the host deque (post evict/consolidate).  Without this the
        device table kept voting evicted cells — the host dict is the
        authority.  No-op off-GPU or when the row isn't present."""
        if not self._gpu:
            return
        c = self.config
        dk_t = torch.tensor([_s64(key)], dtype=torch.int64,
                            device=self._dev)
        pk = self._epi_hash(dk_t)
        se, fe = self._epi_probe(pk)
        if not bool(fe[0]):
            return
        s = int(se[0])
        gt = self._gt
        oi = gt.row_epi
        dq = self._epi.get(key)
        ER = self._epi_pos.shape[1]
        if not dq:                               # key fully evicted
            gt.tkeys[oi, s] = 0
            gt.traw[oi, s] = 0
            gt.tocc[oi, s] = 0.0
            self._epi_pos[s].fill_(-1)
            self._epi_succ[s].fill_(-1)
            if len(c.cert_orders):
                self._epi_cert[s].zero_()
            self._epi_cnt[s] = 0
            gt.used[oi] -= 1
            return
        pos = torch.full((ER,), -1, dtype=torch.int64,
                         device=self._dev)
        succ = torch.full((ER,), -1, dtype=torch.int32,
                          device=self._dev)
        LC = len(c.cert_orders)
        cert = torch.zeros((ER, LC), dtype=torch.int64,
                           device=self._dev) if LC else None
        for j, ent in enumerate(list(dq)[-ER:]):
            pos[j] = int(ent[0])
            succ[j] = int(ent[1])
            if LC:
                for l, cv in enumerate(ent[2] or ()):
                    cert[j, l] = _s64(cv)
        self._epi_pos[s] = pos
        self._epi_succ[s] = succ
        if LC:
            self._epi_cert[s] = cert
        self._epi_cnt[s] = min(len(dq), ER)
        gt.tocc[oi, s] = float(self._epi_cnt[s])

    def _epi_evict(self) -> None:
        """Utility-bounded episodic memory: when over ``epi_total_cap``,
        scan the oldest ``epi_evict_scan`` fifo candidates and evict the
        lowest-utility entries — score = retrieval hits (cells that
        actually get read dominate) + a small recency term.  FIFO is the
        degenerate case (scan=1).  Every drop is journaled (``eE``)."""
        c = self.config
        n_inv = 1.0 / max(self._count, 1)
        while self._epi_n > c.epi_total_cap and self._epi_fifo:
            over = self._epi_n - c.epi_total_cap
            take = min(len(self._epi_fifo),
                       max(c.epi_evict_scan, over))
            batch = [self._epi_fifo.popleft() for _ in range(take)]
            live = [kp for kp in batch
                    if any(e[0] == kp[1]
                           for e in self._epi.get(kp[0], ()))]
            if len(live) <= over:                # mostly tombstones
                for key, pos in live:
                    self._epi_drop(key, pos)
                continue
            scored = sorted(
                live,
                key=lambda kp: (self._epi_hits.get(kp[0], 0)
                                * c.epi_hit_w + kp[1] * n_inv))
            for key, pos in scored[:over]:       # lowest utility out
                self._epi_drop(key, pos)
            for kp in reversed(scored[over:]):   # survivors stay oldest
                self._epi_fifo.appendleft(kp)

    def _epi_remove(self, key: int, pos: int) -> bool:
        """Undo helper: drop the entry for context-end ``pos`` under
        ``key`` from the host index."""
        dq = self._epi.get(key)
        if not dq:
            return False
        for ent in dq:
            if ent[0] == pos:
                dq.remove(ent)
                self._epi_n -= 1
                break
        else:
            return False
        if not dq:
            del self._epi[key]
        return True

    def _uni_p(self, t: int, denom: float) -> float:
        return (self._uni[t] + self.config.uni_alpha) / denom

    def _channel_prob(self, ch: str, x: int, keys: dict[int, int],
                      votes: dict[int, float] | None,
                      denom: float,
                      kvotes: dict[int, float] | None = None) -> float:
        """Channel c's predicted probability of token x (for Hedge)."""
        c = self.config
        if ch == "uni":
            return self._uni_p(x, denom)
        if ch == "knn":
            if not kvotes:
                return self._uni_p(x, denom)
            vt = sum(kvotes.values())
            return (kvotes.get(x, 0.0) + c.alpha
                    * self._uni_p(x, denom)) / (vt + c.alpha)
        if ch == "recency":
            age = self._count - 1 - self._last_seen.get(x, -10 ** 9)
            return c.recency_decay ** age if 0 <= age < c.recency_span else 1e-3
        if ch == "fatigue":
            # rewards novelty: ~1 when x wasn't just seen, ~0 when repeated
            age = self._count - 1 - self._last_seen.get(x, -10 ** 9)
            if 0 <= age < c.fatigue_span:
                return 1.0 - c.fatigue_decay ** age
            return 1.0
        if ch == "topic":
            a = self._A[x]
            cn = float(self._c.norm())
            an = float(a.norm())
            cos = float(a @ self._c) / (an * cn + 1e-8)
            return 1.0 / (1.0 + math.exp(-c.topic_beta * cos))
        if ch == "sem":
            a = self._A[x]
            pn = float(self._proto.norm())
            an = float(a.norm())
            cos = float(a @ self._proto) / (an * pn + 1e-8)
            return 1.0 / (1.0 + math.exp(-c.sem_beta * cos))
        if ch == "csem":
            # contextual fingerprint prob: how similar is x's feature to
            # what THIS context has historically preceded
            row = self._csem_read()
            if row is None:
                return self._uni_p(x, denom)
            ax = self._A[x] @ self._Pcsem
            rn = float(row.norm())
            an = float(ax.norm())
            cos = float(ax @ row) / (an * rn + 1e-8)
            return 1.0 / (1.0 + math.exp(-c.csem_beta * cos))
        if ch == "deep":
            if not votes:
                return self._uni_p(x, denom)
            vt = sum(votes.values())
            return (votes.get(x, 0.0) + c.alpha * self._uni_p(x, denom)) \
                / (vt + c.alpha)
        if ch == "backoff":
            # longest order with enough evidence, confidence-discounted
            best_k, btot = -1, 0.0
            for k, key in keys.items():
                t = self._totals[k].get(key, 0.0)
                if t >= c.backoff_min_tot and k > best_k:
                    best_k, btot = k, t
            if best_k < 0:
                return self._uni_p(x, denom)
            cell = self._tables[best_k].get(keys[best_k]) or {}
            conf = btot / (btot + c.backoff_kappa)
            return conf * (cell.get(x, 0.0)
                           + c.alpha * self._uni_p(x, denom)) \
                / (btot + c.alpha)
        # order channel "o<k>"
        k = int(ch[1:])
        key = keys.get(k, -1)
        cell = self._tables[k].get(key)
        if not cell:
            return self._uni_p(x, denom)
        tot = self._totals[k].get(key, 0.0)
        return (cell.get(x, 0.0) + c.alpha * self._uni_p(x, denom)) \
            / (tot + c.alpha)

    def _deep_votes(self, end: int) -> dict[int, float]:
        """Seed-match + backward extension → successor votes.

        Entries are (ctx_end, succ, certs) tuples.  Positions still in
        the ring get exact backward extension; positions that scrolled
        out (``pos < floor``) vote by certificate: weight = the deepest
        cert order matching the current context's suffix hash, else the
        bare seed order.  The index key IS the validation for archived
        entries (2^-64 collision) — deep memory is unbounded."""
        c = self.config
        key = self._suffix_key(c.deep_order, end)
        if key is None:
            return {}
        positions = self._epi.get(key)
        if not positions:
            return {}
        cap = c.ring_capacity
        floor = max(0, self._count - cap)
        cur_certs = {L: self._suffix_key(L, end)
                     for L in c.cert_orders} if c.cert_orders else {}
        votes: dict[int, float] = {}
        examined = 0
        for ent in reversed(positions):
            pos, succ = ent[0], ent[1]
            if pos >= end:
                continue              # context end not yet writable
            if pos >= floor:
                # on-ring: recompute seed hash (ring bytes may have
                # been overwritten mid-wrap) then extend backwards
                if self._raw_hash(c.deep_order, pos) != key:
                    continue
                m = c.deep_order
                ext = 0
                while (ext < c.deep_max_extend and pos - m >= floor
                       and end - m >= self._ep_start
                       and self._ring[(pos - m) % cap]
                       == self._ring[(end - m) % cap]):
                    ext += 1
                    m += 1
                w = float(m)
            else:
                # archived: certificate-graded vote
                w = float(c.deep_order)
                certs = ent[2] if len(ent) > 2 else ()
                for L, cv in zip(c.cert_orders, certs):
                    if cv and cur_certs.get(L) == cv and L > w:
                        w = float(L)
            if succ < 0:              # legacy entry — read ring
                if pos + 1 >= self._count or pos + 1 < floor:
                    continue
                succ = int(self._ring[(pos + 1) % cap])
            votes[succ] = votes.get(succ, 0.0) + w
            examined += 1
            if examined >= c.deep_max_hits:
                break
        if examined:
            # retrieval hit — feeds utility-scored eviction: cells that
            # actually get read are worth keeping
            self._epi_hits[key] = self._epi_hits.get(key, 0) + examined
        return votes

    def _knn_votes(self, k: int | None = None) -> dict[int, float]:
        """Top-k fingerprint-similar contexts → successor votes.

        One bank matvec + topk; weights = relu(cos)^gamma, floored at
        ``knn_min_sim``.  Returns {} when the bank is empty."""
        c = self.config
        if not c.use_knn or self._knn_cur == 0:
            return {}
        q = self._knn_fp_of(self._c).to(self._knn_fp.dtype)
        # scan only the live ring prefix (bucketed) — the padded tail
        # is pos=-1 anyway, so slicing is exact, just cheaper
        b = self._knn_rows()
        if b <= 0:
            return {}
        # matvec in bank dtype ([b,d]@[d] — fp16 halves the read cost),
        # then widen the [b] result for stable top-k
        sims = (self._knn_fp[:b] @ q).float()
        sims = torch.where(self._knn_pos[:b] >= 0, sims,
                           torch.full((), -1e9,
                                      device=self._dev))
        kk = int(min(k or c.knn_k, max(self._knn_cur, 1),
                     c.knn_capacity))
        if kk <= 0:
            return {}
        top = torch.topk(sims, kk)
        votes: dict[int, float] = {}
        for s, i in zip(top.values.tolist(), top.indices.tolist()):
            if s <= c.knn_min_sim:
                break
            t = int(self._knn_succ[i])
            if t >= 0:
                votes[t] = votes.get(t, 0.0) + max(s, 0.0) \
                    ** c.knn_gamma
        return votes

    # ── csem helpers ────────────────────────────────────────────────────

    def _csem_read(self, key: int | None = None) -> torch.Tensor | None:
        """Fingerprint row for the current (or given) context key."""
        c = self.config
        if not c.use_csem:
            return None
        if key is None:
            key = self._suffix_key(c.csem_order, self._count - 1)
            if key is None:
                return None
        if self._gpu:
            pk = _mix64(torch.tensor([_s64(key)], dtype=torch.int64,
                                     device=self._dev) * _MIX1
                        + self._gt.ROW_CSEM_SALT)
            slot, found = self._gt._probe(
                self._gt.tkeys[self._gt.row_csem], pk)
            if not bool(found[0]):
                return None
            return self._csem_rows[slot[0]].float()
        row = self._csem_cpu.get(key)
        return None if row is None else torch.from_numpy(row).to(self._dev)

    def _csem_write(self, x: int, journal: bool = True,
                    scale: float = 1.0) -> None:
        """EMA this context's fingerprint toward follower's feature.
        ``scale`` = surprise gate (lr_eff = csem_lr * scale)."""
        c = self.config
        if not c.use_csem:
            return
        end = self._count - 1
        key = self._suffix_key(c.csem_order, end)
        if key is None:
            return
        self._csem_write_key(key, x, scale=scale, journal=journal)

    def _csem_write_key(self, key: int, x: int, scale: float = 1.0,
                        journal: bool = True) -> None:
        """Key-addressed csem write — same EMA as ``_csem_write`` but
        for an explicit context key (teach()/manual injection)."""
        c = self.config
        ax = (self._A[x] @ self._Pcsem).float()
        if self._gpu:
            gt = self._gt
            rcs = gt.row_csem
            pk = _mix64(torch.tensor([_s64(key)], dtype=torch.int64,
                                     device=self._dev) * _MIX1
                        + gt.ROW_CSEM_SALT)
            slot, found, evi = gt._probe_row_w(rcs, pk)
            s = int(slot[0])
            prev = (self._csem_rows[s].clone()
                    if bool(found[0]) else None)
            if not bool(found[0]):
                gt.tkeys[rcs, s] = pk[0]
                gt.traw[rcs, s] = _s64(key)
                if not bool(evi[0]):
                    gt.used[rcs] += 1
            cur = (self._csem_rows[s].float() if prev is not None
                   else torch.zeros(c.csem_dim, device=self._dev))
            lr_eff = c.csem_lr * scale
            self._csem_rows[s] = (cur * (1.0 - lr_eff)
                                  + ax.half() * lr_eff).half()
            if journal:
                self.journal.append(
                    ("s", int(pk[0]), s,
                     None if prev is None else prev.cpu().numpy()))
            return
        prev = self._csem_cpu.get(key)
        cur = (torch.from_numpy(prev).to(self._dev) if prev is not None
               else torch.zeros(c.csem_dim, device=self._dev))
        lr_eff = c.csem_lr * scale
        new = (cur * (1.0 - lr_eff) + ax * lr_eff).cpu().numpy()
        self._csem_cpu[key] = new.astype(np.float32)
        if journal:
            self.journal.append(("s", key, -1, prev))

    # ── GPU live paths ─────────────────────────────────────────────────

    def _keys_dev(self, end: torch.Tensor) -> tuple[
            torch.Tensor, torch.Tensor]:
        """(keys [O] int64, valid [O] bool) at device scalar ``end``.
        Validity is a separate mask — raw u64 keys can be negative in
        int64 bit pattern, so a sign test would be wrong."""
        c = self.config
        cap = c.ring_capacity
        ks = self._orders_g
        he = self._Hg.index_select(0, (end % cap).reshape(1)).squeeze(0)
        valid = (end - ks + 1 >= self._ep_g) & (end >= ks - 1) \
            & (end >= 0)
        keys = he - self._Hg[(end - ks) % cap] \
            * self._pows_g[ks]
        return keys, valid

    def _suffix_dev(self, k: int, end: torch.Tensor) -> torch.Tensor:
        """Raw-order-k suffix hash at device scalar ``end`` (no episode
        gate — caller masks)."""
        cap = self.config.ring_capacity
        return self._Hg.index_select(0, (end % cap).reshape(1)) \
            .squeeze(0) \
            - self._Hg.index_select(
                0, ((end - k) % cap).reshape(1)).squeeze(0) \
            * self._pows_g[k]

    def _epi_probe(self, pk: torch.Tensor) -> tuple[
            torch.Tensor, torch.Tensor]:
        """Eager probe on the episodic row of the shared key matrix.
        ``pk`` must already be mixed with ROW_EPI_SALT."""
        return self._gt._probe(self._gt.tkeys[self._gt.row_epi], pk)

    def _epi_hash(self, dk: torch.Tensor) -> torch.Tensor:
        return _mix64(dk * _MIX1 + self._gt.ROW_EPI_SALT)

    def _csem_hash(self, ck: torch.Tensor) -> torch.Tensor:
        return _mix64(ck * _MIX1 + self._gt.ROW_CSEM_SALT)

    def _deep_votes_dev(self, end: torch.Tensor, floor: torch.Tensor,
                        se: torch.Tensor | None = None,
                        fe: torch.Tensor | None = None,
                        dk: torch.Tensor | None = None
                        ) -> torch.Tensor:
        """Deep episodic votes [V] — tensor-only, graph-safe.  ``se``/
        ``fe``/``dk`` may be precomputed by a combined probe."""
        c = self.config
        cap = c.ring_capacity
        if dk is None:
            dk = self._suffix_dev(c.deep_order, end).squeeze(0)
        if se is None:
            pk = self._epi_hash(dk.reshape(1))
            se, fe = self._epi_probe(pk)
            se = se.reshape(1)
        pos_row = self._epi_pos.index_select(0, se).squeeze(0)  # [ER]
        if self._use_triton:
            # one launch evaluates all ER cells — seed check, backward
            # extension, cert grading, and the vote scatter fused
            votes = torch.zeros(self.vocab_size, dtype=torch.float32,
                                device=self._dev)
            LC = len(c.cert_orders)
            succ_r = self._epi_succ.index_select(0, se).squeeze(0)
            if LC:
                certs = self._epi_cert.index_select(
                    0, se).squeeze(0)                        # [ER, LC]
                cur = torch.stack(
                    [self._suffix_dev(L, end).squeeze(0)
                     for L in c.cert_orders])                # [LC]
                cw = self._cert_g
            else:
                certs = torch.zeros((c.epi_row, 1),
                                    dtype=torch.int64, device=self._dev)
                cur = torch.zeros(1, dtype=torch.int64,
                                  device=self._dev)
                cw = cur
            _flux_deep_votes_kernel[(c.epi_row,)](
                pos_row, succ_r, certs, self._Hg, self._ringg,
                cur, cw, votes,
                end, floor, self._ep_g, dk, self._pow_deep, fe,
                CAP=cap, DO=c.deep_order, MX=c.deep_max_extend,
                LC=max(LC, 1), ER=c.epi_row)
            return votes
        valid = (end - c.deep_order + 1 >= self._ep_g) \
            & (end >= c.deep_order - 1) & (end >= 0)
        # fe matters: on a probe miss ``se`` still points at a slot
        # owned by an unrelated key — its cells must not vote
        live = fe & (pos_row >= 0) & (pos_row < end) & valid
        on_ring = live & (pos_row >= floor)
        # validate seed hash only where ring bytes still exist —
        # overwritten slots mismatch and drop out
        hv = self._Hg[pos_row.clamp(min=0) % cap] \
            - self._Hg[(pos_row - c.deep_order)
                       .clamp(min=0) % cap] \
            * self._pows_g[c.deep_order]
        on_ring &= hv == dk
        # backward extension as ONE vectorized op: eq[m,j] = token at
        # pos_j-m equals token at end-m; cumprod over m = running AND
        # -> sum gives the contiguous match length per candidate.
        mx = c.deep_max_extend
        ms = torch.arange(1, mx + 1, dtype=torch.int64,
                          device=self._dev)              # [mx]
        pi = pos_row[None, :] - ms[:, None]              # [mx, ER]
        ei = end - ms                                    # [mx]
        ok = (pi >= floor) & (pi >= 0) \
            & (ei >= self._ep_g)[:, None] & (ei >= 0)[:, None]
        a = self._ringg[pi.clamp(min=0) % cap]
        b = self._ringg.index_select(
            0, (ei.clamp(min=0) % cap).reshape(-1))      # [mx]
        eq = (a.long() == b[:, None].long()) & ok
        runlen = eq.long().cumprod(0).sum(0)             # [ER]
        mlen_ring = torch.where(on_ring,
                                runlen + c.deep_order,
                                torch.zeros_like(runlen))
        # archived entries (pos < floor): certificate-graded weight —
        # stored suffix hashes at cert_orders vs the current context's;
        # the index key match alone guarantees at least seed-order mass
        arch = live & (pos_row < floor)
        base_w = torch.full((), c.deep_order, dtype=torch.int64,
                            device=self._dev)
        mlen_arch = torch.where(arch, base_w.expand_as(runlen),
                                torch.zeros_like(runlen))
        if c.cert_orders:
            certs = self._epi_cert.index_select(0, se)   # [1, ER, L]
            cur = torch.stack([self._suffix_dev(L, end).squeeze(0)
                               for L in c.cert_orders])  # [L]
            cmatch = (certs == cur[None, None, :]) & (certs != 0)
            cw = (cmatch.long()
                  * self._cert_g[None, None, :]).amax(dim=2).squeeze(0)
            mlen_arch = torch.where(
                arch, torch.maximum(cw, base_w),
                torch.zeros_like(runlen))
        # successor comes from the stored entry — never the ring — so
        # votes survive ring wrap (and empty succ=-1 is masked out)
        succ = self._epi_succ.index_select(0, se).squeeze(0) \
            .long().clamp(0, self.vocab_size - 1)
        have_succ = self._epi_succ.index_select(0, se) \
            .squeeze(0) >= 0
        mlen = torch.where(have_succ,
                           mlen_ring + mlen_arch,
                           torch.zeros_like(runlen))
        votes = torch.zeros(self.vocab_size, dtype=torch.float32,
                            device=self._dev)
        votes.index_add_(0, succ, mlen.float())
        return votes

    def _probes_at(self, end: torch.Tensor) -> tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
            torch.Tensor, torch.Tensor]:
        """ONE probe loop resolving every table at context ``end``:
        orders [O], csem row, epi row -> slot/found [R] each, plus the
        validity masks and the raw deep key."""
        c = self.config
        gt = self._gt
        keys, live_k = self._keys_dev(end)                 # [O]
        ck = self._suffix_dev(c.csem_order, end)
        dk = self._suffix_dev(c.deep_order, end)
        allk = torch.cat([keys, ck.reshape(1), dk.reshape(1)])
        pk = gt.ctx_hash_all(allk)                         # [R]
        slot, found, evict = gt.probe_write(pk)
        cok = (end - c.csem_order + 1 >= self._ep_g) \
            & (end >= c.csem_order - 1) & (end >= 0)
        dok = (end - c.deep_order + 1 >= self._ep_g) \
            & (end >= c.deep_order - 1) & (end >= 0)
        return slot, found, evict, live_k, cok, dok, \
            dk.squeeze(0), pk

    def _logits_dev(self, end: torch.Tensor,
                    slot: torch.Tensor, found: torch.Tensor,
                    live_k: torch.Tensor, cok: torch.Tensor,
                    dok: torch.Tensor, dk: torch.Tensor
                    ) -> torch.Tensor:
        """Assemble [V] logits from a combined ``_probes_at`` result."""
        c = self.config
        V = self.vocab_size
        gt = self._gt
        O = gt.O
        denom = (self._tot_g + c.uni_alpha * V).float()
        uni_pv = (self._unig + c.uni_alpha) / denom
        log_uni = self._log_uni_num - torch.log(denom)
        logits = log_uni.clone()
        floor = torch.clamp(end + 1 - c.ring_capacity, min=0)

        oslot, ofound = slot[:O], found[:O]
        hit = ofound & live_k
        rows_t = gt.ttoks[gt._ordx, oslot]             # [O, M]
        rows_c = gt.tcnt[gt._ordx, oslot]              # [O, M]
        tots = gt.ttot[gt._ordx, oslot]                # [O]
        wo = self._wg[:O].float()[:, None]             # [O,1]
        if c.evidence_lift:
            # effective weight = hedge prior + evidence confidence —
            # KN-style interpolation; a massive cell speaks for itself
            wo = wo + (tots / (tots + c.backoff_kappa))[:, None]
        tok_l = rows_t.long().clamp(0, V - 1)
        row_live = (rows_t >= 0) & (rows_t < V) & hit[:, None]
        p = (rows_c + c.alpha * uni_pv[tok_l]) \
            / (tots[:, None] + c.alpha)
        contrib = wo * (torch.log(p.clamp(min=1e-12))
                        - log_uni[tok_l])
        contrib = torch.where(row_live, contrib,
                              torch.zeros((), device=self._dev))
        logits.index_add_(0, tok_l.reshape(-1),
                          contrib.reshape(-1))

        # backoff channel (∞-gram): the longest live order with enough
        # evidence, confidence-discounted — hedged alongside the rest
        if c.use_backoff:
            ok_bo = hit & (tots >= c.backoff_min_tot)
            kstar = torch.where(
                ok_bo, self._orders_g.float(),
                torch.full((), -1.0, device=self._dev)).argmax()
            bo_ok = ok_bo.index_select(
                0, kstar.reshape(1)).squeeze(0)
            frow = (kstar * gt.S
                    + oslot.index_select(
                        0, kstar.reshape(1)).squeeze(0))
            rt_k = gt.ttoks.view(-1, gt.M).index_select(
                0, frow.reshape(1)).squeeze(0)          # [M]
            rc_k = gt.tcnt.view(-1, gt.M).index_select(
                0, frow.reshape(1)).squeeze(0)
            tot_k = gt.ttot.view(-1).index_select(
                0, frow.reshape(1)).squeeze(0)
            conf = tot_k / (tot_k + c.backoff_kappa)
            tl = rt_k.long().clamp(min=0)
            rl = rt_k >= 0
            p_b = conf * (rc_k + c.alpha * uni_pv[tl]) \
                / (tot_k + c.alpha)
            wb = self._wg[self._chix["backoff"]].float()
            if c.evidence_lift:
                wb = wb + conf
            logits.index_add_(
                0, tl,
                torch.where(
                    rl & bo_ok,
                    wb * (torch.log(p_b.clamp(min=1e-12))
                          - log_uni[tl]),
                    torch.zeros((), device=self._dev)))

        votes = self._deep_votes_dev(
            end, floor,
            se=slot[gt.row_epi].reshape(1),
            fe=found[gt.row_epi].reshape(1), dk=dk)
        vt = votes.sum()
        # retrieval-hit flag for host-side utility accounting ( eviction
        # keeps cells that actually get read )
        self._dhit_g.copy_(
            ((vt > 0) & found[gt.row_epi]).reshape(1).long())
        wd = self._wg[self._chix["deep"]].float()
        if c.evidence_lift:
            wd = wd + vt / (vt + c.backoff_kappa)
        p_d = (votes + c.alpha * uni_pv) / (vt + c.alpha)
        logits += torch.where(
            vt > 0, wd * (torch.log(p_d.clamp(min=1e-12))
                          - log_uni),
            torch.zeros((), device=self._dev))

        # topic: cosine readout on top candidates (norms from _An)
        cn = self._c.norm()
        wt = self._wg[self._chix["topic"]].float()
        cand = torch.topk(logits, min(256, V)).indices
        a = self._A[cand]
        an = self._An.index_select(0, cand)
        cos = (a @ self._c) / (an * cn + 1e-8)
        logits[cand] += torch.where(
            cn > 1e-8, wt * c.topic_beta * cos,
            torch.zeros((), device=self._dev))

        # sem + csem share _Ac = A @ Pcsem (rank-1 maintained): JL
        # projection preserves cosine direction, so both channels score
        # in the 64-dim space instead of a fresh [V,192] matvec
        pv = self._proto @ self._Pcsem                    # [64]
        pvn = pv.norm()
        sims = (self._Ac @ pv) / (self._Acn * pvn + 1e-8)

        # csem: per-context fingerprint over full vocab
        cs = slot[gt.row_csem].reshape(1)
        cf = found[gt.row_csem].reshape(1)
        row = self._csem_rows.index_select(0, cs).squeeze(0).float()
        rn = row.norm()
        csims = (self._Ac @ row) / (self._Acn * rn + 1e-8)

        # knn: top-k fingerprint-similar past contexts vote their
        # stored successors — approximate-match episodic memory.
        # Only the live ring prefix is scanned (bucketed — static shape
        # inside a captured graph; unwritten padding stays pos=-1).
        kv = torch.zeros(V, dtype=torch.float32, device=self._dev)
        kvt = torch.zeros((), dtype=torch.float32, device=self._dev)
        kb = self._knn_rows()
        if c.use_knn and kb > 0:
            qk = self._c @ self._Pcsem
            qkn = qk.norm()
            qk = (qk / (qkn + 1e-8)).to(self._knn_fp.dtype)
            ksims = (self._knn_fp[:kb] @ qk).float()
            ksims = torch.where(
                self._knn_pos[:kb] >= 0, ksims,
                torch.full((), -1e9, device=self._dev))
            top = torch.topk(ksims, min(c.knn_k, kb))
            keep = top.values > c.knn_min_sim
            wv = (top.values.clamp(min=0.0) ** c.knn_gamma) * keep
            ksucc = self._knn_succ.index_select(
                0, top.indices).long().clamp(0, V - 1)
            kv.index_add_(0, ksucc, wv)
            kvt = kv.sum()

        if self._use_triton:
            # one fused V-map: sem + csem + knn mixture + recency +
            # fatigue — replaces ~25 elementwise kernels per predict
            _flux_logit_tail_kernel[(triton.cdiv(V, 1024),)](
                logits, log_uni, uni_pv, sims, csims, kv,
                self._seen_g, self._wg, pvn, rn, cf, cok, kvt, end,
                V,
                SEMX=self._chix["sem"], CSEMX=self._chix["csem"],
                KNNX=self._chix["knn"], RECX=self._chix["recency"],
                FATX=self._chix["fatigue"],
                USE_SEM=bool(c.use_sem), USE_CSEM=bool(c.use_csem),
                BETA_SEM=c.sem_beta, BETA_CSEM=c.csem_beta,
                ALPHA=c.alpha, KAPPA=c.backoff_kappa,
                LIFT=bool(c.evidence_lift),
                SPAN_R=c.recency_span,
                LDEC_R=math.log(c.recency_decay),
                SPAN_F=c.fatigue_span,
                LDEC_F=math.log(c.fatigue_decay),
                FAT_STR=c.fatigue_strength,
                BLOCK=1024)
            return logits

        ws = self._wg[self._chix["sem"]].float() * float(c.use_sem)
        logits += torch.where(
            pvn > 1e-8, ws * c.sem_beta * sims,
            torch.zeros((), device=self._dev))

        wc = self._wg[self._chix["csem"]].float() * float(c.use_csem)
        logits += torch.where(
            cf.squeeze() & cok & (rn > 1e-8),
            wc * c.csem_beta * csims,
            torch.zeros((), device=self._dev))

        if c.use_knn:
            p_k = (kv + c.alpha * uni_pv) / (kvt + c.alpha)
            wk = self._wg[self._chix["knn"]].float()
            if c.evidence_lift:
                wk = wk + kvt / (kvt + c.backoff_kappa)
            logits += torch.where(
                kvt > 0, wk * (torch.log(p_k.clamp(min=1e-12))
                               - log_uni),
                torch.zeros((), device=self._dev))

        # recency + fatigue over full V
        ages = (end - self._seen_g).clamp(min=0)
        rec = torch.where(
            (ages < c.recency_span) & (self._seen_g >= 0),
            c.recency_decay ** ages.float(),
            torch.zeros((), device=self._dev))
        fat = torch.where(
            (ages < c.fatigue_span) & (self._seen_g >= 0),
            c.fatigue_decay ** ages.float(),
            torch.zeros((), device=self._dev))
        logits += self._wg[self._chix["recency"]].float() * rec \
            - self._wg[self._chix["fatigue"]].float() \
            * c.fatigue_strength * fat
        return logits

    def _predict_dev(self, end: torch.Tensor) -> torch.Tensor:
        """Next-token logits [V] — pure tensor ops, no host reads."""
        slot, found, _ev, live_k, cok, dok, dk, _pk = \
            self._probes_at(end)
        return self._logits_dev(end, slot, found, live_k,
                                cok, dok, dk)

    def _step_dev(self) -> None:
        """Fused observe+predict for token ``self._bx``: all live
        writes + next-token logits in one device-side pass.  Zero host
        reads — journal metadata goes out through ``self._bout``."""
        c = self.config
        dev = self._dev
        gt = self._gt
        O, C = gt.O, len(self.channels)
        V = self.vocab_size
        cap = c.ring_capacity
        pos = self._pos_g                        # x lands at pos
        end = pos - 1                            # context end
        x = self._bx
        jcur = self._jcur_t                      # ring slot scalar

        # ── phase-1 probe at context end (pre-write state): orders +
        # csem + epi all resolved by ONE probe loop ─────────────────
        keys, live_k = self._keys_dev(end)
        ck_e = self._suffix_dev(c.csem_order, end)
        dk_e = self._suffix_dev(c.deep_order, end)
        pk1 = gt.ctx_hash_all(torch.cat(
            [keys, ck_e.reshape(1), dk_e.reshape(1)]))
        slot1, found1, evict1 = gt.probe_write(pk1)
        slot, found = slot1[:O], found1[:O]
        hit = found & live_k
        rows_t = gt.ttoks[gt._ordx, slot]
        rows_c = gt.tcnt[gt._ordx, slot]
        tots = gt.ttot[gt._ordx, slot]
        cs, cf = slot1[gt.row_csem].reshape(1), \
            found1[gt.row_csem].reshape(1)

        denom = self._tot_g + c.uni_alpha * V
        uni_x = (self._unig.index_select(0, x.reshape(1))
                 .squeeze(0) + c.uni_alpha) / denom

        # ── hedge rewards (pre-write channel probs of x) ───────────
        rw = torch.empty(C, dtype=torch.float64, device=dev)
        m_x = (rows_t == x.to(torch.int32))
        cnt_x = (rows_c.double() * m_x).sum(dim=1)
        p_o = torch.where(hit & (tots > 0),
                          (cnt_x + c.alpha * uni_x)
                          / (tots.double() + c.alpha),
                          uni_x.expand(O))
        rw[:O] = torch.where(live_k, p_o, uni_x.expand(O))
        floor = torch.clamp(pos - cap, min=0)
        votes = self._deep_votes_dev(
            end, floor, se=slot1[gt.row_epi].reshape(1),
            fe=found1[gt.row_epi].reshape(1), dk=dk_e)
        vt = votes.sum().double()
        vx = votes.index_select(0, x.reshape(1)).squeeze(0).double()
        rw[self._chix["deep"]] = torch.where(
            vt > 0, (vx + c.alpha * uni_x)
            / (vt + c.alpha), uni_x)
        a_x = self._A.index_select(0, x.reshape(1)).squeeze(0)
        cn = self._c.norm()
        cos_t = (a_x @ self._c) \
            / (a_x.norm() * cn + 1e-8)
        rw[self._chix["topic"]] = torch.sigmoid(
            c.topic_beta * cos_t).double()
        pn = self._proto.norm()
        cos_s = (a_x @ self._proto) \
            / (a_x.norm() * pn + 1e-8)
        rw[self._chix["sem"]] = torch.sigmoid(
            c.sem_beta * cos_s).double()
        # csem reward: cosine vs this context's fingerprint row
        cok = (end - c.csem_order + 1 >= self._ep_g) \
            & (end >= c.csem_order - 1) & (end >= 0)
        ax_c = (a_x @ self._Pcsem).double()
        row = self._csem_rows.index_select(0, cs).squeeze(0).double()
        rn = row.norm()
        cos_c = (ax_c @ row) / (ax_c.norm() * rn + 1e-8)
        rw[self._chix["csem"]] = torch.where(
            cf.squeeze() & cok & (rn > 1e-8),
            torch.sigmoid(c.csem_beta * cos_c).double(), uni_x)
        # knn reward: top-k fingerprint neighbours' successor vote on x.
        # fp_store reuses the query fingerprint for the bank write below
        if c.use_knn:
            qk = self._c @ self._Pcsem
            qkn = qk.norm()
            fp_store = (qk / (qkn + 1e-8)).to(self._knn_fp.dtype)
            # live ring prefix only — bucketed so the slice stays
            # shape-static inside the captured graph
            kb = self._knn_rows()
            sims = (self._knn_fp[:kb] @ fp_store).float()
            sims = torch.where(
                self._knn_pos[:kb] >= 0, sims,
                torch.full((), -1e9, device=dev))
            ktop = torch.topk(sims, min(c.knn_k, kb))
            kkeep = ktop.values > c.knn_min_sim
            kwv = (ktop.values.clamp(min=0.0) ** c.knn_gamma) * kkeep
            ksucc = self._knn_succ.index_select(
                0, ktop.indices).long().clamp(0, V - 1)
            kv = torch.zeros(V, dtype=torch.float32, device=dev)
            kv.index_add_(0, ksucc, kwv)
            kvt = kv.sum().double()
            kvx = kv.index_select(0, x.reshape(1)).squeeze(0).double()
            rw[self._chix["knn"]] = torch.where(
                kvt > 0, (kvx + c.alpha * uni_x)
                / (kvt + c.alpha), uni_x)
        else:
            fp_store = torch.zeros(c.csem_dim, device=dev,
                                   dtype=self._knn_fp.dtype)
            rw[self._chix["knn"]] = uni_x
        age = (end - self._seen_g.index_select(
            0, x.reshape(1)).squeeze(0)).double()
        inrec = (age >= 0) & (age < c.recency_span)
        rw[self._chix["recency"]] = torch.where(
            inrec, c.recency_decay ** age.clamp(min=0),
            self._const_1e3)
        infat = (age >= 0) & (age < c.fatigue_span)
        rw[self._chix["fatigue"]] = torch.where(
            infat, 1.0 - c.fatigue_decay ** age.clamp(min=0),
            self._const_1)
        rw[self._chix["uni"]] = uni_x
        # backoff reward: prob of x under the longest evidenced order
        if c.use_backoff:
            ok_bo = hit & (tots >= c.backoff_min_tot)
            kstar = torch.where(
                ok_bo, self._orders_g.double(),
                torch.full((), -1.0, dtype=torch.float64,
                           device=dev)).argmax()
            bo_hit = ok_bo.index_select(
                0, kstar.reshape(1)).squeeze(0)
            tot_k = tots.index_select(
                0, kstar.reshape(1)).squeeze(0).double()
            cnt_k = cnt_x.index_select(
                0, kstar.reshape(1)).squeeze(0)
            conf = tot_k / (tot_k + c.backoff_kappa)
            rw[self._chix["backoff"]] = torch.where(
                bo_hit,
                conf * (cnt_k + c.alpha * uni_x)
                / (tot_k + c.alpha), uni_x)
        else:
            rw[self._chix["backoff"]] = uni_x
        # hedge update on device weights (AdaHedge: lr adapts to the
        # accumulated mixability gap — constant regret on easy streams)
        self._w_ring_g.index_copy_(
            0, jcur.reshape(1),
            torch.cat([self._wg, self._hmg.reshape(1)]).unsqueeze(0))
        pmix = (self._wg * rw).sum()
        if c.adahedge:
            self._hmg.add_(rw.max() - pmix)
            eta = torch.minimum(
                self._const_eta_max,
                self._const_lnC / self._hmg.clamp(min=1e-9))
        else:
            eta = self._const_hedge_lr
        self._wg.mul_(torch.exp(eta * (rw - pmix)))
        self._wg.clamp_(c.hedge_min, c.hedge_max)
        self._wg.div_(self._wg.sum())

        # surprise: magnitude of this token's writes (1 - p_mix)
        sv = ((torch.clamp(self._const_1 - pmix,
                           min=c.surprise_floor)
               if c.surprise_gate else self._const_1)).float()

        # ── gap flag: unseen context OR high surprise ──────────────
        gk_i = gt.kidx.get(c.gap_probe_order, -1)
        gap = torch.zeros((), dtype=torch.int64, device=dev)
        if gk_i >= 0:
            gap = (live_k[gk_i]
                   & (~hit[gk_i] | (sv.double() > c.gap_surprise))
                   ).long()

        # ── writes ─────────────────────────────────────────────────
        # order rows (branchless; keys written only on new+valid)
        new = (~found) & live_k
        ev = evict1[:O] & live_k               # victim rows get wiped
        ins = new & ~ev                        # empty-slot inserts
        flat_key = gt._ordx * gt.S + slot
        # snapshot victim rows for exact undo (fixed slots in _bev_*)
        EVI, EVF = self._bev_i.shape[1], self._bev_f.shape[1]
        pad_i = EVI - 3 - gt.M
        pad_f = EVF - 1 - gt.M
        zOi = torch.zeros(O, pad_i, dtype=torch.int64, device=dev)
        bev_i_o = torch.cat([
            flat_key[:, None],
            gt.traw.view(-1).index_select(0, flat_key)[:, None],
            torch.zeros(O, 1, dtype=torch.int64, device=dev),
            gt.ttoks.view(-1, gt.M).index_select(
                0, flat_key).long(), zOi], dim=1)
        bev_f_o = torch.cat([
            gt.ttot.view(-1).index_select(0, flat_key)[:, None],
            gt.tcnt.view(-1, gt.M).index_select(0, flat_key),
            torch.zeros(O, pad_f, device=dev)], dim=1)
        self._bev_i[:O] = torch.where(
            ev[:, None].expand(O, EVI), bev_i_o, self._bev_i[:O])
        self._bev_f[:O] = torch.where(
            ev[:, None].expand(O, EVF), bev_f_o, self._bev_f[:O])
        gt.tkeys.view(-1).index_copy_(
            0, flat_key,
            torch.where(new, pk1[:O],
                        gt.tkeys.view(-1).index_select(0, flat_key)))
        gt.traw.view(-1).index_copy_(
            0, flat_key,
            torch.where(new, keys,
                        gt.traw.view(-1).index_select(0, flat_key)))
        allm = torch.arange(gt.M, dtype=torch.int64, device=dev)
        flat_row = (gt._ordx[:, None] * (gt.S * gt.M)
                    + slot[:, None] * gt.M + allm[None, :])
        neg1 = torch.full((), -1, dtype=torch.int32, device=dev)
        zO = torch.zeros((), device=dev)
        tt_v = gt.ttoks.view(-1)
        tc_v = gt.tcnt.view(-1)
        fr = flat_row.reshape(-1)
        tt_v.index_copy_(
            0, fr, torch.where(ev[:, None].expand(O, gt.M)
                               .reshape(-1), neg1,
                               tt_v.index_select(0, fr)))
        tc_v.index_copy_(
            0, fr, torch.where(ev[:, None].expand(O, gt.M)
                               .reshape(-1), zO,
                               tc_v.index_select(0, fr)))
        gt.ttot.view(-1).index_copy_(
            0, flat_key, torch.where(
                ev, zO,
                gt.ttot.view(-1).index_select(0, flat_key)))
        # re-gather post-wipe rows for the follower write
        rows_t = gt.ttoks[gt._ordx, slot]
        rows_c = gt.tcnt[gt._ordx, slot]
        m_x = (rows_t == x.to(torch.int32))
        tots = gt.ttot[gt._ordx, slot]
        has = m_x.any(dim=1)
        empty = rows_t == -1
        has_empty = empty.any(dim=1)
        col = torch.where(has, m_x.float().argmax(dim=1),
                          torch.where(has_empty,
                                      empty.float().argmax(dim=1),
                                      rows_c.argmin(dim=1)))
        write_tok = (~has) & live_k                  # empty OR evict
        replace = write_tok & (~has_empty)
        # column-evict victims: capture (col, tok, cnt) pre-write into
        # the bev pad slots, and drop the displaced mass from ttot —
        # otherwise row totals inflate and undo can't balance
        vic_c = rows_c.gather(1, col[:, None]).squeeze(1)
        vic_t = rows_t.gather(1, col[:, None]).squeeze(1)
        if pad_i >= 3 and pad_f >= 1:
            rep3 = replace[:, None].expand(O, 3)
            cv_i = torch.cat([
                flat_key[:, None], col[:, None],
                vic_t[:, None].long()], dim=1)
            self._bev_i[:O, EVI - 3:EVI] = torch.where(
                rep3, cv_i, self._bev_i[:O, EVI - 3:EVI])
            self._bev_f[:O, EVF - 1] = torch.where(
                replace, vic_c, self._bev_f[:O, EVF - 1])
        rt_flat = gt.ttoks.view(-1)
        rc_flat = gt.tcnt.view(-1)
        flat = gt._ordx * (gt.S * gt.M) + slot * gt.M + col
        rt_flat.index_copy_(
            0, flat,
            torch.where(write_tok, x.to(torch.int32).expand(O),
                        rt_flat.index_select(0, flat)))
        zeroO = torch.zeros(O, dtype=torch.float32, device=dev)
        sO = sv.float().expand(O)      # surprise-scaled write mass
        rc_flat.index_copy_(
            0, flat,
            torch.where(replace, zeroO,
                        rc_flat.index_select(0, flat)))
        rc_flat.index_add_(
            0, flat, torch.where(live_k, sO, zeroO))
        gt.ttot.view(-1).index_add_(
            0, flat_key, torch.where(replace, -vic_c, zeroO))
        gt.ttot.view(-1).index_add_(
            0, flat_key,
            torch.where(live_k, sO, zeroO))
        gt.tocc.view(-1).index_copy_(
            0, flat_key,
            gt.ttot.view(-1).index_select(0, flat_key))

        # unigram
        self._unig.index_add_(
            0, x.reshape(1), torch.ones(1, device=dev))
        self._tot_g += 1.0
        self._log_uni_num.index_copy_(
            0, x.reshape(1),
            torch.log(self._unig.index_select(0, x.reshape(1))
                      + c.uni_alpha))

        # push x into ring + prefix hash (needed before epi write)
        h_prev = self._Hg.index_select(
            0, ((pos - 1) % cap).clamp(min=0).reshape(1)).squeeze(0)
        h_new = torch.where(pos > 0, h_prev * 1099511628211 + x + 1,
                            x + 1)
        self._Hg.index_copy_(0, (pos % cap).reshape(1),
                             h_new.reshape(1))
        self._ringg.index_copy_(0, (pos % cap).reshape(1),
                                x.to(torch.int32).reshape(1))

        # ── deferred episodic write: ``x`` is the successor of the
        # context ending at ``end`` — the entry (ctx_end, succ, certs)
        # rides the phase-1 probe slot, so it votes even after the ring
        # wraps.  Predict reads still come from the phase-2 probe.
        slot2, found2, evict2, _live2, cok2, dok2, dk2, pk2 = \
            self._probes_at(pos)
        dok1 = (end - c.deep_order + 1 >= self._ep_g) \
            & (end >= c.deep_order - 1) & (end >= 0)
        se = slot1[gt.row_epi].reshape(1)
        fe = found1[gt.row_epi].reshape(1)
        ee = (evict1[gt.row_epi] & dok1).reshape(1)
        e_new = (~fe) & dok1
        ekr = gt.tkeys[gt.row_epi]
        err = gt.traw[gt.row_epi]
        ER = self._epi_pos.shape[1]
        LC = len(c.cert_orders)
        # snapshot victim epi row for undo BEFORE keys overwrite it —
        # positions + successors + certs all restore wholesale
        ep_flat = gt.row_epi * gt.S + se                    # [1]
        pad_e = EVI - 3 - ER * (2 + LC)
        ep_i = torch.cat([
            ep_flat[:, None],
            err.index_select(0, se)[:, None],
            self._epi_cnt.index_select(0, se)[:, None],
            self._epi_pos.index_select(0, se),
            self._epi_succ.index_select(0, se).long(),
            self._epi_cert.index_select(0, se).reshape(1, -1)
            if LC else torch.zeros(1, 0, dtype=torch.int64,
                                   device=dev),
            torch.zeros(1, pad_e, dtype=torch.int64,
                        device=dev)], dim=1)
        self._bev_i.index_copy_(
            0, self._brow_epi,
            torch.where(ee[:, None].expand(1, EVI), ep_i,
                        self._bev_i.index_select(
                            0, self._brow_epi)))
        ekr.index_copy_(
            0, se, torch.where(e_new, pk1[gt.row_epi].reshape(1),
                               ekr.index_select(0, se)))
        err.index_copy_(
            0, se, torch.where(e_new, dk_e.reshape(1),
                               err.index_select(0, se)))
        # evicted epi row: positions/succs/certs wiped, count resets
        epv = self._epi_pos
        prow = epv.index_select(0, se)            # [1, ER]
        epv.index_copy_(
            0, se, torch.where(ee[:, None].expand(1, ER),
                               torch.full((1, ER), -1,
                                          dtype=torch.int64,
                                          device=dev), prow))
        svv = self._epi_succ
        svv.index_copy_(
            0, se, torch.where(ee[:, None].expand(1, ER),
                               torch.full((1, ER), -1,
                                          dtype=torch.int32,
                                          device=dev),
                               svv.index_select(0, se)))
        if LC:
            cvv = self._epi_cert
            cvv.index_copy_(
                0, se, torch.where(
                    ee[:, None, None].expand(1, ER, LC),
                    torch.zeros((1, ER, LC), dtype=torch.int64,
                                device=dev),
                    cvv.index_select(0, se)))
        self._epi_cnt.index_copy_(
            0, se, torch.where(ee, torch.zeros(1, dtype=torch.int64,
                                               device=dev),
                               self._epi_cnt.index_select(0, se)))
        col_e = self._epi_cnt.index_select(0, se) % ER
        eflat = se * ER + col_e
        # certificate suffix hashes at `end` for the stored entry —
        # episode-gated: a cert that crosses ep_start stores 0
        if LC:
            certs_e = torch.stack(
                [self._suffix_dev(L, end).squeeze(0)
                 for L in c.cert_orders])
            cok_l = torch.stack(
                [(end - L + 1 >= self._ep_g)
                 & (end >= L - 1) & (end >= 0)
                 for L in c.cert_orders])
            certs_e = torch.where(
                cok_l, certs_e,
                torch.zeros((), dtype=torch.int64, device=dev))
        dok1r = dok1.reshape(1)
        self._epi_pos.view(-1).index_copy_(
            0, eflat,
            torch.where(dok1r, end.reshape(1),
                        self._epi_pos.view(-1).index_select(0, eflat)))
        self._epi_succ.view(-1).index_copy_(
            0, eflat,
            torch.where(dok1r, x.to(torch.int32).reshape(1),
                        self._epi_succ.view(-1).index_select(0, eflat)))
        if LC:
            self._epi_cert.view(-1, LC).index_copy_(
                0, eflat,
                torch.where(dok1r[:, None].expand(1, LC),
                            certs_e.reshape(1, LC),
                            self._epi_cert.view(-1, LC)
                            .index_select(0, eflat)))
        self._epi_cnt.index_add_(
            0, se,
            torch.where(dok1r, torch.ones(1, dtype=torch.int64,
                                          device=dev),
                        torch.zeros(1, dtype=torch.int64,
                                    device=dev)))
        gt.tocc[gt.row_epi].index_copy_(
            0, se, self._epi_cnt.index_select(0, se).float())

        # ── knn bank write: fp of the pre-x context -> successor x ──
        if c.use_knn:
            ksl = (self._knncur_g % c.knn_capacity).reshape(1)
            kprev_pos = self._knn_pos.index_select(0, ksl)
            kprev_succ = self._knn_succ.index_select(0, ksl)
            self._kprev_g.index_copy_(
                0, jcur.reshape(1), self._knn_fp.index_select(0, ksl))
            self._knn_fp.index_copy_(
                0, ksl, fp_store.reshape(1, c.csem_dim))
            self._knn_succ.index_copy_(
                0, ksl, x.to(torch.int32).reshape(1))
            self._knn_pos.index_copy_(0, ksl, end.reshape(1))
            self._knncur_g += 1
        else:
            ksl = torch.full((1,), -1, dtype=torch.int64, device=dev)
            kprev_pos = torch.full((1,), -1, dtype=torch.int64,
                                   device=dev)
            kprev_succ = torch.full((1,), -1, dtype=torch.int64,
                                    device=dev)

        # topic write: delta rule — subtract the current readout along
        # the context direction before committing (error-corrected
        # associative memory; repeated contexts stop re-writing)
        if c.delta_rule:
            chat = self._c / (cn + 1e-8)
            pred = (a_x * chat).sum()
            delta = chat * ((self._const_1.float() - pred)
                            * c.hebb_lr * sv)
        else:
            delta = self._c * (c.hebb_lr * sv)
        self._avec_g.index_copy_(0, jcur.reshape(1),
                                 delta.unsqueeze(0))
        self._A.index_add_(0, x.reshape(1), delta.unsqueeze(0))
        # incremental Ac/An/Acn so predict never rescans A
        ax_new = self._A.index_select(0, x.reshape(1)).squeeze(0)
        self._An.index_copy_(0, x.reshape(1),
                             ax_new.norm().reshape(1))
        self._Ac.index_add_(
            0, x.reshape(1), (delta @ self._Pcsem).unsqueeze(0))
        self._Acn.index_copy_(
            0, x.reshape(1),
            self._Ac.index_select(0, x.reshape(1)).norm(dim=1))
        self._proto.mul_(c.proto_decay).add_(
            ax_new * (1.0 - c.proto_decay))

        # csem write: EMA this context's row toward follower feature
        if c.use_csem:
            cur_row = self._csem_rows.index_select(0, cs)
            prev_row = torch.where(
                cf[0], cur_row.squeeze(0),
                torch.zeros(c.csem_dim, device=dev))
            self._csem_prev.index_copy_(0, jcur.reshape(1),
                                        prev_row.half().unsqueeze(0))
            new_key = (~cf[0]) & cok
            ckr = gt.tkeys[gt.row_csem]
            crr = gt.traw[gt.row_csem]
            # snapshot victim csem row for undo BEFORE keys overwrite
            cev = (evict1[gt.row_csem] & cok).reshape(1)
            cs_flat = gt.row_csem * gt.S + cs               # [1]
            pad_ci = EVI - 2
            pad_cf = EVF - 1 - c.csem_dim
            cs_i = torch.cat([
                cs_flat[:, None],
                crr.index_select(0, cs)[:, None],
                torch.zeros(1, pad_ci, dtype=torch.int64,
                            device=dev)], dim=1)
            cs_f = torch.cat([
                gt.tocc[gt.row_csem].index_select(
                    0, cs)[:, None],
                cur_row.float(),
                torch.zeros(1, pad_cf, device=dev)], dim=1)
            self._bev_i.index_copy_(
                0, self._brow_csem,
                torch.where(cev[:, None].expand(1, EVI), cs_i,
                            self._bev_i.index_select(
                                0, self._brow_csem)))
            self._bev_f.index_copy_(
                0, self._brow_csem,
                torch.where(cev[:, None].expand(1, EVF), cs_f,
                            self._bev_f.index_select(
                                0, self._brow_csem)))
            ckr.index_copy_(
                0, cs, torch.where(
                    new_key, pk1[gt.row_csem].reshape(1),
                    ckr.index_select(0, cs)))
            crr.index_copy_(
                0, cs, torch.where(
                    new_key, ck_e.reshape(1),
                    crr.index_select(0, cs)))
            upd = prev_row * (1.0 - c.csem_lr * sv) \
                + ax_c.float() * (c.csem_lr * sv)
            self._csem_rows.index_copy_(
                0, cs,
                torch.where(cok, upd.half().unsqueeze(0),
                            cur_row))
            gt.tocc[gt.row_csem].index_add_(
                0, cs, torch.where(
                    cok, torch.ones(1, device=dev),
                    torch.zeros(1, device=dev)))

        # recency + context EMA + counters
        self._bout.index_copy_(
            0, self._boff_prevseen,
            self._seen_g.index_select(0, x.reshape(1)))
        self._seen_g.index_copy_(0, x.reshape(1), end.reshape(1))
        self._c.mul_(c.topic_decay).add_(
            self._R.index_select(0, x.reshape(1)).squeeze(0).float()
            * (1.0 - c.topic_decay))
        self._pos_g += 1

        # ── predict at the new context end (reuses phase-2 probe) ──
        self._blogits.copy_(self._logits_dev(
            pos, slot2, found2, _live2, cok2, dok2, dk2))

        # ── readback bundle (one D2H per token, host journals) ─────
        bout = self._bout
        bout[:O] = keys
        # deferred epi insert: host mirrors (key, ctx_end, certs) and
        # `used` bookkeeping come from the PHASE-1 probe
        # NB dk_e is a raw u64 in int64 bit pattern — often negative —
        # so validity must ride a flag bit, not the key's sign
        bout[O] = torch.where(dok1, dk_e, -1)
        bout[O + 1] = torch.where(dok1, se.squeeze(0), -1)
        # bit0 = found, bit1 = evicted, bit2 = dok1 (entry written)
        bout[O + 2] = fe.squeeze(0).long() \
            + ee.squeeze(0).long() * 2 + dok1.long() * 4
        bout[O + 3] = torch.where(cok, cs.squeeze(0), -1)
        bout[O + 4] = cf.squeeze(0).long() \
            + (evict1[gt.row_csem] & cok).long() * 2
        # bit-pack the four [O] flag vectors as vectorized dot products
        # (a per-order `|=` loop costs ~4*O tiny kernels per token)
        bw = self._bitw
        bits = (ins.long() * bw).sum()
        vbits = (live_k.long() * bw).sum()
        ebits = (ev.long() * bw).sum()
        cbits = (replace.long() * bw).sum()
        bout[O + 5] = bits | (vbits << 16) | (ebits << 32) \
            | (cbits << 48)
        bout[O + 6] = gap
        if LC:
            bout[self._boff_cert:self._boff_cert + LC] = certs_e
        bout[self._boff_knn] = ksl[0]
        bout[self._boff_knn + 1] = kprev_pos[0]
        bout[self._boff_knn + 2] = kprev_succ[0]
        bout[self._boff_w:self._boff_w + C] = self._wg.view(torch.int64)
        bout[self._boff_w + C] = sv.double().view(torch.int64)
        bout[self._boff_w + C + 1] = self._hmg.view(torch.int64)
        # predict-time deep key + retrieval-hit flag (utility eviction)
        bout[self._boff_hit] = torch.where(dok2, dk2,
                                           torch.full((), -1,
                                                      dtype=torch.int64,
                                                      device=dev))
        bout[self._boff_hit + 1] = self._dhit_g[0]

    def _fused_step(self, x: int) -> torch.Tensor:
        """Host wrapper for ``_step_dev``: one replay + one D2H read,
        then host-side journal/mirror bookkeeping."""
        c = self.config
        O, C = self._gt.O, len(self.channels)
        self._bx.fill_(x)
        self._jcur_t.fill_(self._jcur)
        if self._graph is not None:
            if self._knn_rows() != self._graph_knn_b:
                # live bank rows crossed the captured bucket — the
                # baked slice would miss them; drop and recapture
                self._graph = None
            else:
                self._graph.replay()
        if self._graph is None:
            if self._warm < 3 or not getattr(self, "_use_cuda_graph", True):
                self._warm += 1
                self._step_dev()                   # eager warmup steps
            else:
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    self._step_dev()               # capture (no exec)
                self._graph = g
                self._graph_knn_b = self._knn_rows()
                self._graph.replay()               # real work for x
        bi = self._bout.tolist()                     # the ONE sync
        pos = self._count
        end = pos - 1
        keys = bi[:O]
        newmask = bi[O + 5] & 0xFFFF
        valmask = (bi[O + 5] >> 16) & 0xFFFF
        jcur = self._jcur
        bw = self._boff_w
        s_h = float(np.asarray(bi[bw + C],
                              dtype=np.int64).view(np.float64))
        self._hedge_mix = float(np.asarray(
            bi[bw + C + 1], dtype=np.int64).view(np.float64))

        # host mirrors (no GPU involvement)
        self._w = np.asarray(
            [int(v) for v in bi[bw:bw + C]],
            dtype=np.int64).view(np.float64).copy()
        self._uni[x] += 1.0
        self._total += 1
        prev_seen = bi[O + 7]
        prev_j = int(prev_seen) if prev_seen >= 0 else -1
        self._last_seen[x] = end
        self._recent.append((end, x))
        cap = c.ring_capacity
        slot = pos % cap
        self._ring[slot] = x
        prev_h = int(self._H[(pos - 1) % cap]) if pos > 0 else 0
        self._H[slot] = (prev_h * 1099511628211 + x + 1) % (1 << 64)
        # deferred epi host entry: (ctx_end, succ, certs) — validity is
        # the dok1 flag bit (raw u64 keys can be negative as int64)
        certs = tuple(bi[self._boff_cert + j] & 0xFFFFFFFFFFFFFFFF
                      for j in range(len(c.cert_orders)))
        epi_ok = bool(bi[O + 2] & 4)
        if epi_ok:
            ekey = bi[O] & 0xFFFFFFFFFFFFFFFF
            dq = self._epi.get(ekey)
            if dq is None:
                dq = self._epi[ekey] = deque(maxlen=c.epi_max_per_key)
            dq.append((end, x, certs))
            self._epi_fifo.append((ekey, end))
            self._epi_n += 1
            self._epi_evict()
        if c.use_knn:
            self._knn_cur += 1          # host mirror of _knncur_g
        # deep-channel retrieval hit — utility signal for eviction
        if bi[self._boff_hit] >= 0 and bi[self._boff_hit + 1]:
            k_ = bi[self._boff_hit] & 0xFFFFFFFFFFFFFFFF
            self._epi_hits[k_] = self._epi_hits.get(k_, 0) + 1
        if bi[O + 6] and c.track_gaps:
            self._gaps.append(pos)
        for i in range(O):
            if (newmask >> i) & 1:
                self._gt.used[i] += 1
        if bi[O + 3] >= 0 and bi[O + 4] == 0:
            self._gt.used[self._gt.row_csem] += 1
        if epi_ok and (bi[O + 2] & 3) == 0:
            self._gt.used[self._gt.row_epi] += 1

        # journal (same op order as the CPU path).  Eviction entries
        # are appended BEFORE the writes that displaced them so LIFO
        # undo restores victims after subtracting the displacers.
        evmask = (bi[O + 5] >> 32) & 0xFFFF
        cemask = (bi[O + 5] >> 48) & 0xFFFF
        epi_ev = bi[O + 2] & 2
        cs_ev = bi[O + 4] & 2
        bev_i = bev_f = None
        if evmask or epi_ev or cs_ev or cemask:
            bev_i = self._bev_i.cpu().numpy()
            bev_f = self._bev_f.cpu().numpy()
        self.journal.append(("h", jcur))
        self.journal.append(("u", x))
        if evmask:
            M = self._gt.M
            for i in range(O):
                if (evmask >> i) & 1:
                    r_ = bev_i[i]
                    self.journal.append(
                        ("ev", int(r_[0]),
                         int(r_[1]) & 0xFFFFFFFFFFFFFFFF,
                         float(bev_f[i][0]),
                         r_[3:3 + M].astype(np.int32).copy(),
                         bev_f[i][1:1 + M].copy()))
        if cemask:
            EVI = self._bev_i.shape[1]
            EVF = self._bev_f.shape[1]
            for i in range(O):
                if (cemask >> i) & 1:
                    r_ = bev_i[i]
                    self.journal.append(
                        ("cv", int(r_[EVI - 3]), int(r_[EVI - 2]),
                         int(r_[EVI - 1]), float(bev_f[i][EVF - 1])))
        for i, k in enumerate(c.orders):
            if (valmask >> i) & 1:
                self.journal.append(
                    ("c", k, keys[i] & 0xFFFFFFFFFFFFFFFF, x, s_h))
        self.journal.append(("a", x, jcur))
        if cs_ev:
            r_ = bev_i[self._gt.row_csem]
            self.journal.append(
                ("evc", int(r_[0]),
                 int(r_[1]) & 0xFFFFFFFFFFFFFFFF,
                 float(bev_f[self._gt.row_csem][0]),
                 bev_f[self._gt.row_csem]
                 [1:1 + c.csem_dim].copy()))
        if bi[O + 3] >= 0:
            self.journal.append(
                ("sg", int(bi[O + 3]),
                 jcur if (bi[O + 4] & 1) else -1))
        self.journal.append(("r", x, prev_j))
        self.journal.append(("t", x, self.tag))
        if epi_ev:
            r_ = self._bev_i[self._gt.row_epi].cpu().numpy() \
                if bev_i is None else bev_i[self._gt.row_epi]
            ER = self._epi_pos.shape[1]
            LC = len(c.cert_orders)
            self.journal.append(
                ("eve", int(r_[0]),
                 int(r_[1]) & 0xFFFFFFFFFFFFFFFF,
                 int(r_[2]), r_[3:3 + ER].copy(),
                 r_[3 + ER:3 + 2 * ER].astype(np.int32).copy(),
                 r_[3 + 2 * ER:3 + ER * (2 + LC)].copy()))
        if epi_ok:
            self.journal.append(
                ("e", bi[O] & 0xFFFFFFFFFFFFFFFF, end))
        if c.use_knn and bi[self._boff_knn] >= 0:
            ppos = bi[self._boff_knn + 1]
            self.journal.append(
                ("k", bi[self._boff_knn], ppos,
                 bi[self._boff_knn + 2],
                 jcur if ppos >= 0 else -1))
        self._count += 1
        self._jcur = (jcur + 1) % c.vec_delta_cap
        return self._blogits

    def _learn_token_gpu(self, x: int) -> None:
        """Single-token learn — fused step (logits discarded)."""
        self._fused_step(x)

    def _predict_gpu(self) -> torch.Tensor:
        """Standalone predict (no learn) — read-only device path."""
        end = torch.tensor(self._count - 1, dtype=torch.int64,
                           device=self._dev)
        return self._predict_dev(end)

    def _learn_token(self, x: int) -> None:
        """All live updates for observing token ``x`` ( journaled )."""
        if self._gpu:
            return self._learn_token_gpu(x)
        c = self.config
        end = self._count - 1               # context ends just before x
        keys = self._keys_at(end) if end >= self._ep_start else {}
        fast = c.fast_ingest and self.tag not in ("live", "gen")
        votes = ({} if fast
                 else (self._deep_votes(end)
                       if end >= self._ep_start else {}))
        kvotes = {} if fast else self._knn_votes()
        denom = self._total + c.uni_alpha * self.vocab_size

        # Hedge update on channel weights (reward = channel prob of x)
        skip = {"deep", "topic", "sem", "knn"} if fast else ()
        rewards = np.array([
            self._uni_p(x, denom) if ch in skip
            else self._channel_prob(ch, x, keys, votes, denom,
                                    kvotes)
            for ch in self.channels])
        p_mix = float((self._w * rewards).sum())
        # AdaHedge: lr adapts to accumulated mixability gap
        if c.adahedge:
            self._hedge_mix += float(rewards.max() - p_mix)
            eta = min(c.hedge_eta_max,
                      math.log(len(self.channels))
                      / max(self._hedge_mix, 1e-9))
        else:
            eta = c.hedge_lr
        factors = np.exp(eta * (rewards - p_mix))
        self.journal.append(
            self.journal.append_w(self._w, self._hedge_mix))
        self._w *= factors
        np.clip(self._w, c.hedge_min, c.hedge_max, out=self._w)
        self._w /= self._w.sum()

        # surprise s: magnitude of this token's writes (Titans-style)
        s = (max(1.0 - p_mix, c.surprise_floor)
             if c.surprise_gate else 1.0)

        # gap tracking: model was surprised here -> teacher target
        if c.track_gaps and keys:
            probe_k = c.gap_probe_order
            key = keys.get(probe_k)
            thin = key is None or \
                self._totals[probe_k].get(key, 0.0) < c.gap_min_total
            if thin or (1.0 - p_mix) > c.gap_surprise:
                self._gaps.append(self._count)

        # unigram
        self._uni[x] += 1.0
        self._total += 1
        self._log_uni_num[x] = math.log(self._uni[x] + c.uni_alpha)
        self.journal.append(("u", x))

        # order tables (surprise-scaled fractional counts) — skipped by
        # predict-only streams: their updates travel in the FluxPacket
        # instead and the single learner applies selected ones post-hoc
        if not self._predict_only:
            for k, key in keys.items():
                tab = self._tables[k]
                cell = tab.get(key)
                if cell is None:
                    cell = tab[key] = {}
                cell[x] = cell.get(x, 0.0) + s
                tots = self._totals[k]
                tots[key] = tots.get(key, 0.0) + s
                self.journal.append(("c", k, key, x, s))
                if len(tab) > c.max_cells_per_order:
                    self._decay_table(tab, tots)

        # topic row: delta-rule write — subtract the current readout
        # along the context direction before committing (error-corrected
        # memory, no double-count of repeated contexts)
        if c.delta_rule:
            cn = float(self._c.norm())
            if cn > 1e-8:
                chat = self._c / cn
                pred = float(self._A[x] @ chat)
                delta_t = chat * ((1.0 - pred) * c.hebb_lr * s)
            else:
                delta_t = torch.zeros_like(self._c)
        else:
            delta_t = self._c * (c.hebb_lr * s)
        if not self._predict_only:
            self._A[x].add_(delta_t)
            self.journal.append(
                self.journal.append_vec(x, delta_t.cpu().numpy()))
            self._csem_write(x, scale=s)
        self._proto.mul_(c.proto_decay).add_(
            self._A[x] * (1.0 - c.proto_decay))

        # recency
        prev = self._last_seen.get(x, -1)
        self._last_seen[x] = end
        self._recent.append((end, x))
        self.journal.append(("r", x, prev))

        # token marker + push
        self.journal.append(("t", x, self.tag))
        self._push(x)

    # ── bulk vectorized ingest (CUDA-primary) ──────────────────────────

    def _ingest_bulk(self, ids: list[int]) -> int:
        """Vectorized chunk ingest on the GPU backend.

        One journal ``bc`` block per call holds a pre-state snapshot plus
        the write arrays — bulk revert is EXACT (snapshot restore + count
        subtraction), better fidelity than per-token EMA inversion.
        Proto EMA uses post-chunk A rows (documented approximation).
        """
        c = self.config
        dev = self._dev
        V = self.vocab_size
        n = len(ids)
        if n == 0:
            return 0
        base = self._count
        cap = c.ring_capacity
        ids_np = np.asarray(ids, dtype=np.int64)
        ids_t = torch.from_numpy(ids_np).to(dev)
        pos = torch.arange(base, base + n, dtype=torch.int64,
                           device=dev)
        slots = pos % cap

        blk = {"tag": self.tag, "n": n, "count": base,
               "ep_start": self._ep_start,
               "c": self._c.clone(), "proto": self._proto.clone(),
               "w": self._w.copy(), "uni": self._uni.copy(),
               "hedge_mix": self._hedge_mix,
               "last_seen": dict(self._last_seen),
               "seen_g": self._seen_g.clone(),
               "unig": self._unig.clone(),
               "log_uni_num": self._log_uni_num.clone(),
               "recent_len": len(self._recent),
               "gaps": len(self._gaps),
               "epi": [], "csem": [], "writes": {}}
        self.journal.append(("bc", blk))

        # ── ring + prefix hashes (closed form, mod 2^64 via int64 wrap)
        # ring/hash writes are chunked to <=cap positions so each
        # fancy-index scatter has UNIQUE slots — duplicate-index
        # assignment order is undefined on CUDA (numpy is last-wins;
        # the device path must match)
        for off in range(0, n, cap):
            mlen_ = min(cap, n - off)
            self._ringg[slots[off:off + mlen_]] = \
                ids_t[off:off + mlen_].to(torch.int32)
        np_idx = (base + np.arange(n)) % cap
        self._ring[np_idx] = ids_np.astype(np.int32)
        P = 1099511628211
        Pinv = pow(P, -1, 1 << 64)
        if Pinv >= 1 << 63:
            Pinv -= 1 << 64          # signed int64 bit pattern
        pinv_i = torch.cumprod(
            torch.full((n,), Pinv, dtype=torch.int64, device=dev), 0)
        pw_i = torch.cumprod(
            torch.full((n,), P, dtype=torch.int64, device=dev),
            0)                                       # P^{i+1}
        s = torch.cumsum((ids_t + 1) * pinv_i, 0)
        h_prev = (torch.zeros((), dtype=torch.int64, device=dev)
                  if base == 0 else self._Hg[(base - 1) % cap])
        # H[base+i] = h_prev*P^{i+1} + sum_j (id_j+1)*P^{i-j}
        #           = P^{i+1} * (s_i + h_prev)   [s_i has P^{-(j+1)}]
        H_blk = pw_i * (s + h_prev)
        # Local absolute-position hash lookup for every suffix window
        # the block needs.  The RING alone is insufficient: reading
        # Hg[(e) % cap] after the write aliases positions older than
        # `cap` inside the block (slot p%cap ends holding the LAST
        # congruent position's hash).  Hfull[i] = H[base - Kmax - 1 + i]
        # — pre-block tail read BEFORE the ring overwrite.
        Kmax = max(int(max(c.orders)), int(c.deep_order),
                   int(max(c.cert_orders, default=0)),
                   int(c.csem_order))
        pre_ix = torch.arange(base - Kmax - 1, base,
                              dtype=torch.int64, device=dev)
        pre_h = torch.where(
            (pre_ix >= 0) & (pre_ix >= base - cap),
            self._Hg[(pre_ix.clamp(min=0)) % cap],
            torch.zeros((), dtype=torch.int64, device=dev))
        Hfull = torch.cat([pre_h, H_blk])   # Hfull[e - base + Kmax + 1]
        for off in range(0, n, cap):
            mlen_ = min(cap, n - off)
            self._Hg[slots[off:off + mlen_]] = H_blk[off:off + mlen_]
        self._H[np_idx] = H_blk.cpu().numpy().astype(np.uint64)

        # ── per-order context keys + table writes ──────────────────────
        # ONE batched read probe over [O,n] keys (pre-write state for
        # hedge rewards + gap detection), then sync-free per-order adds.
        i64 = torch.arange(n, dtype=torch.int64, device=dev)
        O = len(c.orders)
        kvec = self._orders_g                                     # [O]
        posb = pos[None, :].expand(O, n)
        # context for token at pos ENDS at pos-1 (matches live
        # _keys_dev(end=pos-1)): key = suffix of tokens pos-k..pos-1
        valid_o = (posb - kvec[:, None] >= self._ep_start) \
            & (posb >= kvec[:, None])                             # [O,n]
        kk_all = Hfull[(posb - base + Kmax)] - Hfull[
            (posb - kvec[:, None] - base + Kmax).clamp(min=0)] \
            * self._pows_g[kvec][:, None]                         # [O,n]
        pk_all = _mix64(kk_all * _MIX1
                        + self._gt._salt[:O, None])
        rows_f = self._gt._ordx[:, None].expand(O, n).reshape(-1)
        slot_f, found_f = self._gt.probe_fixed(
            pk_all.reshape(-1), rows=rows_f)
        denom0 = float(self._total) + c.uni_alpha * V
        uni_pre = (self._unig[ids_t] + c.uni_alpha) / denom0  # [n]
        uni_rep = uni_pre[None, :].expand(O, n).reshape(-1)
        ids_f = ids_t[None, :].expand(O, n).reshape(-1)
        rt_f = self._gt.ttoks[rows_f, slot_f]                     # [On,M]
        rc_f = self._gt.tcnt[rows_f, slot_f]
        tots_f = self._gt.ttot[rows_f, slot_f]
        cnt_x = (rc_f * (rt_f == ids_f[:, None].to(torch.int32))
                 ).sum(dim=1)                                     # [On]
        p_o = torch.where(
            found_f & (tots_f > 0),
            (cnt_x + c.alpha * uni_rep) / (tots_f + c.alpha),
            uni_rep)
        # per-token surprise: 1 - p_mix under the (constant-w) chunk
        # mixture — order probs are exact; non-order channels fall back
        # to the unigram prior, same as the bulk reward approximation
        w_o = torch.from_numpy(
            self._w[:O].astype(np.float32)).to(dev)[:, None]
        w_rest = float(1.0 - self._w[:O].sum())
        pmix_i = (w_o * p_o.reshape(O, n)).sum(0) \
            + w_rest * uni_pre
        s_v = (torch.clamp(1.0 - pmix_i, min=c.surprise_floor)
               if c.surprise_gate
               else torch.ones(n, device=dev))          # [n]
        # per-position order rewards kept for the chunked hedge update
        # below (weights evolve within the block, not one flat update)
        rw_o = (torch.where(valid_o.reshape(-1), p_o, uni_rep)
                .double().reshape(O, n))               # [O, n]

        # gap detect: unseen context OR high surprise -> teacher target
        if c.track_gaps and c.gap_probe_order in self._gt.kidx:
            gk_i = self._gt.kidx[c.gap_probe_order]
            thin = valid_o[gk_i] & (
                ~found_f.reshape(O, n)[gk_i]
                | (s_v > c.gap_surprise))
            gp = pos[thin]
            if gp.numel():
                self._gaps.extend(gp.cpu().tolist())

        # ONE batched write across all orders, surprise-scaled
        # (distinct-column disambiguation makes it equivalent to
        # sequential adds)
        d_all = s_v[None, :].expand(O, n)
        ins_rows, ev_snap, col_snap = self._gt.add_multi(
            kk_all, ids_t[None, :].expand(O, n), d_all, valid_o)
        # stash the full key/valid/delta matrices — undo re-derives.
        # HOST copies: keeping GPU tensors here would pin VRAM for the
        # journal's lifetime (blocks accumulate during corpus ingest)
        blk["writes_all"] = (kk_all.cpu(), ids_t.cpu(),
                             valid_o.cpu())
        blk["surprise"] = s_v.cpu()
        blk["evicted"] = tuple(
            t.cpu() for t in ev_snap)      # victim rows (undo)
        blk["col_evicted"] = tuple(
            t.cpu() for t in col_snap)     # displaced follower cols
        for oi, cnt in enumerate(ins_rows.tolist()):
            if cnt:
                self._gt.used[oi] += cnt
                if oi < O and self._gt.used[oi] \
                        > 0.85 * c.gpu_table_slots:
                    self._gt.decay(c.orders[oi])
                    self._gt.rehash(c.orders[oi])

        # ── unigram (host + mirror) ────────────────────────────────────
        uniq, cnts = np.unique(ids_np, return_counts=True)
        self._uni[uniq] += cnts
        self._total += n
        self._unig.index_add_(0, ids_t,
                              torch.ones(n, dtype=torch.float32,
                                         device=dev))
        self._log_uni_num = torch.log(self._unig + c.uni_alpha)

        # ── deferred episodic index: ids[i] succeeds ctx-end pos-1 ────
        # entries carry succ + cert suffix hashes → archived (off-ring)
        # positions keep voting; ring wrap no longer kills deep memory
        e_end = pos - 1
        dk_valid = (e_end - c.deep_order + 1 >= self._ep_start) \
            & (e_end >= c.deep_order - 1) & (e_end >= 0)
        di = torch.nonzero(dk_valid).squeeze(1)
        if di.numel():
            ee_ = e_end[di]                                  # ctx ends
            # Hfull (absolute-position lookup) — the ring aliases mod
            # cap for block-internal positions older than cap
            dkk = (Hfull[ee_ - base + Kmax + 1]
                   - Hfull[(ee_ - c.deep_order - base + Kmax + 1)
                          .clamp(min=0)]
                   * self._pows_g[c.deep_order])
            LC = len(c.cert_orders)
            certs_d = (torch.stack([
                        torch.where(
                            (ee_ - L + 1 >= self._ep_start)
                            & (ee_ >= L - 1) & (ee_ >= 0),
                            Hfull[ee_ - base + Kmax + 1]
                            - Hfull[(ee_ - L - base + Kmax + 1)
                                    .clamp(min=0)]
                            * self._pows_g[L],
                            torch.zeros((), dtype=torch.int64,
                                        device=dev))
                        for L in c.cert_orders], dim=1)
                       if LC else None)
            # one D2H for keys + positions + certs + succs
            pack = [dkk, ee_]
            if LC:
                pack.append(certs_d.reshape(-1))           # [nd*LC]
            pack.append(ids_t[di])
            both = torch.cat(pack).cpu().numpy()
            nd = di.numel()
            dk_cpu = both[:nd].astype(np.uint64)
            dp_cpu = both[nd:2 * nd]
            off = 2 * nd
            if LC:
                cert_cpu = both[off:off + nd * LC] \
                    .reshape(nd, LC).astype(np.uint64)
                off += nd * LC
            else:
                cert_cpu = np.zeros((nd, 0), np.uint64)
            succ_cpu = both[off:off + nd]
            for key, p, s_, cv in zip(dk_cpu.tolist(),
                                      dp_cpu.tolist(),
                                      succ_cpu.tolist(),
                                      cert_cpu.tolist()):
                dq = self._epi.get(key)
                if dq is None:
                    dq = self._epi[key] = deque(
                        maxlen=c.epi_max_per_key)
                dq.append((int(p), int(s_), tuple(int(v) for v in cv)))
                self._epi_fifo.append((int(key), int(p)))
                self._epi_n += 1
                blk["epi"].append((int(key), int(p)))
            self._epi_evict()
            # GPU epi mirror — two passes so same-chunk duplicate keys
            # land on the inserted row.  2-choice slot collisions can
            # make a new key lose its row to a same-block sibling; cell
            # writes are gated on fe2 (key actually resident) so a
            # loser never scatters ghost cells onto a foreign row.
            re_ = self._gt.row_epi
            pk_e = self._epi_hash(dkk)
            ekr = self._gt.tkeys[re_]
            err = self._gt.traw[re_]
            se, fe, ee = self._gt._probe_row_w(re_, pk_e)
            enew = ~fe
            ev = enew & ee
            # pre-block snapshot of EVERY candidate slot (both probes)
            # — superset of all rows the block can touch: keys, raw dk,
            # payloads.  Undo restores wholesale => exact, and it
            # subsumes victim-row restore for evictions.
            s1a, s2a = self._gt._slots2(pk_e)
            ust = torch.unique(torch.cat([s1a, s2a]))
            pre_key = ekr.index_select(0, ust)
            pre_raw = err.index_select(0, ust)
            pos_r = self._epi_pos.index_select(0, ust).clone()
            cnt_r = self._epi_cnt.index_select(0, ust).clone()
            suc_r = self._epi_succ.index_select(0, ust).clone()
            crt_r = (self._epi_cert.index_select(0, ust).clone()
                     if LC else None)
            if bool(enew.any()):
                ekr[se[enew]] = pk_e[enew]
                err[se[enew]] = dkk[enew]
                if bool(ev.any()):
                    self._epi_pos[se[ev]] = -1
                    self._epi_succ[se[ev]] = -1
                    if LC:
                        self._epi_cert[se[ev]] = 0
                    self._epi_cnt[se[ev]] = 0
            se2, fe2 = self._epi_probe(pk_e)
            # pass-1 losers (key not found at either candidate — the
            # slot went to a same-block sibling) retry their ALTERNATE
            # candidate if it's still free
            alt = torch.where(se2 == s1a, s2a, s1a)
            retry = ~fe2 & (ekr.index_select(0, alt) == 0)
            if bool(retry.any()):
                ekr[alt[retry]] = pk_e[retry]
                err[alt[retry]] = dkk[retry]
                se2, fe2 = self._epi_probe(pk_e)
            post_key = ekr.index_select(0, ust)
            inc = int(((pre_key == 0) & (post_key != 0)).sum())
            if inc:
                self._gt.used[re_] += inc
            blk["epi_gpu"] = (dk_cpu, dp_cpu, ust.cpu(),
                              pos_r.cpu(), cnt_r.cpu(),
                              pre_key.cpu(), pre_raw.cpu(),
                              suc_r.cpu(),
                              (crt_r.cpu() if LC else None))
            # cells only where the key actually occupies the row
            se2w = se2[fe2]
            ee_w = ee_[fe2]
            ids_w = ids_t[di][fe2]
            ER = self._epi_pos.shape[1]
            us, inv_e = torch.unique(se2w, return_inverse=True)
            # occurrence rank per slot — positions sharing a key in
            # this block must land on DISTINCT columns (advanced-index
            # writes with duplicate (slot,col) are last-wins) and the
            # count add must accumulate (index_add_, not +=).
            ar_e = torch.arange(se2w.numel(), dtype=torch.int64,
                                device=dev)
            first_i = torch.full((us.numel(),), se2w.numel(),
                                 dtype=torch.int64, device=dev)
            first_i = first_i.scatter_reduce(0, inv_e, ar_e,
                                             reduce="amin")
            rank_e = ar_e - first_i[inv_e]
            col = (self._epi_cnt[se2w] + rank_e) % ER
            self._epi_pos[se2w, col] = ee_w
            self._epi_succ[se2w, col] = ids_w.to(torch.int32)
            if LC:
                self._epi_cert[se2w, col] = certs_d[fe2]
            self._epi_cnt.index_add_(
                0, se2w, torch.ones_like(se2w))
            self._gt.tocc[re_][se2w] = self._epi_cnt[se2w].float()

        # ── EMAs, exact closed form per chunk ──────────────────────────
        # delta-rule writes: each position's applied delta is stored in
        # the block so undo is an exact index_add_ of the negated delta
        # (no re-derivation; fp32 device tensor, no fp16 rounding)
        CH = c.gpu_bulk_chunk
        pd = c.proto_decay
        blk["ids"] = ids_np
        blk["deltas"] = []
        knn_fps = []
        for c0 in range(0, n, CH):
            xchunk = ids_t[c0:c0 + CH]
            s_chunk = s_v[c0:c0 + CH]
            cprev = self._chunk_cprev(xchunk)      # c_{i-1} per token
            if c.use_knn:
                fp_c = cprev.float() @ self._Pcsem
                knn_fps.append(
                    fp_c / (fp_c.norm(dim=1, keepdim=True) + 1e-8))
            if c.delta_rule:
                chat = cprev / (cprev.norm(dim=1, keepdim=True)
                                + 1e-8)
                ax_c = self._A.index_select(0, xchunk)
                pred = (ax_c * chat.float()).sum(dim=1)
                dc_ = chat.float() * ((1.0 - pred)
                                      * c.hebb_lr * s_chunk)[:, None]
            else:
                dc_ = (cprev.float()
                       * (c.hebb_lr * s_chunk)[:, None])
            self._A.index_add_(0, xchunk, dc_)
            blk["deltas"].append(dc_.cpu())     # host copy (VRAM)
            # keep incremental Ac/An/Acn consistent (predict reads them)
            self._Ac.index_add_(0, xchunk, dc_ @ self._Pcsem)
            axn = self._A.index_select(0, xchunk)
            self._An.index_copy_(0, xchunk, axn.norm(dim=1))
            self._Acn.index_copy_(
                0, xchunk,
                self._Ac.index_select(0, xchunk).norm(dim=1))
            # proto over post-chunk A rows (approximation — documented)
            m = xchunk.numel()
            j = torch.arange(m, dtype=torch.float64, device=dev)
            aj = self._A[xchunk].double()
            pdj = pd ** j
            psum = torch.cumsum(aj * (pd ** -j)[:, None], 0)
            PR = (pd ** (j + 1))[:, None] * self._proto.double() \
                + (1 - pd) * pdj[:, None] * psum
            self._proto = PR[-1].float()

        # ── knn bank bulk write: (fp of ctx end pos-1) -> succ ids[i] ──
        if c.use_knn and knn_fps:
            P = c.knn_capacity
            fpall = torch.cat(knn_fps)                       # [n, d]
            # blocks bigger than the bank keep only the newest P rows
            fpall = fpall[-P:]
            kk_ids = ids_t[-fpall.shape[0]:]
            kk_pos = (pos - 1)[-fpall.shape[0]:]
            kslots = (self._knn_cur + torch.arange(
                fpall.shape[0], dtype=torch.int64,
                device=dev)) % P
            alive = self._knn_pos.index_select(0, kslots) >= 0
            blk["knn"] = (kslots.cpu(), alive.cpu(),
                          self._knn_pos.index_select(0, kslots).cpu(),
                          self._knn_succ.index_select(0, kslots).cpu(),
                          self._knn_fp.index_select(0, kslots).cpu())
            self._knn_fp.index_copy_(
                0, kslots, fpall.to(self._knn_fp.dtype))
            self._knn_succ.index_copy_(0, kslots,
                                       kk_ids.to(torch.int32))
            self._knn_pos.index_copy_(0, kslots, kk_pos)
            self._knn_cur += fpall.shape[0]
            if self._gpu:
                self._knncur_g.fill_(self._knn_cur)

        # ── csem bulk: per-key EMA toward mean of follower features ────
        # context key = suffix ending at pos-1 (excluding x, same as
        # the live _suffix_key(csem_order, end=count-1))
        if c.use_csem:
            ck_valid = (pos - c.csem_order >= self._ep_start) \
                & (pos >= c.csem_order)
            ci = torch.nonzero(ck_valid).squeeze(1)
            if ci.numel():
                ck = (Hfull[pos[ci] - base + Kmax]
                      - Hfull[(pos[ci] - c.csem_order - base
                              + Kmax).clamp(min=0)]
                      * self._pows_g[c.csem_order])
                pk = self._csem_hash(ck)
                rcs = self._gt.row_csem
                ckr = self._gt.tkeys[rcs]
                slot, found, evi = self._gt._probe_row_w(rcs, pk)
                newm = ~found
                ev = newm & evi
                if bool(ev.any()):
                    sv = slot[ev]
                    blk["csem_ev"] = (
                        sv,
                        self._gt.traw[rcs].index_select(
                            0, sv).clone(),
                        self._csem_rows.index_select(
                            0, sv).clone(),
                        self._gt.tocc[rcs].index_select(
                            0, sv).clone())
                if bool(newm.any()):
                    ckr[slot[newm]] = pk[newm]
                    self._gt.traw[rcs][slot[newm]] = ck[newm]
                    if bool(ev.any()):
                        self._csem_rows[slot[ev]] = 0.0
                    self._gt.used[rcs] += int((newm & ~evi).sum())
                # per-slot occurrence counts + surprise-weighted
                # follower feature sums (project AFTER gathering —
                # n×dim matvec, not V×dim).  Effective lr per slot =
                # csem_lr * mean-surprise (chunk approximation).
                Ac = self._A[ids_t[ci]] @ self._Pcsem     # [n, csem_dim]
                s_ci = s_v[ci].float()
                uniq_s, inv = torch.unique(slot, return_inverse=True)
                m_per = torch.bincount(inv).float()
                ssum = torch.bincount(inv, weights=s_ci)
                asum = torch.zeros(uniq_s.numel(), c.csem_dim,
                                   dtype=torch.float32, device=dev)
                asum.index_add_(0, inv, Ac.float() * s_ci[:, None])
                amean = asum / ssum.clamp(min=1e-9)[:, None]
                rows = self._csem_rows[uniq_s].float()   # pre-block rows
                slot_new = torch.zeros(uniq_s.numel(), dtype=torch.bool,
                                       device=dev)
                slot_new[inv[newm]] = True
                rows_np = rows.cpu().numpy()
                for i_u, s_ in enumerate(uniq_s.tolist()):
                    blk["csem"].append(
                        (int(s_),
                         None if bool(slot_new[i_u]) else rows_np[i_u]))
                lr_eff = c.csem_lr * (ssum / m_per)
                decay_m = (1.0 - lr_eff) ** m_per
                new_rows = rows * decay_m[:, None] \
                    + amean * (1.0 - decay_m)[:, None]
                self._csem_rows[uniq_s] = new_rows.half()
                self._gt.tocc[rcs].index_add_(
                    0, slot, torch.ones(slot.numel(), device=dev))

        # ── recency / fatigue state ────────────────────────────────────
        # exact pre-block last_seen + within-block previous occurrence
        seen_pre = self._seen_g[ids_t].cpu().numpy()
        enc = ids_np * (n + 1) + np.arange(n)
        order = np.argsort(enc, kind="stable")
        i_s = (enc[order] % (n + 1)).astype(np.int64)
        tok_s = (enc[order] // (n + 1)).astype(np.int64)
        same = tok_s[1:] == tok_s[:-1]
        prev_blk = np.full(n, -1, dtype=np.int64)
        prev_blk[i_s[1:][same]] = i_s[:-1][same]
        self._seen_g.scatter_reduce_(0, ids_t, pos, reduce="amax")
        # one D2H for all touched tokens — not .item() per token
        uniq = np.unique(ids_np)
        vals = self._seen_g[
            torch.from_numpy(uniq).to(dev)].cpu().tolist()
        for t, v in zip(uniq.tolist(), vals):
            self._last_seen[int(t)] = int(v)
        self._recent.extend(
            (base + i - 1, int(x)) for i, x in enumerate(ids_np))

        # ── hedge, chunk-approximated (constant-w per chunk) ──────────
        prev_pos = np.maximum(base + prev_blk, seen_pre)
        ages = (base + np.arange(n) - 1) - prev_pos
        rec_r = np.where(ages < c.recency_span,
                         c.recency_decay ** np.clip(ages, 0, None), 1e-3)
        fat_r = np.where(ages < c.fatigue_span,
                         1.0 - c.fatigue_decay
                         ** np.clip(ages, 0, None), 1.0)
        col = {ch: i for i, ch in enumerate(self.channels)}
        nonord = [col[ch] for ch in
                  ("deep", "topic", "sem", "csem", "knn", "backoff")]
        uni_d = uni_pre.double()
        # chunked multiplicative hedge: one update per gpu_bulk_chunk
        # slice so w evolves through the block (constant-w within a
        # chunk only — much closer to the sequential/live path)
        for c0 in range(0, n, CH):
            c1 = min(c0 + CH, n)
            m_ = c1 - c0
            rw = np.zeros(len(self.channels))
            rw[:O] = rw_o[:, c0:c1].sum(dim=1).cpu().numpy()
            u_sum = float(uni_d[c0:c1].sum())
            rw[col["uni"]] = u_sum
            rw[col["recency"]] = float(rec_r[c0:c1].sum())
            rw[col["fatigue"]] = float(fat_r[c0:c1].sum())
            for ci_ in nonord:
                rw[ci_] = u_sum
            rw /= m_
            p_mix = float((self._w * rw).sum())
            # no "h" journal entry: blk["w"] snapshot already holds the
            # pre-block weights — a trailing "h" would be popped as an
            # orphan and double-revert the previous block's update
            if c.adahedge:
                self._hedge_mix += float(rw.max() - p_mix) * m_
                eta = min(c.hedge_eta_max,
                          math.log(len(self.channels))
                          / max(self._hedge_mix, 1e-9))
            else:
                eta = c.hedge_lr
            expo = np.clip(eta * m_ * (rw - p_mix), -30.0, 30.0)
            self._w *= np.exp(expo)
            np.clip(self._w, c.hedge_min, c.hedge_max, out=self._w)
            self._w /= self._w.sum()

        self._count += n
        # device scalars + hedge weights mirrors
        self._pos_g.fill_(self._count)
        self._tot_g.fill_(float(self._total))
        self._wg.copy_(torch.from_numpy(self._w).to(dev))
        self._hmg.fill_(self._hedge_mix)
        return n

    def _chunk_cprev(self, xchunk: torch.Tensor) -> torch.Tensor:
        """Context vector before each token of the chunk, closed form:
        c_i = lam^{i+1}*c0 + (1-lam)*lam^i*cumsum(r_j*lam^{-j}).
        Advances self._c to the chunk end (caller commits it)."""
        c = self.config
        dev = self._dev
        lam = c.topic_decay
        m = xchunk.numel()
        j = torch.arange(m, dtype=torch.float64, device=dev)
        rj = self._R[xchunk].double()
        lamj = lam ** j
        csum = torch.cumsum(rj * (lam ** -j)[:, None], 0)
        C = (lam ** (j + 1))[:, None] * self._c.double() \
            + (1 - lam) * lamj[:, None] * csum
        cprev = torch.cat([self._c[None].double(), C[:-1]])
        self._c = C[-1].float()
        return cprev

    def _undo_block(self, blk: dict) -> None:
        """Exact inverse of one bulk-ingest block (state snapshot)."""
        c = self.config
        dev = self._dev
        base = blk["count"]
        # subtract table writes (rows found by deterministic re-probe);
        # deltas are the surprise-scaled magnitudes applied forward.
        # Block tensors are stored host-side — re-upload on demand.
        kk_all, tt_all, valid_o = (t.to(dev)
                                   for t in blk["writes_all"])
        s_v = blk.get("surprise")
        if s_v is not None:
            s_v = s_v.to(dev)
        for oi, k in enumerate(self.config.orders):
            idx = torch.nonzero(valid_o[oi]).squeeze(1)
            if not idx.numel():
                continue
            kk = kk_all[oi, idx]
            d = (s_v[idx] if s_v is not None
                 else torch.ones(kk.numel(), dtype=torch.float32,
                                 device=dev))
            self._gt.subtract(k, kk, tt_all[idx], d)
        # restore follower columns displaced by evict-min writes
        cv = blk.get("col_evicted")
        if cv is not None and cv[0].numel():
            cells, toks_v, cnts_v = (t.to(dev) for t in cv)
            gt = self._gt
            gt.ttoks.view(-1).index_copy_(
                0, cells, toks_v.to(torch.int32))
            gt.tcnt.view(-1).index_copy_(0, cells, cnts_v)
            gt.ttot.view(-1).index_add_(0, cells // gt.M, cnts_v)
            gt.tocc.copy_(gt.ttot)
        # csem rows
        rcs = self._gt.row_csem
        ckr = self._gt.tkeys[rcs]
        crr = self._gt.traw[rcs]
        for s_, prev in blk["csem"]:
            if prev is None:
                ckr[s_] = 0
                crr[s_] = 0
                self._csem_rows[s_].zero_()
                self._gt.tocc[rcs, s_] = 0
                self._gt.used[rcs] -= 1
            else:
                self._csem_rows[s_] = torch.from_numpy(
                    prev.astype(np.float16)).to(dev)
        # epi positions (deferred entries: remove by ctx_end)
        for key, p in reversed(blk["epi"]):
            self._epi_remove(key, p)
        # knn bank rows: restore overwritten rows, clear fresh ones
        knn_blk = blk.get("knn")
        if knn_blk is not None:
            kslots, kalive, kppos, kpsucc, kpfp = (
                t.to(dev) for t in knn_blk)
            self._knn_pos.index_copy_(0, kslots, torch.where(
                kalive, kppos, torch.full((), -1, dtype=torch.int64,
                                          device=dev)))
            self._knn_succ.index_copy_(0, kslots, torch.where(
                kalive, kpsucc, torch.full((), -1, dtype=torch.int32,
                                           device=dev)))
            self._knn_fp.index_copy_(0, kslots, torch.where(
                kalive[:, None], kpfp,
                torch.zeros((), dtype=self._knn_fp.dtype,
                            device=dev)))
            self._knn_cur -= int(kslots.numel())
            if self._gpu:
                self._knncur_g.fill_(self._knn_cur)
        # topic rows: subtract the exact deltas applied forward
        # (stored fp32 in the block — no re-derivation, bit-exact)
        ids_np = blk["ids"]
        ids_t = torch.from_numpy(ids_np.astype(np.int64)).to(dev)
        deltas = blk.get("deltas")
        if deltas:
            dall = torch.cat([d.to(dev) for d in deltas])
            self._A.index_add_(0, ids_t, -dall)
            self._Ac.index_add_(0, ids_t, -(dall @ self._Pcsem))
            uniq = torch.unique(ids_t)
            self._An.index_copy_(
                0, uniq, self._A.index_select(0, uniq).norm(dim=1))
            self._Acn.index_copy_(
                0, uniq,
                self._Ac.index_select(0, uniq).norm(dim=1))
        else:  # legacy blocks: replay chunk c-vectors, subtract adds
            CH = c.gpu_bulk_chunk
            c_save = self._c
            for ci0, c_start in enumerate(blk["c_starts"]):
                xchunk = ids_t[ci0 * CH:(ci0 + 1) * CH]
                self._c = torch.from_numpy(c_start).to(dev)
                cprev = self._chunk_cprev(xchunk)
                dc_ = (-cprev * c.hebb_lr).float()
                self._A.index_add_(0, xchunk, dc_)
                self._Ac.index_add_(0, xchunk, dc_ @ self._Pcsem)
                axn = self._A.index_select(0, xchunk)
                self._An.index_copy_(0, xchunk, axn.norm(dim=1))
                self._Acn.index_copy_(
                    0, xchunk,
                    self._Ac.index_select(0, xchunk).norm(dim=1))
            self._c = c_save
        # scalar/vector state — tensors may be CPU-resident after a
        # snapshot/load (torch.save drops the device), so re-home them
        self._c = blk["c"].to(dev)
        self._proto = blk["proto"].to(dev)
        self._w = blk["w"]
        self._hedge_mix = blk.get("hedge_mix", 0.0)
        self._uni = blk["uni"]
        self._unig = blk["unig"].to(dev)
        self._log_uni_num = blk["log_uni_num"].to(dev)
        self._total -= blk["n"]
        self._count = base
        self._ep_start = blk["ep_start"]
        self._last_seen = blk["last_seen"]
        self._seen_g = blk["seen_g"].to(dev)
        if self._gpu:
            self._wg = torch.from_numpy(
                blk["w"].astype(np.float64)).to(dev)
            self._hmg.fill_(self._hedge_mix)
            self._pos_g.fill_(base)
            self._ep_g.fill_(blk["ep_start"])
            self._tot_g.fill_(float(self._total))
            eg = blk.get("epi_gpu")
            if eg is not None:
                if len(eg) == 9 and eg[5].dtype != torch.bool:
                    # candidate-slot format: wholesale restore of
                    # (key, raw, pos, cnt, succ, cert) for every slot
                    # either probe could have touched — subsumes
                    # victim-row restore for evictions
                    _dk, _dp, us, pos_r, cnt_r, pre_k, pre_raw, \
                        succ_r, cert_r = eg
                    re_ = self._gt.row_epi
                    us_t = us.to(dev)
                    pk0 = pre_k.to(dev)
                    cur = self._gt.tkeys[re_, us_t]
                    newly = (pk0 == 0) & (cur != 0)
                    self._gt.used[re_] -= int(newly.sum())
                    self._gt.tkeys[re_, us_t] = pk0
                    self._gt.traw[re_, us_t] = pre_raw.to(dev)
                    self._epi_pos[us_t] = pos_r.to(dev)
                    self._epi_cnt[us_t] = cnt_r.to(dev)
                    if succ_r is not None:
                        self._epi_succ[us_t] = succ_r.to(dev)
                    if cert_r is not None:
                        self._epi_cert[us_t] = cert_r.to(dev)
                    self._gt.tocc[re_, us_t] = \
                        cnt_r.to(dev).float()
                elif len(eg) >= 7:
                    # per-slot pre-write snapshot — restore wholesale
                    # (exact incl. ring cursor); newer blocks also
                    # carry the succ/cert payload columns
                    _dk, _dp, us, pos_r, cnt_r, new_u, inc_u = eg[:7]
                    succ_r = eg[7] if len(eg) > 7 else None
                    cert_r = eg[8] if len(eg) > 8 else None
                    re_ = self._gt.row_epi
                    us_t = us.to(dev)
                    nw = new_u.to(dev)
                    if bool(nw.any()):
                        clr = us_t[nw]
                        self._gt.tkeys[re_, clr] = 0
                        self._gt.traw[re_, clr] = 0
                    inc = inc_u.to(dev)
                    self._gt.used[re_] -= int(inc.sum())
                    self._epi_pos[us_t] = pos_r.to(dev)
                    self._epi_cnt[us_t] = cnt_r.to(dev)
                    if succ_r is not None:
                        self._epi_succ[us_t] = succ_r.to(dev)
                    if cert_r is not None:
                        self._epi_cert[us_t] = cert_r.to(dev)
                    self._gt.tocc[re_, us_t] = \
                        cnt_r.to(dev).float()
                else:
                    # legacy (dk, dp) format — clear matching positions
                    dk_np, dp_np = eg
                    pk_e = self._epi_hash(torch.from_numpy(
                        dk_np.astype(np.int64)).to(dev))
                    se, fe = self._epi_probe(pk_e)
                    if bool(fe.any()):
                        rows = self._epi_pos[se[fe]]
                        m_ = rows == torch.from_numpy(
                            dp_np.astype(np.int64)).to(dev)[fe][:, None]
                        rows[m_] = -1
                        self._epi_pos[se[fe]] = rows
            # restore rows this block's inserts evicted (victims were
            # displaced first in forward order -> restored last here)
            evx = blk.get("evicted")
            if evx is not None and evx[0].numel():
                row_ix, raw, toks, cnts, tot = (
                    t.to(dev) for t in evx)
                for j in range(row_ix.numel()):
                    self._gt.restore_row(
                        int(row_ix[j]),
                        int(raw[j]) & 0xFFFFFFFFFFFFFFFF,
                        toks[j], cnts[j], float(tot[j]))
            eev = blk.get("epi_ev")
            if eev is not None and eev[0].numel():
                sv, raw, pos_r, cnt_r = (t.to(dev)
                                         for t in eev[:4])
                succ_r = eev[4].to(dev) if len(eev) > 4 else None
                cert_r = (eev[5].to(dev)
                          if len(eev) > 5 and eev[5] is not None
                          else None)
                re_ = self._gt.row_epi
                for j in range(sv.numel()):
                    # cells written by this block (pos >= base) are
                    # being undone — keep only pre-block positions
                    keep = (pos_r[j] >= 0) & (pos_r[j] < base)
                    cnt_j = int(keep.sum())   # cursor-equivalent mod ER
                    if cnt_j == 0:
                        # victim held only block-era cells — it did
                        # not exist pre-block; restoring it would
                        # create a zombie row (and could displace a
                        # live row via _pick_slot's 2-choice fallback)
                        continue
                    pk = self._epi_hash(raw[j].reshape(1))
                    s, _ = self._gt._pick_slot(re_, pk)
                    if int(self._gt.tkeys[re_, s]) == 0:
                        self._gt.used[re_] += 1
                    self._gt.tkeys[re_, s] = pk[0]
                    self._gt.traw[re_, s] = raw[j]
                    self._epi_pos[s] = torch.where(
                        keep, pos_r[j],
                        torch.full((), -1, dtype=torch.int64,
                                   device=dev))
                    if succ_r is not None:
                        self._epi_succ[s] = torch.where(
                            keep, succ_r[j],
                            torch.full((), -1, dtype=torch.int32,
                                       device=dev))
                    if cert_r is not None:
                        self._epi_cert[s] = torch.where(
                            keep[:, None], cert_r[j],
                            torch.zeros((), dtype=torch.int64,
                                        device=dev))
                    self._epi_cnt[s] = cnt_j
                    self._gt.tocc[re_, s] = float(cnt_j)
            cev = blk.get("csem_ev")
            if cev is not None and cev[0].numel():
                sv, raw, vec_r, oc_r = cev
                rc_ = self._gt.row_csem
                for j in range(sv.numel()):
                    pk = self._csem_hash(raw[j].reshape(1))
                    s, _ = self._gt._pick_slot(rc_, pk)
                    if int(self._gt.tkeys[rc_, s]) == 0:
                        self._gt.used[rc_] += 1
                    self._gt.tkeys[rc_, s] = pk[0]
                    self._gt.traw[rc_, s] = raw[j]
                    self._csem_rows[s] = vec_r[j]
                    self._gt.tocc[rc_, s] = oc_r[j]
        while len(self._recent) > blk["recent_len"]:
            self._recent.pop()
        while len(self._gaps) > blk["gaps"]:
            self._gaps.pop()

    def _decay_table(self, tab: dict[int, dict[int, float]],
                     tots: dict[int, float]) -> None:
        """Halve all counts and drop empties (forgetting + capacity)."""
        dead = []
        for key, cell in tab.items():
            for t in list(cell):
                v = cell[t] * 0.5
                if v < 0.25:
                    del cell[t]
                else:
                    cell[t] = v
            if not cell:
                dead.append(key)
            else:
                tots[key] = sum(cell.values())
        for key in dead:
            del tab[key]
            tots.pop(key, None)

    # ── prediction ──────────────────────────────────────────────────────

    def _predict(self) -> np.ndarray:
        """Next-token logit vector [V] from the current sketch."""
        if self._gpu:
            return self._predict_gpu()
        c = self.config
        end = self._count - 1
        keys = self._keys_at(end) if end >= self._ep_start else {}
        denom = self._total + c.uni_alpha * self.vocab_size
        w = {ch: self._w[i] for i, ch in enumerate(self.channels)}

        # base = log unigram prior = log_numer - log(denom); the numer
        # tensor is maintained per-write so this is one vector subtract.
        log_denom = math.log(denom)
        logits = self._log_uni_num - log_denom      # [V] on self._dev

        def log_uni(t: int) -> float:
            # scalar read from the host-side count array — no GPU sync
            return math.log(self._uni[t] + c.uni_alpha) - log_denom

        lift: dict[int, float] = {}

        # order channels
        for k in c.orders:
            key = keys.get(k)
            if key is None:
                continue
            cell = self._tables[k].get(key)
            if not cell:
                continue
            tot = self._totals[k].get(key, 0.0)
            wk = w[f"o{k}"]
            if c.evidence_lift:
                wk += tot / (tot + c.backoff_kappa)
            for t, cnt in cell.items():
                p = (cnt + c.alpha * self._uni_p(t, denom)) / (tot + c.alpha)
                lift[t] = lift.get(t, 0.0) + wk * (math.log(p) - log_uni(t))

        # backoff channel: longest-evidenced order's distribution,
        # confidence-discounted — one arm instead of O probes
        if c.use_backoff:
            best_k, bkey, btot = -1, -1, 0.0
            for k, key in keys.items():
                t = self._totals[k].get(key, 0.0)
                if t >= c.backoff_min_tot and k > best_k:
                    best_k, bkey, btot = k, key, t
            if best_k >= 0:
                cell = self._tables[best_k].get(bkey) or {}
                conf = btot / (btot + c.backoff_kappa)
                wb = w["backoff"] + (conf if c.evidence_lift else 0.0)
                for t, cnt in cell.items():
                    p = conf * (cnt + c.alpha * self._uni_p(t, denom)) \
                        / (btot + c.alpha)
                    lift[t] = lift.get(t, 0.0) \
                        + wb * (math.log(p) - log_uni(t))

        # deep episodic channel
        votes = self._deep_votes(end)
        if votes:
            vt = sum(votes.values())
            wd = w["deep"] + (vt / (vt + c.backoff_kappa)
                              if c.evidence_lift else 0.0)
            for t, v in votes.items():
                p = (v + c.alpha * self._uni_p(t, denom)) / (vt + c.alpha)
                lift[t] = lift.get(t, 0.0) + wd * (math.log(p) - log_uni(t))

        # knn channel: approximate-match contexts vote their successors
        if c.use_knn:
            kvotes = self._knn_votes()
            if kvotes:
                vt = sum(kvotes.values())
                wk = w["knn"]
                if c.evidence_lift:
                    wk += vt / (vt + c.backoff_kappa)
                for t, v in kvotes.items():
                    p = (v + c.alpha * self._uni_p(t, denom)) \
                        / (vt + c.alpha)
                    lift[t] = lift.get(t, 0.0) \
                        + wk * (math.log(p) - log_uni(t))

        # topic channel: cosine readout on candidates only
        if lift:
            cn = float(self._c.norm())
            if cn > 1e-8:
                wt = w["topic"] * c.topic_beta
                for t in list(lift):
                    a = self._A[t]
                    an = float(a.norm())
                    if an > 1e-8:
                        lift[t] += wt * float(a @ self._c) / (an * cn)

        # semantic-spread channel: score tokens by fingerprint similarity
        # to the expected-next fingerprint — generalizes beyond the exact
        # followers seen in this context.  On CUDA it scores all V (one
        # matvec); on CPU it scores candidates only.
        if c.use_sem:
            pn = float(self._proto.norm())
            if pn > 1e-8:
                ws = w["sem"] * c.sem_beta
                if self._dev.type == "cuda":
                    an = self._A.norm(dim=1)
                    sims = (self._A @ self._proto) / (an * pn + 1e-8)
                    logits = logits + ws * sims
                else:
                    for t in list(lift):
                        a = self._A[t]
                        an = float(a.norm())
                        if an > 1e-8:
                            lift[t] += ws * float(a @ self._proto) \
                                / (an * pn)

        # csem: per-context fingerprint — same trick as sem but keyed by
        # the current context (contextual recombination)
        if c.use_csem and lift:
            row = self._csem_read()
            if row is not None:
                rn = float(row.norm())
                if rn > 1e-8:
                    wc = w["csem"] * c.csem_beta
                    for t in list(lift):
                        ax = self._A[t] @ self._Pcsem
                        an = float(ax.norm())
                        if an > 1e-8:
                            lift[t] += wc * float(ax @ row) / (an * rn)

        # recency channel: additive boost for recently seen tokens
        wr = w["recency"]
        if wr > 1e-4:
            for pos, t in reversed(self._recent):
                age = end - pos
                if age >= c.recency_span:
                    break
                if age >= 0:
                    lift[t] = lift.get(t, 0.0) + wr * c.recency_decay ** age

        # fatigue channel: subtractive penalty for just-emitted tokens —
        # the memory model's intrinsic "you already said that"
        wf = w["fatigue"] * c.fatigue_strength
        if wf > 1e-4:
            for pos, t in reversed(self._recent):
                age = end - pos
                if age >= c.fatigue_span:
                    break
                if age >= 0:
                    lift[t] = lift.get(t, 0.0) - wf * c.fatigue_decay ** age

        if lift:
            idx_t = torch.tensor(list(lift.keys()), dtype=torch.long,
                                 device=self._dev)
            val_t = torch.tensor(list(lift.values()), dtype=torch.float32,
                                 device=self._dev)
            logits.index_add_(0, idx_t, val_t)
        return logits

    # ── public API ──────────────────────────────────────────────────────

    def ingest(self, ids, tag: str | None = None, learn: bool = True) -> int:
        """Stream-train on token ids (instant, hot). Returns #tokens."""
        if isinstance(ids, torch.Tensor):
            ids = ids.reshape(-1).tolist()
        prev_tag, prev_learn = self.tag, self.learning
        if tag is not None:
            self.tag = tag
        self.learning = learn
        if self.learning and self._gpu:
            n = self._ingest_bulk(ids)
        else:
            n = 0
            for x in ids:
                x = int(x)
                if self.learning:
                    self._learn_token(x)
                else:
                    self._push(x)
                n += 1
        self.tag, self.learning = prev_tag, prev_learn
        return n

    def window_tokens(self, end: int, n: int) -> list[int]:
        """The last ``n`` ring tokens ending at absolute pos ``end``
        (end exclusive).  Short/empty near stream start or wrap edge."""
        cap = self.config.ring_capacity
        start = max(0, self._count - cap, end - n)
        return [int(self._ring[i % cap]) for i in range(start, end)]

    def write_counts(self, ctx_end: int, deltas: dict[int, float],
                     tag: str = "distill") -> int:
        """Write fractional follower counts under the context keys valid
        at absolute position ``ctx_end`` (teacher distillation / manual
        knowledge injection).  Journaled under ``tag`` — revertible via
        ``revert_tag``.  Returns number of cell writes.

        Uses RAW suffix hashes (not episode-gated keys): the stored key
        still matches whenever the same token sequence recurs, which is
        what retrieval needs.  Only writes orders whose tokens are still
        in the ring."""
        keys = {}
        floor = max(0, self._count - self.config.ring_capacity)
        for k in self.config.orders:
            if ctx_end - 1 - k + 1 >= floor:
                keys[k] = self._raw_hash(k, ctx_end - 1)
        self.journal.append(("ws", tag))      # write-block start marker
        n = self._write_counts_keys(keys, deltas)
        self.journal.append(("we", tag))      # write-block end marker
        return n

    def _write_counts_keys(self, keys: dict[int, int],
                           deltas: dict[int, float]) -> int:
        """Shared cell writer for ``write_counts``/``teach``: adds each
        delta under every (order, key) pair and journals the ``c``
        entries.  Returns cells written."""
        n = 0
        if self._gpu:
            toks_l = list(deltas.keys())
            ds_l = [float(deltas[t]) for t in toks_l]
            for k, key in keys.items():
                ctx_t = torch.full((len(toks_l),), _s64(key),
                                   dtype=torch.int64, device=self._dev)
                toks_t = torch.tensor(toks_l, dtype=torch.int64,
                                      device=self._dev)
                d_t = torch.tensor(ds_l, dtype=torch.float32,
                                   device=self._dev)
                self._gt.add(k, ctx_t, toks_t, d_t)
                for t, d in deltas.items():
                    self.journal.append(("c", k, key, t, d))
                    n += 1
            return n
        for k, key in keys.items():
            tab = self._tables[k]
            cell = tab.get(key)
            if cell is None:
                cell = tab[key] = {}
            tots = self._totals[k]
            for t, d in deltas.items():
                cell[t] = cell.get(t, 0.0) + d
                tots[key] = tots.get(key, 0.0) + d
                self.journal.append(("c", k, key, t, d))
                n += 1
        return n

    # ── instant hot-training ────────────────────────────────────────────

    @staticmethod
    def _prefix_hashes(ids) -> np.ndarray:
        """Model-compatible rolling prefix hashes over a token list —
        the same recurrence the ring maintains, standalone."""
        h = np.empty(len(ids), dtype=np.uint64)
        prev = 0
        for i, t in enumerate(ids):
            prev = (prev * 1099511628211 + int(t) + 1) % (1 << 64)
            h[i] = prev
        return h

    def _seq_key(self, H: np.ndarray, k: int, e: int) -> int | None:
        """Suffix key of length ``k`` ending at seq index ``e``
        (0-based, seq-local — no ring, no episode gate)."""
        if e - k + 1 < 0 or e < k - 1:
            return None
        return (int(H[e]) - int(H[e - k]) * int(self._pows[k])) \
            % (1 << 64)

    def _context_vector(self, ids: list[int]) -> torch.Tensor:
        """Standalone topic-EMA context vector for arbitrary ids —
        same weighting ``self._c`` accumulates live (the constant
        ``1 - lam`` factor is irrelevant: fingerprints normalize)."""
        c = self.config
        if not ids:
            return torch.zeros(c.topic_dim, device=self._dev)
        idx = torch.tensor(ids, dtype=torch.int64, device=self._dev)
        w = (c.topic_decay ** torch.arange(
            len(ids) - 1, -1, -1, dtype=torch.float32,
            device=self._dev))
        return (self._R[idx].float() * w[:, None]).sum(0)

    def teach(self, context_ids, target_ids, weight: float = 25.0,
              tag: str = "teach", generalize: bool = True) -> dict:
        """Instant hot-training: bind ``context -> target``.

        For every target position, writes ``weight`` follower-count
        under every valid order key of its context (exact recall —
        keys are computed from the token sequence itself, so no ring
        presence is needed).  With ``generalize`` the csem fingerprint
        of each context EMAs toward the target's A-row and one knn bank
        entry binds the context fingerprint to the first target token —
        *similar* contexts then also lift the answer.

        All writes are journaled inside one ``ws``/``we`` block under
        ``tag`` — ``revert_tag(tag)`` un-teaches.  Idempotent-ish:
        repeated calls keep adding count mass.

        Returns {cells, csem, knn} write counts.
        """
        c = self.config
        ctx = [int(t) for t in context_ids]
        tgt = [int(t) for t in target_ids]
        rep = {"cells": 0, "csem": 0, "knn": 0}
        if not tgt:
            return rep
        seq = ctx + tgt
        H = self._prefix_hashes(seq)
        self.journal.append(("ws", tag))
        for i, t in enumerate(tgt):
            e = len(ctx) + i - 1                # context end in seq
            keys = {k: self._seq_key(H, k, e) for k in c.orders}
            keys = {k: v for k, v in keys.items() if v is not None}
            rep["cells"] += self._write_counts_keys(keys, {t: weight})
            if not generalize:
                continue
            ck = self._seq_key(H, c.csem_order, e)
            if ck is not None and c.use_csem:
                self._csem_write_key(ck, t, scale=1.0)
                rep["csem"] += 1
            if i == 0 and c.use_knn and ctx:
                # one bank entry: fp(context) -> first target token
                fp = self._knn_fp_of(self._context_vector(ctx))
                self._knn_write(t, self._count + i, fp=fp)
                if self._gpu:
                    self._jcur = (self._jcur + 1) % c.vec_delta_cap
                rep["knn"] += 1
        self.journal.append(("we", tag))
        return rep

    def teach_text(self, context: str, target: str, tokenizer,
                   **kw) -> dict:
        """``teach`` on raw text (tokenizes context/target)."""
        ctx = tokenizer(context, add_special_tokens=False)
        tgt = tokenizer(target, add_special_tokens=False)
        ctx_ids = getattr(ctx, "input_ids", getattr(ctx, "ids", ctx))
        tgt_ids = getattr(tgt, "input_ids", getattr(tgt, "ids", tgt))
        return self.teach(list(ctx_ids), list(tgt_ids), **kw)

    # ── self-evolution: reinforce / consolidate ───────────────────────

    def reinforce(self, tag: str, gain: float = 1.0) -> dict:
        """Outcome-weighted replay of a tag's learned cell writes.

        Re-applies every ``c`` delta journaled under ``tag`` scaled by
        ``gain`` — the self-play reward hook: ``gain>0`` strengthens
        what a successful tool-use trajectory learned, ``gain<0``
        weakens (cells floor at 0).  The replay is journaled under
        ``reinforce:<tag>`` so reinforcement itself is revertible.
        Forgetting entirely is ``revert_tag(tag)``.
        """
        deltas = []
        cur = None
        for e in self.journal.entries:
            if e[0] == "ws":
                cur = e[1]
            elif e[0] == "we":
                cur = None
            elif e[0] == "c" and cur == tag:
                deltas.append((e[1], e[2], e[3], e[4]))
            elif e[0] == "bc" and cur == tag:
                # bulk-ingest block inside a tagged span: its order-cell
                # writes are (key, succ, surprise) per (order, token) —
                # the "c" entries' vectorized form.  Amplifying them
                # keeps reinforce() live on the cuda_primary path.
                blk = e[1]
                kk_all, tt_all, valid_o = blk["writes_all"]
                s_v = blk.get("surprise")
                for oi, k in enumerate(self.config.orders):
                    idx = torch.nonzero(valid_o[oi]).squeeze(1)
                    for j in idx.tolist():
                        deltas.append(
                            (k, int(kk_all[oi, j]),
                             int(tt_all[j]),
                             float(s_v[j]) if s_v is not None
                             else 1.0))
        if not deltas:
            return {"found": False, "applied": 0, "tag": tag}
        rtag = f"reinforce:{tag}"
        self.journal.append(("ws", rtag))
        for k, key, t, d in deltas:
            self._write_counts_keys({k: key}, {t: d * gain})
        self.journal.append(("we", rtag))
        if gain < 0:
            self._clamp_counts({(k, key) for k, key, _, _ in deltas})
        return {"found": True, "applied": len(deltas), "tag": rtag,
                "gain": gain}

    def apply_packets(self, packets, accept=None) -> dict:
        """Apply stream update packets through the single learning
        process — serialized ``ingest`` of each accepted packet under
        its tag, wrapped in a ``ws``/``we`` write-block so the applied
        writes are visible to ``reinforce(tag)`` (order-cell deltas) and
        wholesale-revertible via ``revert_writes(tag)``/``revert_tag``.

        ``accept``: None (apply all), a bool sequence aligned with
        ``packets``, or a predicate ``accept(packet) -> bool``.  This is
        the gate that decides which concurrent-generation packets are
        worth learning and which are discarded.
        """
        applied = []
        for i, p in enumerate(packets):
            if callable(accept):
                ok = bool(accept(p))
            elif accept is not None:
                ok = bool(accept[i])
            else:
                ok = True
            if not ok:
                continue
            self.journal.append(("ws", p.tag))
            self.ingest(p.ids, tag=p.tag)
            self.journal.append(("we", p.tag))
            applied.append(i)
        return {"applied": applied}

    def _clamp_counts(self, touched: set) -> None:
        """Floor negative cell counts at 0 after negative-gain writes —
        counts are probability masses; negative cells corrupt the
        channel mixtures.  ``touched`` = (order, key) rows to check."""
        if self._gpu:
            self._gt.tcnt.clamp_(min=0)
            self._gt.ttot.copy_(self._gt.tcnt.sum(-1))
            self._gt.tocc.copy_(self._gt.ttot)
            return
        for k, key in touched:
            cell = self._tables[k].get(key)
            if not cell:
                continue
            for t in list(cell):
                if cell[t] < 0:
                    cell[t] = 0.0
            self._totals[k][key] = sum(cell.values())

    def consolidate(self, min_entries: int = 4, min_agree: float = 0.5,
                    mass: float = 1.0, tag: str = "consol") -> dict:
        """Self-distillation: migrate repeated episodic patterns into
        the fast n-gram tables, then free their episodic cells.

        For each episodic key with >= ``min_entries`` cells whose
        successors agree at fraction >= ``min_agree``, the modal
        successor is written into the order tables at every on-ring
        position (``write_counts`` — journaled under ``tag``), and the
        promoted cells are dropped from the episodic deque (journaled
        ``eE`` — revertible).  Off-ring cells stay archived: their
        order keys can't be recomputed from a raw suffix hash, so their
        only faithful store is the episodic row itself.

        This is the math-driven counterpart of SFT: frequently-
        retrieved context->successor pairs promote from expensive
        episodic storage to O(1) n-gram reads."""
        c = self.config
        from collections import Counter
        floor = max(0, self._count - c.ring_capacity)
        rep = {"keys": 0, "promoted": 0, "freed": 0}
        self.journal.append(("ws", tag))
        for key, dq in list(self._epi.items()):
            if len(dq) < min_entries:
                continue
            cnt = Counter(ent[1] for ent in dq if ent[1] >= 0)
            if not cnt:
                continue
            succ, n_agree = cnt.most_common(1)[0]
            if n_agree / len(dq) < min_agree:
                continue
            on_ring = [e for e in dq
                       if floor <= e[0] < self._count]
            if not on_ring:
                continue                     # nothing promotable
            for ent in on_ring:
                # write_counts(ctx_end) keys end at ctx_end-1; the
                # entry's pos IS the last context token -> pos+1
                rep["promoted"] += self.write_counts(
                    int(ent[0]) + 1, {succ: mass}, tag=tag)
            promoted_pos = {e[0] for e in on_ring}
            keep = [e for e in dq if e[0] not in promoted_pos]
            for ent in on_ring:
                self.journal.append(
                    ("eE", key, ent[0], ent[1], ent[2]))
                self._epi_n -= 1
                rep["freed"] += 1
            if keep:
                self._epi[key] = deque(keep,
                                       maxlen=c.epi_max_per_key)
            else:
                del self._epi[key]
                self._epi_hits.pop(key, None)
            self._epi_sync_row(key)
            rep["keys"] += 1
        self.journal.append(("we", tag))
        return rep

    # ── teacher graft ─────────────────────────────────────────────────

    def graft_teacher(self, emb, method: str | None = None) -> dict:
        """Architectural graft: replace the random bipolar feature
        matrix ``_R`` with a projection of a pretrained teacher's
        embedding table (e.g. ForgeLM V2 ``d_embed.weight``).

        Rationale: ``_R[tok]`` is the sole feature source for the
        context sketch ``_c``, the Hebbian topic matrix ``_A``, proto,
        csem rows and knn fingerprints — so semantic embeddings upgrade
        every channel at once (contexts become a bag-of-meanings, knn
        retrieves *semantically* similar contexts, not just token-
        overlap ones).  Still an associative memory — the graft gives
        good features, not compositionality.

        ``emb`` = [V_t, d] tensor or a safetensors/.pt path (the
        [vocab, dim] tensor is auto-detected).  Rows are unit-normed
        (embedding norm encodes frequency, not meaning), projected to
        ``topic_dim`` via PCA (dominant semantic axes) or a seeded
        random projection (JL — preserves pairwise cosines), then
        rescaled to norm sqrt(topic_dim) so Hebbian/delta-rule rates
        calibrated on ±1 bipolar rows stay valid.

        Must run before any learning — learned state is tied to the
        feature space it was written in."""
        c = self.config
        if self._count or self._epi_n or self._knn_cur:
            raise RuntimeError(
                "graft before learning — existing memory cells were "
                "written under the old feature space")
        method = method or c.teacher_embed_method
        E = (_load_teacher_embedding(emb)
             if isinstance(emb, (str, Path)) else emb.float())
        if E.dim() != 2:
            raise ValueError(f"teacher embedding must be 2-D, got "
                             f"{tuple(E.shape)}")
        if E.shape[0] < self.vocab_size:
            raise ValueError(
                f"teacher vocab {E.shape[0]} < flux vocab "
                f"{self.vocab_size} — cannot fabricate semantics")
        E = E[: self.vocab_size]
        En = E / E.norm(dim=1, keepdim=True).clamp(min=1e-8)
        # drop the shared 'commonness' direction: embedding means encode
        # frequency/register, and subtracting it collapses the random-
        # pair cosine baseline to ~0 while semantic pairs keep their
        # signal (measured: cat-dog 0.155 vs 0.002 baseline on V2)
        En = En - En.mean(0, keepdim=True)
        # project on GPU when available (65536x2560 SVD is heavy on CPU)
        pca_dev = self._dev if self._dev.type == "cuda" else E.device
        En = En.to(pca_dev)
        if method == "pca":
            _, _, Vh = torch.pca_lowrank(En, q=c.topic_dim)
            R = En @ Vh                             # [V, topic_dim]
        elif method == "randproj":
            g = torch.Generator(device=En.device).manual_seed(
                c.seed ^ 0x6A7A)
            W = torch.randn(E.shape[1], c.topic_dim, generator=g,
                            device=En.device) / math.sqrt(E.shape[1])
            R = En @ W
        else:
            raise ValueError(f"unknown graft method {method!r}")
        R = R / R.norm(dim=1, keepdim=True).clamp(min=1e-8) \
            * math.sqrt(c.topic_dim)
        self._R = R.float().to(self._dev)
        self._graft = {"src": str(emb), "method": method,
                       "teacher_dim": int(E.shape[1]),
                       "topic_dim": int(c.topic_dim)}
        self.journal.append(("g", dict(self._graft)))
        return self._graft

    def gap_positions(self) -> list[int]:
        """Positions whose contexts were thin when learned (distill
        targets), oldest first.  Positions rewound by revert are
        dropped."""
        return sorted(p for p in self._gaps if p <= self._count)

    def rescan_gaps(self) -> int:
        """Rebuild gap positions by scanning the ring — for snapshots
        taken before gap tracking existed (or after big reverts).
        Measures CURRENT memory confidence for each stored context."""
        self._gaps.clear()
        c = self.config
        k = c.gap_probe_order
        start = max(0, self._count - c.ring_capacity)
        if self._gpu:
            # vectorized: raw probe-order hash per position -> tot probe
            pp = torch.arange(start + 1, self._count, dtype=torch.int64,
                              device=self._dev)
            valid = pp - 1 - k + 1 >= 0
            pp = pp[valid]
            if pp.numel() == 0:
                return 0
            e_ = pp - 1
            keys = (self._Hg[e_ % c.ring_capacity]
                    - self._Hg[(e_ - k) % c.ring_capacity]
                    * self._pows_g[k])
            tots = self._gt.read_tot(k, keys)
            thin = pp[tots < c.gap_min_total]
            self._gaps.extend(thin.cpu().tolist())
            return len(self._gaps)
        tab = self._tables.get(k, {})
        tots = self._totals.get(k, {})
        for pos in range(start + 1, self._count):
            key = self._raw_hash(k, pos - 1)
            if tots.get(key, 0.0) < c.gap_min_total:
                self._gaps.append(pos)
        return len(self._gaps)

    def distill_from(self, score_fn, top_k: int = 5, scale: float = 2.0,
                     max_gaps: int | None = None, batch_size: int = 64,
                     ctx_len: int = 32, tag: str = "distill",
                     progress=None) -> dict:
        """Selective teacher distillation.

        For each gap position, build its ``ctx_len``-token suffix window,
        ask ``score_fn`` for the teacher's top-k next-token
        probabilities, and write them as fractional counts into this
        model's order tables (journaled under ``tag``).

        ``score_fn(contexts: list[list[int]]) -> (top_ids[B,k],
        top_probs[B,k])`` — e.g. a ForgeLM/HF forward returning logits.
        """
        import torch as _t
        gaps = self.gap_positions()
        if not gaps:
            n = self.rescan_gaps()
            if n:
                gaps = self.gap_positions()
        if max_gaps is not None:
            gaps = gaps[:max_gaps]
        written = scored = 0
        t0 = time.perf_counter()
        for i in range(0, len(gaps), batch_size):
            batch = gaps[i:i + batch_size]
            ctxs = [self.window_tokens(p, ctx_len) for p in batch]
            # left-pad short windows to ctx_len (pad with token 0)
            padded = [[0] * (ctx_len - len(c)) + c for c in ctxs]
            top_ids, top_probs = score_fn(padded)
            top_ids = (top_ids.tolist() if isinstance(top_ids, _t.Tensor)
                       else top_ids)
            top_probs = (top_probs.tolist()
                         if isinstance(top_probs, _t.Tensor)
                         else top_probs)
            for pos, ids_row, probs_row in zip(batch, top_ids, top_probs):
                deltas = {int(t): float(p) * scale
                          for t, p in zip(ids_row, probs_row)}
                written += self.write_counts(pos, deltas, tag=tag)
            scored += len(batch)
            if progress:
                progress(scored, len(gaps))
        return {"gaps_scored": scored, "cells_written": written,
                "seconds": time.perf_counter() - t0}

    def fit_text(self, text: str, tokenizer, tag: str = "text") -> int:
        enc = tokenizer(text, add_special_tokens=False)
        ids = getattr(enc, "input_ids", None)
        if ids is None:
            ids = getattr(enc, "ids", enc)
        return self.ingest(list(ids), tag=tag)

    def soft_reset(self) -> None:
        """New episode: reset context sketch, keep all learned memory."""
        self.journal.append(("b", self._ep_start,
                             self._c.detach().cpu().numpy()
                             .astype(np.float16)))
        self._ep_start = self._count
        self._c.zero_()
        self._last_seen.clear()
        self._recent.clear()
        if self._gpu:
            self._seen_g.fill_(-10 ** 9)
            self._ep_g.fill_(self._count)

    def forward(self, idx: torch.Tensor, targets=None,
                past_key_values=None, use_cache: bool = False,
                return_hidden: bool = False, **kw):
        """Engine contract: returns (logits, loss_or_None, None).

        Each input token is observed (live-learned) then used as context
        for the next-position logits — standard causal alignment.
        """
        if isinstance(idx, torch.Tensor):
            rows = idx.tolist()
        else:
            rows = [list(idx)]
        all_logits = []
        losses = []
        tgt_rows = (targets.tolist() if isinstance(targets, torch.Tensor)
                    else None)
        for b, row in enumerate(rows):
            outs = []
            tgts = tgt_rows[b] if tgt_rows else None
            for i, x in enumerate(row):
                x = int(x)
                if self._gpu and self.learning:
                    outs.append(self._fused_step(x).clone())
                else:
                    if self.learning:
                        self._learn_token(x)
                    else:
                        self._push(x)
                    outs.append(self._predict())
                if tgts is not None and i + 1 < len(tgts):
                    lp = torch.log_softmax(outs[-1].float(), dim=-1)
                    losses.append(-float(lp[int(tgts[i + 1])]))
            all_logits.append(
                torch.stack(outs) if outs else
                torch.zeros((0, self.vocab_size), device=self._dev))
        logits = torch.stack(all_logits)
        if isinstance(idx, torch.Tensor) and idx.device != self._dev:
            logits = logits.to(idx.device)
        loss = (torch.tensor(float(np.mean(losses)))
                if losses else None)
        return logits, loss, None

    # ── revert / audit ──────────────────────────────────────────────────

    def _undo_entry(self, e: tuple) -> bool:
        """Apply the inverse of one journal entry. Returns exactness."""
        op = e[0]
        if op == "t":
            # rewind stream head; ring bytes stay (harmless once head
            # moved) and invert the topic EMA exactly:
            # c_i = lam*c_{i-1} + (1-lam)*r_x  =>  c_{i-1} = (c_i -
            # (1-lam)*r_x)/lam
            _, tok, _tag = e
            lam = self.config.topic_decay
            if lam > 1e-9:
                self._c.sub_(self._R[tok].float() * (1.0 - lam)).div_(lam)
            # proto EMA inverse is approximate (A[tok] may have drifted)
            pd = self.config.proto_decay
            if pd > 1e-9:
                self._proto.sub_(self._A[tok] * (1.0 - pd)).div_(pd)
            self._count -= 1
            if self._gpu:
                self._pos_g -= 1
            return True
        if op == "b":
            self._ep_start = e[1]
            self._c.copy_(torch.from_numpy(
                e[2].astype(np.float32)).to(self._dev))
            if self._gpu:
                self._ep_g.fill_(e[1])
            return True
        if op == "u":
            if self._uni[e[1]] >= 1.0:
                self._uni[e[1]] -= 1.0
                self._total -= 1
                self._log_uni_num[e[1]] = math.log(
                    self._uni[e[1]] + self.config.uni_alpha)
                if self._gpu:
                    self._unig[e[1]] -= 1.0
                    self._tot_g -= 1.0
                return True
            return False
        if op == "c":
            _, k, key, tok, delta = e
            if self._gpu:
                ctx_t = torch.tensor([_s64(key)], dtype=torch.int64,
                                     device=self._dev)
                tok_t = torch.tensor([tok], dtype=torch.int64,
                                     device=self._dev)
                d_t = torch.tensor([delta], dtype=torch.float32,
                                   device=self._dev)
                return self._gt.subtract(k, ctx_t, tok_t, d_t) > 0
            cell = self._tables[k].get(key)
            if cell is None or tok not in cell:
                return False
            cell[tok] -= delta
            tots = self._totals[k]
            tots[key] = tots.get(key, 0.0) - delta
            if cell[tok] <= 0.0:
                del cell[tok]
                if not cell:
                    del self._tables[k][key]
                    tots.pop(key, None)
            return True
        if op == "eE":
            # restore an entry evicted by deque-overflow/FIFO-cap —
            # journaled BEFORE the displacing write, so on undo it is
            # re-inserted as the oldest row (appendleft)
            _, key, pos, succ, certs = e
            dq = self._epi.get(key)
            if dq is None:
                dq = self._epi[key] = deque(
                    maxlen=self.config.epi_max_per_key)
            if len(dq) < self.config.epi_max_per_key:
                self._epi_n += 1
            else:
                dq.pop()              # bound: drop newest to fit
            dq.appendleft((pos, succ, tuple(certs)))
            self._epi_fifo.appendleft((key, pos))
            self._epi_sync_row(key)     # GPU row tracks host dict
            return True
        if op == "e":
            _, key, pos = e
            if not self._epi_remove(key, pos):
                return False
            if self._gpu:
                pk = self._epi_hash(torch.tensor(
                    [_s64(key)], dtype=torch.int64, device=self._dev))
                se, fe = self._epi_probe(pk)
                if bool(fe[0]):
                    row = self._epi_pos[se[0]]
                    m = row == pos
                    if bool(m.any()):
                        col = int(m.float().argmax())
                        row[col] = -1
                        self._epi_succ[se[0], col] = -1
                        if col == int(
                                (self._epi_cnt[se[0]] - 1)
                                % self._epi_pos.shape[1]):
                            self._epi_cnt[se[0]] -= 1
            return True
        if op == "k":
            # knn bank row restore.  CPU entries carry the previous row
            # inline: ("k", slot, None|(pos, succ, fp16[d])); GPU
            # entries carry a slot into _kprev_g:
            # ("k", slot, prev_pos, prev_succ, prev_ring_slot|-1)
            slot = e[1]
            if len(e) == 3:
                prev = e[2]
                if prev is None:
                    self._knn_pos[slot] = -1
                    self._knn_succ[slot] = -1
                    self._knn_fp[slot] = 0
                else:
                    ppos, psucc, pfp = prev
                    self._knn_pos[slot] = ppos
                    self._knn_succ[slot] = psucc
                    self._knn_fp[slot] = torch.from_numpy(
                        np.asarray(pfp)).to(self._dev).to(
                            self._knn_fp.dtype)
            else:
                _, slot, ppos, psucc, rslot = e
                if rslot < 0:
                    self._knn_pos[slot] = -1
                    self._knn_succ[slot] = -1
                    self._knn_fp[slot] = 0
                else:
                    self._knn_pos[slot] = ppos
                    self._knn_succ[slot] = psucc
                    self._knn_fp[slot] = self._kprev_g[rslot]
            # every journaled bank write bumped the cursor once —
            # LIFO undo steps it back (exact for sequential writes)
            self._knn_cur = max(0, self._knn_cur - 1)
            if self._gpu:
                self._knncur_g.fill_(self._knn_cur)
            return True
        if op == "r":
            _, tok, prev = e
            if prev < 0:
                self._last_seen.pop(tok, None)
            else:
                self._last_seen[tok] = prev
            if self._gpu:
                self._seen_g[tok] = prev if prev >= 0 else -10 ** 9
            return True
        if op == "s":
            # csem row restore: (pk_or_key, slot_or_-1, prev_row|None)
            _, key_or_pk, slot, prev = e
            if self._gpu:
                if prev is None:
                    self._gt.tkeys[self._gt.row_csem, slot] = 0
                    self._gt.traw[self._gt.row_csem, slot] = 0
                    self._csem_rows[slot].zero_()
                    self._gt.tocc[self._gt.row_csem, slot] = 0
                    self._gt.used[self._gt.row_csem] -= 1
                else:
                    self._csem_rows[slot] = torch.from_numpy(
                        np.asarray(prev, dtype=np.float16)).to(
                            self._dev)
            else:
                if prev is None:
                    self._csem_cpu.pop(key_or_pk, None)
                else:
                    self._csem_cpu[key_or_pk] = np.asarray(
                        prev, dtype=np.float32)
            return True
        if op == "bc":
            self._undo_block(e[1])
            return True
        if op == "sg":
            # gpu csem write: (slot, jcur_ring_slot_or_-1)
            _, slot, rslot = e
            if rslot < 0:
                self._gt.tkeys[self._gt.row_csem, slot] = 0
                self._gt.traw[self._gt.row_csem, slot] = 0
                self._csem_rows[slot].zero_()
                self._gt.tocc[self._gt.row_csem, slot] = 0
                self._gt.used[self._gt.row_csem] -= 1
            else:
                self._csem_rows[slot] = self._csem_prev[rslot]
            return True
        if op == "a":
            _, tok, slot = e
            if slot < 0:
                return False            # vec delta already recycled
            if self._gpu:
                dv = self._avec_g[slot]
                self._A[tok].sub_(dv)
                self._Ac[tok].sub_(dv @ self._Pcsem)
                self._An[tok] = self._A[tok].norm()
                self._Acn[tok] = self._Ac[tok].norm()
            else:
                self._A[tok].sub_(torch.from_numpy(
                    self.journal.vec_ring[slot].astype(np.float32)
                ).to(self._dev))
            return True
        if op == "h":
            slot = e[1]
            if slot < 0:
                return False            # w snapshot already recycled
            if self._gpu:
                self._wg.copy_(self._w_ring_g[slot, :-1])
                self._hmg.fill_(self._w_ring_g[slot, -1])
                self._w = self._wg.cpu().numpy().astype(np.float64)
                self._hedge_mix = float(self._hmg)
            else:
                self._w[:] = self.journal.w_ring[slot, :-1]
                self._hedge_mix = float(
                    self.journal.w_ring[slot, -1])
            return True
        if op in ("ws", "we"):
            return True                 # structural markers — no state
        if op == "cv":
            # restore a follower column displaced by evict-min write
            _, flat, col, tok, cnt = e
            if not self._gpu:
                return False
            oi = int(flat) // self._gt.S
            s = int(flat) % self._gt.S
            if int(self._gt.tkeys[oi, s]) == 0:
                return False        # host row was fully cleared
            self._gt.ttoks[oi, s, col] = int(tok)
            self._gt.tcnt[oi, s, col] = float(cnt)
            self._gt.ttot[oi, s] += float(cnt)
            self._gt.tocc[oi, s] = float(self._gt.ttot[oi, s])
            return True
        if op == "ev":
            # restore an order-row evicted by a fused-step insert
            _, flat, raw, tot, toks, cnts = e
            if not self._gpu:
                return False
            return self._gt.restore_row(
                flat, raw,
                torch.from_numpy(
                    np.asarray(toks, dtype=np.int32)).to(self._dev),
                torch.from_numpy(
                    np.asarray(cnts, dtype=np.float32)).to(self._dev),
                tot)
        if op == "eve":
            # restore an evicted episodic row (pos + succ + certs)
            _, flat, raw, cnt, positions = e[:5]
            if not self._gpu:
                return False
            oi = self._gt.row_epi
            pk = self._epi_hash(torch.tensor(
                [_s64(raw)], dtype=torch.int64, device=self._dev))
            s, exact = self._gt._pick_slot(oi, pk)
            if int(self._gt.tkeys[oi, s]) == 0:
                self._gt.used[oi] += 1
            self._gt.tkeys[oi, s] = pk[0]
            self._gt.traw[oi, s] = _s64(raw)
            self._epi_pos[s] = torch.from_numpy(
                np.asarray(positions, dtype=np.int64)).to(self._dev)
            if len(e) > 5:
                self._epi_succ[s] = torch.from_numpy(
                    np.asarray(e[5], dtype=np.int32)).to(self._dev)
            if len(e) > 6:
                self._epi_cert[s] = torch.from_numpy(
                    np.asarray(e[6]).reshape(
                        self._epi_cert.shape[1:])) .to(self._dev)
            self._epi_cnt[s] = cnt
            self._gt.tocc[oi, s] = float(cnt)
            return exact
        if op == "evc":
            # restore an evicted csem row
            _, flat, raw, oc, vec = e
            if not self._gpu:
                return False
            oi = self._gt.row_csem
            pk = self._csem_hash(torch.tensor(
                [_s64(raw)], dtype=torch.int64, device=self._dev))
            s, exact = self._gt._pick_slot(oi, pk)
            if int(self._gt.tkeys[oi, s]) == 0:
                self._gt.used[oi] += 1
            self._gt.tkeys[oi, s] = pk[0]
            self._gt.traw[oi, s] = _s64(raw)
            self._csem_rows[s] = torch.from_numpy(
                np.asarray(vec, dtype=np.float16)).to(self._dev)
            self._gt.tocc[oi, s] = oc
            return exact
        if op == "g":
            # graft provenance marker — init-time state change, not a
            # journaled write; nothing to undo
            return True
        return False

    def revert_since(self, n_entries: int | None = None,
                     seq_pos: int | None = None) -> dict:
        """Undo journaled writes.

        ``n_entries``: revert at least N journal entries, rounded up to
            a token boundary (a token's writes stay atomic).
        ``seq_pos``: revert until the stream length equals this position
            (i.e. un-learn the last ``count - seq_pos`` tokens).
        Returns a report {reverted, inexact}.
        """
        entries = self.journal.entries
        reverted = inexact = popped = 0
        # Boundaries: "t"/"bc" (token writes), "b" (episode reset), and
        # "ws"/"we"/"g" (position-free write-block markers).  Write-block
        # markers count as boundaries so a seq_pos revert can never
        # descend into a teach/consolidate block below the floor — those
        # ops belong to revert_writes()/revert_tag(), not a position
        # floor.
        while entries:
            at_boundary = entries[-1][0] in ("t", "b", "bc",
                                             "ws", "we", "g")
            if seq_pos is not None and self._count <= seq_pos \
                    and at_boundary:
                break
            if n_entries is not None and popped >= n_entries \
                    and at_boundary:
                break
            if seq_pos is None and n_entries is None:
                break
            e = entries.pop()
            popped += 1
            if self._undo_entry(e):
                reverted += 1
            else:
                inexact += 1
        return {"reverted": reverted, "inexact": inexact,
                "stream_len": self._count}

    def revert_writes(self, tag: str) -> dict:
        """Undo every ``write_counts`` block journaled under ``tag``
        (distill/manual injections — position-independent, no stream
        rewind needed)."""
        entries = self.journal.entries
        reverted = inexact = 0
        i = len(entries) - 1
        while i >= 0:
            e = entries[i]
            if e[0] == "we" and e[1] == tag:
                # undo back to the matching "ws"
                j = i - 1
                depth = 1
                while j >= 0:
                    op = entries[j][0]
                    if op == "we":
                        depth += 1
                    elif op == "ws":
                        depth -= 1
                        if depth == 0:
                            break
                    else:
                        ok = self._undo_entry(entries[j])
                        reverted += ok
                        inexact += not ok
                    j -= 1
                del entries[j:i + 1]          # drop the whole block
                i = j - 1
            else:
                i -= 1
        return {"reverted": reverted, "inexact": inexact,
                "stream_len": self._count}

    def revert_tag(self, tag: str) -> dict:
        """Un-learn every token journaled under ``tag``, plus any
        ``write_counts`` blocks carrying it.

        Token revert is suffix-based for causal consistency: the stream
        is rewound to just before the earliest journaled token carrying
        ``tag``, which also un-learns later tokens.  Use ``snapshot()``
        first if you need finer-grained rollback.
        """
        rep_w = self.revert_writes(tag)
        # walk token markers backward, tracking a virtual stream length:
        # a "bc" block covers positions blk.count..blk.count+n-1, so the
        # virtual head jumps to blk.count when we pass it
        target = None
        v = self._count
        for e in reversed(self.journal.entries):
            if e[0] == "bc":
                blk = e[1]
                if blk["tag"] == tag:
                    target = blk["count"] if target is None \
                        else min(target, blk["count"])
                v = blk["count"]
            elif e[0] == "t":
                v -= 1
                if e[2] == tag:
                    target = v if target is None else min(target, v)
        if target is None:
            rep_w["found"] = rep_w["reverted"] > 0
            rep_w["tag"] = tag
            return rep_w
        rep = self.revert_since(seq_pos=target)
        rep["reverted"] += rep_w["reverted"]
        rep["inexact"] += rep_w["inexact"]
        rep["found"] = True
        rep["tag"] = tag
        return rep

    def audit(self, n: int = 50, tokenizer=None) -> list[dict]:
        """Decode the last ``n`` learned tokens into readable records:
        what token, under which tag, and the text context tail."""
        out = []
        tail_tokens: deque = deque(maxlen=8)
        for e in self.journal.entries[-(n * 16):]:
            if e[0] == "t":
                rec = {"pos": None, "tok": e[1], "tag": e[2]}
                if tokenizer is not None:
                    rec["token"] = tokenizer.decode([e[1]])
                    rec["context"] = tokenizer.decode(list(tail_tokens))
                tail_tokens.append(e[1])
                out.append(rec)
        return out[-n:]

    def writes_since(self, n_entries: int) -> dict:
        """Summary of the last N journal writes, grouped by op+tag."""
        from collections import Counter
        ops: Counter = Counter()
        tags: Counter = Counter()
        for e in self.journal.entries[-n_entries:]:
            ops[e[0]] += 1
            if e[0] == "t":
                tags[e[2]] += 1
        return {"ops": dict(ops), "tags": dict(tags)}

    # ── persistence ─────────────────────────────────────────────────────

    def snapshot(self, path: str | Path) -> Path:
        """Full memory checkpoint (tables + rings + journal metadata)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._snapshot_state(), path)
        return path

    def clone(self) -> "FluxLM":
        """In-memory deep copy of the full learned state — tables, rings,
        episodic index, knn bank, journal, hedge weights, graft.  The
        clone shares no state with the source: use it for parallel
        exploration (batch generation workers) while the canonical
        model's stream stays untouched."""
        import io
        buf = io.BytesIO()
        torch.save(self._snapshot_state(), buf)
        buf.seek(0)
        return FluxLM.load(buf)

    def _snapshot_state(self) -> dict:
        """The serializable state dict behind snapshot() and clone()."""
        state = {
            "config": asdict(self.config),
            "w": self._w,
            "hedge_mix": self._hedge_mix,
            "channels": list(self.channels),
            "uni": self._uni,
            "log_uni_num": self._log_uni_num.cpu(),
            "total": self._total,
            "tables": self._tables if not self._gpu else {},
            "totals": self._totals if not self._gpu else {},
            # GPU backend: portable decoded rows {k: (keys, toks, cnts)}
            "table_triplets": (self._gt.triplets() if self._gpu
                               else None),
            "csem_cpu": self._csem_cpu if not self._gpu else None,
            "csem_rows_gpu": ((self._gt.tkeys[self._gt.row_csem].cpu(),
                               self._csem_rows.cpu())
                              if self._gpu else None),
            "epi": {k: list(v) for k, v in self._epi.items()},
            "epi_n": self._epi_n,
            "epi_fifo": list(self._epi_fifo),
            # knn associative bank: fp16 fingerprints + successors —
            # the approximate-match channel's entire state
            "knn": (self._knn_fp.cpu(), self._knn_succ.cpu(),
                    self._knn_pos.cpu(), self._knn_cur)
                    if self.config.use_knn else None,
            "ring": self._ring,
            "H": self._H,
            "count": self._count,
            "ep_start": self._ep_start,
            "A": self._A.cpu(),
            "c": self._c.cpu(),
            "proto": self._proto.cpu(),
            # teacher graft: the projected embedding matrix itself —
            # `_R` is otherwise re-seeded from config, which would
            # silently revert the graft on load
            "R_graft": ((self._R.cpu(), dict(self._graft))
                        if self._graft else None),
            "last_seen": self._last_seen,
            "recent": list(self._recent),
            "journal_entries": self.journal.entries,
            "journal_dropped": self.journal.dropped,
            "gaps": list(self._gaps),
        }
        return state

    def to_device(self, device: str) -> None:
        """Move the dense readout tensors (A/R/c/proto/log_uni_num) to a
        device.  Sparse memory stays host-resident either way."""
        dev = torch.device(
            device if (device == "cpu" or torch.cuda.is_available())
            else "cpu")
        self._dev = dev
        self.config.device = device
        for name in ("_R", "_A", "_c", "_proto", "_log_uni_num"):
            setattr(self, name, getattr(self, name).to(dev))

    @classmethod
    def load(cls, path: str | Path, device: str | None = None,
             cuda_primary: bool | None = None) -> "FluxLM":
        state = torch.load(path, map_location="cpu", weights_only=False)
        cfg = FluxConfig(**state["config"])
        if device is not None:
            cfg.device = device
        if cuda_primary is not None:
            cfg.cuda_primary = cuda_primary
        # the grafted matrix is stored in the snapshot — blank the path
        # so __init__ doesn't re-read the teacher file (or fail if it's
        # gone); R_graft below restores the actual tensor
        cfg.teacher_embed = ""
        model = cls(cfg)
        rg = state.get("R_graft")
        if rg is not None:
            model._R = rg[0].float().to(model._dev)
            model._graft = dict(rg[1])
        # remap channel weights by name — survives channel list changes
        saved_w = np.asarray(state["w"], dtype=np.float64)
        saved_ch = state.get("channels")
        if saved_ch and len(saved_ch) == len(saved_w):
            lut = dict(zip(saved_ch, saved_w))
            model._w = np.array([lut.get(ch, 1.0) for ch in model.channels])
            model._w /= model._w.sum()
        elif len(saved_w) == len(model.channels):
            model._w = saved_w
        model._hedge_mix = float(state.get("hedge_mix", 0.0))
        if model._gpu:
            model._hmg.fill_(model._hedge_mix)
        model._uni = state["uni"]
        model._total = state["total"]
        model._log_uni_num = state.get(
            "log_uni_num",
            np.log(state["uni"] + model.config.uni_alpha))
        if not isinstance(model._log_uni_num, torch.Tensor):
            model._log_uni_num = torch.from_numpy(
                np.asarray(model._log_uni_num, dtype=np.float32))
        model._log_uni_num = model._log_uni_num.to(model._dev)
        # GPU snapshots store tables={} (state lives in table_triplets)
        # — keep __init__'s per-order skeleton so the triplet rebuild
        # below has its dicts to write into
        saved_tables = {int(k): v for k, v in state["tables"].items()}
        if saved_tables:
            model._tables = saved_tables
        model._totals = state.get("totals") or {
            k: {key: sum(cell.values()) for key, cell in t.items()}
            for k, t in model._tables.items()}
        # GPU backend: prefer decoded triplets; fall back to converting
        # CPU dicts (recompute pair keys + rebuild tot slots)
        if model._gpu:
            tri = state.get("table_triplets")
            if tri:
                model._gt.load_triplets(tri)
            elif model._tables:
                for k, cells in model._tables.items():
                    kk, tt, cc = [], [], []
                    for key, cell in cells.items():
                        for tok, cnt in cell.items():
                            kk.append(key)
                            tt.append(tok)
                            cc.append(cnt)
                    # no sentinel rows — add() accumulates ttot/tocc
                    # from the follower deltas itself; a tok==V cell
                    # would be read as a real follower and index OOB
                    if kk:
                        model._gt.load_triplets(
                            {k: (np.asarray(kk, dtype=np.uint64),
                                 np.asarray(tt, dtype=np.int64),
                                 np.asarray(cc, dtype=np.float32))})
                model._tables = {k: {} for k in model._tables}
                model._totals = {k: {} for k in model._totals}
            csem_g = state.get("csem_rows_gpu")
            if csem_g is not None:
                keys_g, rows_g = csem_g
                rc = model._gt.row_csem
                n_g = min(keys_g.numel(), model._gt.S)
                model._gt.tkeys[rc, :n_g] = keys_g[:n_g].to(
                    model._dev)
                model._csem_rows[:n_g] = rows_g[:n_g].to(
                    model._dev)
                model._gt.used[rc] = int((keys_g != 0).sum())
        else:
            csem_cpu = state.get("csem_cpu")
            if csem_cpu:
                model._csem_cpu = {int(k): np.asarray(v,
                                                     dtype=np.float32)
                                   for k, v in csem_cpu.items()}
            tri = state.get("table_triplets")
            if tri:
                # rebuild CPU dicts from decoded rows (drop tot rows)
                for k, (kk, tt, cc) in tri.items():
                    k = int(k)
                    for key, tok, cnt in zip(kk.tolist(), tt.tolist(),
                                             cc.tolist()):
                        if tok == model.vocab_size:
                            continue
                        cell = model._tables[k].setdefault(key, {})
                        cell[tok] = cell.get(tok, 0.0) + float(cnt)
                model._totals = {
                    k: {key: sum(cell.values())
                        for key, cell in t.items()}
                    for k, t in model._tables.items()}
        model._epi = {int(k): deque(v, maxlen=cfg.epi_max_per_key)
                      for k, v in state["epi"].items()}
        # FIFO + live-count rebuild: prefer saved values; legacy
        # snapshots fall back to re-deriving from the dict (positions
        # sorted so eviction priority approximates insert order)
        model._epi_n = int(state.get("epi_n") or
                           sum(len(d) for d in model._epi.values()))
        fifo = state.get("epi_fifo")
        if fifo is None:
            fifo = sorted(
                ((k, (e[0] if isinstance(e, (tuple, list)) else e))
                 for k, dq in model._epi.items() for e in dq),
                key=lambda kp: kp[1])
        model._epi_fifo = deque(fifo)
        # knn bank (absent in legacy snapshots -> empty bank)
        knn_st = state.get("knn")
        if knn_st is not None and model.config.use_knn:
            kfp, ksucc, kpos, kcur = knn_st
            P = model.config.knn_capacity
            n_r = min(kfp.shape[0], P)
            model._knn_fp[:n_r] = kfp[:n_r].to(
                model._dev).to(model._knn_fp.dtype)
            model._knn_succ[:n_r] = ksucc[:n_r].to(model._dev)
            model._knn_pos[:n_r] = kpos[:n_r].to(model._dev)
            model._knn_cur = min(int(kcur), 1 << 62)
            if model._gpu:
                model._knncur_g.fill_(model._knn_cur)
        model._ring = state["ring"]
        model._H = state["H"]
        model._count = state["count"]
        model._ep_start = state["ep_start"]
        for name, key in (("_A", "A"), ("_c", "c"), ("_proto", "proto")):
            v = state.get(key)
            if v is None:
                v = torch.zeros((cfg.vocab_size, cfg.topic_dim)
                                if key == "A" else (cfg.topic_dim,))
            if not isinstance(v, torch.Tensor):
                v = torch.from_numpy(np.asarray(v))
            setattr(model, name, v.float().to(model._dev))
        model._last_seen = state["last_seen"]
        model._recent = deque(state.get("recent", []))
        if model._gpu:
            # rebuild device mirrors from the restored host state
            model._ringg = torch.from_numpy(
                model._ring.astype(np.int32)).to(model._dev)
            model._Hg = torch.from_numpy(
                model._H.view(np.int64)).to(model._dev)
            model._unig = torch.from_numpy(
                model._uni.astype(np.float32)).to(model._dev)
            model._seen_g.fill_(-10 ** 9)
            for t, p in model._last_seen.items():
                model._seen_g[t] = p
            model._pos_g.fill_(model._count)
            model._ep_g.fill_(model._ep_start)
            model._tot_g.fill_(float(model._total))
            model._wg = torch.from_numpy(
                model._w.astype(np.float64)).to(model._dev)
            # derived A projections/norms — recompute once (the same
            # rank-1 updates that build them live in every write path)
            model._Ac = (model._A @ model._Pcsem).float()
            model._An = model._A.norm(dim=1)
            model._Acn = model._Ac.norm(dim=1)
            # rebuild GPU epi index from the host dict — entries are
            # (pos, succ, certs) tuples; legacy bare-pos snapshots get
            # succ=-1/empty certs
            ks_all, ps_all, ss_all, cs_all = [], [], [], []
            LC = len(cfg.cert_orders)
            for key, dq in model._epi.items():
                for ent in dq:
                    if isinstance(ent, (tuple, list)):
                        p, s_, cv = ent[0], ent[1], \
                            (ent[2] if len(ent) > 2 else ())
                    else:                     # legacy bare position
                        p, s_, cv = ent, -1, ()
                    ks_all.append(key)
                    ps_all.append(p)
                    ss_all.append(s_)
                    cs_all.append(
                        [int(v) if int(v) < (1 << 63)
                         else int(v) - (1 << 64)
                         for v in cv] + [0] * (LC - len(cv)))
            if ks_all:
                kk = torch.tensor(
                    [_s64(k) for k in ks_all], dtype=torch.int64,
                    device=model._dev)
                pp = torch.tensor(ps_all, dtype=torch.int64,
                                  device=model._dev)
                ss = torch.tensor(ss_all, dtype=torch.int32,
                                  device=model._dev)
                cc_ = (torch.tensor(cs_all, dtype=torch.int64,
                                    device=model._dev)
                       if LC else None)
                re_ = model._gt.row_epi
                pk_e = model._epi_hash(kk)
                ekr = model._gt.tkeys[re_]
                err = model._gt.traw[re_]
                se, fe, _ee = model._gt._probe_row_w(re_, pk_e)
                enew = ~fe
                if bool(enew.any()):
                    ekr[se[enew]] = pk_e[enew]
                    err[se[enew]] = kk[enew]
                    model._gt.used[re_] = int(enew.sum())
                se2, _ = model._epi_probe(pk_e)
                ER = model._epi_pos.shape[1]
                # distinct column per position of the same key —
                # occurrence rank within its (duplicate) slot group
                se2_l = se2.tolist()
                rank: dict[int, int] = {}
                cols = []
                for s_ in se2_l:
                    r = rank.get(s_, 0)
                    cols.append((int(model._epi_cnt[s_]) + r) % ER)
                    rank[s_] = r + 1
                col = torch.tensor(cols, dtype=torch.int64,
                                   device=model._dev)
                model._epi_pos[se2, col] = pp
                model._epi_succ[se2, col] = ss
                if LC:
                    model._epi_cert[se2, col] = cc_
                for s_, r in rank.items():
                    model._epi_cnt[s_] += r
                model._gt.tocc[re_][se2] = \
                    model._epi_cnt[se2].float()
        model.journal.entries = list(state.get("journal_entries", []))
        model.journal.dropped = state.get("journal_dropped", 0)
        model._gaps = deque(state.get("gaps", []), maxlen=cfg.gap_cap)
        return model

    # ── diagnostics ─────────────────────────────────────────────────────

    def memory_report(self) -> dict:
        import sys
        cells = sum(len(t) for t in self._tables.values())
        followers = sum(len(cell) for t in self._tables.values()
                        for cell in t.values())
        epi_pos = sum(len(d) for d in self._epi.values())
        if self._gpu:
            c = self.config
            S, M, O = c.gpu_table_slots, c.gpu_row_width, len(c.orders)
            tab_mb = ((O + 2) * S * (8 + 8) + O * S * 4
                      + O * S * M * 8) / 1e6
            csem_mb = (S * 2 * c.csem_dim) / 1e6
            epi_mb = (S * (8 + 8 * c.epi_row)) / 1e6
            ring_mb = (c.vec_delta_cap * (8 * len(self.channels)
                       + 4 * c.topic_dim + 2 * c.csem_dim)) / 1e6
            cells = sum(self._gt.used)
            followers = cells
            approx = {
                "topic_A_MB": self._A.element_size()
                * self._A.nelement() / 1e6,
                "topic_R_MB": self._R.element_size()
                * self._R.nelement() / 1e6,
                "ring_MB": (self._ring.nbytes + self._H.nbytes) / 1e6,
                "gpu_tables_MB": tab_mb,
                "gpu_rows_used": cells,
                "csem_MB": csem_mb,
                "epi_gpu_MB": epi_mb,
                "journal_rings_gpu_MB": ring_mb,
                "epi_keys": len(self._epi),
                "epi_positions": epi_pos,
                "epi_cap": self.config.epi_total_cap,
                "epi_MB_est": (len(self._epi) * 96 + epi_pos * 48) / 1e6,
                "knn_rows": int((self._knn_pos >= 0).sum()),
                "knn_cap": self.config.knn_capacity,
                "knn_MB": (self._knn_fp.nbytes + self._knn_succ.nbytes
                           + self._knn_pos.nbytes) / 1e6,
                "journal_entries": len(self.journal),
                "journal_MB_est": (len(self.journal.entries) * 160
                                   + self.journal.vec_ring.nbytes
                                   + self.journal.w_ring.nbytes) / 1e6,
                "py_overhead_MB": sys.getsizeof(self._tables) / 1e6,
            }
            approx["total_MB_est"] = sum(
                v for k, v in approx.items()
                if k.endswith("_MB") or k.endswith("_MB_est"))
            approx["vram_MB_est"] = (tab_mb + csem_mb + epi_mb
                                     + ring_mb + approx["topic_A_MB"]
                                     + approx["topic_R_MB"]
                                     + approx["knn_MB"] + 20)
            return approx
        approx = {
            "topic_A_MB": self._A.element_size() * self._A.nelement() / 1e6,
            "topic_R_MB": self._R.element_size() * self._R.nelement() / 1e6,
            "ring_MB": (self._ring.nbytes + self._H.nbytes) / 1e6,
            "tables_cells": cells,
            "tables_followers": followers,
            "tables_MB_est": (cells * 96 + followers * 56) / 1e6,
            "epi_keys": len(self._epi),
            "epi_positions": epi_pos,
            "epi_cap": self.config.epi_total_cap,
            "epi_MB_est": (len(self._epi) * 96 + epi_pos * 48) / 1e6,
            "knn_rows": int((self._knn_pos >= 0).sum()),
            "knn_cap": self.config.knn_capacity,
            "knn_MB": (self._knn_fp.nbytes + self._knn_succ.nbytes
                       + self._knn_pos.nbytes) / 1e6,
            "journal_entries": len(self.journal),
            "journal_MB_est": (len(self.journal.entries) * 160
                               + self.journal.vec_ring.nbytes) / 1e6,
            "py_overhead_MB": sys.getsizeof(self._tables) / 1e6,
        }
        approx["total_MB_est"] = sum(v for k, v in approx.items()
                                     if k.endswith("_MB") or k.endswith("_MB_est"))
        return approx

    def channel_weights(self) -> dict:
        return {ch: float(self._w[i])
                for i, ch in enumerate(self.channels)}

    def extra_repr(self) -> str:
        return (f"flux(vocab={self.vocab_size}, cells="
                f"{sum(len(t) for t in self._tables.values())}, "
                f"stream={self._count}, journal={len(self.journal)})")
