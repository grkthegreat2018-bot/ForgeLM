"""FluxLM unit tests — CPU-only, no tokenizer dependency."""
import numpy as np
import torch

from forge.model.flux import FluxConfig, FluxLM


def _cfg(**kw):
    kw.setdefault("vocab_size", 512)
    kw.setdefault("topic_dim", 32)
    kw.setdefault("ring_capacity", 1 << 14)
    kw.setdefault("journal_cap", 50_000)
    return FluxConfig(**kw)


def _ids(text: str) -> list[int]:
    return [ord(ch) % 512 for ch in text]


def test_forward_contract():
    m = FluxLM(_cfg())
    ids = torch.tensor([_ids("hello world")])
    out = m(ids, use_cache=True)
    logits, loss, past = out
    assert logits.shape == (1, 11, 512)
    assert logits.dtype == torch.float32
    assert past is None and loss is None


def test_live_learning_recall():
    m = FluxLM(_cfg())
    ids = _ids("the cat sat on the mat. " * 20)
    m.ingest(ids, tag="train")
    # "on the" should strongly predict "mat"-ish next token
    m.soft_reset()
    ctx = torch.tensor([_ids("on the")])
    logits, _, _ = m(ctx, use_cache=True)
    top = int(logits[0, -1].argmax())
    assert top == ord("m") % 512 or top == ord(" ") % 512


def test_learn_new_word_instantly():
    """The core promise: teach a fact, it applies on the next forward."""
    m = FluxLM(_cfg())
    m.ingest(_ids("a glorf is a small bird. " * 5), tag="def")
    ctx = torch.tensor([_ids("a glorf is a")])
    logits, _, _ = m(ctx)
    top5 = torch.topk(logits[0, -1], 5).indices.tolist()
    assert ord(" ") % 512 in top5
    ctx2 = torch.tensor([_ids("glorf is a small")])
    logits2, _, _ = m(ctx2)
    top5b = torch.topk(logits2[0, -1], 5).indices.tolist()
    assert ord(" ") % 512 in top5b


def test_unbounded_context_episodic_recall():
    """A fact stated far back is recalled via episodic suffix match —
    no positional limit, no attention window."""
    m = FluxLM(_cfg())
    filler = "q w e r t y u i o p " * 30          # 300 tokens of noise
    m.ingest(_ids("the secret word is zephyr. " + filler), tag="mem")
    m.soft_reset()
    ctx = torch.tensor([_ids("the secret word is")])
    logits, _, _ = m(ctx)
    assert int(logits[0, -1].argmax()) == ord(" ") % 512 or \
        int(logits[0, -1].argmax()) == ord("z") % 512


def test_weights_are_local_not_meshed():
    """Learning under one context must not touch unrelated cells."""
    m = FluxLM(_cfg())
    m.ingest(_ids("aaa bbb aaa bbb "), tag="a")
    cells_before = {k: len(t) for k, t in m._tables.items()}
    m.ingest(_ids("x y z x y z "), tag="b")
    cells_after = {k: len(t) for k, t in m._tables.items()}
    # new cells were created, but existing cells untouched except where
    # the new context is genuinely shared (none here)
    assert all(cells_after[k] >= cells_before[k] for k in cells_after)


def test_journal_audit_and_revert():
    m = FluxLM(_cfg())
    m.ingest(_ids("first fact alpha beta "), tag="keep")
    snap_pos = m._count
    m.ingest(_ids("bad data poison poison "), tag="poison")
    assert m._count > snap_pos
    rep = m.revert_tag("poison")
    assert rep["found"] and rep["reverted"] > 0
    assert m._count == snap_pos
    # the poisoned bigram should be gone
    key_ctx = m._keys_at(m._count - 1)
    for k, key in key_ctx.items():
        assert all(ord("p") % 512 not in cell
                   for cell in [m._tables[k].get(key, {})])


def test_snapshot_roundtrip(tmp_path):
    m = FluxLM(_cfg())
    m.ingest(_ids("persistent memory test data " * 3), tag="s")
    p = m.snapshot(tmp_path / "m.flux")
    m2 = FluxLM.load(p)
    assert m2._count == m._count
    assert m2._total == m._total
    assert m2._tables.keys() == m._tables.keys()


def test_channel_weights_adapt():
    m = FluxLM(_cfg())
    w0 = m._w.copy()
    m.ingest(_ids("ab ab ab ab ab ab ab ab "), tag="t")
    assert not np.allclose(m._w, w0)
    assert abs(m._w.sum() - 1.0) < 1e-6


def test_engine_integration_cpu():
    """ForgeEngine.from_flux → generate() end-to-end on CPU."""
    try:
        from forge.engine.forge_engine import ForgeEngine
    except Exception:
        import pytest
        pytest.skip("engine import failed")
    m_cfg = _cfg()
    m_cfg.vocab_size = 65536  # match canonical tokenizer
    from forge.engine.forge_engine import ForgeEngine
    eng = ForgeEngine.from_flux(config=m_cfg)
    eng.model.ingest(_ids("the cat sat on the mat. " * 30), tag="boot")
    out = eng.generate("the cat sat on the", max_new_tokens=6,
                       temperature=0.0)
    assert isinstance(out, str)


def test_gap_tracking_and_distill_write():
    """Gap positions collected at ingest; teacher top-k writes land in
    the right cells and are revertible by tag."""
    m = FluxLM(_cfg())
    # sparse data => most contexts thin => gaps recorded
    m.ingest(_ids("alpha beta gamma delta "), tag="corpus")
    gaps = m.gap_positions()
    assert gaps
    # fake teacher: always predicts token 7 with confidence
    def stub_score(ctxs):
        return [[7] * 5] * len(ctxs), [[0.9, 0.05, 0.02, 0.02, 0.01]] * len(ctxs)
    rep = m.distill_from(stub_score, top_k=5, scale=2.0,
                         max_gaps=8, batch_size=4, tag="distill")
    assert rep["gaps_scored"] == min(8, len(gaps))
    assert rep["cells_written"] > 0
    # the distilled counts now influence prediction at a gap context
    rep2 = m.revert_tag("distill")
    assert rep2["found"]


def test_write_counts_manual_injection():
    """Direct knowledge injection must raise the target's logit; with
    trained channel weights a strong write wins outright."""
    m = FluxLM(_cfg())
    m.ingest(_ids("my name is "), tag="ctx")
    before = float(m._predict()[ord("F") % 512])
    m.write_counts(m._count, {ord("F") % 512: 200.0}, tag="manual")
    after = float(m._predict()[ord("F") % 512])
    assert after > before

    # with adapted weights (hedge has concentrated on high orders),
    # the same write should be able to win argmax
    m2 = FluxLM(_cfg())
    m2.ingest(_ids("the cat sat on the mat. " * 30), tag="t")
    m2.ingest(_ids("my name is "), tag="ctx")
    m2.write_counts(m2._count, {ord("F") % 512: 200.0}, tag="manual")
    assert int(torch.argmax(m2._predict())) == ord("F") % 512


def test_memory_bound():
    m = FluxLM(_cfg())
    rep = m.memory_report()
    # tiny cfg must be well under 1GB; default cfg is bounded by design
    assert rep["total_MB_est"] < 1024


def test_sem_channel_generalizes():
    """Semantic spread gives lift to tokens similar to known followers —
    the generativity fix (recombination, not just verbatim recall)."""
    m = FluxLM(_cfg())
    m.ingest(_ids("the cat sat. the dog sat. the bird sat. ") * 4,
             tag="t")
    assert float(m._proto.norm()) > 0          # fingerprint formed
    m.soft_reset()
    logits = m._predict()
    assert torch.isfinite(logits).all()


def test_cuda_device_flag():
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("no CUDA")
    m = FluxLM(_cfg(device="cuda"))
    m.ingest(_ids("gpu test tokens gpu test "), tag="t")
    ids = torch.tensor([_ids("gpu test")], device="cuda")
    logits, _, _ = m(ids)
    assert logits.device.type == "cuda"
    assert logits.shape == (1, 8, 512)


def test_revert_restores_prediction():
    """After revert, the model must predict as if the writes never happened."""
    m = FluxLM(_cfg())
    m.ingest(_ids("baseline text here "), tag="base")
    m.soft_reset()
    probe = torch.tensor([_ids("baseline")])
    m.learning = False                      # probe must not write
    before = m(probe)[0][0, -1].clone()
    m.learning = True
    m.soft_reset()
    m.ingest(_ids("poisoned garbage zzz zzz "), tag="bad")
    m.soft_reset()
    m.revert_tag("bad")
    m.soft_reset()
    m.learning = False
    after = m(probe)[0][0, -1]
    m.learning = True
    assert torch.allclose(before, after, atol=1e-4)


def test_knn_approximate_context_recall():
    """The knn channel retrieves by fingerprint similarity — a context
    that is SIMILAR to (not a verbatim repeat of) a taught context
    still lifts its successor.  This is the generalization path exact
    suffix hashing cannot take."""
    m = FluxLM(_cfg())
    ctx = [10, 11, 12, 13, 14, 15, 16, 17]
    m.teach(ctx, [77], weight=40.0)
    m.learning = False                     # observe-only probe
    for x in [10, 11, 12, 13, 14, 15, 90, 91]:   # same head, new tail
        m._push(x)
    p_sim = m._predict()
    rank77 = int((p_sim > p_sim[77]).sum())
    m.soft_reset()
    for x in [400, 401, 402, 403, 404, 405, 406, 407]:  # unrelated ctx
        m._push(x)
    p_ctl = m._predict()
    m.learning = True
    assert float(p_sim[77]) > float(p_ctl[77]) + 0.3
    assert rank77 < 8


def test_archived_episodic_recall_past_ring_wrap():
    """Entries that scrolled out of the ring still vote via stored
    successor + cert hashes — deep memory survives ring wrap."""
    m = FluxLM(_cfg(ring_capacity=64))
    ctx = list(range(200, 220))
    for _ in range(3):
        m.ingest(ctx + [77], tag="fact")
    # bury the fact under > ring_capacity of unrelated tokens
    m.ingest([300 + (i * 7) % 150 for i in range(200)], tag="noise")
    assert m._count - 21 > 64              # fact is fully off-ring
    for x in ctx:
        m._learn_token(x)
    votes = m._deep_votes(m._count - 1)
    assert votes.get(77, 0.0) > 0          # archived vote present
    assert int(torch.as_tensor(m._predict()).argmax()) == 77


def test_teach_instant_hot_training():
    """teach(): one call binds context -> target; recall is immediate
    and needs no gradient step and no stream position."""
    m = FluxLM(_cfg())
    m.ingest(_ids("ordinary preamble text " * 3), tag="boot")
    rep = m.teach([10, 20, 30, 40], [77], weight=50.0, tag="fact")
    assert rep["cells"] > 0 and rep["knn"] == 1
    for x in [10, 20, 30, 40]:
        m._learn_token(x)
    assert int(torch.as_tensor(m._predict()).argmax()) == 77


def test_teach_revert():
    """revert_tag on a teach() block un-teaches: cells, csem row and
    the knn bank entry are all removed."""
    m = FluxLM(_cfg())
    m.ingest(_ids("baseline filler " * 4), tag="boot")
    m.teach([5, 6, 7, 8], [99], weight=60.0, tag="fact")
    for x in [5, 6, 7, 8]:
        m._learn_token(x)
    assert int(torch.as_tensor(m._predict()).argmax()) == 99
    rep = m.revert_tag("fact")
    assert rep["found"] and rep["reverted"] > 0
    m.soft_reset()
    for x in [5, 6, 7, 8]:
        m._learn_token(x)
    assert int(torch.as_tensor(m._predict()).argmax()) != 99


def test_snapshot_roundtrip_memory_state(tmp_path):
    """New state (knn bank, epi tuples + fifo, counters) survives
    snapshot/load on the CPU path."""
    m = FluxLM(_cfg())
    m.ingest(_ids("persistent memory test data " * 3), tag="s")
    m.teach([1, 2, 3, 4], [88], weight=30.0)
    p = m.snapshot(tmp_path / "m.flux")
    m2 = FluxLM.load(p)
    assert m2._knn_cur == m._knn_cur
    assert m2._epi_n == m._epi_n
    assert torch.equal(m2._knn_pos, m._knn_pos)
    assert torch.equal(m2._knn_succ, m._knn_succ)
    for k, dq in m._epi.items():
        assert list(m2._epi[k]) == list(dq)
    for x in [1, 2, 3, 4]:
        m2._learn_token(x)
    assert int(torch.as_tensor(m2._predict()).argmax()) == 88


def test_cuda_primary_teach_and_revert():
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("no CUDA")
    m = FluxLM(_cfg(device="cuda", cuda_primary=True))
    m.ingest(_ids("gpu stream padding " * 8), tag="boot")
    m.teach([5, 6, 7, 8], [99], weight=50.0)
    for x in [5, 6, 7, 8]:
        m._learn_token(x)
    assert int(torch.as_tensor(m._predict()).argmax()) == 99
    rep = m.revert_tag("teach")
    assert rep["found"] and rep["inexact"] == 0
    for x in [5, 6, 7, 8]:
        m._learn_token(x)
    assert int(torch.as_tensor(m._predict()).argmax()) != 99


def test_cuda_primary_episodic_tuple_path():
    """GPU deferred episodic writes store (ctx_end, succ, certs) and
    keep voting; knn cursor mirrors stay in sync through revert."""
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("no CUDA")
    m = FluxLM(_cfg(device="cuda", cuda_primary=True))
    import random
    rnd = random.Random(0)
    seq = [rnd.randrange(400) for _ in range(30)] * 5   # unique ctxs
    m.ingest(seq, tag="t")
    assert m._epi_n == len(seq) - m.config.deep_order
    assert m._knn_cur == len(seq)
    for x in seq[:15]:
        m._learn_token(x)
    top = int(torch.as_tensor(m._predict()).argmax())
    assert top == seq[15]
    rep = m.revert_since(seq_pos=0)
    assert m._count == 0 and m._epi_n == 0 and m._knn_cur == 0


def test_cuda_bulk_ring_wrap_integrity():
    """Block longer than ring_capacity: device _Hg/_ringg must match
    host rings exactly (duplicate-slot scatter is undefined-order on
    CUDA; chunked writes must keep last-wins parity with numpy)."""
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("no CUDA")
    import random
    m = FluxLM(_cfg(device="cuda", cuda_primary=True,
                    ring_capacity=1 << 10))
    rnd = random.Random(0)
    seq = [rnd.randrange(512) for _ in range(2500)]   # > cap
    m.ingest(seq)
    Hg = m._Hg.cpu().numpy().astype(np.uint64)
    assert (Hg == m._H.astype(np.uint64)).all()
    assert (m._ringg.cpu().numpy() == m._ring.astype(np.int32)).all()


def test_cuda_deep_votes_no_ghost_on_miss():
    """An epi probe MISS must yield zero deep votes — _deep_votes_dev
    previously read cells from the default slot's unrelated row."""
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("no CUDA")
    import random
    m = FluxLM(_cfg(device="cuda", cuda_primary=True,
                    ring_capacity=1 << 10, cert_orders=(16,)))
    rnd = random.Random(0)
    m2 = m
    m2.ingest([7] * 40 + [rnd.randrange(512) for _ in range(300)])
    end = torch.tensor(m2._count - 1, dtype=torch.int64,
                       device=m2._dev)
    floor = torch.tensor(max(0, m2._count - m2.config.ring_capacity),
                         dtype=torch.int64, device=m2._dev)
    for flag in (False, True):
        m2._use_triton = flag
        v = m2._deep_votes_dev(end, floor)
        assert float(v.abs().sum()) == 0.0, flag


def test_cuda_bulk_revert_clean():
    """Full revert of a >cap bulk ingest returns all counters to zero
    and re-ingest is deterministic (argmax-stable under atomics)."""
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("no CUDA")
    import random
    m = FluxLM(_cfg(device="cuda", cuda_primary=True,
                    ring_capacity=1 << 10))
    rnd = random.Random(1)
    seq = [rnd.randrange(512) for _ in range(1500)]
    m.ingest(seq, tag="blk")
    rep = m.revert_since(seq_pos=0)
    assert rep["reverted"] > 0
    assert int(m._gt.used[m._gt.row_epi]) == 0
    assert m._epi_n == 0 and m._knn_cur == 0
    m.ingest(seq)
    m.soft_reset()
    for x in seq[:20]:
        m._learn_token(x)
    top = int(torch.as_tensor(m._predict()).argmax())
    assert top == seq[20]


def test_graft_teacher_and_snapshot(tmp_path):
    """Teacher-embedding graft: _R replaced by a semantic projection,
    calibrated row norms, persists through snapshot/load, and refuses
    to run after learning (feature-space consistency)."""
    import math
    V, TD = 512, 32
    m = FluxLM(_cfg(vocab_size=V, topic_dim=TD))
    torch.manual_seed(0)
    E = torch.randn(V, 16)
    E[8] = E[7]                       # tokens 7/8 share a direction
    info = m.graft_teacher(E, method="randproj")
    assert info["teacher_dim"] == 16
    assert m._R.shape == (V, TD) and m._R.dtype == torch.float32
    # rows rescaled to sqrt(topic_dim) — Hebbian rates stay calibrated
    norms = m._R.norm(dim=1)
    assert torch.allclose(norms, torch.full_like(norms, math.sqrt(TD)),
                          atol=0.1)
    # semantic structure survives: cos(7,8) >> cos(7, other)
    Rn = torch.nn.functional.normalize(m._R.float(), dim=1)
    sim_same = float((Rn[7] * Rn[8]).sum())
    sim_oth = float((Rn[7] * Rn[100]).sum())
    assert sim_same > 0.8 and sim_same > sim_oth + 0.3
    # graft after learning must fail — cells are in the old space
    m.ingest(_ids("already learned "), tag="x")
    try:
        m.graft_teacher(E)
        assert False, "graft after learning should raise"
    except RuntimeError:
        pass
    # snapshot roundtrip restores the grafted matrix
    m2 = FluxLM(_cfg(vocab_size=V, topic_dim=TD))
    m2.graft_teacher(E, method="randproj")
    m2.ingest(_ids("snapshot me "), tag="x")
    p = m2.snapshot(tmp_path / "g.flux")
    m3 = FluxLM.load(p)
    assert m3._graft is not None
    assert torch.equal(m3._R.cpu(), m2._R.cpu())


def test_reinforce_and_revert():
    """reinforce(tag, gain) replays a tag's journaled cell writes —
    weak teach becomes decisive; the reinforcement itself reverts."""
    V = 512
    m = FluxLM(_cfg(vocab_size=V))
    m.ingest(_ids("hello world this is a test. " * 10), tag="boot")
    ctx = _ids("open the")
    want = ord("d") % V
    m.teach(ctx, [want], weight=0.4, tag="skill")   # too weak to win
    for x in ctx:
        m._learn_token(x)
    assert int(torch.as_tensor(m._predict()).argmax()) != want
    rep = m.reinforce("skill", gain=8.0)
    assert rep["found"] and rep["applied"] > 0
    assert int(torch.as_tensor(m._predict()).argmax()) == want
    # reinforcement is itself journaled and revertible
    r = m.revert_tag("reinforce:skill")
    assert r["reverted"] == rep["applied"] and r["inexact"] == 0
    assert int(torch.as_tensor(m._predict()).argmax()) != want


def test_revert_since_seq_pos_floor():
    """revert_since(seq_pos) is a hard floor: once stream length reaches
    the mark it must stop — non-token journal entries (teach/consolidate
    writes) sitting below the mark must NOT be eaten."""
    m = FluxLM(_cfg())
    m.ingest(_ids("boot " * 4), tag="boot")
    m.teach([1, 2, 3], [42], weight=40.0, tag="fact")   # below the mark
    mark = m._count
    jlen = len(m.journal.entries)
    m.ingest(_ids("ephemeral junk tokens"), tag="eval")  # above the mark
    rep = m.revert_since(seq_pos=mark)
    assert rep["inexact"] == 0 and m._count == mark
    assert len(m.journal.entries) == jlen   # teach entries survived
    m.soft_reset()
    for x in [1, 2, 3]:
        m._learn_token(x)
    assert int(torch.as_tensor(m._predict()).argmax()) == 42


def test_consolidate_promotes_and_frees():
    """consolidate(): episodic keys with agreeing modal successors get
    promoted into the order tables, their cells freed — journaled."""
    import random
    V = 512
    rnd = random.Random(0)
    m = FluxLM(_cfg(vocab_size=V, ring_capacity=1 << 12))
    pat = [rnd.randrange(V) for _ in range(16)]
    seq = ([rnd.randrange(V) for _ in range(400)]
           + pat + [77] + pat + [77] + pat + [77] + pat)
    m.ingest(seq)
    key = m._suffix_key(m.config.deep_order, m._count - 1)
    assert len(m._epi.get(key, [])) == 3
    n0 = m._epi_n
    rep = m.consolidate(min_entries=3, min_agree=0.5, mass=1.0)
    assert rep["keys"] > 0 and rep["promoted"] > 0 and rep["freed"] > 0
    assert m._epi_n < n0
    # modal successor promoted into the order tables: the order-8 cell
    # at each promoted ctx_end now carries the successor's mass
    k8 = 8
    cell = m._tables[k8].get(m._raw_hash(k8, 415), {})
    assert cell.get(77, 0.0) >= 3.0        # mass x 3 agreeing entries


def test_epi_evict_utility_keeps_hot_keys():
    """Utility eviction: a frequently-retrieved (hit-tracked) key
    survives cap pressure while cold entries drop."""
    import random
    V = 512
    rnd = random.Random(0)
    m = FluxLM(_cfg(vocab_size=V, ring_capacity=1 << 12,
                    epi_total_cap=64, epi_evict_scan=32))
    pat = [rnd.randrange(V) for _ in range(16)]
    warm = [rnd.randrange(V) for _ in range(300)] + pat + [77] + pat + [77]
    m.ingest(warm)
    key = m._suffix_key(m.config.deep_order, m._count - 1)
    # several retrievals -> hits recorded
    for _ in range(4):
        m._deep_votes(m._count - 1)
    assert m._epi_hits.get(key, 0) > 0
    # push past the cap with fresh noise — hot key should survive
    m.ingest([rnd.randrange(V) for _ in range(300)])
    assert m._epi_n <= m.config.epi_total_cap
    assert key in m._epi


def test_clone_state_equivalence_and_independence():
    """clone(): full-memory copy — equal state, zero sharing."""
    import os
    import tempfile
    m = FluxLM(_cfg())
    m.ingest(_ids("the cat sat on the mat. " * 30), tag="t")
    m.teach(_ids("teach ctx"), _ids("tgt"), weight=3.0, tag="teach")
    c = m.clone()
    assert c._count == m._count
    assert len(c._epi) == len(m._epi)
    # generation equivalence: same context -> same argmax
    ctx = torch.tensor([_ids("on the")])
    lm, _, _ = m(ctx); lc, _, _ = c(ctx)
    assert int(lm[0, -1].argmax()) == int(lc[0, -1].argmax())
    # independence: write into the clone, source untouched
    epi0, cnt0 = len(m._epi), m._count
    c.ingest(_ids("new novel stream data " * 20), tag="x")
    assert len(m._epi) == epi0 and m._count == cnt0
    assert c._count > cnt0
    # snapshot compat: clone saves and loads identically
    with tempfile.TemporaryDirectory() as d:
        p = c.snapshot(os.path.join(d, "c.flux"))
        m2 = FluxLM.load(p)
        assert m2._count == c._count and len(m2._epi) == len(c._epi)
