# FluxStreamPool — multi-stream predict + single-learner packets (R52-10)

## Design (user-approved direction)
- N predict-only token streams on SHARED read-only memory + ONE serialized
  learning process applying selected update packets post-hoc.
- Streams never write tables/epi/knn/csem/A/journal. Per-stream state:
  pos, small ring/hashes (stream tokens only at pos>=C; canonical reads
  via shared `_ringg`/`_Hg` for pos<C — hybrid gather), seen[V] int64,
  unig[V] f32 + loguni[V] f32 + tot, c[D], proto[D], hedge w[Cch] f64 + hmg.
  ep_start_i = C (soft_reset semantics, matches clone workers).
- VRAM/stream ≈ cap_s*12B + V*16B ≈ 1.2MB @cap_s=1024,V=64k → 64 streams
  ~75MB vs ~1.4GB/clone. No serialize/load, no journal, no revert.
- Packet = {tag, prompt_ids, gen_ids}. Apply = ws → ingest(ids,tag) → we.
- `FluxLM.apply_packets(packets, accept)` = the single learning process.

## Latent bug confirmed
- `reinforce(tag)` collects only "c" entries inside ws/we spans.
  `ingest()` emits "c" w/o ws/we (CPU) or one "bc" block (GPU) →
  reinforce-after-ingest = silent no-op. RSI `_flux_finish` affected.
- Fix: (a) apply_packets wraps ingest in ws/we; (b) reinforce collects
  "bc" block order-deltas inside span (writes_all+surprise) and amplifies
  via _gt.add + journaled "c" entries (revertible).

## Implementation
1. forge/model/flux_stream.py — FluxPacket, FluxStreamPool
   - GPU: batched probes/logits [N,V] port of _probes_at/_logits_dev +
     batched reward (hedge) port of _step_dev rewards; hybrid hash reads.
   - CPU: _StreamCtx facade (__getattr__→model; own _H,_ring (copy),
     _last_seen,_recent,_c,_proto,_w,_hedge_mix,_uni,_total,_log_uni_num,
     _count,_ep_start,_gaps,journal=null,_predict_only=True,learning=False)
     + FluxLM._learn_token(facade,x) / FluxLM._predict(facade).
   - generate(): feed prompts lockstep → per-step logits → per-stream
     temp/top-p/top-k/rep-penalty/seed/processors → eos/stop → packets.
2. flux.py: `_predict_only` guards in _learn_token (order cells, A-delta,
   _csem_write); reinforce() bc-deltas; apply_packets().
3. engine_generation._generate_batch_flux → pool (streams=N prompts),
   pass logits_processors, stash self.last_flux_packets,
   apply_flux_packets(accept) helper. flux_workers_device kept (cpu|cuda
   selects pool device only when model is cpu? — streams run on model._dev;
   keep flag as compat no-op / document).
4. session_manager._run_batched: FluxLM → engine.generate_batch path.
5. infinite_loop: _flux_eval_acc → batched predict-only eval;
   _flux_finish → apply packet of first good attempt under tag + reinforce
   (fix) + teach_text on fail (unchanged).
6. _free_engine: unlink flux_sleep_*.flux after sleep(2).
7. tests + bench + AGENTS.md.

## CPU facade predict_only gates in _learn_token
- order-table writes, A[x].add_+append_vec journal, _csem_write → gated.
- uni/_last_seen/_recent/_proto/hedge/gaps → per-facade state, kept.
- _push epi/knn writes → facade.learning=False skips them naturally.
