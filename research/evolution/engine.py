"""ForgeEvolve engine: the phase loop.

  Phase 1: GENERATE  — N generators produce candidates (batched GPU forward)
  Phase 2: FILTER    — surrogate predicts scores, top-K selected (GPU)
  Phase 3: SCORE     — real evaluation of K candidates (domain-specific)
  Phase 4: TRAIN     — update generators + surrogate + archive (GPU)
  Phase 5: REPEAT

CUDA: generators + surrogate run on GPU when available. Domain evaluation
may use GPU independently (e.g., quant domain does FP4 ops on GPU).

Usage:
  from research.evolution import ForgeEvolve, ForgeEvolveConfig
  from research.evolution.domains.synthetic import SyntheticDomain

  cfg = ForgeEvolveConfig(
      domain=SyntheticDomain(...),
      n_generators=500,
      filter_ratio=20,
      generations=50,
  )
  engine = ForgeEvolve(cfg)
  results = engine.run()
"""
from __future__ import annotations

import torch
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .generators import GeneratorConfig, GeneratorPopulation, TemplateGenerator
from .surrogate import SurrogateModel
from .archive import MapElitesArchive
from .trainer import GeneratorTrainer
from .database import FindingsDB


# ── CPU worker globals (set by _cpu_worker_init in each worker process) ──
_CPU_DOMAIN = None


def _cpu_worker_init(domain_cls, seq_len: int):
    """Initialize a CPU worker process with its own domain copy.

    Runs once per worker at pool creation. Creates a CPU-only domain
    so the worker can evaluate configs without touching the GPU.
    Handles both heavy domains (seq_len, seed, device args) and
    lightweight domains (no args).
    """
    global _CPU_DOMAIN
    import torch
    torch.set_num_threads(2)
    # Try with full args first, fall back to no-arg constructor
    try:
        _CPU_DOMAIN = domain_cls(seq_len=seq_len, seed=43, device=torch.device("cpu"))
    except TypeError:
        _CPU_DOMAIN = domain_cls()


def _cpu_eval(config: dict) -> dict:
    """Evaluate a config on CPU in a worker process."""
    global _CPU_DOMAIN
    return _CPU_DOMAIN.evaluate(config)


@dataclass
class ForgeEvolveConfig:
    """Configuration for a ForgeEvolve run."""
    # Domain (provides evaluate + config space)
    domain: Any  # BaseDomain

    # Generator population
    n_generators: int = 500
    noise_dim: int = 16
    context_dim: int = 32
    hidden_dim: int = 64
    mutation_rate: float = 0.1

    # Filtering
    filter_ratio: int = 20        # N_generated / N_evaluated
    min_evaluate: int = 10        # minimum candidates to evaluate per gen
    max_evaluate: int = 100       # maximum candidates to evaluate per gen

    # Training
    generator_lr: float = 1e-3
    surrogate_lr: float = 1e-3
    surrogate_mode: str = "mlp"   # "mlp" or "gp"

    # Search
    generations: int = 50
    exploration: float = 0.3      # 0=exploit, 1=explore
    time_budget_s: float = 0      # 0 = no limit

    # Device
    device: str = "auto"          # "auto", "cuda", "cpu"

    # Database
    db_path: str = "forge_evolve.db"
    run_id: str = ""              # empty = auto-generate
    warm_start: bool = True       # load past findings for this domain

    # Parallelism
    parallel_eval: int = 1        # GPU eval parallelism

    # Logging
    verbose: bool = True
    log_every: int = 1


class ForgeEvolve:
    """The main evolutionary search engine."""

    def __init__(self, cfg: ForgeEvolveConfig):
        self.cfg = cfg
        self.domain = cfg.domain
        self.output_dim = self.domain.output_dim()

        # Device selection
        if cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif cfg.device == "cuda":
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Build generator population (batched, on device)
        gen_cfg = GeneratorConfig(
            noise_dim=cfg.noise_dim,
            context_dim=cfg.context_dim,
            hidden_dim=cfg.hidden_dim,
            output_dim=self.output_dim,
            n_generators=cfg.n_generators,
            mutation_rate=cfg.mutation_rate,
        )

        template = None
        discrete_choices = self.domain.discrete_choices()
        if discrete_choices:
            template = TemplateGenerator(discrete_choices)

        self.population = GeneratorPopulation(gen_cfg, template, device=self.device)

        # Surrogate (on device)
        self.surrogate = SurrogateModel(
            input_dim=self.output_dim,
            mode=cfg.surrogate_mode,
            lr=cfg.surrogate_lr,
            device=self.device,
        )

        # Archive
        self.archive = MapElitesArchive(self.domain.behavioral_dims())

        # Trainer (uses batched generator, on device)
        self.trainer = GeneratorTrainer(
            self.population.batched_gen, lr=cfg.generator_lr,
            device=self.device,
        )

        # State
        self.generation = 0
        self.all_results: list[dict] = []
        self.max_results: int = 10000  # cap to prevent OOM on long runs
        self.discoveries: list[dict] = []
        self.start_time = 0.0

        # CPU/GPU pair processing: separate processes for true parallelism
        # GPU process: generators + surrogate + GPU-side evals
        # CPU processes: CPU-side evals (bypass GIL entirely)
        self.cpu_pool = None
        self.n_cpu_workers = 0
        if cfg.parallel_eval > 1 and self.device.type == "cuda":
            try:
                import multiprocessing as mp
                # Use spawn to avoid CUDA fork issues
                ctx = mp.get_context("spawn")
                self.n_cpu_workers = cfg.parallel_eval - 1
                self.cpu_pool = ctx.Pool(
                    processes=self.n_cpu_workers,
                    initializer=_cpu_worker_init,
                    initargs=(type(self.domain), self.domain.seq_len if hasattr(self.domain, "seq_len") else 2048),
                )
            except Exception as e:
                if cfg.verbose:
                    print(f"[CPU pool] Failed to start: {e}, falling back to GPU-only")
                self.cpu_pool = None
                self.n_cpu_workers = 0

        # Database
        self.db = FindingsDB(cfg.db_path)
        self.run_id = cfg.run_id or f"{self.domain.name()}_{int(time.time())}"
        self._warm_started = False
        self._pending_discoveries = []  # batched DB saves

        # Warm-start from past findings
        if cfg.warm_start:
            n_seeded = self.db.seed_from_past(
                self.domain.name(),
                self.population.batched_gen,
                self.surrogate,
            )
            if n_seeded > 0:
                self._warm_started = True
                self._log(f"Warm-started from {n_seeded} past findings "
                          f"(generators + surrogate loaded)")
            else:
                # Cross-domain transfer: if no same-domain history, borrow
                # generators from a different domain with matching output_dim
                od = self.domain.output_dim()
                n_xfer = self.db.seed_cross_domain(
                    self.domain.name(),
                    self.population.batched_gen,
                    self.surrogate,
                    od,
                )
                if n_xfer > 0:
                    self._warm_started = True
                    self._log(f"Cross-domain transfer from {n_xfer} source domain(s) "
                              f"(output_dim={od})")
            # Also load past discoveries as archive seeds
            past = self.db.query_best_configs(self.domain.name(), limit=20)
            for p in past:
                result = self.domain.evaluate(p["config"])
                self.archive.add(p["config"], result["score"],
                                 result.get("behavioral", (0,)), -1)
            if past:
                self._log(f"Loaded {len(past)} past discoveries into archive")

        # Seed from domain's known-good configs
        seed_configs = self.domain.seed_configs()
        if seed_configs:
            self._log(f"Evaluating {len(seed_configs)} seed configs...")
            seed_params = []
            seed_scores = []
            for sc in seed_configs:
                result = self.domain.evaluate(sc)
                self.archive.add(sc, result["score"],
                                 result.get("behavioral", (0,)), -1)
                self.all_results.append({
                    "generation": -1,
                    "config": sc,
                    "score": result["score"],
                    "behavioral": result.get("behavioral", (0,)),
                    "metadata": result.get("metadata", {}),
                })
                # Encode for surrogate training
                params = self.domain.encode(sc)
                if params is not None:
                    seed_params.append(params)
                    seed_scores.append(result["score"])

            # Train surrogate on seeds immediately (not random anymore)
            if seed_params:
                param_tensor = torch.stack(seed_params).to(self.device)
                score_tensor = torch.tensor(seed_scores, dtype=torch.float32)
                self.surrogate.train(param_tensor, score_tensor, epochs=5)
                self._log(f"Surrogate trained on {len(seed_params)} seeds "
                          f"(best seed score={max(seed_scores):.4f})")

            # Update context with best seed
            if seed_scores:
                best_idx = int(np.argmax(seed_scores))
                self.population.update_context(seed_params[best_idx])

    def _gpu_batch_eval(self, configs: list[dict]) -> list[dict]:
        """Evaluate configs on GPU with memory-aware chunking.

        Profiles the first eval to measure per-eval GPU memory delta,
        then chunks the batch so that peak VRAM stays under a safety
        threshold. Falls back to sequential for lightweight domains.
        """
        import torch

        if len(configs) <= 2:
            return [self._safe_eval(cfg) for cfg in configs]

        # ── Profile first eval: time + memory delta ──
        if not hasattr(self, '_eval_profile'):
            self._eval_profile = None
        if self._eval_profile is None:
            # Warmup call (first CUDA call triggers kernel JIT, ignore timing)
            self._safe_eval(configs[0])
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                mem_before = torch.cuda.memory_allocated()
            # Second call for accurate timing
            t0 = time.time()
            self._safe_eval(configs[0])
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            ms = (time.time() - t0) * 1000
            if self.device.type == "cuda":
                mem_delta = torch.cuda.memory_allocated() - mem_before
                peak = torch.cuda.max_memory_allocated()
                total = torch.cuda.get_device_properties(0).total_memory
                # Reserve 2GB for generators + model + overhead
                safe_budget = max(total - 2 * 1024**3, 512 * 1024**2)
                # Per-eval memory cost (use peak delta, conservative)
                mem_per_eval = max(mem_delta, peak - mem_before, 1)
                # How many evals can we fit in the safety budget?
                chunk_size = max(1, int(safe_budget / (mem_per_eval * 1.5)))
            else:
                chunk_size = len(configs)
            self._eval_profile = {"ms": ms, "mem_per": mem_per_eval if self.device.type == "cuda" else 0,
                                  "chunk": chunk_size}

        chunk = self._eval_profile["chunk"]
        ms = self._eval_profile["ms"]

        # Sequential for fast domains (<10ms), chunked for heavy
        if ms < 10 and chunk >= len(configs):
            return [self._safe_eval(cfg) for cfg in configs]

        # Chunked evaluation with memory cleanup between chunks
        results = []
        for i in range(0, len(configs), chunk):
            batch = configs[i:i + chunk]
            if ms < 10:
                # Fast domain: just run sequentially
                results.extend(self._safe_eval(cfg) for cfg in batch)
            else:
                # Heavy domain: threaded within chunk, sync between chunks
                results.extend(self._threaded_eval(batch))
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        return results

    def _safe_eval(self, config: dict) -> dict:
        """Evaluate a single config with OOM recovery."""
        try:
            return self.domain.evaluate(config)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                import torch
                torch.cuda.empty_cache()
                # Return a penalty score so the search doesn't crash
                return {"score": -1e9, "behavioral": (0.0,) * len(self.domain.behavioral_dims()),
                        "metadata": {"oom": True}}
            raise

    def _threaded_eval(self, configs: list[dict]) -> list[dict]:
        """Evaluate configs using threads with CUDA streams for overlap."""
        from concurrent.futures import ThreadPoolExecutor
        import torch

        n_threads = min(4, len(configs))
        results = [None] * len(configs)

        def _eval_one(idx_cfg):
            idx, cfg = idx_cfg
            try:
                stream = torch.cuda.Stream() if self.device.type == "cuda" else None
                if stream:
                    with torch.cuda.stream(stream):
                        result = self.domain.evaluate(cfg)
                    stream.synchronize()
                else:
                    result = self.domain.evaluate(cfg)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    result = {"score": -1e9, "behavioral": (0.0,) * len(self.domain.behavioral_dims()),
                              "metadata": {"oom": True}}
                else:
                    raise
            return idx, result

        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            for idx, result in pool.map(_eval_one, enumerate(configs)):
                results[idx] = result

        return results

    def _log(self, msg: str):
        if self.cfg.verbose:
            elapsed = time.time() - self.start_time if self.start_time else 0
            print(f"[Gen {self.generation:3d} | {elapsed:6.1f}s] {msg}")

    def run(self) -> dict:
        """Run the full evolutionary search loop."""
        self.start_time = time.time()
        self._log(f"Starting ForgeEvolve: {self.cfg.n_generators} generators, "
                  f"filter_ratio={self.cfg.filter_ratio}, "
                  f"domain={self.domain.name()}, device={self.device}"
                  + (f", CPU_pool={self.n_cpu_workers} workers" if self.cpu_pool else ""))

        try:
            for gen in range(self.cfg.generations):
                self.generation = gen

                # Clear CUDA cache periodically to prevent fragmentation OOM
                if gen > 0 and gen % 5 == 0 and self.device.type == "cuda":
                    torch.cuda.empty_cache()

                if self.cfg.time_budget_s > 0:
                    elapsed = time.time() - self.start_time
                    if elapsed > self.cfg.time_budget_s:
                        self._log(f"Time budget exceeded ({elapsed:.0f}s)")
                        break

                # ── PHASE 1: GENERATE (batched GPU forward) ──
                t0 = time.time()
                candidates = self.population.generate(n_per_gen=1)
                n_generated = len(candidates)
                t_gen = time.time() - t0

                # ── PHASE 2: FILTER (surrogate on GPU + epsilon-greedy exploration) ──
                t0 = time.time()
                neural_cands = [c for c in candidates if c.get("params") is not None]
                template_cands = [c for c in candidates if c.get("template_config") is not None]

                n_eval = max(self.cfg.min_evaluate,
                             min(self.cfg.max_evaluate, n_generated // self.cfg.filter_ratio))

                n_template_eval = len(template_cands)
                n_neural_eval = max(0, n_eval - n_template_eval)

                if neural_cands and n_neural_eval > 0:
                    # Epsilon-greedy: 30% of eval budget goes to random candidates
                    # Adaptive: more exploration early, less later
                    epsilon = max(0.1, 0.3 - gen * 0.005)
                    n_random = max(1, int(n_neural_eval * epsilon))
                    n_surrogate = n_neural_eval - n_random

                    # Surrogate-filtered selection
                    param_tensor = torch.stack([c["params"] for c in neural_cands]).to(self.device)
                    top_indices = self.surrogate.filter_top_k(param_tensor, n_surrogate)
                    selected_neural = [neural_cands[i] for i in top_indices.tolist()]

                    # Random selection (exploration)
                    remaining = [i for i in range(len(neural_cands)) if i not in top_indices.tolist()]
                    if remaining and n_random > 0:
                        random_indices = np.random.choice(remaining,
                                                          size=min(n_random, len(remaining)),
                                                          replace=False)
                        selected_neural += [neural_cands[i] for i in random_indices]
                else:
                    selected_neural = []

                selected = selected_neural + template_cands
                t_filter = time.time() - t0

                # ── PHASE 3: SCORE (domain evaluation, GPU+CPU parallel) ──
                t0 = time.time()
                scores = []
                behavioral_list = []
                configs = []

                # Decode all configs first — batch GPU→CPU transfer to avoid 50 sync points
                # Collect all param tensors, move to CPU in one call, then decode
                param_tensors = []
                param_indices = []
                for ci, cand in enumerate(selected):
                    if "params" in cand and cand["params"] is not None:
                        param_tensors.append(cand["params"])
                        param_indices.append(ci)

                if param_tensors:
                    # Single GPU→CPU transfer for all params at once
                    batched_params = torch.stack(param_tensors).detach().cpu()

                decoded = [None] * len(selected)
                for ci, cand in enumerate(selected):
                    if "params" in cand and cand["params"] is not None:
                        # Use pre-transferred CPU tensor
                        pi = param_indices.index(ci)
                        decoded[ci] = self.domain.decode(batched_params[pi])
                    else:
                        tc = cand["template_config"]
                        try:
                            config = self.domain.decode(self.domain.encode(tc))
                            config.update(tc)
                        except Exception:
                            config = tc
                        decoded[ci] = config

                if self.cpu_pool is not None and len(decoded) >= 8:
                    # Split: GPU gets 40%, CPU workers get 60% (CPU has more cores)
                    n_cpu = int(len(decoded) * 0.6)
                    n_gpu = len(decoded) - n_cpu
                    gpu_configs = decoded[:n_gpu]
                    cpu_configs = decoded[n_gpu:]

                    # Submit CPU batch asynchronously (non-blocking, chunked)
                    chunksize = max(1, len(cpu_configs) // (self.n_cpu_workers * 2))
                    cpu_result_async = self.cpu_pool.map_async(
                        _cpu_eval, cpu_configs, chunksize=chunksize,
                    )

                    # Run GPU batch with thread pool + CUDA streams for overlap
                    gpu_results = self._gpu_batch_eval(gpu_configs)

                    # Wait for CPU workers to finish
                    cpu_results = cpu_result_async.get()

                    # Re-merge in order
                    results = gpu_results + cpu_results
                    configs = gpu_configs + cpu_configs
                else:
                    # GPU-only: use threaded batch eval for kernel overlap
                    results = self._gpu_batch_eval(decoded)
                    configs = decoded

                for config, result in zip(configs, results):
                    scores.append(result["score"])
                    behavioral_list.append(result.get("behavioral", (0,)))

                    self.all_results.append({
                        "generation": gen,
                        "config": config,
                        "score": result["score"],
                        "behavioral": result.get("behavioral", (0,)),
                        "metadata": result.get("metadata", {}),
                    })

                # Cap results to prevent OOM on long runs
                if len(self.all_results) > self.max_results:
                    self.all_results = self.all_results[-self.max_results:]

                t_score = time.time() - t0

                # ── PHASE 4: TRAIN (GPU updates) ──
                t0 = time.time()

                # Update archive + save new discoveries to DB
                new_discoveries = 0
                new_discovery_list = []
                for config, score, behavioral in zip(configs, scores, behavioral_list):
                    added = self.archive.add(config, score, behavioral, gen)
                    if added:
                        new_discoveries += 1
                        self.discoveries.append({
                            "generation": gen,
                            "config": config,
                            "score": score,
                            "behavioral": behavioral,
                        })
                        new_discovery_list.append({
                            "generation": gen,
                            "config": config,
                            "score": score,
                            "behavioral": behavioral,
                            "metadata": self.all_results[-1].get("metadata", {}),
                        })

                # Save new discoveries to database (batched — flush every 10 gens)
                if new_discovery_list:
                    self._pending_discoveries.extend(new_discovery_list)
                    if (gen + 1) % 10 == 0 or gen == self.cfg.generations - 1:
                        self.db.save_discoveries(self.run_id, self.domain.name(),
                                                 self._pending_discoveries)
                        self._pending_discoveries = []

                # Update surrogate
                if neural_cands and selected_neural:
                    eval_params = torch.stack([c["params"] for c in selected_neural]).detach()
                    eval_scores = torch.tensor(
                        scores[:len(selected_neural)], dtype=torch.float32
                    )
                    self.surrogate.train(eval_params, eval_scores)
                    self.surrogate.generation = gen

                # Update generators (REINFORCE) — pass context
                self.trainer.update(selected, scores, self.population.context)

                # Evolve population
                gen_indices = [c.get("generator_idx", -1) for c in selected]
                mutated = self.population.evolve(scores, gen_indices)

                # Notify trainer of mutations so it can reset optimizer state
                for idx in mutated:
                    self.trainer.notify_mutation(idx)

                # Update context: UCB-selected parent (not just best)
                # 70% UCB selection, 30% best (exploit known good)
                if np.random.random() < 0.7:
                    parent = self.archive.sample_elite_ucb(current_gen=gen, c=1.0)
                else:
                    parent = self.archive.best_entry
                if parent is None:
                    parent = self.archive.best_entry
                if parent:
                    parent_params = self.domain.encode(parent.config)
                    if parent_params is not None:
                        self.population.update_context(parent_params)

                t_train = time.time() - t0

                # Incremental generator/surrogate checkpoint every 10 gens
                if (gen + 1) % 10 == 0:
                    self.db.save_generators(self.run_id, self.domain.name(),
                                            self.population.batched_gen)
                    self.db.save_surrogate(self.run_id, self.domain.name(), self.surrogate)

                # ── LOG ──
                if gen % self.cfg.log_every == 0:
                    self._log(
                        f"gen={t_gen:.3f}s filter={t_filter:.3f}s "
                        f"score={t_score:.2f}s train={t_train:.3f}s | "
                        f"evaluated={len(selected)}/{n_generated} | "
                        f"new={new_discoveries} | "
                        f"{self.archive.summary()}"
                    )

            # Final summary
            elapsed = time.time() - self.start_time
            self._log(f"Done: {self.generation+1} generations, "
                      f"{len(self.all_results)} evaluations, "
                      f"{len(self.discoveries)} discoveries, "
                      f"{elapsed:.1f}s total")

            # Save final state to database
            # Flush any remaining pending discoveries
            if self._pending_discoveries:
                self.db.save_discoveries(self.run_id, self.domain.name(),
                                         self._pending_discoveries)
                self._pending_discoveries = []

            results = {
                "generations": self.generation + 1,
                "total_evaluations": len(self.all_results),
                "discoveries": len(self.discoveries),
                "best_score": self.archive.best_score,
                "best_config": self.archive.best_entry.config if self.archive.best_entry else None,
                "device": str(self.device),
            }
            self.db.save_run(self.run_id, self.domain.name(),
                             {"n_generators": self.cfg.n_generators,
                              "filter_ratio": self.cfg.filter_ratio,
                              "generations": self.cfg.generations},
                             results, self.start_time)
            self.db.save_generators(self.run_id, self.domain.name(),
                                    self.population.batched_gen)
            self.db.save_surrogate(self.run_id, self.domain.name(), self.surrogate)
            self._log(f"Saved to database: {self.cfg.db_path} (run_id={self.run_id})")

            results = {
                "generations": self.generation + 1,
                "total_evaluations": len(self.all_results),
                "discoveries": len(self.discoveries),
                "archive_coverage": self.archive.coverage(),
                "best_score": self.archive.best_score,
                "best_config": self.archive.best_entry.config if self.archive.best_entry else None,
                "pareto_front": [
                    {"config": e.config, "score": e.score, "behavioral": e.behavioral,
                     "generation": e.generation, "metadata": e.metadata}
                    for e in self.archive.get_pareto_front()
                ],
                "time_s": elapsed,
                "all_results": self.all_results,
                "discoveries_list": self.discoveries,
                "archive": self.archive,
                "device": str(self.device),
                "run_id": self.run_id,
                "warm_started": self._warm_started,
            }
        finally:
            # Clean up CPU worker pool
            if self.cpu_pool is not None:
                self.cpu_pool.close()
                self.cpu_pool.join()
                self.cpu_pool = None

        return results

    def guide(self, direction: str = "explore",
              seed_configs: list[dict] | None = None,
              constraints: dict | None = None,
              time_budget: str | float | None = None):
        """Steer the search mid-run (applies on next generation)."""
        if direction == "explore":
            self.cfg.exploration = 0.8
            self.cfg.mutation_rate = 0.3
        elif direction == "exploit":
            self.cfg.exploration = 0.1
            self.cfg.mutation_rate = 0.02

        if seed_configs:
            for config in seed_configs:
                result = self.domain.evaluate(config)
                self.archive.add(config, result["score"],
                                 result.get("behavioral", (0,)), self.generation)

        if time_budget:
            if isinstance(time_budget, str):
                import re
                m = re.match(r"(\d+(?:\.\d+)?)\s*(s|m|h)", time_budget)
                if m:
                    val, unit = float(m.group(1)), m.group(2)
                    self.cfg.time_budget_s = val * {"s": 1, "m": 60, "h": 3600}[unit]
            else:
                self.cfg.time_budget_s = time_budget
