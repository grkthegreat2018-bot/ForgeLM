"""Persistent storage for ForgeEvolve findings using SQLite.

Tables:
  - runs: metadata about each search run (domain, config, timestamps)
  - discoveries: configs that entered the MAP-Elites archive
  - evaluations: every (config, score) pair ever evaluated
  - generators: saved generator weights (for resuming/cross-run transfer)
  - surrogate: saved surrogate weights (for warm-starting)

Usage:
  db = FindingsDB("forge_evolve.db")
  db.save_run(run_id, config, results)
  db.save_discoveries(run_id, discoveries_list)
  db.save_generators(run_id, batched_gen)
  db.load_generators(run_id, batched_gen)  # warm-start
  past = db.query_discoveries(domain="quant", min_score=-2.0)
"""
from __future__ import annotations

import sqlite3
import json
import pickle
import time
import torch
import numpy as np
from pathlib import Path
from typing import Any, Optional


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays and scalars (avoids default=str)."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


def _dumps(obj) -> str:
    """JSON dumps with numpy support."""
    return json.dumps(obj, default=_NumpyEncoder().default)


class FindingsDB:
    """SQLite-backed persistent storage for ForgeEvolve findings."""

    def __init__(self, db_path: str | Path = "forge_evolve.db"):
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path)
        # WAL mode: non-blocking reads, async writes (10-100x faster commits)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA cache_size=-10000")  # 10MB cache
        self._init_tables()

    def _init_tables(self):
        c = self.conn.cursor()

        c.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                domain TEXT NOT NULL,
                config_json TEXT,
                start_time REAL,
                end_time REAL,
                generations INTEGER,
                total_evals INTEGER,
                discoveries INTEGER,
                best_score REAL,
                best_config_json TEXT,
                device TEXT
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS discoveries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                domain TEXT NOT NULL,
                generation INTEGER,
                config_json TEXT NOT NULL,
                score REAL NOT NULL,
                behavioral_json TEXT,
                metadata_json TEXT,
                timestamp REAL DEFAULT (strftime('%s','now')),
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
        """)

        c.execute("""
            CREATE INDEX IF NOT EXISTS idx_discoveries_domain
            ON discoveries(domain)
        """)
        c.execute("""
            CREATE INDEX IF NOT EXISTS idx_discoveries_score
            ON discoveries(domain, score DESC)
        """)
        c.execute("""
            CREATE INDEX IF NOT EXISTS idx_discoveries_dedup
            ON discoveries(domain, config_json)
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                domain TEXT NOT NULL,
                generation INTEGER,
                config_json TEXT NOT NULL,
                score REAL NOT NULL,
                metadata_json TEXT,
                timestamp REAL DEFAULT (strftime('%s','now')),
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS generators (
                run_id TEXT PRIMARY KEY,
                domain TEXT NOT NULL,
                weights_blob BLOB NOT NULL,
                fitness_blob BLOB NOT NULL,
                config_json TEXT,
                generation INTEGER,
                timestamp REAL DEFAULT (strftime('%s','now'))
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS surrogate (
                run_id TEXT PRIMARY KEY,
                domain TEXT NOT NULL,
                weights_blob BLOB NOT NULL,
                n_trained INTEGER,
                config_json TEXT,
                timestamp REAL DEFAULT (strftime('%s','now'))
            )
        """)

        # Canonical generators/surrogate: the best-ever weights per domain,
        # persisted across ALL runs and profiles. Not keyed by run_id —
        # keyed by domain only. Updated only when a run beats the stored
        # best_score. This is the "permanent knowledge" layer.
        c.execute("""
            CREATE TABLE IF NOT EXISTS canonical_generators (
                domain TEXT PRIMARY KEY,
                weights_blob BLOB NOT NULL,
                fitness_blob BLOB NOT NULL,
                config_json TEXT,
                best_score REAL NOT NULL,
                run_id TEXT,
                timestamp REAL DEFAULT (strftime('%s','now'))
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS canonical_surrogate (
                domain TEXT PRIMARY KEY,
                weights_blob BLOB NOT NULL,
                n_trained INTEGER,
                config_json TEXT,
                best_score REAL NOT NULL,
                run_id TEXT,
                timestamp REAL DEFAULT (strftime('%s','now'))
            )
        """)

        self.conn.commit()

    def save_run(self, run_id: str, domain: str, config: dict,
                 results: dict, start_time: float):
        """Save or update a run record."""
        c = self.conn.cursor()
        c.execute("""
            INSERT OR REPLACE INTO runs
            (run_id, domain, config_json, start_time, end_time,
             generations, total_evals, discoveries, best_score,
             best_config_json, device)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id, domain, _dumps(config),
            start_time, time.time(),
            results.get("generations", 0),
            results.get("total_evaluations", 0),
            results.get("discoveries", 0),
            results.get("best_score", 0),
            _dumps(results.get("best_config")),
            results.get("device", "unknown"),
        ))
        self.conn.commit()

    def save_discoveries(self, run_id: str, domain: str,
                         discoveries: list[dict]):
        """Save discoveries from a run. One row per unique (domain, config).
        If a config already exists, UPDATE its score if the new one is higher
        (don't insert a duplicate). This keeps the DB strictly unique."""
        c = self.conn.cursor()
        n_saved = 0
        n_updated = 0
        n_skipped = 0
        for d in discoveries:
            config_json = _dumps(d.get("config"))
            score = d.get("score", 0)
            # Check if this exact config already exists for this domain
            c.execute(
                "SELECT id, score FROM discoveries WHERE domain=? AND config_json=? "
                "ORDER BY score DESC LIMIT 1",
                (domain, config_json))
            row = c.fetchone()
            if row is not None:
                existing_id, existing_score = row[0], row[1]
                if score > existing_score:
                    # Update the existing row with the better score + metadata
                    c.execute("""
                        UPDATE discoveries
                        SET run_id=?, generation=?, score=?,
                            behavioral_json=?, metadata_json=?
                        WHERE id=?
                    """, (
                        run_id, d.get("generation", 0), score,
                        _dumps(d.get("behavioral")),
                        _dumps(d.get("metadata", {})),
                        existing_id,
                    ))
                    n_updated += 1
                else:
                    n_skipped += 1
                continue
            c.execute("""
                INSERT INTO discoveries
                (run_id, domain, generation, config_json, score,
                 behavioral_json, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id, domain, d.get("generation", 0),
                config_json,
                score,
                _dumps(d.get("behavioral")),
                _dumps(d.get("metadata", {})),
            ))
            n_saved += 1
        self.conn.commit()
        return n_saved, n_skipped + n_updated

    def save_evaluation(self, run_id: str, domain: str, generation: int,
                        config: dict, score: float, metadata: dict):
        """Save a single evaluation (for incremental saves during run)."""
        c = self.conn.cursor()
        c.execute("""
            INSERT INTO evaluations
            (run_id, domain, generation, config_json, score, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            run_id, domain, generation,
            _dumps(config), score,
            _dumps(metadata),
        ))
        self.conn.commit()

    def save_generators(self, run_id: str, domain: str, batched_gen):
        """Save generator weights for resuming or cross-run transfer."""
        weights = {}
        for name, p in batched_gen.named_parameters():
            weights[name] = p.detach().cpu().numpy()
        fitness = batched_gen.fitness_ema.detach().cpu().numpy()

        config = {
            "noise_dim": batched_gen.cfg.noise_dim,
            "context_dim": batched_gen.cfg.context_dim,
            "hidden_dim": batched_gen.cfg.hidden_dim,
            "output_dim": batched_gen.cfg.output_dim,
            "n_generators": batched_gen.cfg.n_generators,
        }

        c = self.conn.cursor()
        c.execute("""
            INSERT OR REPLACE INTO generators
            (run_id, domain, weights_blob, fitness_blob, config_json, generation)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            run_id, domain,
            pickle.dumps(weights),
            pickle.dumps(fitness),
            json.dumps(config),
            0,  # TODO: track actual generation
        ))
        self.conn.commit()

    def load_generators(self, run_id: str, batched_gen) -> bool:
        """Load generator weights from a past run. Returns True if loaded."""
        c = self.conn.cursor()
        c.execute("SELECT weights_blob, fitness_blob FROM generators WHERE run_id=?",
                  (run_id,))
        row = c.fetchone()
        if row is None:
            return False

        weights = pickle.loads(row[0])
        fitness = pickle.loads(row[1])

        with torch.no_grad():
            for name, p in batched_gen.named_parameters():
                if name in weights:
                    saved = torch.from_numpy(weights[name]).to(p.device)
                    if saved.shape != p.shape:
                        continue  # skip mismatched shapes (e.g. different output_dim)
                    p.copy_(saved)
            if fitness.shape == batched_gen.fitness_ema.shape:
                batched_gen.fitness_ema.copy_(
                    torch.from_numpy(fitness).to(batched_gen.fitness_ema.device)
                )
        return True

    def save_surrogate(self, run_id: str, domain: str, surrogate):
        """Save surrogate network weights for warm-starting."""
        if surrogate.mode != "mlp":
            return  # only MLP supported

        weights = {}
        for name, p in surrogate.net.named_parameters():
            weights[name] = p.detach().cpu().numpy()

        c = self.conn.cursor()
        c.execute("""
            INSERT OR REPLACE INTO surrogate
            (run_id, domain, weights_blob, n_trained, config_json)
            VALUES (?, ?, ?, ?, ?)
        """, (
            run_id, domain,
            pickle.dumps(weights),
            surrogate.n_trained,
            json.dumps({"mode": surrogate.mode, "input_dim": surrogate.input_dim}),
        ))
        self.conn.commit()

    def load_surrogate(self, run_id: str, surrogate) -> bool:
        """Load surrogate weights. Returns True if loaded."""
        if surrogate.mode != "mlp":
            return False

        c = self.conn.cursor()
        c.execute("SELECT weights_blob, n_trained FROM surrogate WHERE run_id=?",
                  (run_id,))
        row = c.fetchone()
        if row is None:
            return False

        weights = pickle.loads(row[0])
        surrogate.n_trained = row[1]

        with torch.no_grad():
            for name, p in surrogate.net.named_parameters():
                if name in weights:
                    saved = torch.from_numpy(weights[name]).to(p.device)
                    if saved.shape != p.shape:
                        continue  # skip mismatched shapes (e.g. different input_dim)
                    p.copy_(saved)
        return True

    # ── Canonical generators/surrogate (permanent knowledge layer) ──────

    def save_canonical_generators(self, domain: str, batched_gen,
                                  best_score: float, run_id: str) -> bool:
        """Save generators as canonical for this domain if best_score beats
        the stored value. Returns True if updated."""
        c = self.conn.cursor()
        c.execute("SELECT best_score FROM canonical_generators WHERE domain=?",
                  (domain,))
        row = c.fetchone()
        if row is not None and row[0] >= best_score:
            return False  # stored is better, don't overwrite

        weights = {}
        for name, p in batched_gen.named_parameters():
            weights[name] = p.detach().cpu().numpy()
        fitness = batched_gen.fitness_ema.detach().cpu().numpy()
        config = {
            "noise_dim": batched_gen.cfg.noise_dim,
            "context_dim": batched_gen.cfg.context_dim,
            "hidden_dim": batched_gen.cfg.hidden_dim,
            "output_dim": batched_gen.cfg.output_dim,
            "n_generators": batched_gen.cfg.n_generators,
        }
        c.execute("""
            INSERT OR REPLACE INTO canonical_generators
            (domain, weights_blob, fitness_blob, config_json, best_score, run_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (domain, pickle.dumps(weights), pickle.dumps(fitness),
              json.dumps(config), best_score, run_id))
        self.conn.commit()
        return True

    def load_canonical_generators(self, domain: str, batched_gen) -> bool:
        """Load canonical generators for a domain. Returns True if loaded."""
        c = self.conn.cursor()
        c.execute("SELECT weights_blob, fitness_blob FROM canonical_generators WHERE domain=?",
                  (domain,))
        row = c.fetchone()
        if row is None:
            return False
        weights = pickle.loads(row[0])
        fitness = pickle.loads(row[1])
        with torch.no_grad():
            for name, p in batched_gen.named_parameters():
                if name in weights:
                    saved = torch.from_numpy(weights[name]).to(p.device)
                    if saved.shape == p.shape:
                        p.copy_(saved)
            if fitness.shape == batched_gen.fitness_ema.shape:
                batched_gen.fitness_ema.copy_(
                    torch.from_numpy(fitness).to(batched_gen.fitness_ema.device))
        return True

    def save_canonical_surrogate(self, domain: str, surrogate,
                                 best_score: float, run_id: str) -> bool:
        """Save surrogate as canonical for this domain if best_score beats
        the stored value. Returns True if updated."""
        if surrogate.mode != "mlp":
            return False
        c = self.conn.cursor()
        c.execute("SELECT best_score FROM canonical_surrogate WHERE domain=?",
                  (domain,))
        row = c.fetchone()
        if row is not None and row[0] >= best_score:
            return False

        weights = {}
        for name, p in surrogate.net.named_parameters():
            weights[name] = p.detach().cpu().numpy()
        c.execute("""
            INSERT OR REPLACE INTO canonical_surrogate
            (domain, weights_blob, n_trained, config_json, best_score, run_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (domain, pickle.dumps(weights), surrogate.n_trained,
              json.dumps({"mode": surrogate.mode, "input_dim": surrogate.input_dim}),
              best_score, run_id))
        self.conn.commit()
        return True

    def load_canonical_surrogate(self, domain: str, surrogate) -> bool:
        """Load canonical surrogate for a domain. Returns True if loaded."""
        if surrogate.mode != "mlp":
            return False
        c = self.conn.cursor()
        c.execute("SELECT weights_blob, n_trained FROM canonical_surrogate WHERE domain=?",
                  (domain,))
        row = c.fetchone()
        if row is None:
            return False
        weights = pickle.loads(row[0])
        surrogate.n_trained = row[1]
        with torch.no_grad():
            for name, p in surrogate.net.named_parameters():
                if name in weights:
                    saved = torch.from_numpy(weights[name]).to(p.device)
                    if saved.shape == p.shape:
                        p.copy_(saved)
        return True

    def get_canonical_best_score(self, domain: str) -> float | None:
        """Get the best score ever stored canonically for a domain."""
        c = self.conn.cursor()
        c.execute("SELECT best_score FROM canonical_generators WHERE domain=?",
                  (domain,))
        row = c.fetchone()
        return row[0] if row else None

    def query_discoveries(self, domain: str, min_score: float = -1e9,
                          limit: int = 100) -> list[dict]:
        """Query past discoveries for a domain, sorted by score."""
        c = self.conn.cursor()
        c.execute("""
            SELECT config_json, score, behavioral_json, metadata_json, run_id, generation
            FROM discoveries
            WHERE domain=? AND score >= ?
            ORDER BY score DESC
            LIMIT ?
        """, (domain, min_score, limit))
        rows = c.fetchall()

        return [{
            "config": json.loads(r[0]),
            "score": r[1],
            "behavioral": json.loads(r[2]) if r[2] else None,
            "metadata": json.loads(r[3]) if r[3] else {},
            "run_id": r[4],
            "generation": r[5],
        } for r in rows]

    def query_best_configs(self, domain: str, limit: int = 10) -> list[dict]:
        """Get the best configs ever found for a domain."""
        return self.query_discoveries(domain, min_score=-1e9, limit=limit)

    def seed_from_past(self, domain: str, batched_gen, surrogate,
                       n_seed: int = 5) -> int:
        """Warm-start generators + surrogate from best past findings.

        Loads the best past configs, evaluates them to set context,
        and loads surrogate weights if available.

        Returns number of seeds loaded.
        """
        # Load best past discoveries
        past = self.query_best_configs(domain, limit=n_seed)
        if not past:
            return 0

        # Load surrogate weights from most recent run
        c = self.conn.cursor()
        c.execute("""
            SELECT run_id FROM surrogate WHERE domain=?
            ORDER BY timestamp DESC LIMIT 1
        """, (domain,))
        surr_row = c.fetchone()
        if surr_row:
            self.load_surrogate(surr_row[0], surrogate)

        # Load generator weights from most recent run
        c.execute("""
            SELECT run_id FROM generators WHERE domain=?
            ORDER BY timestamp DESC LIMIT 1
        """, (domain,))
        gen_row = c.fetchone()
        if gen_row:
            self.load_generators(gen_row[0], batched_gen)

        return len(past)

    def seed_cross_domain(self, domain: str, batched_gen, surrogate,
                          output_dim: int) -> int:
        """Cross-domain transfer: load generators from a different domain
        with the same output_dim. The generator learns the parameter
        distribution, which transfers across domains with similar dim.

        Returns number of source domains transferred from, or 0 if none.
        """
        c = self.conn.cursor()

        # Find the best-scoring run from ANY domain with matching output_dim
        # by checking the generator weights shape
        c.execute("""
            SELECT g.run_id, g.domain, g.weights_blob, g.fitness_blob,
                   r.best_score
            FROM generators g
            JOIN runs r ON g.run_id = r.run_id
            WHERE g.domain != ?
            ORDER BY r.best_score DESC
            LIMIT 10
        """, (domain,))
        rows = c.fetchall()

        transferred = 0
        for row in rows:
            run_id, src_domain, weights_blob, fitness_blob, best_score = row
            try:
                weights = pickle.loads(weights_blob)
                # Check if output_dim matches (W2 last dim = output_dim)
                w2_key = [k for k in weights if k.endswith("W2") or "W2" in k]
                if not w2_key:
                    continue
                src_out_dim = weights[w2_key[0]].shape[-1]
                if src_out_dim != output_dim:
                    continue

                # Load weights — this transfers learned parameter distribution
                with torch.no_grad():
                    for name, p in batched_gen.named_parameters():
                        if name in weights:
                            src = torch.from_numpy(weights[name])
                            if src.shape == p.shape:
                                p.copy_(src.to(p.device))
                    fitness = pickle.loads(fitness_blob)
                    if fitness.shape == batched_gen.fitness_ema.shape:
                        batched_gen.fitness_ema.copy_(
                            torch.from_numpy(fitness).to(batched_gen.fitness_ema.device))

                # Add small noise to break symmetry (domain-specific adaptation)
                    for p in batched_gen.parameters():
                        p.add_(torch.randn_like(p) * 0.05)

                transferred += 1
                # Only transfer from the best source (first match)
                break
            except Exception as e:
                print(f"  [ForgeEvolve] Cross-domain transfer failed: {e}")
                continue

        return transferred

    def list_runs(self, domain: str | None = None) -> list[dict]:
        """List all runs, optionally filtered by domain."""
        c = self.conn.cursor()
        if domain:
            c.execute("""
                SELECT run_id, domain, generations, total_evals,
                       discoveries, best_score, device, end_time
                FROM runs WHERE domain=? ORDER BY end_time DESC
            """, (domain,))
        else:
            c.execute("""
                SELECT run_id, domain, generations, total_evals,
                       discoveries, best_score, device, end_time
                FROM runs ORDER BY end_time DESC
            """)
        rows = c.fetchall()
        return [{
            "run_id": r[0], "domain": r[1], "generations": r[2],
            "total_evals": r[3], "discoveries": r[4], "best_score": r[5],
            "device": r[6], "end_time": r[7],
        } for r in rows]

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        if hasattr(self, 'conn'):
            self.close()

