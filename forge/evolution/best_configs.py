"""Best-config export/import — the live link between ForgeEvolve and
production code (critique F21/NC8).

``python -m forge.evolution --apply-best`` reads ``forge_evolve.db`` and
writes the highest-scoring config per domain to
``forge/evolution/best_configs.json``.  Production entrypoints
(``sft_train.py``) call :func:`apply_to_namespace` to override argparse
defaults for ``# evolution-discovered`` parameters — the frozen comments
become a live file instead.

Safety: only params listed in :data:`SFT_PARAM_MAP` are applied, and only
when the user did not pass the flag explicitly.  Values matching
documented *reverted* promotions (see AGENTS.md "Reverted promotions")
are refused so a stale DB cannot silently re-promote a known-bad value.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Canonical output — versioned inside the package so it travels with the code.
BEST_CONFIGS_PATH = Path(__file__).resolve().parent / "best_configs.json"

# Allowlist mapping argparse dest → (evolution domain, config key, type).
# Only params on this list are ever overridden — everything else in the DB
# stays advisory.  Keep this in sync with the ``# evolution-discovered``
# comments in sft_train.py.
SFT_PARAM_MAP: dict[str, tuple[str, str, type]] = {
    "grad_accum": ("grad_accum_config", "accum_steps", int),
    "sync_freq": ("grad_accum_config", "sync_freq", int),
    "focal_gamma": ("loss_config", "focal_gamma", float),
}

# Reverted-promotion guards (AGENTS.md "Reverted promotions"): conditions
# under which an evolved value must NOT be applied.  Each entry is
# (argparse dest, predicate) — if predicate(value) is True the value is
# refused with a warning.
REVERTED_GUARDS: dict[str, object] = {
    # label_smoothing > 0.2 was a scoring artifact — reverted to 0.1.
    "label_smoothing_eps": lambda v: v > 0.2,
    # warmup_steps == 0 broke training stability — reverted to 500/20.
    "warmup_steps": lambda v: v == 0,
}


def export_best_configs(
    db_path: str = "forge_evolve.db",
    out_path: str | Path | None = None,
    domains: list[str] | None = None,
) -> dict[str, dict]:
    """Read the findings DB and write the best config per domain to JSON.

    Returns the exported mapping ``{domain: {"config": ..., "score": ...}}``.
    """
    from .database import FindingsDB
    from .domain_spec import list_specs
    from .domains import DOMAINS

    out = Path(out_path) if out_path else BEST_CONFIGS_PATH
    db = FindingsDB(db_path)
    try:
        domain_names = domains or sorted(set(list_specs()) | set(DOMAINS.keys()))
        exported: dict[str, dict] = {}
        for name in domain_names:
            best = db.query_best_configs(name, limit=1)
            if not best:
                continue
            row = best[0]
            if row.get("config") is None:
                continue
            exported[name] = {
                "config": row["config"],
                "score": row.get("score"),
            }
    finally:
        close = getattr(db, "close", None)
        if callable(close):
            close()

    payload = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "db_path": str(db_path),
        "domains": exported,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return exported


def load_best_configs(path: str | Path | None = None) -> dict[str, dict]:
    """Load the exported best-configs file. Returns {} if absent/corrupt."""
    p = Path(path) if path else BEST_CONFIGS_PATH
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data.get("domains", {})


def apply_to_namespace(args, defaults: dict, param_map: dict | None = None) -> list[str]:
    """Override ``args`` attributes with evolution-best values.

    Only params in ``param_map`` (default :data:`SFT_PARAM_MAP`) are
    considered, only when the current value still equals the argparse
    default (explicit user flags always win), and only when the value
    passes the :data:`REVERTED_GUARDS` checks.

    Returns a list of human-readable strings describing each override
    applied (for logging).
    """
    param_map = param_map or SFT_PARAM_MAP
    domains = load_best_configs()
    if not domains:
        return []
    applied: list[str] = []
    for dest, (domain, key, typ) in param_map.items():
        entry = domains.get(domain)
        if not entry or key not in (entry.get("config") or {}):
            continue
        current = getattr(args, dest, None)
        if current is None or current != defaults.get(dest):
            continue  # user passed the flag explicitly — never override
        try:
            value = typ(entry["config"][key])
        except (TypeError, ValueError):
            continue
        guard = REVERTED_GUARDS.get(dest)
        if guard is not None and guard(value):
            logger.warning(
                "Evolution value for --%s (%r) matches a reverted promotion; "
                "keeping default %r", dest.replace("_", "-"), value, current)
            continue
        setattr(args, dest, value)
        applied.append(
            f"--{dest.replace('_', '-')}={value} "
            f"(domain={domain}, score={entry.get('score')})")
    return applied
