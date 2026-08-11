"""Index of model checkpoints + registered ModelConfigs.

Scans research/checkpoints/ for *.safetensors / *.pt and pulls the registry
from research.config.MODEL_CONFIGS without importing torch at scan time.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .status_reader import project_root

logger = logging.getLogger(__name__)

CKPT_EXTS = (".safetensors", ".pt", ".bin", ".gguf")
META_SUFFIXES = (".meta.json", ".json")


def _human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


@dataclass
class ModelEntry:
    name: str
    path: str
    size_bytes: int
    size_label: str
    ext: str
    config_name: Optional[str] = None
    config: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    modified: float = 0.0

    @property
    def is_safetensors(self) -> bool:
        return self.ext == ".safetensors"


@dataclass
class ConfigEntry:
    name: str
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: Optional[int]
    vocab_size: int
    attn_type: str
    ffn_type: str
    max_seq_len: int
    params_label: str


def _estimate_params(cfg: dict) -> str:
    d = cfg.get("d_model", 0)
    L = cfg.get("n_layers", 0)
    V = cfg.get("vocab_size", 0)
    inter = cfg.get("intermediate_size") or (8 * d // 3 if d else 0)
    # rough: embed + L*(qkv + ffn) + head
    embed = V * d
    per_layer = 3 * d * d + 2 * d * inter + d * d  # qkv-ish + ffn
    total = embed + L * per_layer + V * d
    if total <= 0:
        return "—"
    if total >= 1e9:
        return f"{total/1e9:.2f} B"
    if total >= 1e6:
        return f"{total/1e6:.1f} M"
    return f"{total/1e3:.0f} K"


class ModelsIndex:
    """Scans checkpoints + reads config registry lazily."""

    def __init__(self) -> None:
        self._configs: dict[str, dict] = {}
        self._loaded = False
        self._scan_sig: Optional[tuple] = None  # dir-mtime signature of last scan
        self._scan_cache: list[ModelEntry] = []

    def _ensure_configs(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            from research.config import MODEL_CONFIGS  # type: ignore
            for name, cfg in MODEL_CONFIGS.items():
                self._configs[name] = {
                    "d_model": cfg.d_model,
                    "n_layers": cfg.n_layers,
                    "n_heads": cfg.n_heads,
                    "n_kv_heads": cfg.n_kv_heads,
                    "vocab_size": cfg.vocab_size,
                    "attn_type": cfg.attn_type,
                    "ffn_type": cfg.ffn_type,
                    "max_seq_len": cfg.max_seq_len,
                    "intermediate_size": cfg.intermediate_size,
                }
        except Exception as e:
            # Fallback: parse config.py statically so we don't require torch.
            logger.warning("MODEL_CONFIGS import failed, using static parse: %s", e)
            try:
                self._configs = _static_parse_configs()
            except Exception as e2:
                logger.warning("static config parse failed: %s", e2)
                self._configs = {}

    def configs(self) -> list[ConfigEntry]:
        self._ensure_configs()
        out: list[ConfigEntry] = []
        for name, c in sorted(self._configs.items()):
            out.append(ConfigEntry(
                name=name, d_model=c.get("d_model", 0),
                n_layers=c.get("n_layers", 0), n_heads=c.get("n_heads", 0),
                n_kv_heads=c.get("n_kv_heads"),
                vocab_size=c.get("vocab_size", 0),
                attn_type=c.get("attn_type", "—"),
                ffn_type=c.get("ffn_type", "—"),
                max_seq_len=c.get("max_seq_len", 0),
                params_label=_estimate_params(c),
            ))
        return out

    @staticmethod
    def _dir_signature(ckpt_dir: Path) -> Optional[tuple]:
        """Cheap change signature: dir mtime + mtimes of immediate subdirs.

        New runs create subdirectories (bumps ckpt_dir mtime); new files
        inside a run bump that subdir's mtime. One-level scandir stays O(dirs).
        """
        if not ckpt_dir.is_dir():
            return None
        try:
            sig = [ckpt_dir.stat().st_mtime]
            for entry in os.scandir(ckpt_dir):
                if entry.is_dir():
                    sig.append(entry.stat().st_mtime)
            return tuple(sig)
        except OSError as e:
            logger.warning("checkpoint dir scan failed: %s", e)
            return None

    def models(self) -> list[ModelEntry]:
        root = project_root()
        ckpt_dir = root / "research" / "checkpoints"
        out: list[ModelEntry] = []
        if not ckpt_dir.is_dir():
            self._scan_sig = None
            self._scan_cache = []
            return out
        sig = self._dir_signature(ckpt_dir)
        if sig is not None and sig == self._scan_sig:
            return self._scan_cache
        for p in sorted(ckpt_dir.rglob("*")):
            if p.is_dir():
                continue
            if p.suffix.lower() not in CKPT_EXTS:
                continue
            try:
                st = p.stat()
            except Exception as e:
                logger.warning("stat failed for checkpoint %s: %s", p, e)
                continue
            meta: dict = {}
            for suf in META_SUFFIXES:
                mp = Path(str(p) + suf) if suf == ".meta.json" else p.with_suffix(suf)
                if mp.is_file():
                    try:
                        with open(mp, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                        break
                    except Exception as e:
                        logger.warning("failed to parse metadata %s: %s", mp, e)
                        meta = {}
            cfg_name = meta.get("config") or meta.get("config_name")
            cfg = self._configs.get(cfg_name, {}) if cfg_name else {}
            out.append(ModelEntry(
                name=p.name, path=str(p.relative_to(root)).replace("\\", "/"),
                size_bytes=st.st_size, size_label=_human_bytes(st.st_size),
                ext=p.suffix.lower(), config_name=cfg_name, config=cfg, meta=meta,
                modified=st.st_mtime,
            ))
        out.sort(key=lambda m: m.modified, reverse=True)
        self._scan_sig = sig
        self._scan_cache = out
        return out


def _static_parse_configs() -> dict[str, dict]:
    """Best-effort regex parse of research/config.py to avoid importing torch."""
    cfg_path = project_root() / "research" / "config.py"
    if not cfg_path.is_file():
        return {}
    text = cfg_path.read_text(encoding="utf-8")
    out: dict[str, dict] = {}
    # Split on top-level config entries: "name": ModelConfig( ... )
    import re
    pat = re.compile(r'"([^"]+)"\s*:\s*ModelConfig\((.*?)\)\s*,?\s*\n', re.S)
    for m in pat.finditer(text):
        name, body = m.group(1), m.group(2)
        d: dict = {}
        for key in ("d_model", "n_layers", "n_heads", "n_kv_heads", "vocab_size",
                    "intermediate_size", "max_seq_len"):
            km = re.search(rf"\b{key}\s*=\s*([0-9None]+)", body)
            if km:
                v = km.group(1)
                d[key] = None if v == "None" else int(v)
        for key in ("attn_type", "ffn_type"):
            km = re.search(rf'\b{key}\s*=\s*"([^"]+)"', body)
            if km:
                d[key] = km.group(1)
        out[name] = d
    return out
