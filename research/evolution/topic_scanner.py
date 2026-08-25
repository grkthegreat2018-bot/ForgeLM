"""Topic scanner â€” auto-discovers new optimization targets from the codebase.

Scans the ForgeAI codebase for:
  1. Feature flags in activation.py (42 use_* flags)
  2. Config parameters in config.py (ModelConfig fields)
  3. Architecture keys in research/keys/ (89 Key classes)
  4. KV cache strategies in kv_backend.py
  5. Decoding strategies in decoding.py
  6. Quantization modes in forge_engine.py
  7. Scheduler policies in inference/scheduler/
  8. Training optimizers in training/optim/

For each discovered topic, checks if an evolution domain already covers it.
Topics without domains are candidates for new domain creation.

This removes the manual domain creation bottleneck: the system auto-discovers
what can be optimized, and the LLM domain generator creates domains for the
uncovered topics.
"""
from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class OptimizationTopic:
    """A discovered optimization target in the codebase."""
    name: str
    category: str  # "feature_flag", "config_param", "arch_key", "kv_strategy", etc.
    source_file: str
    description: str
    parameters: list[str] = field(default_factory=list)  # tunable param names
    has_domain: bool = False  # whether an evolution domain already covers this
    domain_name: str = ""  # name of covering domain if has_domain=True


class TopicScanner:
    """Scans the codebase for optimization targets.

    Runs at startup to build a catalog of everything that CAN be optimized.
    The LLM domain generator uses this to create domains for uncovered topics.
    """

    def __init__(self, project_root: str):
        self.root = Path(project_root)
        self.topics: list[OptimizationTopic] = []

    def scan_all(self) -> list[OptimizationTopic]:
        """Run all scanners and return the full topic catalog."""
        self.topics = []
        self._scan_feature_flags()
        self._scan_config_params()
        self._scan_arch_keys()
        self._scan_kv_strategies()
        self._scan_decoding_strategies()
        self._scan_quant_modes()
        self._scan_schedulers()
        self._scan_training_optimizers()
        self._mark_covered_topics()
        return self.topics

    def _scan_feature_flags(self):
        """Scan activation.py for use_* feature flags."""
        path = self.root / "research" / "inference" / "activation.py"
        if not path.exists():
            return
        content = path.read_text(encoding="utf-8")
        # Find all use_* boolean fields
        for match in re.finditer(r'(use_\w+)\s*:\s*bool\s*=\s*(\w+)', content):
            name = match.group(1)
            self.topics.append(OptimizationTopic(
                name=name,
                category="feature_flag",
                source_file=str(path),
                description=f"Feature flag: {name} (boolean toggle for inference feature)",
                parameters=[name, "enabled"],
            ))

    def _scan_config_params(self):
        """Scan config.py for ModelConfig numeric/string fields."""
        path = self.root / "research" / "config.py"
        if not path.exists():
            return
        content = path.read_text(encoding="utf-8")
        # Find dataclass fields with defaults (numeric or string)
        for match in re.finditer(
            r'(\w+)\s*:\s*(int|float|str)\s*=\s*([^\n]+)', content
        ):
            name = match.group(1)
            if name.startswith("_"):
                continue
            self.topics.append(OptimizationTopic(
                name=name,
                category="config_param",
                source_file=str(path),
                description=f"ModelConfig parameter: {name}",
                parameters=[name],
            ))

    def _scan_arch_keys(self):
        """Scan research/keys/ for Key classes."""
        keys_dir = self.root / "research" / "keys"
        if not keys_dir.exists():
            return
        for py_file in keys_dir.rglob("*_key.py"):
            content = py_file.read_text(encoding="utf-8")
            # Find class definitions ending in "Key"
            for match in re.finditer(r'class\s+(\w+Key)\b', content):
                name = match.group(1)
                rel_path = str(py_file.relative_to(self.root))
                self.topics.append(OptimizationTopic(
                    name=name,
                    category="arch_key",
                    source_file=rel_path,
                    description=f"Architecture key: {name} (structural optimization)",
                    parameters=["enabled", "rank", "gate_init"],
                ))

    def _scan_kv_strategies(self):
        """Scan kv_backend.py for KV cache strategy names."""
        path = self.root / "research" / "inference" / "kv_backend.py"
        if not path.exists():
            return
        content = path.read_text(encoding="utf-8")
        # Find strategy class names
        for match in re.finditer(r'class\s+(\w+(?:KV|Cache|Eviction)\w*)\b', content):
            name = match.group(1)
            self.topics.append(OptimizationTopic(
                name=name,
                category="kv_strategy",
                source_file=str(path),
                description=f"KV cache strategy: {name}",
                parameters=["budget", "block_size", "compression_ratio"],
            ))

    def _scan_decoding_strategies(self):
        """Scan decoding.py for decoding strategy classes."""
        path = self.root / "research" / "inference" / "decoding.py"
        if not path.exists():
            return
        content = path.read_text(encoding="utf-8")
        for match in re.finditer(r'class\s+(\w*(?:Decoding|Decode|Spec)\w*)\b', content):
            name = match.group(1)
            self.topics.append(OptimizationTopic(
                name=name,
                category="decoding_strategy",
                source_file=str(path),
                description=f"Decoding strategy: {name}",
                parameters=["temperature", "top_p", "top_k", "k"],
            ))

    def _scan_quant_modes(self):
        """Scan quant/ directory for quantization methods."""
        quant_dir = self.root / "research" / "inference" / "quant"
        if not quant_dir.exists():
            return
        for py_file in quant_dir.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            content = py_file.read_text(encoding="utf-8")
            for match in re.finditer(r'class\s+(\w*(?:Quant|Linear)\w*)\b', content):
                name = match.group(1)
                self.topics.append(OptimizationTopic(
                    name=name,
                    category="quant_mode",
                    source_file=str(py_file.relative_to(self.root)),
                    description=f"Quantization method: {name}",
                    parameters=["bits", "group_size", "mode", "alpha"],
                ))

    def _scan_schedulers(self):
        """Scan inference/scheduler/ for scheduler policies."""
        sched_dir = self.root / "research" / "inference" / "scheduler"
        if not sched_dir.exists():
            return
        for py_file in sched_dir.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            content = py_file.read_text(encoding="utf-8")
            for match in re.finditer(r'class\s+(\w+(?:Scheduler|Serve|Route)\w*)\b', content):
                name = match.group(1)
                self.topics.append(OptimizationTopic(
                    name=name,
                    category="scheduler",
                    source_file=str(py_file.relative_to(self.root)),
                    description=f"Scheduling policy: {name}",
                    parameters=["batch_size", "priority", "timeout"],
                ))

    def _scan_training_optimizers(self):
        """Scan training/optim/ for optimizer classes."""
        opt_dir = self.root / "research" / "training" / "optim"
        if not opt_dir.exists():
            return
        for py_file in opt_dir.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            content = py_file.read_text(encoding="utf-8")
            for match in re.finditer(r'class\s+(\w*(?:Optim|Adam|Lion|Muon|Forge)\w*)\b', content):
                name = match.group(1)
                self.topics.append(OptimizationTopic(
                    name=name,
                    category="training_optimizer",
                    source_file=str(py_file.relative_to(self.root)),
                    description=f"Training optimizer: {name}",
                    parameters=["lr", "betas", "weight_decay", "eps"],
                ))

    def _mark_covered_topics(self):
        """Check which topics already have evolution domains."""
        try:
            from research.evolution.domains import DOMAINS
            domain_names = set(DOMAINS.keys())
        except ImportError:
            domain_names = set()

        # Also check domain class names (not just registry keys)
        try:
            from research.evolution.domains import list_domains
            all_domain_names = set(list_domains())
        except ImportError:
            all_domain_names = set()

        for topic in self.topics:
            # Check if any domain name contains the topic name (fuzzy match)
            topic_lower = topic.name.lower().replace("_", "")
            for dname in all_domain_names:
                dname_lower = dname.lower().replace("_", "")
                if topic_lower in dname_lower or dname_lower in topic_lower:
                    topic.has_domain = True
                    topic.domain_name = dname
                    break

    def get_uncovered_topics(self) -> list[OptimizationTopic]:
        """Return topics that don't have evolution domains yet."""
        return [t for t in self.topics if not t.has_domain]

    def get_summary(self) -> dict:
        """Return a summary of the scan results."""
        by_cat = {}
        for t in self.topics:
            by_cat.setdefault(t.category, {"total": 0, "covered": 0, "uncovered": 0})
            by_cat[t.category]["total"] += 1
            if t.has_domain:
                by_cat[t.category]["covered"] += 1
            else:
                by_cat[t.category]["uncovered"] += 1
        return {
            "total_topics": len(self.topics),
            "covered": sum(1 for t in self.topics if t.has_domain),
            "uncovered": sum(1 for t in self.topics if not t.has_domain),
            "by_category": by_cat,
        }

