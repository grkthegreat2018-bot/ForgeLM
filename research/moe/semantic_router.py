"""Semantic embedding-based router replacing keyword-matching ExpertRouter.

Uses the model's hidden states (or logits as fallback) to compute mean-pooled
embeddings of topic descriptions and queries, then routes by cosine similarity.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    from loguru import logger as _log
except ImportError:
    import logging
    _log = logging.getLogger(__name__)

DEFAULT_TOPIC_DESCRIPTIONS = {
    "python_algorithms": "Python programming algorithms: sorting, searching, recursion, dynamic programming, data structures, fibonacci, factorial, prime numbers, graph traversal",
    "math_arithmetic": "Mathematical arithmetic: calculations, number theory, factorials, GCD, LCM, prime factorization, modular arithmetic, basic algebra",
    "python_strings": "Python string manipulation: reversing, parsing, pattern matching, regex, text processing, character counting, palindromes",
    "python_general": "General Python programming: functions, loops, conditionals, lists, dictionaries, file I/O, error handling, object-oriented programming",
    "python_file_io": "Python file input/output: reading files, writing files, CSV processing, JSON handling, file parsing",
    "python_math": "Python mathematical computing: numpy, calculations, numerical methods, statistics, linear algebra",
    "python_oop": "Python object-oriented programming: classes, inheritance, polymorphism, encapsulation, design patterns",
    "coding": "Software development: code generation, debugging, refactoring, testing, algorithms, data structures, programming languages",
    "math": "Mathematics: algebra, calculus, geometry, statistics, probability, number theory, proofs, equations",
    "algorithms": "Algorithm design and analysis: complexity, sorting, searching, graph algorithms, dynamic programming, greedy algorithms",
    "theory": "Theoretical reasoning: explanation, analysis, logic, proofs, conceptual understanding, critical thinking",
    "creativity": "Creative writing: stories, poems, brainstorming, imaginative text, narrative generation",
    "tool_use": "Tool usage and function calling: API calls, command execution, tool selection, automation, scripting",
    "token_efficiency": "Concise efficient responses: brief answers, minimal tokens, clear and direct communication",
    "general": "General knowledge and assistance: questions, explanations, help, advice, information",
}

MAX_LEN = 128
LOW_CONF_THRESHOLD = 0.3


class SemanticRouter:
    def __init__(self, model, tokenizer, topic_descriptions: dict[str, str], device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.topic_embeddings: dict[str, torch.Tensor] = {}
        self._supports_hidden = self._check_hidden_support()
        self._register_batch(topic_descriptions)

    def _check_hidden_support(self) -> bool:
        """Probe whether forward returns hidden states alongside logits."""
        try:
            self.model.eval()
            dummy = torch.zeros((1, 2), dtype=torch.long, device=self.device)
            with torch.no_grad():
                out = self.model(dummy)
            if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
                return True
        except Exception as e:
            _log.warning(f"[SemanticRouter] hidden-state probe failed ({e}); using logits fallback")
        return False

    def _embed(self, texts: list[str]) -> torch.Tensor:
        """Compute mean-pooled embeddings for a batch of texts."""
        self.model.eval()
        enc = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN
        )
        input_ids = enc["input_ids"].to(self.device)
        with torch.no_grad():
            out = self.model(input_ids)
        if self._supports_hidden and isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
            hidden = out[1]
            if isinstance(hidden, (list, tuple)):
                hidden = hidden[-1]
            mask = enc.get("attention_mask")
            if mask is not None:
                mask = mask.to(self.device).unsqueeze(-1).float()
                hidden = hidden * mask
                emb = hidden.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            else:
                emb = hidden.mean(dim=1)
        else:
            logits = out[0] if isinstance(out, tuple) else out
            emb = logits.float().mean(dim=1)
        return emb.float()

    def _register_batch(self, topic_descriptions: dict[str, str]) -> None:
        if not topic_descriptions:
            return
        topics = list(topic_descriptions.keys())
        descs = [topic_descriptions[t] for t in topics]
        embs = self._embed(descs)
        for t, e in zip(topics, embs):
            self.topic_embeddings[t] = e.detach()

    def register_topic(self, topic: str, description: str) -> None:
        self.topic_embeddings[topic] = self._embed([description])[0].detach()

    def _similarities(self, query: str) -> dict[str, float]:
        q = self._embed([query])[0]
        sims = {}
        for t, e in self.topic_embeddings.items():
            sims[t] = F.cosine_similarity(q.unsqueeze(0), e.unsqueeze(0)).item()
        return sims

    def classify(self, query: str) -> str:
        sims = self._similarities(query)
        if not sims:
            return "general"
        best_topic, best_sim = max(sims.items(), key=lambda kv: kv[1])
        return best_topic if best_sim >= LOW_CONF_THRESHOLD else "general"

    def classify_multi(self, query: str, top_n: int = 2) -> list[str]:
        sims = self._similarities(query)
        ranked = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)
        return [t for t, _ in ranked[:top_n]]

    def list_topics(self) -> list[str]:
        return sorted(self.topic_embeddings.keys())

    def save(self, path: str) -> None:
        torch.save(
            {"topic_embeddings": {t: e.cpu() for t, e in self.topic_embeddings.items()}},
            path,
        )

    @classmethod
    def load(cls, path: str, model, tokenizer, device: str = "cuda") -> "SemanticRouter":
        data = torch.load(path, map_location=device)
        inst = cls(model, tokenizer, {}, device=device)
        for t, e in data["topic_embeddings"].items():
            inst.topic_embeddings[t] = e.to(device)
        return inst
