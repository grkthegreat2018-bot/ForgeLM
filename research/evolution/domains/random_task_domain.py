"""RandomTaskDomain — generates random math/algorithm/logic problems for the
gen model to solve, then scores correctness + focus + speed.

This domain breaks the standard param→config→simulate mold because the
"config" being searched is not domain parameters but the GEN MODEL'S
generation parameters (temperature, max_tokens, top_p, top_k). The search
optimizes how the model is *prompted/sampled*, not what the model is.

Flow per evaluate(config):
  1. Generate a fresh random problem (varying difficulty / ops / distractions)
     and store it on ``self.current_problem``.
  2. Build a prompt from the problem and call ``self._gen_model`` with the
     generation params decoded from ``config``.
  3. Parse the model's answer text + wall-clock time.
  4. Run the matching simulator (random_math/algorithm/logic_simulate) which
     reads ``self.current_problem`` + ``config["answer"]`` and returns raw
     metrics.
  5. Compose the final score via RewardGuard using the domain's JSON spec.

The domain works WITHOUT a gen model set (for testing): a dummy solver
returns "0" (wrong → score 0) so the wiring can be exercised end-to-end.
"""
from __future__ import annotations

import random
import time
from typing import Any, Callable, Optional

import torch

from . import BaseDomain
from ..reward_guard import RewardGuard
from ..simulators import get_simulator
from ..simulators.random_task_sim import _safe_eval


# Generation-parameter ranges (must match the JSON specs).
_GEN_PARAM_RANGES = {
    "temperature": (0.0, 1.5),
    "max_tokens": (10, 200),
    "top_p": (0.1, 1.0),
    "top_k": (1, 100),
}
_GEN_PARAM_ORDER = ["temperature", "max_tokens", "top_p", "top_k"]

# Maps task_kind → (spec name, simulator name, default difficulty range)
_KIND_MAP = {
    "math": ("random_math", "random_math_simulate"),
    "algorithm": ("random_algorithm", "random_algorithm_simulate"),
    "logic": ("random_logic", "random_logic_simulate"),
}

# Distraction sentences prepended to the prompt when enabled.
_DISTRACTIONS = [
    "The sky is blue today.",
    "Bob has 3 apples in his basket.",
    "The train arrives at noon.",
    "Pizza is popular on Fridays.",
    "The weather is nice and warm.",
    "A cat sits on the windowsill.",
    "The library closes at 9pm.",
    "Mountains are tall and cold.",
]


class RandomTaskDomain(BaseDomain):
    """A domain that generates random problems and scores the gen model's answer.

    Args:
        task_kind: "math" | "algorithm" | "logic"
        seed: RNG seed for reproducible problem generation
        difficulty_range: (low, high) inclusive difficulty levels to sample
        include_distractions: whether to inject irrelevant sentences
        distraction_prob: probability a given problem gets distractions
        device: torch device
    """

    def __init__(
        self,
        task_kind: str = "math",
        seed: int = 42,
        difficulty_range: tuple[int, int] = (1, 5),
        include_distractions: bool = True,
        distraction_prob: float = 0.5,
        device=None,
    ):
        super().__init__()
        if task_kind not in _KIND_MAP:
            raise ValueError(f"Unknown task_kind '{task_kind}'. "
                             f"Choose from {list(_KIND_MAP.keys())}")
        self.task_kind = task_kind
        self._spec_name, self._sim_name = _KIND_MAP[task_kind]
        # Lazy import to avoid a circular import with domain_spec (which
        # imports from .domains at its own top level).
        from ..domain_spec import load_spec
        self.spec = load_spec(self._spec_name)
        self._simulator_fn: Callable = get_simulator(self._sim_name)
        self._guard = RewardGuard(self.spec.scoring)
        self._guard._behavioral_names = [b[0] for b in self.spec.behavioral_dims]

        self._rng = random.Random(seed)
        self.difficulty_range = difficulty_range
        self.include_distractions = include_distractions
        self.distraction_prob = distraction_prob

        # Gen model — set externally via set_gen_model(). None → dummy solver.
        self._gen_model: Optional[Any] = None

        # The current problem dict, read by the simulator. Set fresh each
        # evaluate() call.
        self.current_problem: dict[str, Any] = {}

        if device is not None:
            self._device = torch.device(device) if not isinstance(device, torch.device) else device

    # ------------------------------------------------------------------
    # Gen model plumbing
    # ------------------------------------------------------------------

    def set_gen_model(self, llm_gen_model: Any) -> None:
        """Attach a gen model used to answer problems during evaluate().

        The gen model may be:
          - a callable: gen_model(prompt: str, **gen_params) -> str
          - an object with a .generate(prompt: str, **gen_params) -> str method
        """
        self._gen_model = llm_gen_model

    def _call_gen_model(self, prompt: str, gen_params: dict) -> tuple[str, float]:
        """Call the gen model (or dummy solver) and return (answer_text, time_s)."""
        if self._gen_model is None:
            # Dummy solver: always wrong ("0"). Lets wiring be tested without
            # an LLM. Score will be 0 for any non-zero real answer.
            return "0", 0.0
        t0 = time.perf_counter()
        gm = self._gen_model
        if hasattr(gm, "generate") and callable(gm.generate):
            answer = gm.generate(prompt, **gen_params)
        elif callable(gm):
            answer = gm(prompt, **gen_params)
        else:
            raise TypeError("gen_model must be callable or expose .generate()")
        elapsed = time.perf_counter() - t0
        if not isinstance(answer, str):
            answer = str(answer)
        return answer, elapsed

    # ------------------------------------------------------------------
    # BaseDomain interface
    # ------------------------------------------------------------------

    def name(self) -> str:
        return self.spec.name

    def output_dim(self) -> int:
        return len(_GEN_PARAM_ORDER)

    def behavioral_dims(self) -> list[tuple]:
        return [("correctness", 2, 0, 1), ("difficulty", 5, 1, 5)]

    def decode(self, params: torch.Tensor) -> dict[str, Any]:
        """Map [0,1] params → generation-param config dict."""
        p = params.detach().cpu().numpy()
        config = {}
        for i, name in enumerate(_GEN_PARAM_ORDER):
            lo, hi = _GEN_PARAM_RANGES[name]
            v = float(min(max(float(p[i]), 0.0), 1.0))
            if name in ("max_tokens", "top_k"):
                config[name] = int(round(lo + v * (hi - lo)))
            else:
                config[name] = float(lo + v * (hi - lo))
        return config

    def encode(self, config: dict[str, Any]) -> torch.Tensor:
        """Map generation-param config dict → [0,1] params tensor."""
        out = []
        for name in _GEN_PARAM_ORDER:
            lo, hi = _GEN_PARAM_RANGES[name]
            v = float(config.get(name, lo))
            if hi == lo:
                out.append(0.0)
            else:
                out.append(min(max((v - lo) / (hi - lo), 0.0), 1.0))
        return torch.tensor(out, dtype=torch.float32)

    def discrete_choices(self) -> dict[str, list] | None:
        return None

    def seed_configs(self) -> list[dict[str, Any]]:
        # A couple of sensible generation-param seeds to bootstrap the archive.
        return [
            {"temperature": 0.0, "max_tokens": 50, "top_p": 1.0, "top_k": 1},
            {"temperature": 0.3, "max_tokens": 100, "top_p": 0.9, "top_k": 40},
        ]

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, config: dict[str, Any]) -> dict:
        """Generate a random problem, query the gen model, and score it."""
        # 1. Fresh random problem
        problem = self._generate_problem()
        self.current_problem = problem

        # 2. Build prompt + call gen model
        prompt = problem["prompt"]
        gen_params = {k: config.get(k, _GEN_PARAM_RANGES[k][0]) for k in _GEN_PARAM_ORDER}
        answer_text, time_s = self._call_gen_model(prompt, gen_params)
        problem["time_s"] = time_s

        # 3. Run simulator → raw metrics
        sim_config = {"answer": answer_text, "time_s": time_s}
        metrics = self._simulator_fn(sim_config, domain=self)

        # 4. Compose final score via RewardGuard
        result = self._guard.score(config, metrics)
        # Carry the problem into metadata for debugging / DB inspection.
        result["metadata"]["problem"] = problem.get("prompt", "")
        result["metadata"]["real_answer"] = problem.get("real_answer")
        result["metadata"]["model_answer"] = answer_text
        return result

    # ------------------------------------------------------------------
    # Problem generation
    # ------------------------------------------------------------------

    def _generate_problem(self) -> dict[str, Any]:
        """Generate one random problem of this domain's kind."""
        low, high = self.difficulty_range
        difficulty = self._rng.randint(low, high)
        has_distractions = (self.include_distractions
                            and self._rng.random() < self.distraction_prob)
        if self.task_kind == "math":
            base = self._gen_math_problem(difficulty)
        elif self.task_kind == "algorithm":
            base = self._gen_algorithm_problem(difficulty)
        else:  # logic
            base = self._gen_logic_problem(difficulty)
        base["kind"] = self.task_kind
        base["difficulty"] = difficulty
        base["has_distractions"] = has_distractions
        # Inject distractions into the prompt text.
        if has_distractions:
            base["prompt"] = self._inject_distractions(base["prompt"])
        return base

    def _inject_distractions(self, prompt: str) -> str:
        """Prepend 1-3 irrelevant sentences to the prompt."""
        n = self._rng.randint(1, 3)
        chosen = self._rng.sample(_DISTRACTIONS, min(n, len(_DISTRACTIONS)))
        noise = " ".join(chosen) + " "
        return noise + prompt

    # ---- math --------------------------------------------------------

    def _gen_math_problem(self, difficulty: int) -> dict:
        ops = ["+", "-", "*", "/", "%", "**"]
        if difficulty <= 1:
            a = self._rng.randint(1, 20)
            b = self._rng.randint(1, 20)
            op = self._rng.choice(["+", "-", "*"])
            expr = f"{a} {op} {b}"
        elif difficulty == 2:
            terms = [self._rng.randint(1, 20) for _ in range(3)]
            o = [self._rng.choice(["+", "-", "*"]) for _ in range(2)]
            expr = f"{terms[0]} {o[0]} {terms[1]} {o[1]} {terms[2]}"
        elif difficulty == 3:
            terms = [self._rng.randint(1, 15) for _ in range(4)]
            o = [self._rng.choice(["+", "-", "*", "/"]) for _ in range(3)]
            expr = (f"{terms[0]} {o[0]} {terms[1]} {o[1]} {terms[2]} {o[2]} {terms[3]}")
        elif difficulty == 4:
            terms = [self._rng.randint(1, 12) for _ in range(4)]
            o = [self._rng.choice(["+", "-", "*"]) for _ in range(3)]
            # Wrap a sub-expression in parentheses.
            expr = (f"({terms[0]} {o[0]} {terms[1]}) {o[1]} {terms[2]} {o[2]} {terms[3]}")
        else:  # 5
            terms = [self._rng.randint(1, 8) for _ in range(4)]
            o = [self._rng.choice(["+", "-", "*", "**"]) for _ in range(3)]
            expr = (f"({terms[0]} {o[0]} {terms[1]}) {o[1]} ({terms[2]} {o[2]} {terms[3]})")
        real_answer = _safe_eval(expr)
        prompt = f"What is {expr} = ? Reply with just the number."
        return {"prompt": prompt, "expression": expr, "real_answer": real_answer}

    # ---- algorithm ---------------------------------------------------

    def _gen_algorithm_problem(self, difficulty: int) -> dict:
        kind = self._rng.choice(["fibonacci", "sort", "sum_even", "reverse"])
        if kind == "fibonacci":
            n = self._rng.randint(2, max(5, difficulty * 2))
            real = _fib(n)
            prompt = (f"What is the {n}th Fibonacci number? "
                      f"(F(1)=1, F(2)=1) Reply with just the number.")
            return {"prompt": prompt, "algo": "fibonacci", "n": n, "real_answer": float(real)}
        if kind == "sort":
            length = self._rng.randint(3, 4 + difficulty)
            nums = [self._rng.randint(1, 50) for _ in range(length)]
            pos = self._rng.randint(1, length)
            sorted_nums = sorted(nums)
            real = sorted_nums[pos - 1]
            prompt = (f"Sort these numbers: {nums}. "
                      f"What is the {pos}th element (1-indexed)? "
                      f"Reply with just the number.")
            return {"prompt": prompt, "algo": "sort", "nums": nums,
                    "pos": pos, "real_answer": float(real)}
        if kind == "sum_even":
            length = self._rng.randint(4, 4 + difficulty * 2)
            nums = [self._rng.randint(1, 30) for _ in range(length)]
            real = sum(x for x in nums if x % 2 == 0)
            prompt = (f"What is the sum of even numbers in {nums}? "
                      f"Reply with just the number.")
            return {"prompt": prompt, "algo": "sum_even", "nums": nums,
                    "real_answer": float(real)}
        # reverse
        words = ["hello", "world", "python", "forge", "evolve", "matrix",
                 "tensor", "neural", "logic", "random"]
        word = self._rng.choice(words)
        pos = self._rng.randint(1, len(word))
        rev = word[::-1]
        real = ord(rev[pos - 1])
        prompt = (f"Reverse the word '{word}'. "
                  f"What is the {pos}th character (1-indexed)? "
                  f"Reply with its ASCII code (a number).")
        return {"prompt": prompt, "algo": "reverse", "word": word,
                "pos": pos, "real_answer": float(real)}

    # ---- logic -------------------------------------------------------

    def _gen_logic_problem(self, difficulty: int) -> dict:
        kind = self._rng.choice(["transitivity", "syllogism"])
        if kind == "transitivity":
            labels = ["A", "B", "C", "D", "E"]
            n = min(2 + difficulty // 2, len(labels) - 1)
            chain = labels[: n + 1]
            # Build "A > B and B > C and ..." relations.
            rels = " and ".join(f"{chain[i]} > {chain[i+1]}"
                                for i in range(n))
            ask_a, ask_b = chain[0], chain[-1]
            # Always true for a '>' chain: first > last.
            real = 1.0
            prompt = (f"If {rels}, is {ask_a} > {ask_b}? (yes/no) "
                      f"Reply with just yes or no.")
            return {"prompt": prompt, "logic": "transitivity",
                    "real_answer": real}
        # syllogism
        subjects = [("cats", "animals", "Whiskers"),
                    ("dogs", "mammals", "Rex"),
                    ("birds", "creatures", "Tweety"),
                    ("fish", "swimmers", "Nemo")]
        cat, genus, member = self._rng.choice(subjects)
        real = 1.0
        prompt = (f"All {cat} are {genus}. {member} is a {cat[:-1]}. "
                  f"Is {member} a {genus[:-1]}? (yes/no) "
                  f"Reply with just yes or no.")
        return {"prompt": prompt, "logic": "syllogism",
                "real_answer": real}

    # ------------------------------------------------------------------
    # CPU copy for parallel evaluation
    # ------------------------------------------------------------------

    def to_cpu(self) -> "RandomTaskDomain":
        return RandomTaskDomain(
            task_kind=self.task_kind,
            seed=self._rng.randint(0, 1_000_000),
            difficulty_range=self.difficulty_range,
            include_distractions=self.include_distractions,
            distraction_prob=self.distraction_prob,
            device=torch.device("cpu"),
        )


# ---------------------------------------------------------------------------
# Pure helpers (kept at module scope so they're testable in isolation)
# ---------------------------------------------------------------------------

def _fib(n: int) -> int:
    """nth Fibonacci (1-indexed: F(1)=1, F(2)=1)."""
    if n <= 0:
        return 0
    a, b = 1, 1
    for _ in range(2, n):
        a, b = b, a + b
    return b if n > 1 else a
