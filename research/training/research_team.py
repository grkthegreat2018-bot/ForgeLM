"""Autonomous research team — 32 LFM agents self-organize around vague goals.

Agents:
- Self-assign roles based on the goal (no pre-assigned roles)
- Communicate freely via message bus
- Generate training data (JSONL) + documentation (Markdown) continuously
- Run safe sandbox Python scripts
- Search the web for current info
- Save everything to a shared database
- NO ARTIFICIAL CAPS — runs continuously until Ctrl+C

Usage:
    python -m research.training.research_team --goal "best LLM practices 2026"
    python -m research.training.research_team --goals "math discoveries 2026,quantum computing breakthroughs"
    python -m research.training.research_team --list-goals  # show built-in goals
    Ctrl+C to stop — all data saved continuously
"""
import asyncio
import json
import os
import random
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx

from research.paths import LMSTUDIO_API as LMSTUDIO_BASE, PROJECT_ROOT
from research.training.web_scraper import search_all, search_all_parallel

# ── Built-in vague goals ─────────────────────────────────────────────────────

BUILTIN_GOALS = [
    "best LLM training practices: LoRA, QLoRA, fine-tuning tricks",
    "efficient inference techniques: KV cache, quantization, batching",
    "AI agent frameworks: LangGraph, CrewAI, AutoGen comparison",
    "efficient training techniques for small models under 2B params",
    "speculative decoding: EAGLE, Medusa, MTP implementation",
    "tool use and function calling patterns for LLMs",
    "synthetic data generation methods for fine-tuning",
    "mixture of experts architecture for small models",
    "distillation techniques from large to small models",
    "RLHF and DPO training strategies for small models",
    "best coding patterns for AI applications",
    "edge AI and on-device inference optimization",
    "RAG retrieval augmented generation best practices",
    "prompt engineering techniques for code generation",
    "attention mechanism optimization: FlashAttention, GQA, MQA",
]

# ── Message bus ──────────────────────────────────────────────────────────────

@dataclass
class Message:
    sender: str
    recipient: str       # "#3", "all", "team"
    content: str
    msg_type: str        # chat, draft, critique, question, result, tool, doc, role
    timestamp: float = field(default_factory=time.time)


class MessageBus:
    def __init__(self, log_path: Path = None):
        self._messages: list[Message] = []
        self._subscribers: dict[str, asyncio.Queue] = {}
        self.log_file = None
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_file = open(log_path, "w", encoding="utf-8")

    async def publish(self, msg: Message):
        self._messages.append(msg)
        if self.log_file:
            self.log_file.write(json.dumps({
                "sender": msg.sender, "recipient": msg.recipient,
                "type": msg.msg_type, "content": msg.content[:300],
                "ts": msg.timestamp,
            }, ensure_ascii=False) + "\n")
            self.log_file.flush()
        if msg.recipient == "all" or msg.recipient == "team":
            for name, q in self._subscribers.items():
                if name != msg.sender:
                    await q.put(msg)
        elif msg.recipient in self._subscribers:
            await self._subscribers[msg.recipient].put(msg)
        self._print(msg)

    def _print(self, msg: Message):
        colors = {
            "chat": "\033[37m", "draft": "\033[36m", "critique": "\033[33m",
            "question": "\033[35m", "result": "\033[32m", "tool": "\033[34m",
            "doc": "\033[32m", "role": "\033[93m", "search": "\033[34m",
        }
        c = colors.get(msg.msg_type, "\033[37m")
        r = "\033[0m"
        arrow = "->all" if msg.recipient in ("all", "team") else f"->{msg.recipient}"
        content = msg.content[:100].replace("\n", " ")
        print(f"  {c}{msg.sender}{arrow} [{msg.msg_type}]{r} {content}")

    def subscribe(self, name: str) -> asyncio.Queue:
        q = asyncio.Queue()
        self._subscribers[name] = q
        return q

    def history(self, limit=30) -> list[Message]:
        return self._messages[-limit:]


# ── Safe sandbox ─────────────────────────────────────────────────────────────

SAFE_BUILTINS = {
    "len": len, "str": str, "int": int, "float": float, "bool": bool,
    "list": list, "dict": dict, "set": set, "tuple": tuple,
    "range": range, "sum": sum, "min": min, "max": max, "abs": abs,
    "round": round, "sorted": sorted, "reversed": reversed,
    "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
    "any": any, "all": all, "print": print,
    "True": True, "False": False, "None": None,
    "isinstance": isinstance, "type": type, "hash": hash,
    "open": open,
    "math": __import__("math"),
    "json": __import__("json"),
    "statistics": __import__("statistics"),
    "itertools": __import__("itertools"),
    "collections": __import__("collections"),
    "re": __import__("re"),
}

# Whitelisted modules for sandbox import
_SANDBOX_MODULES = {"math", "json", "statistics", "itertools", "collections", "re",
                    "random", "datetime", "decimal", "fractions"}


def _sandbox_import(name, *args, **kwargs):
    """Restricted import — only allow whitelisted modules."""
    if name in _SANDBOX_MODULES:
        return __import__(name, *args, **kwargs)
    raise ImportError(f"Module '{name}' not allowed in sandbox. Use: {', '.join(sorted(_SANDBOX_MODULES))}")


SAFE_BUILTINS["__import__"] = _sandbox_import


def run_sandbox(code: str, timeout: int = 5) -> str:
    """Run Python code in a safe sandbox. Returns output or error."""
    import io, contextlib
    buf = io.StringIO()
    globals_dict = {"__builtins__": SAFE_BUILTINS, "__name__": "__sandbox__"}
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            exec(code, globals_dict)
        return buf.getvalue().strip() or "(no output)"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


# ── Database ─────────────────────────────────────────────────────────────────

class Database:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    async def save(self, record: dict):
        async with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    async def count(self) -> int:
        if not self.path.exists():
            return 0
        return sum(1 for _ in open(self.path, encoding="utf-8"))


# ── Agent ────────────────────────────────────────────────────────────────────

class Agent:
    """Autonomous LFM agent that self-assigns its role."""

    def __init__(self, idx: int, bus: MessageBus, db: Database,
                 docs_dir: Path, lfm_sem: asyncio.Semaphore,
                 scrape_sem: asyncio.Semaphore,
                 model: str = "forgeai/lfm2.5-1.2b",
                 heavy_sem: asyncio.Semaphore = None,
                 heavy_model: str = None):
        self.name = f"#{idx+1}"
        self.idx = idx
        self.bus = bus
        self.db = db
        self.docs_dir = docs_dir
        self.lfm_sem = lfm_sem
        self.scrape_sem = scrape_sem
        self.model = model  # default model for this agent
        self.heavy_sem = heavy_sem  # semaphore for heavy model (Qwen)
        self.heavy_model = heavy_model  # heavy model name
        self.inbox = bus.subscribe(self.name)
        self.seed = random.randint(0, 999999)
        self._call_count = 0
        self.role: Optional[str] = None
        self.role_desc: Optional[str] = None
        self.context: list[dict] = []
        self.client: httpx.AsyncClient = None
        self.goal: str = ""

    async def think(self, prompt: str, temp: float = 0.4,
                    max_tokens: int = 768, use_heavy: bool = False) -> str:
        """Think using either the light model (LFM) or heavy model (Qwen).

        LFM = fast grunt work: search queries, questions, simple answers
        Qwen = heavy reasoning: theorizing, deep analysis, code generation
        """
        messages = self.context[-8:] + [{"role": "user", "content": prompt}]
        call_seed = (self.seed + self._call_count) % 1000000
        self._call_count += 1

        # Route to correct backend
        if use_heavy and self.heavy_model and self.heavy_sem:
            model_name = self.heavy_model
            sem = self.heavy_sem
            timeout = 180.0
            base_url = "http://127.0.0.1:8080/v1"  # llama-server (Qwen)
        else:
            model_name = self.model
            sem = self.lfm_sem
            timeout = 90.0
            base_url = "http://127.0.0.1:1235/v1"  # ForgeAI server (LFM)

        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temp,
            "max_tokens": max_tokens,
            "seed": call_seed,
        }
        async with sem:
            resp = await self.client.post(
                f"{base_url}/chat/completions", json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()

    async def say(self, content: str, recipient: str = "all",
                  msg_type: str = "chat"):
        msg = Message(sender=self.name, recipient=recipient,
                      content=content, msg_type=msg_type)
        await self.bus.publish(msg)
        self.context.append({"role": "assistant", "content": content})

    async def listen(self, timeout: float = 3.0) -> Optional[Message]:
        try:
            return await asyncio.wait_for(self.inbox.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    async def web_search(self, query: str) -> str:
        await self.say(f"Searching: {query}", msg_type="search")
        async with self.scrape_sem:
            results = await asyncio.to_thread(search_all_parallel, query, 3)
        content = "\n\n".join(r.content for r in results if r.content)
        return content[:5000]

    async def save_training(self, q: str, a: str, topic: str, tags: list[str] = None):
        # Trim artifacts from Q&A
        q = self._trim_artifacts(q).strip()
        a = self._trim_artifacts(a).strip()
        # Dedup check — similarity-based clustering (not just exact match)
        if hasattr(self.db, '_questions') and self._is_near_duplicate(q, self.db._questions):
            await self.say(f"Skip near-dup: {q[:50]}...", msg_type="chat")
            return False
        record = {
            "agent": self.name, "role": self.role, "topic": topic,
            "goal": self.goal, "question": q, "answer": a,
            "tags": tags or [], "seed": self.seed, "timestamp": time.time(),
        }
        await self.db.save(record)
        if not hasattr(self.db, '_questions'):
            self.db._questions = set()
        self.db._questions.add(q)
        await self.say(f"Saved training data: {q[:60]}...", msg_type="result")
        return True

    @staticmethod
    def _is_near_duplicate(q: str, existing: set, threshold: float = 0.90) -> bool:
        """Check if question is too similar to any existing one.
        Uses content-word overlap (ignores stop words) with a high threshold
        to only reject near-identical questions, not related ones.
        """
        # Stop words that don't carry meaning — ignore them in comparison
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "what", "how", "why", "when", "where", "which", "who", "whom",
            "do", "does", "did", "can", "could", "should", "would", "will",
            "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
            "and", "or", "but", "not", "no", "if", "then", "that", "this",
            "these", "those", "it", "its", "they", "them", "their", "we",
            "you", "your", "i", "me", "my", "our", "has", "have", "had",
            "been", "about", "into", "than", "so", "such", "very", "more",
            "most", "some", "any", "all", "both", "each", "few", "other",
        }
        q_tokens = set(q.lower().split()) - stop_words
        if len(q_tokens) < 3:
            return False
        for eq in existing:
            eq_tokens = set(eq.lower().split()) - stop_words
            if not eq_tokens:
                continue
            overlap = len(q_tokens & eq_tokens) / max(len(q_tokens), len(eq_tokens))
            if overlap > threshold:
                return True
        return False

    async def save_doc(self, title: str, content: str, topic: str):
        safe_title = title.replace(" ", "_").replace("/", "_")[:60]
        path = self.docs_dir / f"{safe_title}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        header = f"# {title}\n\n*Generated by agent {self.name} ({self.role})*\n*Goal: {self.goal}*\n*Topic: {topic}*\n\n---\n\n"
        # Strip LFM output artifacts
        clean = self._trim_artifacts(content)
        with open(path, "w", encoding="utf-8") as f:
            f.write(header + clean)
        await self.say(f"Saved doc: {safe_title}.md", msg_type="doc")

    @staticmethod
    def _trim_artifacts(text: str) -> str:
        """Remove LFM output artifacts like [End of Response], [End of Document]."""
        import re
        # Remove trailing artifacts
        text = re.sub(r'\s*\[End of (?:Response|Document|output|Output)\]\s*$', '', text)
        text = re.sub(r'\s*\[End of [^\]]*\]\s*$', '', text)
        # Remove inline artifacts
        text = re.sub(r'\n\[End of (?:Response|Document|output|Output)\].*$', '', text, flags=re.MULTILINE)
        # Remove prompt leakage
        text = re.sub(r'^(TRAINING_Q|TRAINING_A|DOC_TITLE|DOC_BODY):.*$', '', text, flags=re.MULTILINE)
        # Remove "Saved training data:" / "Saved doc:" leakage
        text = re.sub(r'^Saved (training data|doc):.*$', '', text, flags=re.MULTILINE)
        # Trim trailing whitespace
        return text.rstrip() + "\n"

    def run_script(self, code: str) -> str:
        return run_sandbox(code)

    async def decide_role(self, goal: str, taken_roles: list[str]) -> str:
        """Self-assign a role based on the goal and what's already taken."""
        taken_str = ", ".join(taken_roles) if taken_roles else "none yet"
        # Suggest diverse roles to prevent all agents picking the same one
        suggestions = [
            "web_researcher", "data_curator", "doc_writer", "fact_checker",
            "code_example_writer", "critic", "synthesizer", "question_generator",
            "analogist", "summarizer", "edge_case_finder", "practitioner",
            "theorist", "debugger", "comparator", "tutorial_writer",
            "benchmark_designer", "safety_auditor", "optimization_expert",
            "architecture_reviewer", "historian", "contrarian", "tester",
            "coder", "verifier", "simplifier", "detailer", "connector",
            "challenger", "reviewer", "interviewer", "antipattern_finder",
            "scalability_analyst", "security_reviewer",
            "simplifier", "detailer", "connector", "challenger", "reviewer",
            "interviewer", "antipattern_finder", "scalability_analyst",
            "security_reviewer",
        ]
        available = [r for r in suggestions if r not in taken_roles]
        suggest_str = ", ".join(available[:8])
        prompt = (
            f"Goal: {goal}\n"
            f"Roles already taken: {taken_str}\n"
            f"Available roles: {suggest_str}\n"
            f"You are agent {self.name}. Pick a role NOT already taken. "
            f"Reply with just: role_name - one sentence description\n"
            f"Example: data_curator - organizes findings into structured datasets"
        )
        response = await self.think(prompt, temp=0.6, max_tokens=80)
        # Parse and force uniqueness
        if " - " in response:
            parts = response.split(" - ", 1)
            self.role = parts[0].strip().lower().replace(" ", "_")
            self.role_desc = parts[1].strip()
        else:
            self.role = response.strip().lower().replace(" ", "_")[:30]
            self.role_desc = response.strip()
        # If role already taken, force-assign from available
        if self.role in taken_roles and available:
            self.role = available[0]
            self.role_desc = f"auto-assigned {self.role}"
        self.goal = goal
        await self.say(f"I'll be: {self.role} - {self.role_desc}", msg_type="role")
        return self.role


# ── Research Team ────────────────────────────────────────────────────────────

class ResearchTeam:
    """32 autonomous agents researching vague goals."""

    def __init__(self, goals: list[str], n_agents: int = 16,
                 lfm_parallel: int = 16, output_dir: Path = None,
                 auto_goals: bool = False,
                 heavy_model: str = None, heavy_parallel: int = 4):
        self.goals = goals
        self.n_agents = n_agents
        self.lfm_parallel = lfm_parallel
        self.auto_goals = auto_goals
        self.heavy_model = heavy_model  # e.g. "qwen2.5-3b-instruct-abliterated"
        self.heavy_parallel = heavy_parallel
        self.goal_pool = list(goals)
        self.output_dir = output_dir or PROJECT_ROOT / "research" / "data" / "research_team"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.docs_dir = self.output_dir / "docs"
        self.docs_dir.mkdir(exist_ok=True)

        ts = int(time.time())
        self.bus = MessageBus(self.output_dir / f"session_{ts}.log")
        self.db = Database(self.output_dir / f"training_data_{ts}.jsonl")
        self.lfm_sem = asyncio.Semaphore(lfm_parallel)
        self.scrape_sem = asyncio.Semaphore(16)  # 5 free providers can handle this
        # Heavy model semaphore — low parallelism to avoid VRAM pressure
        self.heavy_sem = asyncio.Semaphore(heavy_parallel) if heavy_model else None

        self.agents: list[Agent] = [
            Agent(i, self.bus, self.db, self.docs_dir, self.lfm_sem, self.scrape_sem,
                  heavy_sem=self.heavy_sem, heavy_model=self.heavy_model)
            for i in range(n_agents)
        ]

    async def run(self) -> dict:
        """Run continuously — no artificial caps. Ctrl+C to stop.

        Agents loop forever:
        1. Pick a goal (round-robin or self-select)
        2. Research (web search + share)
        3. Generate training data + docs
        4. Cross-review
        5. Back to step 1 with a new angle
        """
        print(f"\n{'='*70}")
        print(f"  RESEARCH TEAM: {self.n_agents} agents (CONTINUOUS MODE)")
        print(f"  ForgeAI LFM2.5-1.2B (batched, port 1235, parallel={self.lfm_parallel})")
        if self.heavy_model:
            print(f"  Qwen2.5-3B (port 8080, parallel={self.heavy_parallel})")
            print(f"    -> Qwen: theorizing, deep analysis, code generation, refinement")
            print(f"    -> LFM:  search, questions, answers, voting, reviews")
        print(f"  Goals: {len(self.goals)}")
        for g in self.goals:
            print(f"    - {g}")
        print(f"  Output: {self.output_dir}")
        print(f"  Press Ctrl+C to stop. Data saved continuously.")
        print(f"{'='*70}\n")

        async with httpx.AsyncClient() as client:
            for a in self.agents:
                a.client = client

            # Phase 1: Assign roles (concurrent — pre-divide from role list)
            print(f"  Phase 1: Assigning roles...", flush=True)
            role_list = [
                "web_researcher", "data_curator", "doc_writer", "fact_checker",
                "code_example_writer", "critic", "synthesizer", "question_generator",
                "analogist", "summarizer", "edge_case_finder", "practitioner",
                "theorist", "debugger", "comparator", "tutorial_writer",
                "benchmark_designer", "safety_auditor", "optimization_expert",
                "architecture_reviewer", "historian", "contrarian", "tester",
                "coder", "verifier", "simplifier", "detailer", "connector",
                "challenger", "reviewer", "interviewer", "antipattern_finder",
            ]
            taken = []
            for i, a in enumerate(self.agents):
                a.role = role_list[i % len(role_list)]
                a.role_desc = f"{a.role} for research team"
                a.goal = self.goals[0] if self.goals else "general knowledge"
                taken.append(a.role)
            print(f"  Roles: {', '.join(taken[:16])}{'...' if len(taken) > 16 else ''}\n")

            # Phase 1b: Generate initial goal pool if agents should self-select
            if self.auto_goals:
                print(f"  Phase 1b: Agents generating training data goals...", flush=True)
                self.goal_pool = await self._generate_goal_pool()
                print(f"  Goal pool: {len(self.goal_pool)} goals\n")
            else:
                self.goal_pool = list(self.goals)

            # Continuous generation loop — adversarial voting architecture
            round_num = 0
            try:
                while True:
                    round_num += 1

                    # Select goal: from pool, refresh every 10 rounds
                    if self.auto_goals and (round_num % 10 == 0):
                        print(f"  Refreshing goal pool...", flush=True)
                        new_goals = await self._generate_goal_pool()
                        self.goal_pool.extend(new_goals)
                        # Keep last 30 goals
                        self.goal_pool = self.goal_pool[-30:]
                        print(f"  Goal pool: {len(self.goal_pool)} goals\n")

                    goal = self.goal_pool[(round_num - 1) % len(self.goal_pool)]

                    # Vote: should this be an R&D round?
                    is_rd_round = await self._vote_rd_round(round_num)

                    if is_rd_round:
                        print(f"\n{'='*70}")
                        print(f"  R&D ROUND {round_num} | Goal: {goal}")
                        print(f"{'='*70}\n")
                        await self._run_rd_round(goal, round_num)
                    else:
                        print(f"\n{'='*70}")
                        print(f"  ROUND {round_num} | Goal: {goal}")
                        print(f"{'='*70}\n")
                        await self._run_adversarial_round(goal, round_num)

                    count = await self.db.count()
                    docs = list(self.docs_dir.glob("*.md"))
                    print(f"\n  [Round {round_num}] Total: {count} training records, {len(docs)} docs\n")

            except (KeyboardInterrupt, asyncio.CancelledError):
                pass
            except Exception as e:
                print(f"\n  ERROR in round {round_num}: {e}")
                import traceback
                traceback.print_exc()
            finally:
                print(f"\n\n{'='*70}")
                print(f"  STOPPING — saving final state...")
                count = await self.db.count()
                docs = list(self.docs_dir.glob("*.md"))
                print(f"  Training data: {count} records in {self.db.path.name}")
                print(f"  Documentation: {len(docs)} files in {self.docs_dir}/")
                print(f"  Log: {self.bus.log_file.name if self.bus.log_file else 'none'}")
                print(f"  Rounds completed: {round_num}")
                print(f"{'='*70}")
                return {"training_count": count, "doc_count": len(docs), "rounds": round_num}

    async def _generate_goal_pool(self) -> list[str]:
        """Agents generate diverse training data goals covering all domains
        needed for a general-purpose LLM.

        Each agent proposes goals from its role's perspective. Goals are
        deduplicated and cover: coding, math, reasoning, knowledge, writing,
        instruction following, summarization, and more.
        """
        # Domain categories for a coding/research/theory focused LLM
        domain_categories = [
            "coding: algorithms, data structures, complexity analysis, optimization",
            "coding: debugging, error handling, edge cases, defensive programming",
            "coding: code review, refactoring, design patterns, clean code",
            "coding: system design, architecture, scalability, distributed systems",
            "coding: Python idioms, async programming, type hints, testing",
            "coding: PyTorch, TensorFlow, CUDA, GPU programming patterns",
            "research: ML model architectures, training techniques, fine-tuning",
            "research: inference optimization, quantization, KV cache, batching",
            "research: evaluation metrics, benchmarking, ablation studies",
            "research: paper analysis, reproducing results, experiment design",
            "research: data pipelines, tokenization, dataset curation",
            "research: RLHF, DPO, alignment, safety in LLMs",
            "theory: linear algebra, matrix operations, eigenvalues, SVD",
            "theory: probability, statistics, Bayesian methods, distributions",
            "theory: calculus, gradients, optimization theory, convergence",
            "theory: information theory, entropy, KL divergence, mutual information",
            "theory: computational complexity, Big-O, NP-completeness, algorithms",
            "theory: logic, formal proofs, type theory, category theory basics",
            "creative coding: code golf, elegant solutions, one-liners, clever tricks",
            "creative reasoning: lateral thinking puzzles, edge case discovery",
            "creative architecture: novel ML architectures, experimental designs",
        ]

        # Each agent generates 2-3 goals from random categories
        import random
        gen_coros = []
        for agent in self.agents:
            cats = random.sample(domain_categories, min(3, len(domain_categories)))
            gen_coros.append(self._agent_generate_goals(agent, cats))

        results = await asyncio.gather(*gen_coros, return_exceptions=True)

        # Collect and dedup goals
        all_goals = []
        seen = set()
        for r in results:
            if isinstance(r, list):
                for g in r:
                    g = g.strip().strip('"').strip("*").strip()
                    if g and len(g) > 10 and not Agent._is_near_duplicate(g, seen):
                        seen.add(g)
                        all_goals.append(g)

        # Ensure minimum diversity — if agents didn't cover enough categories,
        # add fallback goals
        if len(all_goals) < 10:
            fallback = [
                "coding: implement and optimize binary search tree operations in Python",
                "coding: debug and fix common async/await pitfalls in Python",
                "coding: design a scalable distributed cache with consistent hashing",
                "research: compare LoRA vs QLoRA fine-tuning memory and speed trade-offs",
                "research: analyze KV cache quantization impact on inference latency",
                "research: reproduce benchmark results from a recent ML paper",
                "theory: prove that gradient descent converges for convex functions",
                "theory: derive the backpropagation equations for a multi-layer network",
                "theory: analyze time and space complexity of sorting algorithms",
                "theory: explain SVD and its application in dimensionality reduction",
            ]
            for g in fallback:
                if not Agent._is_near_duplicate(g, seen):
                    seen.add(g)
                    all_goals.append(g)

        return all_goals[:20]  # cap at 20 goals

    async def _agent_generate_goals(self, agent: Agent, categories: list[str]) -> list[str]:
        """Agent generates 2-3 specific training data goals from given categories."""
        cats_text = "\n".join(f"- {c}" for c in categories)
        prompt = (
            f"You are {agent.name}, a {agent.role} on a data generation team.\n"
            f"We're building training data for a general-purpose LLM.\n\n"
            f"Your assigned categories:\n{cats_text}\n\n"
            f"Generate 2-3 SPECIFIC training data goals from your categories.\n"
            f"Each goal should be a topic that produces diverse Q&A pairs.\n"
            f"Format: 'category: specific topic description'\n"
            f"Examples:\n"
            f"  coding: implement binary search tree with delete operation\n"
            f"  math: solve quadratic equations using the quadratic formula\n"
            f"  reasoning: logical deduction puzzles with premises and conclusions\n\n"
            f"Reply with ONE goal per line. No numbering, no bullets."
        )
        try:
            resp = await agent.think(prompt, temp=0.8, max_tokens=256)
            resp = agent._trim_artifacts(resp).strip()
            goals = [g.strip() for g in resp.split("\n") if g.strip() and len(g.strip()) > 10]
            await agent.say(f"Generated {len(goals)} goals", msg_type="chat")
            return goals[:3]
        except Exception:
            return []

    async def _vote_rd_round(self, round_num: int) -> bool:
        """Decide whether the next round should be an R&D round.

        Every 3rd round is auto R&D. Otherwise, use a weighted random
        choice (30% R&D, 70% normal) since the 1.2B model always votes yes.
        """
        # Every 3rd round is automatically R&D
        if round_num > 0 and round_num % 3 == 0:
            print(f"  Vote: Auto R&D round (every 3rd round)")
            return True

        # Weighted random — the 1.2B model can't vote meaningfully
        import random
        decision = random.random() < 0.3
        print(f"  Vote: {'R&D' if decision else 'normal'} round (weighted random)")
        return decision

    async def _agent_vote_rd(self, agent: Agent, round_num: int) -> bool:
        """Single agent votes on whether to do an R&D round."""
        import random
        # Add random context to break the always-yes bias
        last_mode = random.choice(["R&D", "normal", "normal", "normal"])
        prompt = (
            f"You are {agent.name}, a {agent.role} on a research team.\n"
            f"Round {round_num} is about to start. Last round was {last_mode}.\n\n"
            f"R&D round: deep analysis, script testing, fewer records.\n"
            f"Normal round: broad questions, voting, more records.\n\n"
            f"Which mode would be more useful right now? Reply 'R&D' or 'normal'."
        )
        try:
            resp = await agent.think(prompt, temp=0.7, max_tokens=16)
            return "r&d" in resp.lower() or "rd" in resp.lower()
        except Exception:
            return False

    async def _run_rd_round(self, goal: str, round_num: int):
        """R&D round: deep collaborative thinking + script testing.

        Structure:
        1. All agents search web for the deepest technical content on the goal
        2. Each agent writes a deep analysis (long-form, 1000+ words)
        3. Coder/verifier agents write and run verification scripts
        4. Agents review and build on each other's analyses
        5. Best analyses are saved as training data with full reasoning chains
        """
        import random

        # Phase 1: Deep web research — fetch full papers
        print(f"  R&D Phase 1: Deep web research (full papers)...", flush=True)
        research_coros = [self._agent_deep_research(a, goal, round_num) for a in self.agents]
        results = await asyncio.gather(*research_coros, return_exceptions=True)

        # Collect content, pick the richest
        all_content = []
        for i, r in enumerate(results):
            if isinstance(r, str) and len(r) > 100:
                all_content.append((self.agents[i], r))

        # Share the best content with all agents
        if all_content:
            all_content.sort(key=lambda x: len(x[1]), reverse=True)
            best_content = all_content[0][1][:4000]
            print(f"  R&D Phase 1: Best content: {len(best_content)} chars from {all_content[0][0].name}")
        else:
            best_content = ""
            print(f"  R&D Phase 1: No content fetched, using model knowledge")

        # Phase 2: Each agent writes a deep analysis
        print(f"  R&D Phase 2: Agents writing deep analyses...", flush=True)
        analysis_coros = []
        for agent in self.agents:
            analysis_coros.append(self._agent_deep_analysis(agent, goal, best_content, round_num))
        analyses_raw = await asyncio.gather(*analysis_coros, return_exceptions=True)

        analyses = []
        for i, a in enumerate(analyses_raw):
            if isinstance(a, tuple) and a[1] and len(a[1]) > 200:
                analyses.append((self.agents[i], a[0], a[1]))  # (agent, question, analysis)

        print(f"  R&D Phase 2: {len(analyses)} deep analyses written")

        # Phase 3: Coder/verifier agents write and run verification scripts
        print(f"  R&D Phase 3: Script testing and verification...", flush=True)
        coder_agents = [a for a in self.agents if a.role in
                        ("coder", "verifier", "tester", "debugger",
                         "benchmark_designer", "fact_checker")]
        script_coros = []
        for agent, question, analysis in analyses[:len(coder_agents)]:
            coder = coder_agents[len(script_coros) % len(coder_agents)]
            script_coros.append(self._agent_rd_script_test(coder, question, analysis, goal, round_num))

        script_results = await asyncio.gather(*script_coros, return_exceptions=True)
        verified_count = sum(1 for s in script_results if isinstance(s, str) and len(s) > 50)
        print(f"  R&D Phase 3: {verified_count} scripts tested")

        # Phase 4: Agents review and build on each other's analyses
        print(f"  R&D Phase 4: Collaborative review and extension...", flush=True)
        review_coros = []
        for i, (agent, question, analysis) in enumerate(analyses):
            # Pick 2 other agents to review and extend
            reviewers = [a for a in self.agents if a.name != agent.name]
            reviewers = random.sample(reviewers, min(2, len(reviewers)))
            for reviewer in reviewers:
                review_coros.append(self._agent_rd_review(
                    reviewer, question, analysis, goal, round_num))

        reviews_raw = await asyncio.gather(*review_coros, return_exceptions=True)

        # Phase 5: Save the best analyses + reviews as training data
        print(f"  R&D Phase 5: Saving R&D training data...", flush=True)
        save_coros = []
        for i, (agent, question, analysis) in enumerate(analyses):
            # Collect reviews for this analysis
            analysis_reviews = []
            idx = 0
            for j, (a2, q2, an2) in enumerate(analyses):
                if j == i:
                    for _ in range(2):
                        if idx < len(reviews_raw) and isinstance(reviews_raw[idx], str):
                            analysis_reviews.append(reviews_raw[idx])
                        idx += 1
                elif j < i:
                    idx += 2  # skip reviews for earlier analyses

            # Build final training record: question + analysis + reviews + script
            final_answer = analysis
            if analysis_reviews:
                final_answer += "\n\n--- Peer Review ---\n"
                for r in analysis_reviews:
                    final_answer += f"\n{r[:1000]}\n"

            # Add script results if available
            if i < len(script_results) and isinstance(script_results[i], str) and len(script_results[i]) > 50:
                final_answer += f"\n\n--- Verification Script ---\n{script_results[i]}"

            save_coros.append(agent.save_training(
                question, final_answer, goal,
                tags=[agent.role, f"round{round_num}", "rd", "collaborative"]))

        await asyncio.gather(*save_coros, return_exceptions=True)

        # Save a collaborative doc
        if analyses:
            top_analysis = max(analyses, key=lambda x: len(x[2]))
            agent, question, analysis = top_analysis
            await agent.save_doc(
                f"R&D_Deep_Analysis_R{round_num}",
                f"# R&D Round {round_num}: {goal}\n\n"
                f"## Key Question\n\n{question}\n\n"
                f"## Deep Analysis (by {agent.name})\n\n{analysis}\n",
                goal)

    async def _agent_deep_research(self, agent: Agent, goal: str, round_num: int) -> str:
        """Agent does deep web research — fetches full papers, not just snippets.

        Each agent gets a role-specific sub-topic to ensure diversity.
        """
        agent.goal = goal
        # Role-specific angle to force diverse searches
        role_angles = {
            "web_researcher": "latest research papers and benchmarks",
            "data_curator": "dataset quality and curation strategies",
            "doc_writer": "documentation and tutorials",
            "fact_checker": "common misconceptions and corrections",
            "code_example_writer": "code implementation examples",
            "critic": "limitations and failure modes",
            "synthesizer": "combining multiple techniques together",
            "question_generator": "edge cases and unusual scenarios",
            "analogist": "comparisons with other fields",
            "summarizer": "key findings and takeaways",
            "edge_case_finder": "edge cases and boundary conditions",
            "practitioner": "real-world deployment experiences",
            "theorist": "theoretical foundations and math",
            "debugger": "debugging and troubleshooting",
            "comparator": "comparison of different approaches",
            "tutorial_writer": "step-by-step tutorial",
            "benchmark_designer": "benchmark methodology and metrics",
            "safety_auditor": "safety and robustness concerns",
            "optimization_expert": "performance optimization techniques",
            "architecture_reviewer": "architecture design patterns",
            "historian": "historical context and evolution",
            "contrarian": "counterarguments and alternatives",
            "tester": "testing strategies and validation",
            "coder": "production code patterns",
            "verifier": "verification and validation methods",
            "simplifier": "minimal viable implementation",
            "detailer": "low-level implementation details",
            "connector": "connections to other domains",
            "challenger": "challenging assumptions",
            "reviewer": "peer review and critique",
            "interviewer": "interview-style deep questions",
            "antipattern_finder": "anti-patterns and common mistakes",
        }
        angle = role_angles.get(agent.role, "general overview")
        prompt = (
            f"Goal: {goal}\n"
            f"You are {agent.name}, a {agent.role}. R&D Round {round_num}.\n"
            f"Your specific angle: {angle}\n"
            f"Generate a search query focused on YOUR ANGLE.\n"
            f"Reply with ONLY the search query, no quotes."
        )
        try:
            query = await agent.think(prompt, temp=0.7, max_tokens=64)
            query = query.strip().strip('"').strip("'")
            # Use ar5iv for full papers + other providers
            async with agent.scrape_sem:
                content = await asyncio.to_thread(search_all_parallel, query, 3)
            combined = "\n\n".join(r.content for r in content if r.content)
            await agent.say(f"Deep research [{agent.role}]: {len(combined)} chars", msg_type="search")
            return combined[:6000]
        except Exception as e:
            await agent.say(f"Deep research failed: {str(e)[:50]}", msg_type="chat")
            return ""

    async def _agent_deep_analysis(self, agent: Agent, goal: str,
                                   content: str, round_num: int) -> tuple:
        """Agent writes a deep long-form analysis (aim for 1000+ words)."""
        content_section = f"Research content:\n{content[:3000]}\n\n" if content else ""
        prompt = (
            f"Goal: {goal}\n"
            f"You are {agent.name}, a {agent.role}. R&D Round {round_num}.\n\n"
            f"{content_section}"
            f"Write a DEEP TECHNICAL ANALYSIS on one specific aspect of this goal.\n"
            f"Requirements:\n"
            f"- 800+ words of detailed technical content\n"
            f"- Explain HOW things work at a mechanism level\n"
            f"- Explain WHY certain approaches are better\n"
            f"- Include specific numbers, parameters, and trade-offs\n"
            f"- Include code examples where relevant (real APIs only)\n"
            f"- NO quantum computing, NO sci-fi, NO hallucinated APIs\n"
            f"- Use only real APIs: PyTorch, HuggingFace Transformers, numpy, etc.\n"
            f"- Do NOT invent library names that don't exist\n"
            f"- Ground your analysis in the research content\n"
            f"- Do NOT write a title, heading, 'Line 1:', or '###'\n"
            f"- Do NOT start with 'Certainly', 'Sure', 'Let me', 'Here is', 'The task at hand'\n"
            f"- Do NOT include word counts or meta-commentary\n\n"
            f"Start with a specific technical question (ending with ?).\n"
            f"Then write your full analysis below it.\n"
        )
        try:
            # Use heavy model (Qwen) for deep analysis — it's better at reasoning
            response = await agent.think(prompt, temp=0.5, max_tokens=2048, use_heavy=True)
            response = agent._trim_artifacts(response).strip()
            # Strip common preambles the 1.2B model adds
            preambles = [
                "Certainly! Let's dive into", "Certainly! Let's focus on",
                "Certainly! Let's", "Sure! Let's", "Let me", "I'll",
                "Here is", "Here's", "**Deep Technical Analysis**",
                "The task at hand is to", "The task at hand",
                "### Technical Analysis", "### Deep Technical",
                "Deep Technical Analysis:", "Title:",
                "Certainly! Here's", "Sure! Here's",
            ]
            for pre in preambles:
                if response.lower().startswith(pre.lower()):
                    response = response[len(pre):].lstrip(": *#\n ").strip()
            # Strip markdown headers at start
            import re as _re
            response = _re.sub(r'^#{1,4}\s+.*?\n', '', response).strip()
            # Strip trailing word counts and filler
            response = _re.sub(r'\n*\(Word count:.*?\)\s*$', '', response, flags=_re.DOTALL)
            response = _re.sub(r'\n*Let me know if.*$', '', response, flags=_re.DOTALL)
            # Find the first question mark — that's the question
            qm_idx = response.find("?")
            if qm_idx > 0 and qm_idx < 300:
                # Find start of the question sentence
                before_q = response[:qm_idx]
                # Look for sentence start (after newline or start)
                last_break = max(before_q.rfind("\n"), before_q.rfind(". "), 0)
                question = before_q[last_break:].strip().strip("*").strip('"') + "?"
                analysis = response[qm_idx + 1:].strip()
            else:
                # No question found, split by first line
                lines = response.split("\n", 1)
                question = lines[0].strip().strip('"')[:200]
                analysis = lines[1].strip() if len(lines) > 1 else response
            # Clean question
            question = question.strip("*").strip()
            if not question.endswith("?"):
                question += "?"
            await agent.say(f"Analysis: {len(analysis)} chars on '{question[:40]}'", msg_type="draft")
            return (question, analysis)
        except Exception as e:
            await agent.say(f"Analysis failed: {str(e)[:50]}", msg_type="chat")
            return (None, None)

    async def _agent_rd_script_test(self, agent: Agent, question: str,
                                    analysis: str, goal: str, round_num: int) -> str:
        """Coder agent writes and runs a verification script for the analysis."""
        prompt = (
            f"You are {agent.name}, a {agent.role}. R&D Round {round_num}.\n\n"
            f"Question: {question}\n"
            f"Analysis to verify:\n{analysis[:1500]}\n\n"
            f"Write a Python script (max 20 lines) to verify or demonstrate a key claim.\n"
            f"Use only: math, json, statistics, itertools, collections, re, random.\n"
            f"No numpy, no torch, no external packages.\n"
            f"Reply with ONLY the Python code."
        )
        try:
            # Use heavy model for code generation — Qwen2.5 is much better at code
            code = await agent.think(prompt, temp=0.3, max_tokens=256, use_heavy=True)
            code = code.strip()
            if code.startswith("```"):
                code = code.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            await agent.say(f"Running R&D script...", msg_type="tool")
            output = agent.run_script(code)
            await agent.say(f"Script output: {output[:80]}", msg_type="tool")

            # Interpret results
            interp_prompt = (
                f"Script:\n{code[:400]}\n\nOutput:\n{output[:300]}\n\n"
                f"Does this confirm or refute the analysis? Explain in 2-3 sentences."
            )
            interpretation = await agent.think(interp_prompt, temp=0.3, max_tokens=256)
            await agent.say(f"Verified: {interpretation[:60]}", msg_type="result")

            return f"```python\n{code}\n```\n\nOutput:\n```\n{output}\n```\n\nAnalysis: {interpretation}"
        except Exception as e:
            await agent.say(f"Script test failed: {str(e)[:50]}", msg_type="chat")
            return ""

    async def _agent_rd_review(self, agent: Agent, question: str,
                               analysis: str, goal: str, round_num: int) -> str:
        """Agent reviews and extends a teammate's analysis."""
        prompt = (
            f"You are {agent.name}, a {agent.role}. R&D Round {round_num}.\n\n"
            f"Question: {question}\n"
            f"Teammate's analysis:\n{analysis[:2000]}\n\n"
            f"Review this analysis and EXTEND it with:\n"
            f"- What did they miss or get wrong?\n"
            f"- What additional technical details can you add?\n"
            f"- What are the practical implications?\n"
            f"Write 200-400 words of extension. Be specific and technical."
        )
        try:
            review = await agent.think(prompt, temp=0.4, max_tokens=512)
            review = agent._trim_artifacts(review).strip()
            await agent.say(f"Review: {len(review)} chars", msg_type="critique")
            return review
        except Exception:
            return ""

    async def _run_adversarial_round(self, goal: str, round_num: int):
        """5-phase adversarial round:
        1. All agents search web + generate questions
        2. Dedup questions via clustering
        2.5. Evol-Instruct: mutate questions to increase complexity/diversity
        3. 3 agents answer each question independently (with diverse styles)
        4. Judge agents vote on best answer
        5. Winner gets refined via reflection, theorized, code verified
        """
        import random

        # Phase 1: All agents search web and generate a question
        print(f"  Phase 1: Agents researching and generating questions...")
        research_coros = [self._agent_research_question(a, goal, round_num) for a in self.agents]
        results = await asyncio.gather(*research_coros, return_exceptions=True)

        # Collect (agent, question, content) tuples
        questions = []
        for i, r in enumerate(results):
            if isinstance(r, tuple) and r[0]:
                questions.append((self.agents[i], r[0], r[1]))

        # Phase 2: Dedup questions
        unique_questions = []
        seen = set()
        for agent, q, content in questions:
            if not Agent._is_near_duplicate(q, seen):
                seen.add(q)
                unique_questions.append((agent, q, content))

        print(f"  Phase 2: {len(questions)} questions -> {len(unique_questions)} unique")
        if not unique_questions:
            return

        # Phase 2.5: Evol-Instruct — mutate some questions to increase complexity
        print(f"  Phase 2.5: Evolving questions for complexity...")
        evolve_coros = []
        for agent, q, content in unique_questions[:16]:  # evolve first 16
            evolver = random.choice(self.agents)
            evolve_coros.append(self._agent_evolve_question(evolver, q, content, goal, round_num))
        evolved = await asyncio.gather(*evolve_coros, return_exceptions=True)

        # Add evolved questions (if not near-dup)
        for e in evolved:
            if isinstance(e, tuple) and e[0] and not Agent._is_near_duplicate(e[0], seen):
                seen.add(e[0])
                unique_questions.append((e[1] if len(e) > 1 else unique_questions[0][0], e[0], e[2] if len(e) > 2 else ""))

        print(f"  Phase 2.5: +{len([e for e in evolved if isinstance(e, tuple) and e[0]])} evolved questions")

        # Phase 3: 3 agents answer each question independently (with diverse styles)
        print(f"  Phase 3: {len(unique_questions)} questions x 3 answers each...")
        answer_coros = []
        for agent, q, content in unique_questions:
            answerers = [a for a in self.agents if a.name != agent.name]
            answerers = random.sample(answerers, min(3, len(answerers)))
            for ans_agent in answerers:
                answer_coros.append(self._agent_answer_question(ans_agent, q, content, goal, round_num))

        answers_raw = await asyncio.gather(*answer_coros, return_exceptions=True)

        # Group answers by question
        answers_by_q = {}
        idx = 0
        for agent, q, content in unique_questions:
            answers_by_q[q] = {"content": content, "answers": [], "asker": agent}
            for _ in range(min(3, len([a for a in self.agents if a.name != agent.name]))):
                if idx < len(answers_raw) and not isinstance(answers_raw[idx], Exception):
                    answers_by_q[q]["answers"].append((answers_raw[idx][0], answers_raw[idx][1]))
                idx += 1

        # Phase 4: Judge agents vote on best answer
        print(f"  Phase 4: Judging {len(answers_by_q)} question answer sets...")
        judge_coros = []
        for q, data in answers_by_q.items():
            if len(data["answers"]) >= 2:
                judge = random.choice([a for a in self.agents if a.name != data["asker"].name])
                judge_coros.append(self._agent_judge(judge, q, data, goal, round_num))

        judged = await asyncio.gather(*judge_coros, return_exceptions=True)

        # Phase 5: Save winners + reflect/refine + theorize + code verify
        print(f"  Phase 5: Saving winners, refining, theorizing, verifying...")
        save_coros = []
        for q, data in answers_by_q.items():
            if len(data["answers"]) >= 2:
                winner = None
                for j in judged:
                    if isinstance(j, tuple) and j[0] == q:
                        winner = j[1]
                        break
                if winner is not None:
                    save_coros.append(self._save_winner_with_reflection(
                        data["asker"], q, winner, data["content"], goal, round_num))

        await asyncio.gather(*save_coros, return_exceptions=True)

    async def _agent_evolve_question(self, agent: Agent, question: str,
                                     content: str, goal: str, round_num: int):
        """Evol-Instruct: mutate a question to increase complexity or diversity.

        Mutation types (from Evol-Instruct literature):
        - ADD_CONSTRAINTS: add specific constraints to the question
        - DEEPEN: increase reasoning depth required
        - CONCRETIZE: make the question more specific with real parameters
        - INCREASE_REASONING: require multi-step reasoning
        - SWITCH_TOPIC: change topic while keeping similar difficulty
        """
        import random
        mutation = random.choice(["ADD_CONSTRAINTS", "DEEPEN", "CONCRETIZE",
                                  "INCREASE_REASONING", "SWITCH_TOPIC"])
        mutation_desc = {
            "ADD_CONSTRAINTS": "Add 1-2 specific constraints (e.g., 'on a 12GB GPU', 'with batch size 8')",
            "DEEPEN": "Make the question require deeper analysis (e.g., 'and explain the trade-offs')",
            "CONCRETIZE": "Make it more specific with real numbers/models (e.g., 'for a 1.2B model')",
            "INCREASE_REASONING": "Require multi-step reasoning (e.g., 'compare X and Y, then recommend Z')",
            "SWITCH_TOPIC": "Change to a related but different sub-topic",
        }[mutation]
        prompt = (
            f"Original question: {question}\n\n"
            f"Mutation type: {mutation}\n"
            f"Instruction: {mutation_desc}\n\n"
            f"Create an EVOLVED version of this question that is more complex and specific.\n"
            f"Keep it grounded in real, implementable techniques.\n"
            f"NO quantum computing, NO sci-fi.\n"
            f"Reply with ONLY the evolved question."
        )
        try:
            evolved = await agent.think(prompt, temp=0.6, max_tokens=128)
            evolved = agent._trim_artifacts(evolved).strip().strip('"')
            await agent.say(f"Evolved [{mutation}]: {evolved[:50]}", msg_type="search")
            return (evolved, agent, content)
        except Exception:
            return (None, agent, content)

    async def _save_winner_with_reflection(self, asker: Agent, question: str,
                                           winner: tuple, content: str,
                                           goal: str, round_num: int):
        """Save the winning answer, then reflect and refine it.

        Based on MAMM-Refine and Logical DA:
        1. A reviewer identifies weaknesses in the winning answer
        2. A refiner improves the answer based on the critique
        3. The refined answer is saved (replacing the original)
        4. Then theorize and code-verify the refined answer
        """
        import random
        winner_name, winner_answer = winner
        winner_agent = next((a for a in self.agents if a.name == winner_name), asker)

        # Step 1: Reviewer critiques the winning answer
        reviewer = random.choice([a for a in self.agents if a.name != winner_name])
        critique_prompt = (
            f"Question: {question}\n\n"
            f"Answer to review:\n{winner_answer[:1500]}\n\n"
            f"You are {reviewer.name}, a {reviewer.role}.\n"
            f"Identify 2-3 SPECIFIC weaknesses in this answer:\n"
            f"- Missing technical details?\n"
            f"- Vague claims without evidence?\n"
            f"- Hallucinated APIs or fake numbers?\n"
            f"- Missing code examples?\n"
            f"- Insufficient explanation of HOW/WHY?\n"
            f"Reply with ONLY the critique, 2-3 bullet points."
        )
        try:
            critique = await reviewer.think(critique_prompt, temp=0.3, max_tokens=256)
            await reviewer.say(f"Critique: {critique[:80]}", msg_type="critique")

            # Step 2: Refiner improves the answer based on critique
            refiner = random.choice([a for a in self.agents if a.name != winner_name and a.name != reviewer.name])
            refine_prompt = (
                f"Question: {question}\n\n"
                f"Original answer:\n{winner_answer[:1500]}\n\n"
                f"Critique from reviewer:\n{critique}\n\n"
                f"Write a COMPLETE, STANDALONE answer to the question.\n"
                f"Address the critique but do NOT mention it.\n"
                f"Requirements:\n"
                f"- Start directly with the answer — no preambles, no 'Certainly', no 'Here is'\n"
                f"- Be detailed and specific (aim for 400+ words)\n"
                f"- Include code examples where relevant\n"
                f"- Explain HOW it works and WHY\n"
                f"- Use only real APIs: PyTorch, HuggingFace Transformers, numpy, etc.\n"
                f"- Do NOT invent library names that don't exist\n"
                f"- Do NOT include word counts, critique responses, or meta-commentary\n"
                f"- Do NOT end with 'Let me know' or similar conversational filler\n"
                f"Reply with ONLY the answer text."
            )
            # Use heavy model for refinement — Qwen produces better quality answers
            refined = await refiner.think(refine_prompt, temp=0.4, max_tokens=1024, use_heavy=True)
            refined = refiner._trim_artifacts(refined).strip()
            # Strip common preambles and meta-commentary
            for pre in ["Certainly! Here's", "Certainly! Let's", "Sure! Here's",
                        "Here is the", "Here's the", "Great question!",
                        "Certainly! Here is", "Sure! Let's"]:
                if refined.lower().startswith(pre.lower()):
                    refined = refined[len(pre):].lstrip(": ,").strip()
            # Strip trailing word counts and filler
            import re as _re
            refined = _re.sub(r'\n*\(Word count:.*?\)\s*$', '', refined, flags=_re.DOTALL)
            refined = _re.sub(r'\n*Critique from reviewer:.*$', '', refined, flags=_re.DOTALL)
            refined = _re.sub(r'\n*Let me know if.*$', '', refined, flags=_re.DOTALL)
            refined = refined.strip()
            await refiner.say(f"Refined answer: {len(refined)} chars", msg_type="result")

            # Use refined answer if it's longer and more detailed
            final_answer = refined if len(refined) > len(winner_answer) * 0.8 else winner_answer

            # Save the refined Q&A
            saved = await asker.save_training(question, final_answer, goal,
                                              tags=[asker.role, f"round{round_num}", "voted", "refined"])
            if saved:
                await asker.save_doc(
                    f"Refined_{question[:40]}_R{round_num}",
                    f"## Question\n\n{question}\n\n"
                    f"## Answer (refined by {refiner.name})\n\n{final_answer}\n\n"
                    f"## Critique (by {reviewer.name})\n\n{critique}\n",
                    goal)

                # Theorize the refined answer
                await self._agent_theorize(refiner, goal, round_num)

                # Code verify if coder role
                if refiner.role in ("coder", "verifier", "tester", "debugger",
                                    "benchmark_designer", "fact_checker"):
                    await self._agent_code_verify(refiner, goal, round_num)

        except Exception as e:
            # Fallback: save original winner without refinement
            saved = await asker.save_training(question, winner_answer, goal,
                                              tags=[asker.role, f"round{round_num}", "voted"])
            if saved:
                await asker.save_doc(
                    f"Voted_{question[:40]}_R{round_num}",
                    f"## Question\n\n{question}\n\n## Answer (by {winner_name})\n\n{winner_answer}\n",
                    goal)

    async def _agent_research_question(self, agent: Agent, goal: str, round_num: int):
        """Agent searches web and generates ONE question. Returns (question, content).

        Each agent gets a role-specific angle to ensure diverse search queries.
        Without this, the 1.2B model generates identical queries for all 32 agents.
        """
        agent.goal = goal
        # Role-specific angles to force diverse search queries
        role_angles = {
            "web_researcher": "latest research papers and benchmark results",
            "data_curator": "dataset preparation and tokenization strategies",
            "doc_writer": "official documentation and API references",
            "fact_checker": "common myths vs verified facts",
            "code_example_writer": "working code examples and snippets",
            "critic": "known limitations and failure modes",
            "synthesizer": "combining multiple techniques together",
            "question_generator": "unusual edge cases and corner cases",
            "analogist": "comparisons with similar techniques in other fields",
            "summarizer": "key findings from recent surveys",
            "edge_case_finder": "boundary conditions and extreme cases",
            "practitioner": "real-world deployment logs and post-mortems",
            "theorist": "mathematical foundations and proofs",
            "debugger": "common bugs and troubleshooting steps",
            "comparator": "head-to-head comparison of 2-3 approaches",
            "tutorial_writer": "step-by-step implementation guide",
            "benchmark_designer": "evaluation metrics and methodology",
            "safety_auditor": "safety risks and mitigation strategies",
            "optimization_expert": "speed and memory optimization tricks",
            "architecture_reviewer": "system design and architecture patterns",
            "historian": "how this technique evolved over time",
            "contrarian": "why this approach might be wrong",
            "tester": "testing and validation strategies",
            "coder": "production-grade implementation details",
            "verifier": "how to verify correctness",
            "simplifier": "minimal viable implementation",
            "detailer": "low-level parameter tuning",
            "connector": "connections to other ML domains",
            "challenger": "challenging common assumptions",
            "reviewer": "peer review of published claims",
            "interviewer": "deep dive interview questions",
            "antipattern_finder": "common mistakes and anti-patterns",
        }
        angle = role_angles.get(agent.role, "general technical overview")
        angle_prompt = (
            f"Goal: {goal}\n"
            f"You are {agent.name}, a {agent.role}. Round {round_num}.\n"
            f"YOUR SPECIFIC ANGLE: {angle}\n"
            f"Search for content about this topic from YOUR ANGLE.\n"
            f"CONSTRAINTS:\n"
            f"- Focus on REAL, implementable techniques\n"
            f"- NO quantum computing, NO sci-fi\n"
            f"- Be specific: hyperparameters, code patterns, tricks\n"
            f"Generate a web search query from your angle. Reply with ONLY the query."
        )
        try:
            query = await agent.think(angle_prompt, temp=0.7, max_tokens=64)
            content = await agent.web_search(query.strip().strip('"').strip("'"))

            # Generate a question grounded in the content
            content_section = f"Web content:\n{content[:2000]}\n\n" if content else ""
            q_prompt = (
                f"Goal: {goal}\n"
                f"You are {agent.name}, a {agent.role}.\n"
                f"{content_section}"
                f"Based on the web content above, generate ONE specific technical question.\n"
                f"The question must be grounded in the content, not hallucinated.\n"
                f"NO meta-questions about 'designing Q&A pairs'.\n"
                f"Reply with ONLY the question."
            )
            question = await agent.think(q_prompt, temp=0.4, max_tokens=128)
            question = agent._trim_artifacts(question).strip().strip('"')
            await agent.say(f"Q: {question[:60]}", msg_type="search")
            return (question, content)
        except Exception as e:
            await agent.say(f"Research failed: {str(e)[:60]}", msg_type="chat")
            return (None, None)

    async def _agent_answer_question(self, agent: Agent, question: str,
                                     content: str, goal: str, round_num: int):
        """Agent writes an answer to a question. Returns (agent_name, answer).

        Enforces response diversity: each answerer gets a different style/length/format
        to maximize microscopic diversity (token distribution in responses).
        Research shows this has the STRONGEST correlation with model performance.
        """
        content_section = f"Web content for reference:\n{content[:2000]}\n\n" if content else ""
        import random
        # Diverse formats and styles for microscopic diversity
        fmt = random.choice(["qa", "code_completion", "debugging", "comparison", "tutorial",
                             "analysis", "pros_cons", "step_by_step"])
        style = random.choice(["technical", "conversational", "academic", "practical"])
        length_target = random.choice(["concise (200-300 words)", "detailed (400-600 words)",
                                       "comprehensive (600-1000 words)"])
        fmt_desc = {
            "qa": "direct answer with explanation",
            "code_completion": "complete a code snippet with explanation",
            "debugging": "identify and fix a bug, explain why",
            "comparison": "compare 2-3 approaches with trade-offs table",
            "tutorial": "step-by-step tutorial with code examples",
            "analysis": "deep analysis of the mechanism and trade-offs",
            "pros_cons": "list pros and cons with explanation",
            "step_by_step": "numbered steps with rationale for each",
        }[fmt]
        style_desc = {
            "technical": "use precise terminology, include code snippets",
            "conversational": "explain like talking to a colleague, accessible",
            "academic": "formal tone, cite evidence, structured argument",
            "practical": "focus on real-world implementation, include gotchas",
        }[style]
        prompt = (
            f"Goal: {goal}\n"
            f"You are {agent.name}, a {agent.role}.\n"
            f"Answer this question: {question}\n\n"
            f"{content_section}"
            f"CONSTRAINTS:\n"
            f"- Ground your answer in the web content above\n"
            f"- DO NOT hallucinate APIs — only reference real, known APIs (PyTorch, HF Transformers, etc.)\n"
            f"- Explain HOW it works and WHY\n"
            f"- Include specific numbers, parameters, or code patterns\n"
            f"- Format: {fmt} ({fmt_desc})\n"
            f"- Style: {style} ({style_desc})\n"
            f"- Length: {length_target}\n"
            f"Reply with ONLY your answer."
        )
        try:
            answer = await agent.think(prompt, temp=0.4, max_tokens=1024)
            answer = agent._trim_artifacts(answer).strip()
            return (agent.name, answer)
        except Exception as e:
            return (agent.name, f"Error: {e}")

    async def _agent_judge(self, judge: Agent, question: str, data: dict,
                           goal: str, round_num: int):
        """Judge agent picks the best answer from multiple candidates.

        Uses shuffled letter labels (A, B, C) instead of numbers to avoid
        the small-model recency bias of always picking the last option.
        """
        import random
        # Shuffle answers and label with letters to avoid position bias
        shuffled = list(enumerate(data["answers"]))
        random.shuffle(shuffled)
        answers_text = ""
        label_to_idx = {}
        for label, (orig_idx, (ans_name, ans_text)) in zip(["A", "B", "C"], shuffled):
            answers_text += f"\nAnswer {label} (by {ans_name}):\n{ans_text[:500]}\n"
            label_to_idx[label] = orig_idx

        prompt = (
            f"You are {judge.name}, a {judge.role}. You are judging answers.\n"
            f"Question: {question}\n\n"
            f"Candidate answers:{answers_text}\n\n"
            f"Pick the BEST answer based on:\n"
            f"1. Accuracy — no hallucinated APIs or fake numbers\n"
            f"2. Grounding — uses facts from web content\n"
            f"3. Clarity — explains HOW and WHY\n"
            f"4. Specificity — concrete details, not vague\n"
            f"Reply with ONLY the letter (A, B, or C) of the best answer."
        )
        try:
            choice = await judge.think(prompt, temp=0.3, max_tokens=16)
            choice = choice.strip().strip(".").upper().strip()
            # Extract first letter A/B/C found
            label = None
            for ch in choice:
                if ch in label_to_idx:
                    label = ch
                    break
            if label is None:
                # Fallback: random pick instead of always-first
                label = random.choice(list(label_to_idx.keys()))
            idx = label_to_idx[label]
            winner_name, winner_answer = data["answers"][idx]
            await judge.say(f"Vote: {label} ({winner_name})", msg_type="critique")
            return (question, (winner_name, winner_answer))
        except Exception:
            # Random fallback instead of always-first
            import random as _r
            idx = _r.randint(0, len(data["answers"]) - 1)
            return (question, data["answers"][idx])

    async def _save_winner(self, asker: Agent, question: str, winner: tuple,
                           content: str, goal: str, round_num: int):
        """Save the winning answer, then theorize and code-verify it."""
        winner_name, winner_answer = winner
        # Find the winner agent
        winner_agent = next((a for a in self.agents if a.name == winner_name), asker)

        # Save the winning Q&A
        saved = await asker.save_training(question, winner_answer, goal,
                                          tags=[asker.role, f"round{round_num}", "voted"])
        if saved and winner_answer:
            await asker.save_doc(
                f"Voted_{question[:40]}_R{round_num}",
                f"## Question\n\n{question}\n\n## Answer (by {winner_name})\n\n{winner_answer}\n",
                goal)

            # Theorize
            await self._agent_theorize(winner_agent, goal, round_num)

            # Code verify if coder role
            if winner_agent.role in ("coder", "verifier", "tester", "debugger",
                                      "benchmark_designer", "fact_checker"):
                await self._agent_code_verify(winner_agent, goal, round_num)

    async def _agent_work_cycle(self, agent: Agent, goal: str, round_num: int):
        """One agent's continuous work cycle: research → generate → theorize → review.

        Each round, the agent picks a fresh angle on the goal.
        """
        # Update agent's goal for this round
        agent.goal = goal
        # Research with a fresh angle each round — GROUNDED, practical only
        angle_prompt = (
            f"Goal: {goal}\n"
            f"You are {agent.name}, a {agent.role}. Round {round_num}.\n"
            f"Find a NEW SPECIFIC aspect of this topic.\n"
            f"CONSTRAINTS:\n"
            f"- Focus on REAL, implementable techniques (not theoretical/future)\n"
            f"- NO quantum computing, NO sci-fi, NO hypothetical breakthroughs\n"
            f"- Think: specific hyperparameters, specific code patterns, specific tricks\n"
            f"- DO NOT search for 'best practices' or 'how to design' — search for SPECIFIC techniques\n"
            f"- Example: 'LoRA rank 16 vs 32 memory usage' not 'best LoRA practices'\n"
            f"Generate a web search query for a SPECIFIC technical detail. Reply with ONLY the query."
        )
        try:
            query = await agent.think(angle_prompt, temp=0.5 + (round_num * 0.05) % 0.3, max_tokens=64)
            content = await agent.web_search(query.strip().strip('"'))

            if content:
                # Summarize and share — grounded findings only
                summary = await agent.think(
                    f"Summarize the PRACTICAL, implementable findings from this content in 2 sentences.\n"
                    f"Ignore any theoretical or futuristic claims.\n"
                    f"Content:\n{content[:2000]}",
                    temp=0.3, max_tokens=128)
                await agent.say(f"[R{round_num}] {summary[:150]}", msg_type="chat")

                # Generate training data
                saved = await self._agent_generate(agent, goal, content, round_num)

                # Theorize: refine and explain how/why
                if saved:
                    await self._agent_theorize(agent, goal, round_num)

                    # Coder/verifier roles also run scripts to confirm
                    if agent.role in ("coder", "verifier", "tester", "debugger",
                                      "benchmark_designer", "fact_checker"):
                        await self._agent_code_verify(agent, goal, round_num)

            # Review a teammate's work
            await self._agent_review(agent, goal)

        except Exception as e:
            await agent.say(f"Work cycle error: {str(e)[:80]}", msg_type="chat")

    async def _agent_research(self, agent: Agent, goal: str):
        """Agent searches the web and shares findings with the team."""
        prompt = (
            f"Goal: {goal}\n"
            f"You are {agent.name}, a {agent.role}.\n"
            f"Generate a web search query to find cutting-edge info about this goal. "
            f"Reply with ONLY the search query."
        )
        try:
            query = await agent.think(prompt, temp=0.4, max_tokens=64)
            content = await agent.web_search(query.strip().strip('"'))
            if content:
                # Summarize key findings and share with team
                summary_prompt = (
                    f"You found this web content about '{goal}':\n{content[:3000]}\n\n"
                    f"Summarize the 3 most important findings in 1-2 sentences each. "
                    f"Share with your team."
                )
                summary = await agent.think(summary_prompt, temp=0.3, max_tokens=256)
                await agent.say(f"Findings: {summary[:200]}", msg_type="chat")
        except Exception as e:
            await agent.say(f"Research failed: {e}", msg_type="chat")

    async def _agent_generate(self, agent: Agent, goal: str,
                              content: str = None, round_num: int = 1) -> bool:
        """Agent generates training data and/or documentation."""
        content_section = f"Web content (use these facts, cite them):\n{content[:3000]}\n\n" if content else ""
        # Show existing questions so agent avoids duplicates
        existing_qs = list(getattr(self.db, '_questions', set()))[-10:]
        existing_section = ""
        if existing_qs:
            existing_section = (
                f"Questions already asked (DO NOT repeat or rephrase these):\n"
                + "\n".join(f"- {q[:80]}" for q in existing_qs)
                + "\n\n"
            )
        # Randomly pick format and difficulty for diversity
        import random
        fmt = random.choice(["qa", "code_completion", "debugging", "comparison", "tutorial"])
        difficulty = random.choice(["easy", "medium", "hard"])
        fmt_desc = {
            "qa": "a direct question and detailed answer",
            "code_completion": "a code snippet with a TODO that the answer completes",
            "debugging": "a buggy code snippet that the answer fixes and explains",
            "comparison": "a comparison of 2-3 approaches with trade-offs",
            "tutorial": "a step-by-step tutorial with code examples",
        }[fmt]
        diff_desc = {
            "easy": "basic definition or simple concept",
            "medium": "practical application with specific parameters",
            "hard": "design trade-off requiring analysis of multiple factors",
        }[difficulty]
        prompt = (
            f"Goal: {goal}\n"
            f"You are {agent.name}, a {agent.role}. {agent.role_desc}\n"
            f"Round {round_num}.\n\n"
            f"CONSTRAINTS:\n"
            f"- Focus on REAL, implementable techniques\n"
            f"- NO quantum computing, NO sci-fi, NO hypothetical breakthroughs\n"
            f"- The answer must be GROUNDED in the web content above — cite specific facts\n"
            f"- DO NOT hallucinate APIs or function names — only use real, known APIs\n"
            f"- The answer must explain HOW and WHY something works\n"
            f"- Include concrete details: specific techniques, numbers, code patterns\n"
            f"- DO NOT ask meta-questions about 'designing Q&A pairs' or 'training pipelines'\n"
            f"- Ask SPECIFIC technical questions about the actual topic\n\n"
            f"FORMAT: {fmt} — {fmt_desc}\n"
            f"DIFFICULTY: {difficulty} — {diff_desc}\n\n"
            f"{existing_section}"
            f"{content_section}"
            f"Create a SPECIFIC technical Q&A pair in the {fmt} format at {difficulty} difficulty.\n"
            f"The question must be DIFFERENT from all previous ones.\n"
            f"The answer must be grounded in the web content and explain HOW/WHY.\n"
            f"If the web content doesn't cover a topic, pick a different angle.\n\n"
            f"Format your response as:\n"
            f"TRAINING_Q: <specific technical question>\n"
            f"TRAINING_A: <detailed answer grounded in web content, explaining HOW and WHY>\n"
            f"DOC_TITLE: <title>\n"
            f"DOC_BODY: <markdown content>\n"
        )
        try:
            response = await agent.think(prompt, temp=0.5, max_tokens=1024)
            q, a, doc_title, doc_body = self._parse_response(response)

            if q and a:
                await agent.save_training(q, a, goal, tags=[agent.role, f"round{round_num}"])
            if doc_title and doc_body:
                await agent.save_doc(f"{doc_title}_R{round_num}", doc_body, goal)
            return bool(q and a)
        except Exception as e:
            await agent.say(f"Generation failed: {e}", msg_type="chat")
            return False

    async def _agent_theorize(self, agent: Agent, goal: str, round_num: int):
        """Theorizing phase: refine the generated Q&A and explain how/why.

        The agent reviews its own output and adds:
        - WHY the technique/approach works (mechanism)
        - HOW it can be implemented (concrete steps)
        - What evidence supports it
        - What trade-offs exist
        """
        # Get the last question this agent generated
        existing_qs = list(getattr(self.db, '_questions', set()))
        if not existing_qs:
            return
        last_q = existing_qs[-1] if existing_qs[-1] else existing_qs[-2]

        prompt = (
            f"Goal: {goal}\n"
            f"You are {agent.name}, a {agent.role}. Round {round_num}.\n\n"
            f"You just generated this training question:\n"
            f"Q: {last_q[:200]}\n\n"
            f"Now THEORIZE about it:\n"
            f"1. WHY does this approach work? What is the underlying mechanism?\n"
            f"2. HOW would you implement it? What are the concrete steps?\n"
            f"3. What EVIDENCE or data supports this?\n"
            f"4. What are the TRADE-OFFS or limitations?\n\n"
            f"Be specific and practical. No quantum computing, no sci-fi.\n"
            f"Reply with your analysis in 4-6 sentences."
        )
        try:
            # Use heavy model for theorizing — requires deeper reasoning
            theory = await agent.think(prompt, temp=0.3, max_tokens=256, use_heavy=True)
            await agent.say(f"Theory: {theory[:120]}", msg_type="draft")

            # Save the theorized refinement as additional training data
            # This creates a "why/how" follow-up Q&A
            refine_q = f"Why does the approach to '{last_q[:80]}' work, and what are its trade-offs?"
            refine_a = theory
            await agent.save_training(refine_q, refine_a, goal,
                                      tags=[agent.role, f"round{round_num}", "theory"])
        except Exception as e:
            await agent.say(f"Theory failed: {e}", msg_type="chat")

    async def _agent_code_verify(self, agent: Agent, goal: str, round_num: int):
        """Coder/verifier phase: run scripts to confirm findings.

        The agent writes a small Python script to test or demonstrate
        the claim from the theorize phase, runs it in the sandbox,
        and saves the verified result as training data.
        """
        existing_qs = list(getattr(self.db, '_questions', set()))
        if not existing_qs:
            return
        last_q = existing_qs[-1] if existing_qs[-1] else existing_qs[-2]

        # Step 1: Agent writes a verification script
        script_prompt = (
            f"Goal: {goal}\n"
            f"You are {agent.name}, a {agent.role}. Round {round_num}.\n\n"
            f"A teammate theorized about:\nQ: {last_q[:200]}\n\n"
            f"Write a SHORT Python script (max 15 lines) to verify or demonstrate "
            f"the core claim. Use only basic Python (math, lists, dicts, strings).\n"
            f"No imports. No file access. Just pure computation.\n"
            f"Examples: calculate memory savings, simulate a technique, verify a formula.\n\n"
            f"Reply with ONLY the Python code, no explanation."
        )
        try:
            # Use heavy model for code generation — Qwen2.5 is much better at code
            code = await agent.think(script_prompt, temp=0.3, max_tokens=256, use_heavy=True)
            # Strip markdown code fences if present
            code = code.strip()
            if code.startswith("```"):
                code = code.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            await agent.say(f"Running verification script...", msg_type="tool")

            # Step 2: Run the script in sandbox
            output = agent.run_script(code)

            await agent.say(f"Script output: {output[:80]}", msg_type="tool")

            # Step 3: Agent interprets the results
            interpret_prompt = (
                f"You ran this Python script to verify a claim about:\n"
                f"Q: {last_q[:150]}\n\n"
                f"Script:\n{code[:500]}\n\n"
                f"Output:\n{output[:500]}\n\n"
                f"Interpret the results in 3-4 sentences:\n"
                f"1. Does the output CONFIRM or REFUTE the claim?\n"
                f"2. What specific numbers or patterns support this?\n"
                f"3. What does this tell us about HOW and WHY the technique works?\n"
                f"Be concrete — cite the actual output values."
            )
            interpretation = await agent.think(interpret_prompt, temp=0.3, max_tokens=256)
            await agent.say(f"Verified: {interpretation[:100]}", msg_type="result")

            # Step 4: Save verified training data
            verify_q = f"For '{last_q[:80]}', verify with code: does this actually work as claimed?"
            verify_a = (
                f"Verification script:\n```python\n{code[:300]}\n```\n\n"
                f"Output:\n```\n{output[:300]}\n```\n\n"
                f"Analysis: {interpretation}"
            )
            await agent.save_training(verify_q, verify_a, goal,
                                      tags=[agent.role, f"round{round_num}", "verified"])

            # Also save the script as a doc for reference
            await agent.save_doc(
                f"Verification_{last_q[:40]}_R{round_num}",
                f"## Verification Script\n\n```python\n{code}\n```\n\n"
                f"## Output\n\n```\n{output}\n```\n\n"
                f"## Analysis\n\n{interpretation}\n",
                goal)

        except Exception as e:
            await agent.say(f"Code verify failed: {e}", msg_type="chat")

    def _parse_response(self, response: str) -> tuple:
        """Parse structured response from agent."""
        q = a = doc_title = doc_body = None
        lines = response.split("\n")
        current_section = None
        current_lines = []

        for line in lines:
            if line.startswith("TRAINING_Q:"):
                if current_section == "TRAINING_A":
                    a = "\n".join(current_lines).strip()
                elif current_section == "DOC_BODY":
                    doc_body = "\n".join(current_lines).strip()
                current_section = "TRAINING_Q"
                current_lines = [line[len("TRAINING_Q:"):].strip()]
            elif line.startswith("TRAINING_A:"):
                if current_section == "TRAINING_Q":
                    q = "\n".join(current_lines).strip()
                current_section = "TRAINING_A"
                current_lines = [line[len("TRAINING_A:"):].strip()]
            elif line.startswith("DOC_TITLE:"):
                if current_section == "TRAINING_A":
                    a = "\n".join(current_lines).strip()
                current_section = "DOC_TITLE"
                current_lines = [line[len("DOC_TITLE:"):].strip()]
            elif line.startswith("DOC_BODY:"):
                if current_section == "DOC_TITLE":
                    doc_title = "\n".join(current_lines).strip()
                elif current_section == "TRAINING_A":
                    a = "\n".join(current_lines).strip()
                current_section = "DOC_BODY"
                current_lines = [line[len("DOC_BODY:"):].strip()]
            else:
                current_lines.append(line)

        # Flush last section
        if current_section == "TRAINING_A":
            a = "\n".join(current_lines).strip()
        elif current_section == "DOC_BODY":
            doc_body = "\n".join(current_lines).strip()
        elif current_section == "TRAINING_Q":
            q = "\n".join(current_lines).strip()

        return q, a, doc_title, doc_body

    async def _agent_review(self, agent: Agent, goal: str):
        """Agent reviews team's work and suggests improvements."""
        history = self.bus.history(limit=20)
        recent = [m for m in history if m.msg_type in ("result", "doc", "draft")
                  and m.sender != agent.name]
        if not recent:
            return

        recent_text = "\n".join(f"[{m.sender}]: {m.content[:100]}" for m in recent[:5])
        prompt = (
            f"Goal: {goal}\n"
            f"You are {agent.name}, a {agent.role}.\n"
            f"Your teammates produced:\n{recent_text}\n\n"
            f"Give brief feedback or suggest one improvement (1-2 sentences)."
        )
        try:
            feedback = await agent.think(prompt, temp=0.3, max_tokens=128)
            await agent.say(f"Review: {feedback[:100]}", msg_type="critique")
        except Exception:
            pass


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Autonomous LFM research team")
    parser.add_argument("--goal", type=str, default=None,
                        help="Single research goal (vague, open-ended)")
    parser.add_argument("--goals", type=str, default=None,
                        help="Comma-separated list of goals")
    parser.add_argument("--agents", type=int, default=16, help="Number of LFM agents")
    parser.add_argument("--lfm-parallel", type=int, default=16,
                        help="Max concurrent LFM calls (ForgeAI server port 1235)")
    parser.add_argument("--list-goals", action="store_true", help="List built-in goals and exit")
    parser.add_argument("--auto-goals", action="store_true",
                        help="Agents self-generate diverse training data goals")
    parser.add_argument("--heavy-model", type=str, default="qwen2.5-3b-instruct-abliterated",
                        help="Heavy model name (llama-server port 8080)")
    parser.add_argument("--heavy-parallel", type=int, default=4,
                        help="Max concurrent heavy model calls")
    args = parser.parse_args()

    if args.list_goals:
        print("Built-in goals:")
        for i, g in enumerate(BUILTIN_GOALS, 1):
            print(f"  {i}. {g}")
        return

    # Determine goals
    if args.goals:
        goals = [g.strip() for g in args.goals.split(",")]
    elif args.goal:
        goals = [args.goal]
    else:
        goals = BUILTIN_GOALS[:5]  # default: first 5

    team = ResearchTeam(goals=goals, n_agents=args.agents,
                        lfm_parallel=args.lfm_parallel,
                        auto_goals=args.auto_goals,
                        heavy_model=args.heavy_model,
                        heavy_parallel=args.heavy_parallel)
    asyncio.run(team.run())


if __name__ == "__main__":
    main()
