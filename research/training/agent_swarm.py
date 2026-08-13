"""Multi-agent collaboration system for training data generation (v2).

Orchestrator-worker pattern with structured outputs, tool-calling schema,
and token budgets. Connects to ForgeAI inference server (not LM Studio).

Key improvements over v1:
- Structured outputs (Pydantic models) eliminate "Reply with ONLY..." drift
- OpenAI function-calling schema lets models choose tools dynamically
- Token budgets (4096/agent) and max_turns (10) prevent runaway agents
- Critique bug fixed: agents excluded by name, not index
- Orchestrator-worker replaces fixed 5-phase pipeline

Usage:
    from research.training.agent_swarm import AgentSwarm
    swarm = AgentSwarm(n_agents=8, task="code", n_topics=10)
    await swarm.run()
"""
import asyncio
import json
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx
from pydantic import BaseModel, Field, ValidationError

# Default inference endpoints (overridable via ModelConfig)
# Both GGUFs are served by LM Studio from a single endpoint.
from research.paths import LMSTUDIO_API as LMSTUDIO_BASE
FORGE_API = LMSTUDIO_BASE       # LM Studio (was ForgeAI server :8000)
QWEN_API = LMSTUDIO_BASE        # LM Studio (was llama-server :8080)

from research.training.web_scraper import search_all

# Qwen has a 4096-token context window — cap content sent to Qwen agents
# at ~2500 chars (leaves room for system prompt + instructions + output).
# LFM has 128K context so it handles raw scraped content + summarization.
QWEN_CONTEXT_CHARS = 2500


# ── Model & Swarm Settings ───────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """A backend model the swarm can use.

    `concurrency` is the max number of simultaneous generations against this
    model — enforced by a per-model asyncio.Semaphore.
    """
    key: str            # short id used in settings, e.g. "qwen", "lfm"
    name: str           # model id sent to the API
    base_url: str       # inference server base URL (OpenAI-compatible /v1)
    concurrency: int = 1


@dataclass
class SwarmSettings:
    """Declarative swarm configuration.

    `agents` is a list of (model_key, role) tuples — one entry per agent.
    `master` is the model_key whose agents lead each topic.
    `vote` enables the critique/voting round (disabled in simple mode).
    `all_tools` exposes every tool to every agent.
    `dynamic_goals` lets the master agent generate topics dynamically instead
    of using a fixed topic list.
    """
    models: dict[str, ModelConfig] = field(default_factory=dict)
    agents: list[tuple[str, str]] = field(default_factory=list)
    master: str = "qwen"
    vote: bool = False
    all_tools: bool = True
    dynamic_goals: bool = False


# ── Simple mode preset ───────────────────────────────────────────────────────
# Qwen = master, handles Developer / Researcher / Theorizer / Critique roles.
# LFM2.5 = Fetcher + low-tier worker.
# 1 agent per (model, role); concurrency=1 per model (1 concurrent gen each).
# All tools enabled, no voting round, Qwen is the master.
#
# GGUF models (loaded in LM Studio):
#   qwen: D:/LMstudio/Models/AdvancedDataIntelligence/adi-qwen2.5-14b-glm5.2-general-GGUF/adi-qwen2.5-14b-glm5.2-general-q4_k_m.gguf
#   lfm:  D:/LMstudio/Models/lmstudio-community/LFM2.5-1.2B-Instruct-GGUF/LFM2.5-1.2B-Instruct-Q8_0.gguf
# Qwen 3.6 (downloading) will replace the qwen entry once available.

SIMPLE_MODE = SwarmSettings(
    models={
        "qwen": ModelConfig(
            key="qwen",
            name="adi-qwen2.5-14b-glm5.2-general",
            base_url=QWEN_API,
            concurrency=1,
        ),
        "lfm": ModelConfig(
            key="lfm",
            name="liquid/lfm2.5-1.2b",
            base_url=FORGE_API,
            concurrency=1,
        ),
    },
    agents=[
        ("qwen", "developer"),
        ("qwen", "researcher"),
        ("qwen", "theorizer"),
        ("qwen", "critique"),
        ("lfm", "fetcher"),
        ("lfm", "worker"),
    ],
    master="qwen",
    vote=False,
    all_tools=True,
)

# Perspectives for simple-mode roles (supplements ROLE_PERSPECTIVES).
SIMPLE_ROLE_PERSPECTIVES = {
    "developer":  "Write production-quality code and concrete implementations",
    "researcher": "Find and synthesize the most relevant facts from sources",
    "theorizer":  "Provide the theoretical foundation and explain why it works",
    "critique":   "Find flaws, edge cases, and improvements in teammates' work",
    "fetcher":    "Generate search queries and fetch web content efficiently",
    "worker":     "Summarize large content into compact Qwen-friendly summaries; handle low-tier formatting and lookups",
}

# ── Agent roles ──────────────────────────────────────────────────────────────

ROLES = [
    "questioner", "deep_diver", "pragmatist", "analogist", "critic",
    "architect", "optimizer", "teacher", "interviewer", "debugger",
    "comparator", "summarizer", "explorer", "historian", "reviewer",
    "challenger", "simplifier", "detailer", "tester", "refactorer",
    "theorist", "practitioner", "antipattern", "connector", "troubleshooter",
    "migrator", "scalability", "security", "minimalist", "completeness",
    "contrarian", "synthesizer",
]

ROLE_PERSPECTIVES = {
    "questioner": "Ask the most fundamental question a beginner would have",
    "deep_diver": "Focus on edge cases and advanced usage",
    "pragmatist": "Focus on practical real-world usage",
    "analogist": "Explain using analogies from other domains",
    "critic": "Find common misconceptions and correct them",
    "architect": "Focus on design patterns and structure",
    "optimizer": "Focus on performance and efficiency",
    "teacher": "Create the clearest explanation for a student",
    "interviewer": "Frame as a technical interview question",
    "debugger": "Focus on common bugs and how to fix them",
    "comparator": "Compare alternatives and trade-offs",
    "summarizer": "Distill to the essential 3 points",
    "explorer": "Explore unusual or creative applications",
    "historian": "Explain the evolution and why it exists",
    "reviewer": "Review as if evaluating a PR, suggest improvements",
    "challenger": "Challenge assumptions, what if the opposite is true?",
    "simplifier": "Explain in the fewest words possible",
    "detailer": "Go deep into implementation details",
    "tester": "Focus on testing strategies and examples",
    "refactorer": "Show bad code then refactor to good code",
    "theorist": "Explain the theoretical foundation",
    "practitioner": "Show what a working professional actually does",
    "antipattern": "Show what NOT to do and why",
    "connector": "Connect this to other related concepts",
    "troubleshooter": "Focus on debugging and error messages",
    "migrator": "Show how to migrate from older approaches",
    "scalability": "Focus on scaling and production concerns",
    "security": "Focus on security implications",
    "minimalist": "Show the absolute minimal working example",
    "completeness": "Cover every aspect exhaustively",
    "contrarian": "Argue against the conventional approach",
    "synthesizer": "Synthesize multiple viewpoints into one answer",
}

# ── Message bus ──────────────────────────────────────────────────────────────

@dataclass
class Message:
    """A message on the agent message bus."""
    sender: str       # "#1", "#5", etc.
    recipient: str    # "#3", "all", "synthesizer"
    content: str
    msg_type: str     # "chat", "draft", "critique", "question", "result", "tool"
    timestamp: float = field(default_factory=time.time)


class MessageBus:
    """Shared message bus for agent communication."""

    def __init__(self):
        self._messages: list[Message] = []
        self._subscribers: dict[str, asyncio.Queue] = {}
        self._lock = asyncio.Lock()
        self.log_file = None

    def set_log_file(self, path: Path):
        self.log_file = open(path, "w", encoding="utf-8")

    async def publish(self, msg: Message):
        """Publish a message to the bus."""
        self._messages.append(msg)
        # Log to file
        if self.log_file:
            self.log_file.write(json.dumps({
                "sender": msg.sender, "recipient": msg.recipient,
                "type": msg.msg_type, "content": msg.content[:200],
                "ts": msg.timestamp,
            }, ensure_ascii=False) + "\n")
            self.log_file.flush()
        # Notify subscribers
        if msg.recipient == "all":
            for q in self._subscribers.values():
                await q.put(msg)
        elif msg.recipient in self._subscribers:
            await self._subscribers[msg.recipient].put(msg)
        # Live console output
        self._print(msg)

    def _print(self, msg: Message):
        """Live console output — watch agents work and talk."""
        colors = {
            "chat":     "\033[37m",   # white
            "draft":    "\033[36m",   # cyan
            "critique": "\033[33m",   # yellow
            "question": "\033[35m",   # magenta
            "result":   "\033[32m",   # green
            "tool":     "\033[34m",   # blue
        }
        reset = "\033[0m"
        color = colors.get(msg.msg_type, "\033[37m")
        arrow = "->" if msg.recipient == "all" else f"->{msg.recipient}"
        content = msg.content[:120].replace("\n", " ")
        print(f"  {color}{msg.sender}{arrow} [{msg.msg_type}]{reset} {content}")

    def subscribe(self, agent_name: str) -> asyncio.Queue:
        """Subscribe to messages for an agent."""
        q = asyncio.Queue()
        self._subscribers[agent_name] = q
        return q

    def get_history(self, limit: int = 20) -> list[Message]:
        """Get recent message history."""
        return self._messages[-limit:]


# ── Database (simple JSONL) ──────────────────────────────────────────────────

class AgentDatabase:
    """Simple JSONL database for agents to save results."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    async def save(self, record: dict):
        """Save a record to the database."""
        async with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    async def load_all(self) -> list[dict]:
        """Load all records."""
        if not self.path.exists():
            return []
        records = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))
        return records


# ── Structured Output Models ─────────────────────────────────────────────────

class SearchQuery(BaseModel):
    query: str = Field(description="Search query string")

class GoalList(BaseModel):
    goals: list[str] = Field(description="List of research topics/goals to explore")

class DraftOutput(BaseModel):
    topic: str = Field(description="The topic being addressed")
    question: str = Field(description="Specific question about the topic")
    answer: str = Field(description="Detailed answer from the agent's perspective")

class CritiqueOutput(BaseModel):
    feedback: str = Field(description="1-2 sentence critique of another agent's draft")

class SynthesisOutput(BaseModel):
    question: str = Field(description="Best question combining all perspectives")
    answer: str = Field(description="Best answer combining all drafts")


# ── Tool Schema ──────────────────────────────────────────────────────────────

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information on a topic",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_to_db",
            "description": "Save a training data record to the database",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "response": {"type": "string"},
                },
                "required": ["topic", "response"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Run a safe Python expression and return the result",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python expression"}
                },
                "required": ["code"],
            },
        },
    },
]

# ── Agent ────────────────────────────────────────────────────────────────────

class Agent:
    """A single agent in the swarm with structured outputs and tool calling.

    Connects to ForgeAI inference server. Uses Pydantic models for output
    parsing and OpenAI function-calling schema for tool selection.
    """

    # Per-agent limits
    MAX_TURNS = 10
    TOKEN_BUDGET = 4096

    def __init__(self, idx: int, role: str, bus: MessageBus,
                 db: AgentDatabase, model_cfg: ModelConfig,
                 model_sem: asyncio.Semaphore,
                 scrape_sem: asyncio.Semaphore,
                 is_master: bool = False, all_tools: bool = True):
        self.name = f"#{idx+1}"
        self.role = role
        perspective = ROLE_PERSPECTIVES.get(role)
        if perspective is None:
            perspective = SIMPLE_ROLE_PERSPECTIVES.get(role, "Provide unique insight")
        self.perspective = perspective
        self.seed = random.randint(0, 999999)
        self.bus = bus
        self.db = db
        self.model_cfg = model_cfg
        self.model_sem = model_sem
        self.scrape_sem = scrape_sem
        self.is_master = is_master
        self.all_tools = all_tools
        self.inbox = bus.subscribe(self.name)
        self.context: list[dict] = []  # conversation history
        self.client: httpx.AsyncClient = None
        self._tokens_used: int = 0
        self._turns: int = 0

    async def think(self, prompt: str, temperature: float = None,
                    max_tokens: int = 512, tools: list[dict] = None,
                    output_model: type[BaseModel] = None) -> str:
        """Call the model via its configured backend with optional tool schema.

        Uses standard system/user/assistant format (no sudo-think tags).
        Enforces token budget and turn limit. Per-model semaphore caps
        concurrent generations against each backend.
        """
        if self._turns >= self.MAX_TURNS:
            return "[MAX_TURNS]"
        if self._tokens_used >= self.TOKEN_BUDGET:
            return "[BUDGET_EXCEEDED]"

        temp = temperature if temperature is not None else 0.4
        actual_max = min(max_tokens, self.TOKEN_BUDGET - self._tokens_used)

        # Build system prompt
        master_tag = " (MASTER — you lead the team and make final calls)" if self.is_master else ""
        system_msg = {
            "role": "system",
            "content": (
                f"You are {self.name}, a {self.role}{master_tag}. "
                f"Your perspective: {self.perspective}. "
                f"Be concise and accurate. "
                f"{'Respond with valid JSON matching the required format.' if output_model else ''}"
            ),
        }
        messages = [system_msg] + self.context + [{"role": "user", "content": prompt}]

        payload = {
            "model": self.model_cfg.name,
            "messages": messages,
            "temperature": temp,
            "max_tokens": actual_max,
            "seed": self.seed,
        }
        # Send tools only when we want tool-calling — not when we expect
        # structured JSON output (otherwise the model may emit a tool_call
        # instead of content, yielding an empty response).
        if tools and not output_model:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        # Per-model concurrency cap — retry without tools on 400 (some
        # GGUF models reject the OpenAI tool-calling schema).
        async with self.model_sem:
            for attempt in range(2):
                try:
                    resp = await self.client.post(
                        f"{self.model_cfg.base_url}/chat/completions",
                        json=payload, timeout=120.0)
                    resp.raise_for_status()
                    break
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 400 and "tools" in payload and attempt == 0:
                        payload.pop("tools", None)
                        payload.pop("tool_choice", None)
                        continue
                    raise
            data = resp.json()
            raw_content = data["choices"][0]["message"].get("content")
            content = (raw_content or "").strip()

        self._turns += 1
        self._tokens_used += actual_max

        # Parse structured output if model specified
        if output_model and content:
            content = self._parse_structured(content, output_model)

        return content

    def _parse_structured(self, content: str, model: type[BaseModel]) -> str:
        """Parse model output into structured format, with fallback.

        Tries to extract JSON from the response. If parsing fails,
        returns the raw content with a warning.
        """
        # Try direct JSON parse
        try:
            parsed = model.model_validate_json(content)
            return json.dumps(parsed.model_dump())
        except (ValidationError, ValueError):
            pass

        # Try to extract JSON from code blocks or inline
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            try:
                parsed = model.model_validate_json(json_match.group(1))
                return json.dumps(parsed.model_dump())
            except (ValidationError, ValueError):
                pass

        # Try to find any JSON object in the text
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            try:
                parsed = model.model_validate_json(json_match.group(0))
                return json.dumps(parsed.model_dump())
            except (ValidationError, ValueError):
                pass

        # Fallback: wrap raw content in the expected structure
        try:
            fallback = model(
                topic="unknown",
                question=content[:200] if hasattr(model, 'question') else "",
                answer=content if hasattr(model, 'answer') else "",
            )
            return json.dumps(fallback.model_dump())
        except Exception:
            return content  # Last resort: raw text

    async def say(self, content: str, recipient: str = "all",
                  msg_type: str = "chat"):
        """Send a message to the bus."""
        msg = Message(sender=self.name, recipient=recipient,
                      content=content, msg_type=msg_type)
        await self.bus.publish(msg)
        # Keep in own context
        self.context.append({"role": "assistant", "content": content})

    async def listen(self, timeout: float = 5.0) -> Optional[Message]:
        """Wait for a message from the bus."""
        try:
            return await asyncio.wait_for(self.inbox.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    async def web_search(self, query: str, max_chars: int = 4000) -> str:
        """Tool: search the web and return content."""
        await self.say(f"Searching web: {query}", msg_type="tool")
        async with self.scrape_sem:
            results = search_all(query, n=3)
        content = "\n\n".join(r.content for r in results if r.content)
        return content[:max_chars]

    async def save_to_db(self, record: dict):
        """Tool: save a record to the database."""
        record["agent"] = self.name
        record["role"] = self.role
        record["seed"] = self.seed
        record["timestamp"] = time.time()
        await self.db.save(record)
        await self.say(f"Saved to database", msg_type="tool")

    async def run_python(self, code: str) -> str:
        """Tool: run safe Python code (no imports, no file access)."""
        await self.say(f"Running Python: {code[:60]}...", msg_type="tool")
        # Sandboxed eval — only builtins, no imports
        safe_globals = {"__builtins__": {
            "len": len, "str": str, "int": int, "float": float,
            "list": list, "dict": dict, "range": range, "sum": sum,
            "min": min, "max": max, "abs": abs, "round": round,
            "sorted": sorted, "enumerate": enumerate, "zip": zip,
            "print": print, "True": True, "False": False, "None": None,
        }}
        try:
            result = eval(code, safe_globals, {})
            return str(result)
        except Exception as e:
            return f"Error: {e}"


# ── Swarm ────────────────────────────────────────────────────────────────────

class AgentSwarm:
    """Orchestrates agents collaborating on training data generation.

    Two modes:
    - Legacy (n_agents + lfm_parallel): round-robin roles from ROLES, single
      LFM backend, teams of 4 per topic, with critique/vote round.
    - Settings-driven (settings=SwarmSettings(...)): explicit per-model config,
      per-model concurrency caps, fixed role assignments, optional master and
      voting. SIMPLE_MODE is the canonical preset.
    """

    def __init__(self, n_agents: int = 32, task_type: str = "code",
                 n_topics: int = 10, topics: list[str] = None,
                 output_dir: Path = None, lfm_parallel: int = 32,
                 settings: SwarmSettings = None):
        self.task_type = task_type
        self.n_topics = n_topics
        self.topics = topics or []
        self.output_dir = output_dir or Path("research/data/finetune")
        self.settings = settings

        # Shared infrastructure
        self.bus = MessageBus()
        self.db = AgentDatabase(self.output_dir / f"swarm_{task_type}_{int(time.time())}.jsonl")
        self.bus.set_log_file(self.output_dir / f"swarm_{task_type}_{int(time.time())}.log")
        self.scrape_sem = asyncio.Semaphore(4)

        if settings is not None:
            # Settings-driven mode: per-model semaphores from ModelConfig.concurrency
            self.model_sems: dict[str, asyncio.Semaphore] = {
                k: asyncio.Semaphore(m.concurrency)
                for k, m in settings.models.items()
            }
            self.n_agents = len(settings.agents)
            self.agents: list[Agent] = []
            for i, (model_key, role) in enumerate(settings.agents):
                mcfg = settings.models[model_key]
                is_master = (model_key == settings.master)
                self.agents.append(Agent(
                    i, role, self.bus, self.db,
                    mcfg, self.model_sems[model_key], self.scrape_sem,
                    is_master=is_master, all_tools=settings.all_tools))
        else:
            # Legacy mode: single LFM backend
            self.n_agents = n_agents
            self.lfm_parallel = lfm_parallel
            self.lfm_sem = asyncio.Semaphore(lfm_parallel)
            legacy_cfg = ModelConfig(
                key="lfm", name="lfm2.5-1.2b",
                base_url=FORGE_API, concurrency=lfm_parallel)
            self.model_sems = {"lfm": self.lfm_sem}
            self.agents = []
            for i in range(n_agents):
                role = ROLES[i % len(ROLES)]
                self.agents.append(Agent(
                    i, role, self.bus, self.db,
                    legacy_cfg, self.lfm_sem, self.scrape_sem))

    async def run(self) -> list[dict]:
        """Run the full multi-agent pipeline. Watch live in console."""
        if self.settings is not None:
            return await self._run_settings()

        print(f"\n{'='*70}")
        print(f"  AGENT SWARM: {self.n_agents} agents | {self.n_topics} topics | {self.task_type}")
        print(f"  Agents: {', '.join(a.name for a in self.agents[:8])}... ")
        print(f"{'='*70}\n")

        async with httpx.AsyncClient() as client:
            # Assign shared client to all agents
            for a in self.agents:
                a.client = client

            # Assign topics to agent groups (agents collaborate per topic)
            results = []
            topics_per_round = max(1, self.n_agents // 4)  # 4 agents per topic

            for round_idx in range(0, self.n_topics, topics_per_round):
                batch_topics = self.topics[round_idx:round_idx + topics_per_round]
                if not batch_topics:
                    break

                # Assign 4 agents per topic
                topic_coros = []
                for ti, topic in enumerate(batch_topics):
                    agent_start = ti * 4
                    team = self.agents[agent_start:agent_start + 4]
                    topic_coros.append(self._run_team(topic, team))

                batch_results = await asyncio.gather(*topic_coros)
                results.extend(batch_results)
                print(f"\n  [Round {round_idx // topics_per_round + 1}] "
                      f"Completed {len(batch_results)} topics\n")

        return results

    async def _run_settings(self) -> list[dict]:
        """Settings-driven pipeline: master leads, fetcher scrapes, roles
        execute in order. No voting when settings.vote is False."""
        s = self.settings
        tools = TOOLS_SCHEMA if s.all_tools else None
        # Index agents by role for quick lookup
        by_role: dict[str, Agent] = {}
        for a in self.agents:
            by_role.setdefault(a.role, a)
        master_agents = [a for a in self.agents if a.is_master]
        master = master_agents[0] if master_agents else self.agents[0]

        print(f"\n{'='*70}")
        print(f"  AGENT SWARM (settings mode): {self.n_agents} agents | {self.n_topics} topics")
        for k, m in s.models.items():
            print(f"  model {k}: {m.name} @ {m.base_url} (concurrency={m.concurrency})")
        print(f"  master: {master.name} ({master.role}) | vote={s.vote} | all_tools={s.all_tools}")
        if s.dynamic_goals:
            print(f"  dynamic_goals: ON — master generates topics")
        print(f"{'='*70}\n")

        async with httpx.AsyncClient() as client:
            for a in self.agents:
                a.client = client

            # Determine topics: dynamic generation or fixed list
            if s.dynamic_goals:
                topics = await self._generate_goals(master, tools)
            else:
                topics = self.topics[:self.n_topics]

            if not topics:
                print("  No topics to process, exiting.")
                return []

            results = []
            for ti, topic in enumerate(topics[:self.n_topics]):
                print(f"\n  [Topic {ti+1}/{min(len(topics), self.n_topics)}] {topic}")
                rec = await self._run_simple_topic(topic, by_role, master, tools)
                results.append(rec)
        return results

    async def _generate_goals(self, master: Agent, tools: list[dict]) -> list[str]:
        """Master agent dynamically generates a diverse set of research topics.

        Uses the task_type as a domain hint. Generates up to n_topics goals
        in one call, parses the structured GoalList output.
        """
        domain_hint = {
            "code": "software engineering, programming languages, algorithms, "
                    "systems design, debugging, and developer tooling",
            "llm": "LLM architecture, training, inference, fine-tuning, "
                   "quantization, and AI applications",
            "logic": "mathematical reasoning, logic puzzles, formal proofs, "
                     "and analytical problem-solving",
        }.get(self.task_type, "general technical topics")

        prompt = (
            f"You are the master of a research team generating training data.\n"
            f"Domain: {domain_hint}\n"
            f"Generate {self.n_topics} diverse, specific research topics that "
            f"would produce high-quality training Q&A pairs.\n"
            f"Each topic should be a concrete, searchable phrase (not too vague).\n"
            f"Avoid generic topics — favor specific techniques, patterns, or "
            f"problems that have depth.\n"
            f'Respond with JSON: {{"goals": ["topic1", "topic2", ...]}}'
        )
        raw = await master.think(prompt, temperature=0.7, max_tokens=512,
                                 tools=tools, output_model=GoalList)
        goals = self._extract_goals(raw)
        if not goals:
            # Fallback to fixed topics if generation fails
            print("  [dynamic_goals] Generation failed, falling back to fixed topics")
            return self.topics[:self.n_topics]
        await master.say(f"Generated {len(goals)} goals", msg_type="chat")
        for i, g in enumerate(goals, 1):
            print(f"    {i}. {g}")
        return goals

    def _extract_goals(self, raw: str) -> list[str]:
        """Parse a GoalList JSON or newline-separated list from raw output."""
        if not raw or not isinstance(raw, str):
            return []
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict) and isinstance(obj.get("goals"), list):
                return [str(g).strip() for g in obj["goals"] if str(g).strip()]
        except (ValueError, TypeError):
            pass
        # Fallback: treat each non-empty line as a goal
        lines = [l.strip().lstrip("0123456789.-) ") for l in raw.splitlines()]
        return [l for l in lines if l][:self.n_topics]

    async def _run_simple_topic(self, topic: str, by_role: dict[str, Agent],
                                master: Agent, tools: list[dict]) -> dict:
        """Simple-mode pipeline for one topic.

        1. LFM fetcher generates a search query + scrapes the web (large pull,
           LFM has 128K context).
        2. LFM worker summarizes raw content into a Qwen-friendly compact
           summary (Qwen has 4096 context).
        3. Qwen researcher extracts key facts from the summary.
        4. Qwen theorizer frames the theoretical foundation.
        5. Qwen developer writes the concrete answer/code.
        6. Qwen critique reviews (only if settings.vote is True).
        7. Master saves the final synthesized record.
        Falls back gracefully if a role is missing.
        """
        await master.say(f"Starting topic: {topic}", msg_type="chat")

        # Phase 1: Fetcher generates query + scrapes (LFM, large pull)
        fetcher = by_role.get("fetcher")
        raw_content = ""
        if fetcher is not None:
            q_prompt = (
                f"Generate a web search query about: {topic}\n"
                f'Respond with JSON: {{"query": "<search query>"}}'
            )
            qraw = await fetcher.think(q_prompt, temperature=0.3, max_tokens=64,
                                       tools=tools, output_model=SearchQuery)
            query = self._extract_query(qraw) or topic
            await fetcher.say(f"Query: {query}", msg_type="question")
            # LFM has 128K context — pull up to 16K chars of raw content
            raw_content = await fetcher.web_search(query, max_chars=16000)
            if not raw_content:
                raw_content = ""
        else:
            # No fetcher — do a direct scrape as fallback
            async with self.scrape_sem:
                results = search_all(topic, n=3)
            raw_content = "\n\n".join(r.content for r in results if r.content)[:16000]

        if not raw_content.strip():
            await master.say("No content found, skipping", msg_type="chat")
            return {"topic": topic, "error": "no content", "ok": False}

        await master.say(f"Scraped {len(raw_content)} chars", msg_type="tool")

        # Phase 2: Worker (LFM) summarizes raw content for Qwen
        # Qwen only has 4096 context — LFM compresses the raw scrape down.
        worker = by_role.get("worker")
        content = raw_content
        if worker is not None and len(raw_content) > QWEN_CONTEXT_CHARS:
            sum_prompt = (
                f"Topic: {topic}\n"
                f"Raw web content ({len(raw_content)} chars):\n{raw_content}\n\n"
                f"Summarize this into a dense, information-rich summary "
                f"({QWEN_CONTEXT_CHARS} chars max) preserving all key facts, "
                f"code snippets, and technical details needed to write a "
                f"training Q&A. Do NOT add commentary — just compress."
            )
            content = await worker.think(sum_prompt, temperature=0.2,
                                         max_tokens=1024, tools=tools)
            await worker.say(
                f"Summarized {len(raw_content)} -> {len(content)} chars for Qwen",
                msg_type="tool")
        else:
            # No worker or content already small — truncate to Qwen budget
            content = raw_content[:QWEN_CONTEXT_CHARS]

        # Phase 3: Researcher extracts key facts (Qwen, from summary)
        researcher = by_role.get("researcher")
        facts = ""
        if researcher is not None:
            r_prompt = (
                f"Topic: {topic}\nSummary:\n{content}\n\n"
                f"Extract the 3-5 most important facts a training answer must cover."
            )
            facts = await researcher.think(r_prompt, temperature=0.3,
                                           max_tokens=256, tools=tools)
            await researcher.say(f"Facts: {facts[:120]}", msg_type="draft")

        # Phase 4: Theorizer frames the foundation (Qwen)
        theorizer = by_role.get("theorizer")
        theory = ""
        if theorizer is not None:
            t_prompt = (
                f"Topic: {topic}\nKey facts:\n{facts[:1500]}\n\n"
                f"Explain the theoretical foundation in 3-4 sentences."
            )
            theory = await theorizer.think(t_prompt, temperature=0.4,
                                           max_tokens=256, tools=tools)
            await theorizer.say(f"Theory: {theory[:120]}", msg_type="draft")

        # Phase 5: Developer writes the concrete answer (Qwen)
        developer = by_role.get("developer")
        draft = ""
        if developer is not None:
            d_prompt = self._build_simple_draft_prompt(topic, content, facts, theory)
            draft = await developer.think(d_prompt, temperature=0.4,
                                          max_tokens=512, tools=tools,
                                          output_model=DraftOutput)
            await developer.say(f"Draft ready ({len(draft)} chars)", msg_type="draft")

        if not draft.strip():
            return {"topic": topic, "error": "no draft", "ok": False}

        # Phase 6: Critique (only when voting is enabled)
        critique = by_role.get("critique")
        feedback = ""
        if critique is not None and self.settings.vote:
            c_prompt = (
                f"Topic: {topic}\nDraft:\n{draft[:2000]}\n\n"
                f"As critique, give brief feedback in 1-2 sentences."
            )
            feedback = await critique.think(c_prompt, temperature=0.3,
                                            max_tokens=128, tools=tools,
                                            output_model=CritiqueOutput)
            await critique.say(f"Feedback: {feedback[:100]}", msg_type="critique")

        # Phase 7: Master synthesizes + saves (Qwen — keep within 4096 ctx)
        synth_prompt = (
            f"Topic: {topic}\n"
            f"Facts: {facts[:600]}\n"
            f"Theory: {theory[:600]}\n"
            f"Draft: {draft[:1500]}\n"
            + (f"Feedback: {feedback[:300]}\n" if feedback else "")
            + f"\nProduce the BEST training Q&A combining these. "
            f'Respond with JSON: {{"question": "...", "answer": "..."}}'
        )
        final = await master.think(synth_prompt, temperature=0.3, max_tokens=512,
                                   tools=tools, output_model=SynthesisOutput)

        record = {
            "topic": topic,
            "task_type": self.task_type,
            "response": final,
            "agents": [a.name for a in self.agents],
            "roles": [a.role for a in self.agents],
            "master": master.name,
            "vote": self.settings.vote,
            "ok": True,
        }
        await master.save_to_db(record)
        await master.say("Final synthesis saved to database!", msg_type="result")
        return record

    def _extract_query(self, raw: str) -> str:
        """Pull a query string out of a SearchQuery JSON or raw text."""
        if not raw or not isinstance(raw, str):
            return ""
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict) and obj.get("query"):
                return obj["query"]
        except (ValueError, TypeError):
            pass
        return raw.strip().splitlines()[0] if raw.strip() else ""

    def _build_simple_draft_prompt(self, topic: str, content: str,
                                   facts: str, theory: str) -> str:
        if self.task_type == "code":
            fmt = "coding question with explanation and working code"
        elif self.task_type == "llm":
            fmt = "question about the concept with concise grounded explanation"
        else:
            fmt = "problem with 2-3 step solution"
        return (
            f"Topic: {topic}\n"
            f"Key facts:\n{facts[:800]}\n\n"
            f"Theory:\n{theory[:600]}\n\n"
            f"Summary:\n{content[:1200]}\n\n"
            f"Create the best training Q&A ({fmt}).\n"
            f'Respond with JSON: {{"topic": "...", "question": "...", "answer": "..."}}'
        )

    async def _run_team(self, topic: str, team: list[Agent]) -> dict:
        """A team of 4 agents collaborates on one topic."""
        team_names = ", ".join(a.name for a in team)
        await team[0].say(f"Team {team_names} starting on: {topic}", msg_type="chat")

        # Phase 1: Each agent generates a search query (structured output)
        query_coros = []
        for a in team:
            prompt = (
                f"Generate a web search query about: {topic}\n"
                f"Respond with JSON: {{\"query\": \"<search query>\"}}"
            )
            query_coros.append(a.think(
                prompt, temperature=0.3, max_tokens=64,
                output_model=SearchQuery))
        queries = await asyncio.gather(*query_coros, return_exceptions=True)

        # Share queries
        for i, q in enumerate(queries):
            if isinstance(q, str):
                await team[i].say(f"My query: {q}", msg_type="question")

        # Phase 2: Scrape web (dedup queries)
        unique_queries = list(set(q for q in queries if isinstance(q, str)))[:3]
        content_parts = []
        for q in unique_queries:
            async with self.scrape_sem:
                results = search_all(q, n=2)
            content_parts.append("\n".join(r.content for r in results if r.content))
        content = "\n\n".join(content_parts)

        if not content:
            await team[0].say("No content found, skipping", msg_type="chat")
            return {"topic": topic, "error": "no content", "ok": False}

        await team[0].say(f"Scraped {len(content)} chars from {len(unique_queries)} sources",
                          msg_type="tool")

        # Phase 3: Each agent writes a draft (structured output, no sudo-think)
        draft_coros = []
        for a in team:
            prompt = self._build_draft_prompt(topic, content, a)
            draft_coros.append(a.think(
                prompt, temperature=0.4, max_tokens=512,
                output_model=DraftOutput))
        drafts = await asyncio.gather(*draft_coros, return_exceptions=True)

        # Share drafts
        ok_drafts = []
        for i, d in enumerate(drafts):
            if isinstance(d, str) and d.strip():
                await team[i].say(f"Draft ready ({len(d)} chars)", msg_type="draft")
                ok_drafts.append({"agent": team[i].name, "role": team[i].role, "draft": d})

        if not ok_drafts:
            return {"topic": topic, "error": "no drafts", "ok": False}

        # Phase 4: Agents critique each other (exclude own draft by name)
        critique_coros = []
        for a in team:
            others = [d for d in ok_drafts if d["agent"] != a.name]
            if others:
                prompt = (
                    f"Other agents wrote these drafts about {topic}:\n"
                    + "\n---\n".join(f"[{d['agent']}]: {d['draft'][:500]}" for d in others[:2])
                    + f"\n\nAs {a.name} ({a.role}), give brief feedback in 1-2 sentences."
                    + "\nRespond with JSON: {\"feedback\": \"...\"}"
                )
                critique_coros.append(a.think(
                    prompt, temperature=0.3, max_tokens=128,
                    output_model=CritiqueOutput))
        if critique_coros:
            critiques = await asyncio.gather(*critique_coros, return_exceptions=True)
            for i, c in enumerate(critiques):
                if isinstance(c, str) and c.strip():
                    await team[i].say(f"Feedback: {c[:80]}", msg_type="critique")

        # Phase 5: Synthesizer agent combines best elements (structured output)
        synth_agent = team[-1]  # last agent synthesizes
        drafts_text = "\n\n---\n\n".join(
            f"[{d['agent']} ({d['role']})]: {d['draft']}" for d in ok_drafts)
        synth_prompt = (
            f"Multiple agents wrote drafts about: {topic}\n"
            f"Create the BEST training Q&A by combining the strongest elements.\n\n"
            f"Drafts:\n{drafts_text[:6000]}\n\n"
            f"Respond with JSON: {{\"question\": \"...\", \"answer\": \"...\"}}"
        )
        final = await synth_agent.think(
            synth_prompt, temperature=0.3, max_tokens=512,
            output_model=SynthesisOutput)

        # Save to database
        record = {
            "topic": topic,
            "task_type": self.task_type,
            "response": final,
            "n_agents": len(team),
            "n_drafts": len(ok_drafts),
            "agents": [a.name for a in team],
            "roles": [a.role for a in team],
            "synthesized": True,
            "ok": True,
        }
        await synth_agent.save_to_db(record)
        await synth_agent.say("Final synthesis saved to database!", msg_type="result")

        return record

    def _build_draft_prompt(self, topic: str, content: str, agent: Agent) -> str:
        """Build a draft prompt for an agent (standard format, no sudo-think)."""
        if self.task_type == "code":
            fmt_desc = "coding question with explanation and working code"
        elif self.task_type == "llm":
            fmt_desc = "question about the concept with concise grounded explanation"
        else:
            fmt_desc = "problem with 2-3 step solution"

        return (
            f"Topic: {topic}\n"
            f"Perspective: {agent.perspective}\n\n"
            f"Web content:\n{content[:3000]}\n\n"
            f"Create the best training Q&A from your perspective ({fmt_desc}).\n"
            'Respond with JSON: {"topic": "...", "question": "...", "answer": "..."}'
        )


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Multi-agent swarm for data generation")
    parser.add_argument("--simple", action="store_true",
                        help="Use SIMPLE_MODE preset (Qwen master + LFM fetcher/worker, "
                             "1 concurrent gen per model, no vote, all tools)")
    parser.add_argument("--agents", type=int, default=32, help="Number of agents (legacy mode, max 32)")
    parser.add_argument("--topics", type=int, default=10, help="Number of topics to process")
    parser.add_argument("--task", choices=["code", "llm", "logic"], default="code")
    parser.add_argument("--lfm-parallel", type=int, default=32,
                        help="Max concurrent LFM calls (legacy mode)")
    # Settings-mode overrides (applied on top of SIMPLE_MODE)
    parser.add_argument("--qwen-concurrency", type=int, default=None,
                        help="Override Qwen concurrent generations (simple mode)")
    parser.add_argument("--lfm-concurrency", type=int, default=None,
                        help="Override LFM concurrent generations (simple mode)")
    parser.add_argument("--qwen-model", type=str, default=None,
                        help="Override Qwen model id (simple mode)")
    parser.add_argument("--lfm-model", type=str, default=None,
                        help="Override LFM model id (simple mode)")
    parser.add_argument("--vote", action="store_true",
                        help="Enable critique/vote round in simple mode (off by default)")
    parser.add_argument("--dynamic-goals", action="store_true",
                        help="Master agent generates topics dynamically (simple mode)")
    args = parser.parse_args()

    # Topic lists (same as data_gen.py) — used as fallback or fixed list
    from research.training.data_gen import WEB_TOPICS
    topics = WEB_TOPICS.get(args.task, [])[:args.topics]

    # In dynamic-goals mode, topics are generated at runtime; still load
    # the fixed list as fallback.
    if not topics and not (args.simple and args.dynamic_goals):
        print(f"No topics for task type: {args.task}")
        return

    settings = None
    if args.simple:
        # Clone SIMPLE_MODE and apply overrides
        import copy
        settings = copy.deepcopy(SIMPLE_MODE)
        settings.vote = args.vote
        settings.dynamic_goals = args.dynamic_goals
        if args.qwen_concurrency is not None:
            settings.models["qwen"].concurrency = args.qwen_concurrency
        if args.lfm_concurrency is not None:
            settings.models["lfm"].concurrency = args.lfm_concurrency
        if args.qwen_model is not None:
            settings.models["qwen"].name = args.qwen_model
        if args.lfm_model is not None:
            settings.models["lfm"].name = args.lfm_model

    swarm = AgentSwarm(
        n_agents=args.agents,
        task_type=args.task,
        n_topics=len(topics),
        topics=topics,
        lfm_parallel=args.lfm_parallel,
        settings=settings,
    )

    t0 = time.time()
    results = asyncio.run(swarm.run())
    t1 = time.time()

    ok = sum(1 for r in results if r.get("ok"))
    print(f"\n{'='*70}")
    print(f"  SWARM COMPLETE: {ok}/{len(results)} topics in {t1-t0:.1f}s")
    print(f"  Database: {swarm.db.path}")
    print(f"  Log: {swarm.bus.log_file.name}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
