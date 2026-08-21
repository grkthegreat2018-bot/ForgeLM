"""Tool registry for the discovery self-play loop.

⚠️  THIS IS THE SELF-PLAY DISCOVERY TOOL REGISTRY — NOT the inference tool
    registry. These tools are used during self-play training only.
    For inference (model serving), use research/inference/engine_tools.py
    which has a separate, non-overlapping set of tools (library_*, engine_*,
    math_eval, web_search via Tavily/Exa, file_*, etc.).

Each tool is a plain function (args dict) -> result dict. The loop parses the
LLM's tool-call JSON, dispatches here, and feeds the JSON result back.

Tools exposed to the LLM during self-play:
  think          — record a train-of-thought entry (theorizing)
  sudo_think     — meta-reasoning about its own process / strategy
  run_script     — execute Python in a sandboxed subprocess (timeout, no net)
  web_search     — internet research via DuckDuckGo HTML (returns snippets)
  save_research  — persist a research finding to the DB
  propose_theory — log a hypothesis
  update_theory  — change a theory's status / evidence tallies
  record_discovery — record a confirmed finding
  query_db       — read-only SELECT against the LLM's own memory
  migrate_schema — LLM-initiated, audited DDL (additive only)
  finish_session — end the current discovery session

`run_script` reuses SandboxExecutor from self_play_sandbox for safe execution.
`web_search` uses stdlib urllib + DuckDuckGo HTML so no API key is needed.
"""
from __future__ import annotations

import json
import re
import time
from html import unescape
from typing import Any, Callable
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

from research.self_play.discovery.discovery_db import DiscoveryDB


# ── sandboxed script execution ────────────────────────────────────────
def _make_executor():
    """Lazily build a SandboxExecutor (heavy import deferred)."""
    from research.self_play.self_play_sandbox import SandboxExecutor
    return SandboxExecutor(timeout_s=8.0, memory_limit_mb=512,
                           use_persistent=False)


_EXEC = None


def _run_script(code: str) -> dict:
    global _EXEC
    if _EXEC is None:
        _EXEC = _make_executor()
    t0 = time.time()
    try:
        res = _EXEC.execute(code, expected_output=None)
        return {
            "stdout": (res.get("stdout") or "")[:4000],
            "stderr": (res.get("stderr") or "")[:4000],
            "returncode": res.get("returncode", -1),
            "exec_ms": round((time.time() - t0) * 1000, 1),
            "ok": res.get("returncode") == 0,
        }
    except Exception as e:  # sandbox itself blew up
        return {"stdout": "", "stderr": f"sandbox error: {e}",
                "returncode": -1, "exec_ms": round((time.time() - t0) * 1000, 1),
                "ok": False}


# ── web search (DuckDuckGo HTML, no API key) ──────────────────────────
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
_RE_RESULT = re.compile(
    r'<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?'
    r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', re.DOTALL)
_RE_TAG = re.compile(r"<[^>]+>")


def _web_search(query: str, n: int = 5) -> dict:
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    req = Request(url, headers={"User-Agent": _UA, "Accept-Language": "en"})
    try:
        with urlopen(req, timeout=12) as r:
            html = r.read().decode("utf-8", "ignore")
    except Exception as e:
        return {"results": [], "error": f"fetch failed: {e}"}
    results = []
    for m in _RE_RESULT.finditer(html):
        if len(results) >= n:
            break
        href = unescape(m.group(1))
        # DDG wraps URLs in a redirect; strip the leading //duckduckgo.com/l/?uddg=
        if "uddg=" in href:
            from urllib.parse import parse_qs, urlparse
            qs = parse_qs(urlparse(href).query)
            href = qs.get("uddg", [href])[0]
        title = unescape(_RE_TAG.sub("", m.group(2))).strip()
        snippet = unescape(_RE_TAG.sub("", m.group(3))).strip()
        results.append({"url": href, "title": title[:200], "snippet": snippet[:400]})
    return {"results": results, "error": None if results else "no results parsed"}


# ── Wikipedia API (free, no key) ──────────────────────────────────────
def _wikipedia_search(query: str, n: int = 3) -> dict:
    """Search Wikipedia via the REST API. Returns summaries."""
    try:
        search_url = (f"https://en.wikipedia.org/w/api.php?action=query&list=search"
                      f"&format=json&srlimit={n}&srsearch={quote_plus(query)}")
        req = Request(search_url, headers={"User-Agent": _UA})
        with urlopen(req, timeout=12) as r:
            data = json.loads(r.read().decode("utf-8", "ignore"))
        items = data.get("query", {}).get("search", [])
        results = []
        for item in items[:n]:
            title = item.get("title", "")
            snippet = _RE_TAG.sub("", item.get("snippet", "")).strip()
            results.append({"title": title, "snippet": snippet[:400],
                            "url": f"https://en.wikipedia.org/wiki/{quote_plus(title)}"})
        return {"results": results, "error": None if results else "no results"}
    except Exception as e:
        return {"results": [], "error": str(e)}


# ── arXiv API (free, no key) ──────────────────────────────────────────
def _arxiv_search(query: str, n: int = 3) -> dict:
    """Search arXiv for academic papers."""
    try:
        url = (f"http://export.arxiv.org/api/query?search_query=all:{quote_plus(query)}"
               f"&start=0&max_results={n}")
        req = Request(url, headers={"User-Agent": _UA})
        with urlopen(req, timeout=15) as r:
            xml = r.read().decode("utf-8", "ignore")
        # Parse atom feed entries (lightweight regex, no lxml dependency)
        entries = re.findall(r"<entry>(.*?)</entry>", xml, re.DOTALL)
        results = []
        for entry in entries[:n]:
            title = re.search(r"<title>(.*?)</title>", entry, re.DOTALL)
            summary = re.search(r"<summary>(.*?)</summary>", entry, re.DOTALL)
            link = re.search(r'<id>(.*?)</id>', entry, re.DOTALL)
            published = re.search(r"<published>(.*?)</published>", entry, re.DOTALL)
            if title:
                results.append({
                    "title": _RE_TAG.sub("", title.group(1)).strip()[:200],
                    "summary": summary.group(1).strip()[:400] if summary else "",
                    "url": link.group(1).strip() if link else "",
                    "published": published.group(1)[:10] if published else "",
                })
        return {"results": results, "error": None if results else "no results"}
    except Exception as e:
        return {"results": [], "error": str(e)}


# ── fetch URL (extract text from any web page) ────────────────────────
def _fetch_url(url: str, max_chars: int = 2000) -> dict:
    """Fetch a URL and extract readable text (strip HTML tags)."""
    try:
        req = Request(url, headers={"User-Agent": _UA, "Accept-Language": "en"})
        with urlopen(req, timeout=12) as r:
            html = r.read().decode("utf-8", "ignore")
        # Remove scripts, styles, tags
        html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
        text = _RE_TAG.sub(" ", html)
        text = re.sub(r"\s+", " ", text).strip()
        return {"text": text[:max_chars], "url": url, "chars": len(text), "error": None}
    except Exception as e:
        return {"text": "", "url": url, "error": str(e)}


# ── calculate (safe math evaluation) ──────────────────────────────────
def _calculate(code: str) -> dict:
    """Evaluate a math expression safely via the sandbox."""
    # Wrap in print() so output is captured
    wrapped = f"print({code})" if not code.strip().startswith("print") else code
    return _run_script(wrapped)


# ── tool registry ─────────────────────────────────────────────────────
class ToolRegistry:
    """Holds bound tools + their schemas for the system prompt.

    Tools close over a DiscoveryDB and the current session_id so the LLM
    doesn't have to pass them every call.
    """

    def __init__(self, db: DiscoveryDB, session_id: str):
        self.db = db
        self.session_id = session_id
        self.tools: dict[str, Callable[[dict], dict]] = {}
        self.schemas: list[dict] = []
        self._register_all()

    def _register(self, name: str, desc: str, params: dict,
                  fn: Callable[[dict], dict]) -> None:
        self.tools[name] = fn
        self.schemas.append({"name": name, "description": desc, "parameters": params})

    def _register_all(self) -> None:
        db, sid = self.db, self.session_id

        def think(args: dict) -> dict:
            tid = db.add_thought(sid, "think", args.get("content", ""),
                                 confidence=args.get("confidence"))
            db.emit("think", {"id": tid, "content": args.get("content", "")[:200]}, sid)
            return {"thought_id": tid, "saved": True}

        def sudo_think(args: dict) -> dict:
            tid = db.add_thought(sid, "sudo_think", args.get("content", ""))
            db.emit("sudo_think", {"content": args.get("content", "")[:200]}, sid)
            return {"thought_id": tid, "saved": True}

        def run_script(args: dict) -> dict:
            code = args.get("code", "")
            res = _run_script(code)
            sid_field = db.add_script(sid, code, stdout=res["stdout"],
                                      stderr=res["stderr"], returncode=res["returncode"],
                                      exec_ms=res["exec_ms"])
            db.emit("run_script", {"id": sid_field, "ok": res["ok"],
                                   "stderr": res["stderr"][:200]}, sid)
            res["script_id"] = sid_field
            return res

        def web_search(args: dict) -> dict:
            res = _web_search(args.get("query", ""), n=int(args.get("n", 5)))
            db.emit("web_search", {"query": args.get("query", ""),
                                   "n_results": len(res["results"])}, sid)
            return res

        def wikipedia_search(args: dict) -> dict:
            res = _wikipedia_search(args.get("query", ""), n=int(args.get("n", 3)))
            db.emit("wikipedia_search", {"query": args.get("query", ""),
                                         "n_results": len(res["results"])}, sid)
            return res

        def arxiv_search(args: dict) -> dict:
            res = _arxiv_search(args.get("query", ""), n=int(args.get("n", 3)))
            db.emit("arxiv_search", {"query": args.get("query", ""),
                                     "n_results": len(res["results"])}, sid)
            return res

        def fetch_url(args: dict) -> dict:
            res = _fetch_url(args.get("url", ""), max_chars=int(args.get("max_chars", 2000)))
            db.emit("fetch_url", {"url": args.get("url", "")[:120],
                                  "chars": res.get("chars", 0)}, sid)
            return res

        def calculate(args: dict) -> dict:
            res = _calculate(args.get("code", args.get("expression", "")))
            db.emit("calculate", {"ok": res.get("ok", False),
                                  "stdout": res.get("stdout", "")[:100]}, sid)
            return res

        def save_research(args: dict) -> dict:
            rid = db.add_research(sid, args.get("query", ""),
                                  url=args.get("url"), title=args.get("title"),
                                  summary=args.get("summary"),
                                  raw_snippet=args.get("snippet"))
            db.emit("save_research", {"id": rid, "query": args.get("query", "")[:120]}, sid)
            return {"research_id": rid, "saved": True}

        def propose_theory(args: dict) -> dict:
            tid = db.add_theory(sid, args.get("statement", ""),
                                notes=args.get("notes", ""))
            db.emit("propose_theory", {"id": tid,
                    "statement": args.get("statement", "")[:160]}, sid)
            return {"theory_id": tid, "saved": True}

        def update_theory(args: dict) -> dict:
            db.update_theory(int(args["theory_id"]), status=args.get("status"),
                             delta_for=int(args.get("evidence_for", 0)),
                             delta_against=int(args.get("evidence_against", 0)),
                             notes=args.get("notes"))
            db.emit("update_theory", args, sid)
            return {"updated": True}

        def record_discovery(args: dict) -> dict:
            did = db.add_discovery(sid, args.get("summary", ""),
                                   theory_id=args.get("theory_id"),
                                   confidence=args.get("confidence"))
            db.emit("record_discovery", {"id": did,
                    "summary": args.get("summary", "")[:200]}, sid)
            return {"discovery_id": did, "saved": True}

        def query_db(args: dict) -> dict:
            try:
                rows = db.query(args.get("sql", ""), tuple(args.get("params", [])))
                return {"rows": rows[:50], "n": len(rows), "truncated": len(rows) > 50}
            except Exception as e:
                return {"rows": [], "error": str(e)}

        def migrate_schema(args: dict) -> dict:
            res = db.migrate_schema(args.get("sql", ""), reason=args.get("reason", ""),
                                    session_id=sid)
            db.emit("migrate_schema", res, sid)
            return res

        def finish_session(args: dict) -> dict:
            db.end_session(sid, args.get("summary", ""))
            db.emit("finish_session", {"summary": args.get("summary", "")[:200]}, sid)
            return {"finished": True}

        def set_goal(args: dict) -> dict:
            """Let the model set its own goal for the current session.

            The model proposes a goal, which is recorded in the DB. This enables
            self-directed exploration where the model decides what to investigate.
            """
            goal = args.get("goal", "")
            tid = db.add_thought(sid, "self_goal", goal, confidence=0.9)
            db.emit("set_goal", {"goal": goal[:200], "thought_id": tid}, sid)
            return {"goal_id": tid, "saved": True,
                    "note": "Goal recorded. Now pursue it using your tools."}

        def ask_clarification(args: dict) -> dict:
            """Let the model ask a clarifying question about the task.

            Unlike outputting text (which ends the task), this tool records
            the question and provides a synthetic response, allowing the
            loop to continue. This teaches the model to seek clarification
            when tasks are ambiguous (SynthAgent, ACL 2026).
            """
            question = args.get("question", "")
            tid = db.add_thought(sid, "clarification", question, confidence=0.5)
            db.emit("ask_clarification", {"question": question[:200]}, sid)
            # Provide a generic encouraging response — the model should
            # proceed with its best interpretation
            return {
                "question_id": tid,
                "response": "Proceed with your best interpretation of the task. "
                            "Make reasonable assumptions and document them with think.",
                "note": "Clarification recorded. Continue working on the task.",
            }

        def summarize_context(args: dict) -> dict:
            """Let the model proactively summarize its own context.

            The model writes a summary of what it's learned so far, which
            gets saved to the DB. This helps with long conversations where
            the model needs to consolidate findings before continuing.
            """
            summary = args.get("summary", "")
            tid = db.add_thought(sid, "context_summary", summary,
                                 confidence=args.get("confidence", 0.8))
            db.emit("summarize_context", {"summary": summary[:200]}, sid)
            return {"saved": True, "thought_id": tid,
                    "note": "Summary saved. Use this to consolidate findings before continuing."}

        self._register("think", "Record a train-of-thought idea or observation.",
                       {"content": "string", "confidence": "number 0-1 optional"}, think)
        self._register("sudo_think",
                       "Meta-reason about your own process: what to explore next, "
                       "whether your strategy is working, what to change.",
                       {"content": "string"}, sudo_think)
        self._register("run_script",
                       "Execute Python code in a sandbox (8s timeout, no network). "
                       "Use for experiments, calculations, probing ideas.",
                       {"code": "string"}, run_script)
        self._register("web_search",
                       "Search the internet via DuckDuckGo. Returns result snippets.",
                       {"query": "string", "n": "int optional, default 5"}, web_search)
        self._register("wikipedia_search",
                       "Search Wikipedia for encyclopedic knowledge. Returns article summaries. "
                       "Best for factual questions, definitions, history, science.",
                       {"query": "string", "n": "int optional, default 3"}, wikipedia_search)
        self._register("arxiv_search",
                       "Search arXiv for academic papers on AI, ML, CS, physics, math. "
                       "Returns titles, abstracts, and links. Best for research questions.",
                       {"query": "string", "n": "int optional, default 3"}, arxiv_search)
        self._register("fetch_url",
                       "Fetch a web page and extract its text content. Use to read articles, "
                       "documentation, or pages found via web_search.",
                       {"url": "string", "max_chars": "int optional, default 2000"}, fetch_url)
        self._register("calculate",
                       "Evaluate a math expression or short Python calculation. "
                       "Example: calculate('2**10 + 3*5') or calculate('sum(range(100))').",
                       {"code": "string"}, calculate)
        self._register("save_research",
                       "Persist a web research finding to your database.",
                       {"query": "string", "url": "string", "title": "string",
                        "summary": "string", "snippet": "string"}, save_research)
        self._register("propose_theory",
                       "Log a hypothesis to track. Status starts 'open'.",
                       {"statement": "string", "notes": "string optional"}, propose_theory)
        self._register("update_theory",
                       "Update a theory's status/evidence. status: open|supported|refuted|abandoned.",
                       {"theory_id": "int", "status": "string optional",
                        "evidence_for": "int optional", "evidence_against": "int optional",
                        "notes": "string optional"}, update_theory)
        self._register("record_discovery",
                       "Record a confirmed finding you've verified.",
                       {"summary": "string", "theory_id": "int optional",
                        "confidence": "number 0-1 optional"}, record_discovery)
        self._register("query_db",
                       "Read-only SELECT/WITH against your own memory. Inspect past "
                       "thoughts, scripts, theories, research, discoveries.",
                       {"sql": "string", "params": "list optional"}, query_db)
        self._register("migrate_schema",
                       "Add/refactor database tables (CREATE/ALTER/INDEX/VIEW only; "
                       "DROP TABLE and DML are blocked). Audited.",
                       {"sql": "string", "reason": "string"}, migrate_schema)
        self._register("summarize_context",
                       "Summarize what you've learned so far in this session. "
                       "Use when the conversation is getting long and you need to "
                       "consolidate findings before continuing. The summary is saved "
                       "to your memory and can be retrieved with query_db.",
                       {"summary": "string", "confidence": "number 0-1 optional"},
                       summarize_context)
        self._register("finish_session",
                       "End this discovery session with a summary.",
                       {"summary": "string"}, finish_session)
        self._register("set_goal",
                       "Set your own goal for this session. Use this to propose what you want "
                       "to investigate or accomplish, then pursue it with your other tools. "
                       "This enables self-directed exploration and learning.",
                       {"goal": "string"}, set_goal)
        self._register("ask_clarification",
                       "Ask a clarifying question about the task if something is ambiguous. "
                       "Unlike outputting text (which ends the task), this records your "
                       "question and lets you continue working. Use when the task is unclear.",
                       {"question": "string"}, ask_clarification)

    def call(self, name: str, args: dict) -> dict:
        fn = self.tools.get(name)
        if fn is None:
            return {"error": f"unknown tool: {name}"}
        try:
            return fn(args or {})
        except Exception as e:
            return {"error": f"tool '{name}' failed: {e}"}

    def prompt_block(self) -> str:
        """JSON description of all tools (legacy — use tool_definitions for LFM2.5)."""
        return json.dumps(self.schemas, indent=2, ensure_ascii=False)

    def tool_definitions(self) -> list[dict]:
        """Return tool definitions in LFM2.5 format for the chat template.

        LFM2.5 expects: [{"name": "...", "description": "...", "parameters": {...}}]
        where parameters is a JSON schema-like dict.
        """
        defs = []
        for s in self.schemas:
            params = {"type": "object", "properties": {}}
            # The schemas store params as a descriptive string; we expose them
            # as free-form string properties so the model can pass any args.
            for pname, pdesc in s.get("parameters", {}).items():
                ptype = "string"
                if "int" in pdesc.lower() or "number" in pdesc.lower():
                    ptype = "string"  # keep as string — model outputs Pythonic literals
                params["properties"][pname] = {"type": ptype, "description": pdesc}
            defs.append({
                "name": s["name"],
                "description": s["description"],
                "parameters": params,
            })
        return defs
