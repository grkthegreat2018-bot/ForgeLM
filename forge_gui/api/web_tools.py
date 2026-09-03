"""Keyless web tools for the ForgeAI agent harness.

Exposes ``web_search``, ``web_fetch``, ``wikipedia_search``, and
``arxiv_search`` to the model so the agent can do real-time research
without any API key. All HTTP is stdlib ``urllib`` GET-only — no
side-effecting requests, no auth, no dependencies.

The search/fetch primitives reuse the proven DuckDuckGo-HTML / Wikipedia /
arXiv / tag-stripping implementation from
``research/self_play/discovery/discovery_tools.py`` (battle-tested in the
self-play discovery loop). They are duplicated here rather than imported
because the discovery versions are private (``_``-prefixed) and coupled to
the self-play DB emit pattern; the agent harness is a separate subsystem
(``forge_gui/api/``) that must not depend on ``research/self_play/``.

Safety:
- Only ``http``/``https`` URLs are accepted for ``web_fetch`` —
  ``javascript:``, ``file:``, ``data:``, ``ftp:`` etc. are rejected
  before any request is made.
- All requests are GET with a fixed browser User-Agent and a hard timeout.
- Output is capped (``MAX_FETCH_CHARS`` / ``MAX_SNIPPET``) to keep the
  tool result inside the engine's KV-cache budget.
"""
from __future__ import annotations

import json
import re
from html import unescape
from typing import Any, Optional
from urllib.parse import parse_qs, quote_plus, urlparse
from urllib.request import Request, urlopen

# ── constants ───────────────────────────────────────────────────────────
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
_TIMEOUT_S = 12
_MAX_FETCH_CHARS = 4000   # cap fetched page text
_MAX_SNIPPET = 400        # cap each search-result snippet
_MAX_TITLE = 200          # cap each search-result title
_DEFAULT_N = 5            # default number of search results

# DuckDuckGo HTML result-block regex (proven in discovery_tools).
_RE_RESULT = re.compile(
    r'<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?'
    r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', re.DOTALL)
_RE_TAG = re.compile(r"<[^>]+>")

# URL schemes permitted for web_fetch (block javascript:/file:/data:/...).
_ALLOWED_SCHEMES = {"http", "https"}


def _is_safe_url(url: str) -> bool:
    """True iff the URL has an http/https scheme."""
    try:
        return urlparse(url).scheme.lower() in _ALLOWED_SCHEMES
    except Exception:
        return False


def _strip_ddg_redirect(href: str) -> str:
    """DuckDuckGo wraps result URLs in a redirect; unwrap ``uddg=``."""
    if "uddg=" in href:
        qs = parse_qs(urlparse(href).query)
        return qs.get("uddg", [href])[0]
    return href


# ── primitives ──────────────────────────────────────────────────────────
def _web_search(query: str, n: int = _DEFAULT_N) -> dict:
    """DuckDuckGo HTML search (no API key). Returns {results, error}."""
    if not query.strip():
        return {"results": [], "error": "empty query"}
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    req = Request(url, headers={"User-Agent": _UA, "Accept-Language": "en"})
    try:
        with urlopen(req, timeout=_TIMEOUT_S) as r:
            html = r.read().decode("utf-8", "ignore")
    except Exception as e:
        return {"results": [], "error": f"fetch failed: {e}"}
    results = []
    for m in _RE_RESULT.finditer(html):
        if len(results) >= n:
            break
        href = unescape(_strip_ddg_redirect(m.group(1)))
        title = unescape(_RE_TAG.sub("", m.group(2))).strip()[:_MAX_TITLE]
        snippet = unescape(_RE_TAG.sub("", m.group(3))).strip()[:_MAX_SNIPPET]
        results.append({"url": href, "title": title, "snippet": snippet})
    return {"results": results,
            "error": None if results else "no results parsed"}


def _wikipedia_search(query: str, n: int = 3) -> dict:
    """Wikipedia REST API search (no key). Returns {results, error}."""
    if not query.strip():
        return {"results": [], "error": "empty query"}
    search_url = (
        f"https://en.wikipedia.org/w/api.php?action=query&list=search"
        f"&format=json&srlimit={n}&srsearch={quote_plus(query)}")
    req = Request(search_url, headers={"User-Agent": _UA})
    try:
        with urlopen(req, timeout=_TIMEOUT_S) as r:
            data = json.loads(r.read().decode("utf-8", "ignore"))
    except Exception as e:
        return {"results": [], "error": str(e)}
    items = data.get("query", {}).get("search", [])
    results = []
    for item in items[:n]:
        title = item.get("title", "")
        snippet = _RE_TAG.sub("", item.get("snippet", "")).strip()[:_MAX_SNIPPET]
        results.append({
            "title": title, "snippet": snippet,
            "url": f"https://en.wikipedia.org/wiki/{quote_plus(title)}"})
    return {"results": results,
            "error": None if results else "no results"}


def _arxiv_search(query: str, n: int = 3) -> dict:
    """arXiv API search (no key). Returns {results, error}."""
    if not query.strip():
        return {"results": [], "error": "empty query"}
    url = (f"http://export.arxiv.org/api/query?search_query=all:"
           f"{quote_plus(query)}&start=0&max_results={n}")
    req = Request(url, headers={"User-Agent": _UA})
    try:
        with urlopen(req, timeout=_TIMEOUT_S + 3) as r:
            xml = r.read().decode("utf-8", "ignore")
    except Exception as e:
        return {"results": [], "error": str(e)}
    entries = re.findall(r"<entry>(.*?)</entry>", xml, re.DOTALL)
    results = []
    for entry in entries[:n]:
        title = re.search(r"<title>(.*?)</title>", entry, re.DOTALL)
        summary = re.search(r"<summary>(.*?)</summary>", entry, re.DOTALL)
        link = re.search(r"<id>(.*?)</id>", entry, re.DOTALL)
        published = re.search(r"<published>(.*?)</published>", entry, re.DOTALL)
        if title:
            results.append({
                "title": _RE_TAG.sub("", title.group(1)).strip()[:_MAX_TITLE],
                "summary": summary.group(1).strip()[:_MAX_SNIPPET] if summary else "",
                "url": link.group(1).strip() if link else "",
                "published": published.group(1)[:10] if published else "",
            })
    return {"results": results,
            "error": None if results else "no results"}


def _web_fetch(url: str, max_chars: int = _MAX_FETCH_CHARS) -> dict:
    """Fetch a URL and extract readable text (strip HTML). GET-only.

    Rejects non-http(s) schemes before any network call.
    """
    if not url.strip():
        return {"text": "", "url": url, "error": "empty url"}
    if not _is_safe_url(url):
        return {"text": "", "url": url,
                "error": f"unsupported URL scheme (only {sorted(_ALLOWED_SCHEMES)})"}
    req = Request(url, headers={"User-Agent": _UA, "Accept-Language": "en"})
    try:
        with urlopen(req, timeout=_TIMEOUT_S) as r:
            html = r.read().decode("utf-8", "ignore")
    except Exception as e:
        return {"text": "", "url": url, "error": f"fetch failed: {e}"}
    # Drop scripts/styles, then all tags, then collapse whitespace.
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html,
                  flags=re.DOTALL | re.IGNORECASE)
    text = _RE_TAG.sub(" ", html)
    text = re.sub(r"\s+", " ", text).strip()
    return {"text": text[:max_chars], "url": url,
            "chars": len(text), "truncated": len(text) > max_chars,
            "error": None}


# ── tool definitions (OpenAI function-calling shape) ────────────────────
_WEB_TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for real-time info, docs, and news. "
                "Returns {url, title, snippet} per result. Use web_fetch "
                "to read a full page from a result url. No API key needed."),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "search query"},
                    "n": {"type": "integer",
                          "description": "max results (default 5, max 10)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": (
                "Fetch a URL and return its text content. Use to read "
                "full articles or docs found via web_search. Only http(s) "
                "URLs accepted. HTML tags stripped, output truncated."),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "http(s) URL to fetch"},
                    "max_chars": {"type": "integer",
                                  "description": "max chars to return (default 4000)"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wikipedia_search",
            "description": (
                "Search Wikipedia (free API). Returns {title, snippet, url} "
                "summaries. Best for encyclopedic / factual background."),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "search query"},
                    "n": {"type": "integer",
                          "description": "max results (default 3)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "arxiv_search",
            "description": (
                "Search arXiv for academic papers (free API). Returns "
                "{title, summary, url, published}. Best for ML/AI/math "
                "research papers."),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "search query"},
                    "n": {"type": "integer",
                          "description": "max results (default 3)"},
                },
                "required": ["query"],
            },
        },
    },
]


def web_tool_defs() -> list[dict]:
    """Return the OpenAI-style tool definition list for the web tools."""
    return [dict(d) for d in _WEB_TOOL_DEFS]


# ── manager ─────────────────────────────────────────────────────────────
class WebTools:
    """Stateless holder for the web tool implementations.

    Mirrors the ``TimeManager`` / ``BackupManager`` pattern so the
    :class:`ToolHarness` can hold a single instance and dispatch to it.
    All methods are safe to call from any thread (each opens its own
    ``urlopen`` with a hard timeout).
    """

    NAMES = frozenset(d["function"]["name"] for d in _WEB_TOOL_DEFS)

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def execute(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a web tool call. Returns the raw result dict.

        On success the ``error`` key (which the primitives set to ``None``)
        is removed so the harness ``"error" in result`` ok-check works.
        """
        if not self.enabled:
            return {"error": "web tools disabled"}
        n = int(args.get("n", _DEFAULT_N))
        # clamp n to a sane range to avoid huge responses
        n = max(1, min(n, 10))
        if name == "web_search":
            res = _web_search(args.get("query", ""), n=n)
        elif name == "web_fetch":
            max_chars = int(args.get("max_chars", _MAX_FETCH_CHARS))
            max_chars = max(200, min(max_chars, 8000))
            res = _web_fetch(args.get("url", ""), max_chars=max_chars)
        elif name == "wikipedia_search":
            res = _wikipedia_search(args.get("query", ""), n=n)
        elif name == "arxiv_search":
            res = _arxiv_search(args.get("query", ""), n=n)
        else:
            return {"error": f"unknown web tool: {name}"}
        # normalize: drop a falsy error so the harness ok-check passes
        if not res.get("error"):
            res.pop("error", None)
        return res
