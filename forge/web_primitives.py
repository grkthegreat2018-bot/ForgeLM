"""Shared web fetch/parse primitives (critique F20).

Extracted from the duplicated logic in:
- ``forge_gui/api/web_tools.py`` (agent harness)
- ``forge/self_play/discovery/discovery_tools.py`` (self-play discovery)

Stdlib-only (urllib, re, html.parser) — no third-party deps, no imports from
forge_gui or forge.self_play. Both subsystems import from here to avoid
duplicating URL safety checks, HTTP fetching, DuckDuckGo HTML parsing, and
HTML-to-text stripping.
"""
from __future__ import annotations

import json
import re
from html import unescape
from typing import Any
from urllib.parse import parse_qs, quote_plus, urlparse
from urllib.request import Request, urlopen

# ── constants ───────────────────────────────────────────────────────────
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
TIMEOUT_S = 12
MAX_FETCH_CHARS = 4000
MAX_SNIPPET = 400
MAX_TITLE = 200
DEFAULT_N = 5

# DuckDuckGo HTML result-block regex.
RE_RESULT = re.compile(
    r'<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?'
    r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', re.DOTALL)
RE_TAG = re.compile(r"<[^>]+>")

# URL schemes permitted for web_fetch (block javascript:/file:/data:/...).
ALLOWED_SCHEMES = {"http", "https"}


def is_safe_url(url: str) -> bool:
    """True iff the URL has an http/https scheme."""
    try:
        return urlparse(url).scheme.lower() in ALLOWED_SCHEMES
    except Exception:
        return False


def strip_ddg_redirect(href: str) -> str:
    """DuckDuckGo wraps result URLs in a redirect; unwrap ``uddg=``."""
    if "uddg=" in href:
        qs = parse_qs(urlparse(href).query)
        return qs.get("uddg", [href])[0]
    return href


def html_to_text(html: str) -> str:
    """Strip HTML tags and collapse whitespace to a single line."""
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html,
                  flags=re.DOTALL | re.IGNORECASE)
    text = RE_TAG.sub(" ", html)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def http_get(url: str, timeout: float = TIMEOUT_S,
             accept_language: str = "en") -> str:
    """GET a URL and return decoded text. Raises on network error."""
    req = Request(url, headers={"User-Agent": UA, "Accept-Language": accept_language})
    with urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "ignore")


def parse_ddg_html(html: str, n: int = DEFAULT_N) -> list[dict[str, str]]:
    """Parse DuckDuckGo HTML results into a list of {url, title, snippet}."""
    results: list[dict[str, str]] = []
    for m in RE_RESULT.finditer(html):
        if len(results) >= n:
            break
        href = unescape(m.group(1))
        href = strip_ddg_redirect(href)
        title = unescape(RE_TAG.sub("", m.group(2))).strip()
        snippet = unescape(RE_TAG.sub("", m.group(3))).strip()
        results.append({
            "url": href,
            "title": title[:MAX_TITLE],
            "snippet": snippet[:MAX_SNIPPET],
        })
    return results


def ddg_search(query: str, n: int = DEFAULT_N) -> dict[str, Any]:
    """DuckDuckGo HTML search (no API key). Returns {results, error}."""
    if not query.strip():
        return {"results": [], "error": "empty query"}
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    try:
        html = http_get(url)
    except Exception as e:
        return {"results": [], "error": f"fetch failed: {e}"}
    results = parse_ddg_html(html, n=n)
    return {"results": results, "error": None if results else "no results parsed"}


def wikipedia_search(query: str, n: int = 3) -> dict[str, Any]:
    """Search Wikipedia via the REST API. Returns summaries."""
    try:
        search_url = (
            f"https://en.wikipedia.org/w/api.php?action=query&list=search"
            f"&format=json&srlimit={n}&srsearch={quote_plus(query)}"
        )
        data = json.loads(http_get(search_url, accept_language=""))
        items = data.get("query", {}).get("search", [])
        results = []
        for item in items[:n]:
            title = item.get("title", "")
            snippet = RE_TAG.sub("", item.get("snippet", "")).strip()
            results.append({
                "title": title,
                "snippet": snippet[:MAX_SNIPPET],
                "url": f"https://en.wikipedia.org/wiki/{quote_plus(title)}",
            })
        return {"results": results, "error": None if results else "no results"}
    except Exception as e:
        return {"results": [], "error": str(e)}


def arxiv_search(query: str, n: int = 3) -> dict[str, Any]:
    """Search arXiv for academic papers via the Atom API."""
    try:
        url = (f"http://export.arxiv.org/api/query?search_query=all:{quote_plus(query)}"
               f"&start=0&max_results={n}")
        xml = http_get(url, timeout=15)
        entries = re.findall(r"<entry>(.*?)</entry>", xml, re.DOTALL)
        results = []
        for entry in entries[:n]:
            title = re.search(r"<title>(.*?)</title>", entry, re.DOTALL)
            summary = re.search(r"<summary>(.*?)</summary>", entry, re.DOTALL)
            link = re.search(r'<id>(.*?)</id>', entry, re.DOTALL)
            published = re.search(r"<published>(.*?)</published>", entry, re.DOTALL)
            if title:
                results.append({
                    "title": RE_TAG.sub("", title.group(1)).strip()[:MAX_TITLE],
                    "summary": summary.group(1).strip()[:MAX_SNIPPET] if summary else "",
                    "url": link.group(1).strip() if link else "",
                    "published": published.group(1)[:10] if published else "",
                })
        return {"results": results, "error": None if results else "no results"}
    except Exception as e:
        return {"results": [], "error": str(e)}


def fetch_url(url: str, max_chars: int = MAX_FETCH_CHARS) -> dict[str, Any]:
    """Fetch a URL and extract readable text (strip HTML tags)."""
    if not url.strip():
        return {"text": "", "url": url, "error": "empty url"}
    if not is_safe_url(url):
        return {"text": "", "url": url,
                "error": f"unsupported URL scheme (only {sorted(ALLOWED_SCHEMES)})"}
    try:
        html = http_get(url)
        text = html_to_text(html)
        return {"text": text[:max_chars], "url": url,
                "chars": len(text), "truncated": len(text) > max_chars,
                "error": None}
    except Exception as e:
        return {"text": "", "url": url, "error": f"fetch failed: {e}"}
