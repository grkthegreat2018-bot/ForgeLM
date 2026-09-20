"""Unit tests for forge_gui.api.web_tools (keyless web search/fetch).

No real network calls: ``urllib.request.urlopen`` is monkeypatched to
return canned HTML / JSON / raise errors. Tests cover:
- tool definition shape (OpenAI function-calling schema)
- URL scheme validation (block javascript:/file:/data:, allow http/https)
- DuckDuckGo HTML parsing (result + redirect unwrap)
- Wikipedia / arXiv JSON/XML parsing
- network error handling (graceful {error: ...} dict)
- WebTools.execute dispatch + n clamping
- ToolHarness integration (defs include web tools, dispatch works)
"""
import io
import json
from unittest.mock import patch

import pytest

from forge_gui.api.web_tools import (
    WebTools, web_tool_defs,
)
from forge.web_primitives import (
    arxiv_search as _arxiv_search,
    ddg_search as _web_search,
    fetch_url as _web_fetch,
    is_safe_url as _is_safe_url,
    strip_ddg_redirect as _strip_ddg_redirect,
    wikipedia_search as _wikipedia_search,
)
from forge_gui.api.tool_harness import ToolHarness


# ── helpers ─────────────────────────────────────────────────────────────
class _FakeResp:
    """Minimal context-manager response object mimicking urlopen's return."""

    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fake_urlopen(data: bytes | str):
    """Return a callable that yields a _FakeResp wrapping `data`."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return lambda *a, **kw: _FakeResp(data)


# ── tool definition shape ───────────────────────────────────────────────
def test_web_tool_defs_shape():
    defs = web_tool_defs()
    names = {d["function"]["name"] for d in defs}
    assert {"web_search", "news_search", "web_fetch",
            "wikipedia_search", "arxiv_search"} == names
    for d in defs:
        assert d["type"] == "function"
        params = d["function"]["parameters"]
        assert params["type"] == "object"
        assert "required" in params


def test_web_tools_names_matches_defs():
    def_names = {d["function"]["name"] for d in web_tool_defs()}
    assert WebTools.NAMES == def_names


# ── URL safety ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("url,ok", [
    ("https://example.com", True),
    ("http://example.com/path?q=1", True),
    ("HTTPS://Example.com", True),
    ("javascript:alert(1)", False),
    ("file:///etc/passwd", False),
    ("data:text/html,<script>", False),
    ("ftp://example.com", False),
    ("", False),
])
def test_is_safe_url(url, ok):
    assert _is_safe_url(url) is ok


def test_strip_ddg_redirect():
    wrapped = "//duckduckgo.com/l/?uddg=https%3A%2F%2Freal.com%2Fpage&rut=abc"
    assert _strip_ddg_redirect(wrapped) == "https://real.com/page"
    # non-redirect href passes through unchanged
    assert _strip_ddg_redirect("https://plain.com/x") == "https://plain.com/x"


# ── web_search (DuckDuckGo HTML) ────────────────────────────────────────
_DDG_HTML = """
<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.python.org%2F3%2Flibrary%2Furllib.html">urllib — Python docs</a>
<a class="result__snippet">This module provides a high-level interface ...</a>
<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2F2">Second Result</a>
<a class="result__snippet">Second snippet <b>text</b> here.</a>
"""


def test_web_search_parses_results():
    with patch("forge.web_primitives.urlopen",
               _fake_urlopen(_DDG_HTML)):
        res = _web_search("urllib", n=5)
    assert res["error"] is None
    assert len(res["results"]) == 2
    first = res["results"][0]
    assert first["url"] == "https://docs.python.org/3/library/urllib.html"
    assert first["title"] == "urllib — Python docs"
    assert "high-level interface" in first["snippet"]
    # second snippet has tags stripped
    assert "<b>" not in res["results"][1]["snippet"]


def test_web_search_empty_query():
    res = _web_search("   ")
    assert res["results"] == []
    assert "empty" in res["error"]


def test_web_search_network_error():
    def boom(*a, **kw):
        raise OSError("connection refused")
    with patch("forge.web_primitives.urlopen", boom):
        res = _web_search("anything")
    assert res["results"] == []
    assert "fetch failed" in res["error"]


def test_web_search_no_results():
    with patch("forge.web_primitives.urlopen",
               _fake_urlopen("<html><body>no results here</body></html>")):
        res = _web_search("zzz")
    assert res["results"] == []
    assert "no results" in res["error"]


def test_web_search_filters_ad_redirects():
    # DDG injects sponsored links that use a y.js redirector (destination
    # base64 in u3=, not uddg=) — these must never surface as results.
    html = """
<a class="result__a" href="https://duckduckgo.com/y.js?ad_domain=ads.example&u3=AAAA">Sponsored Junk</a>
<a class="result__snippet">Buy now!</a>
<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Freal.com%2Farticle">Real Article</a>
<a class="result__snippet">Real snippet.</a>
<a class="result__a" href="https://duckduckgo.com/l/?no_uddg=1">Still DDG</a>
<a class="result__snippet">Tracker.</a>
"""
    with patch("forge.web_primitives.urlopen", _fake_urlopen(html)):
        res = _web_search("news", n=5)
    urls = [r["url"] for r in res["results"]]
    assert urls == ["https://real.com/article"]


# ── news_search (Google News RSS) ───────────────────────────────────────
_GNEWS_XML = """<?xml version="1.0"?>
<rss version="2.0"><channel>
<item>
  <title>Big story of the day - Example News</title>
  <link>https://news.google.com/rss/articles/AAAA</link>
  <pubDate>Tue, 01 Jul 2025 12:00:00 GMT</pubDate>
  <source url="https://example.com">Example News</source>
  <description>&lt;a href="..."&gt;Big story&lt;/a&gt; details</description>
</item>
<item>
  <title>Second headline - Other Outlet</title>
  <link>https://news.google.com/rss/articles/BBBB</link>
  <pubDate>Tue, 01 Jul 2025 11:30:00 GMT</pubDate>
  <source url="https://other.com">Other Outlet</source>
  <description>plain description</description>
</item>
</channel></rss>"""


def test_news_search_parses_rss():
    from forge.web_primitives import google_news_search
    with patch("forge.web_primitives.urlopen", _fake_urlopen(_GNEWS_XML)):
        res = google_news_search("top news", n=5)
    assert res["error"] is None
    assert len(res["results"]) == 2
    r = res["results"][0]
    assert r["title"].startswith("Big story of the day")
    assert r["source"] == "Example News"
    assert r["published"] == "Tue, 01 Jul 2025 12:00:00 GMT"
    assert "news.google.com" in r["url"]
    assert "<a" not in r["snippet"]  # description HTML stripped


def test_news_search_network_error():
    from forge.web_primitives import google_news_search

    def boom(*a, **kw):
        raise OSError("timeout")
    with patch("forge.web_primitives.urlopen", boom):
        res = google_news_search("x")
    assert res["results"] == []
    assert "timeout" in res["error"]


# ── web_fetch ───────────────────────────────────────────────────────────
def test_web_fetch_strips_html():
    html = ("<html><head><script>evil()</script><style>x{}</style></head>"
            "<body><p>Hello   world</p></body></html>")
    with patch("forge.web_primitives.urlopen", _fake_urlopen(html)):
        res = _web_fetch("https://example.com")
    assert res["error"] is None
    assert res["url"] == "https://example.com"
    assert "Hello world" in res["text"]
    assert "evil" not in res["text"]      # script stripped
    assert "<" not in res["text"]         # tags stripped


def test_web_fetch_rejects_unsafe_scheme():
    res = _web_fetch("javascript:alert(1)")
    assert res["text"] == ""
    assert "scheme" in res["error"]


def test_web_fetch_rejects_file_scheme():
    res = _web_fetch("file:///etc/passwd")
    assert res["text"] == ""
    assert "scheme" in res["error"]


def test_web_fetch_network_error():
    def boom(*a, **kw):
        raise OSError("404")
    with patch("forge.web_primitives.urlopen", boom):
        res = _web_fetch("https://example.com/missing")
    assert res["text"] == ""
    assert "fetch failed" in res["error"]


def test_web_fetch_truncates():
    html = "<body>" + ("x " * 5000) + "</body>"
    with patch("forge.web_primitives.urlopen", _fake_urlopen(html)):
        res = _web_fetch("https://example.com", max_chars=100)
    assert len(res["text"]) <= 100
    assert res["truncated"] is True


# ── wikipedia_search ────────────────────────────────────────────────────
_WIKI_JSON = json.dumps({
    "query": {"search": [
        {"title": "Python (programming language)", "snippet": "a <i>high-level</i> language"},
        {"title": "Python (genus)", "snippet": "a genus of snakes"},
    ]}
})


def test_wikipedia_search_parses():
    with patch("forge.web_primitives.urlopen", _fake_urlopen(_WIKI_JSON)):
        res = _wikipedia_search("python", n=2)
    assert res["error"] is None
    assert len(res["results"]) == 2
    assert res["results"][0]["title"] == "Python (programming language)"
    assert "<i>" not in res["results"][0]["snippet"]  # tags stripped
    assert "wikipedia.org/wiki/Python" in res["results"][0]["url"]


def test_wikipedia_search_network_error():
    def boom(*a, **kw):
        raise OSError("dns fail")
    with patch("forge.web_primitives.urlopen", boom):
        res = _wikipedia_search("x")
    assert res["results"] == []
    assert "dns fail" in res["error"]


# ── arxiv_search ────────────────────────────────────────────────────────
_ARXIV_XML = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
<entry>
  <title>Attention Is All You Need</title>
  <summary>We propose a new architecture...</summary>
  <id>http://arxiv.org/abs/1706.03762v5</id>
  <published>2017-06-12T18:00:00Z</published>
</entry>
</feed>"""


def test_arxiv_search_parses():
    with patch("forge.web_primitives.urlopen", _fake_urlopen(_ARXIV_XML)):
        res = _arxiv_search("transformer", n=3)
    assert res["error"] is None
    assert len(res["results"]) == 1
    r = res["results"][0]
    assert r["title"] == "Attention Is All You Need"
    assert "new architecture" in r["summary"]
    assert r["url"] == "http://arxiv.org/abs/1706.03762v5"
    assert r["published"] == "2017-06-12"


def test_arxiv_search_network_error():
    def boom(*a, **kw):
        raise OSError("timeout")
    with patch("forge.web_primitives.urlopen", boom):
        res = _arxiv_search("x")
    assert res["results"] == []
    assert "timeout" in res["error"]


# ── WebTools.execute dispatch ───────────────────────────────────────────
def test_webtools_execute_dispatch():
    wt = WebTools()
    with patch("forge.web_primitives.urlopen", _fake_urlopen(_DDG_HTML)):
        res = wt.execute("web_search", {"query": "test"})
    # success → error key popped by execute() normalization
    assert "error" not in res
    assert len(res["results"]) == 2


def test_webtools_execute_unknown_tool():
    wt = WebTools()
    res = wt.execute("not_a_web_tool", {})
    assert "unknown web tool" in res["error"]


def test_webtools_execute_disabled():
    wt = WebTools(enabled=False)
    res = wt.execute("web_search", {"query": "x"})
    assert "disabled" in res["error"]


def test_webtools_execute_clamps_n():
    wt = WebTools()
    captured = {}

    def fake_open(req, timeout=None):
        captured["url"] = req.full_url
        return _FakeResp(b"<html></html>")

    with patch("forge.web_primitives.urlopen", fake_open):
        wt.execute("web_search", {"query": "x", "n": 999})
    # n clamped to 10 → srlimit/max_results stays bounded (DDG ignores extra)
    assert "q=x" in captured["url"]


def test_webtools_execute_clamps_max_chars():
    wt = WebTools()
    html = "<body>" + ("y " * 10000) + "</body>"
    with patch("forge.web_primitives.urlopen", _fake_urlopen(html)):
        res = wt.execute("web_fetch", {"url": "https://x.com", "max_chars": 999999})
    # clamped to 8000
    assert len(res["text"]) <= 8000


# ── ToolHarness integration ─────────────────────────────────────────────
def test_harness_includes_web_tools_when_provided(tmp_path):
    h = ToolHarness(workspace=str(tmp_path), web_tools=WebTools())
    names = {d["function"]["name"] for d in h.tool_defs()}
    assert {"web_search", "news_search", "web_fetch",
            "wikipedia_search", "arxiv_search"} <= names


def test_harness_no_web_tools_when_none(tmp_path):
    h = ToolHarness(workspace=str(tmp_path))
    names = {d["function"]["name"] for d in h.tool_defs()}
    assert "web_search" not in names


def test_harness_dispatches_web_tool(tmp_path):
    h = ToolHarness(workspace=str(tmp_path), web_tools=WebTools())
    with patch("forge.web_primitives.urlopen", _fake_urlopen(_DDG_HTML)):
        rec = h.execute("web_search", {"query": "urllib"})
    assert rec["ok"] is True
    assert rec["name"] == "web_search"
    assert len(rec["result"]["results"]) == 2


def test_harness_web_tool_error_marked_not_ok(tmp_path):
    h = ToolHarness(workspace=str(tmp_path), web_tools=WebTools())
    rec = h.execute("web_fetch", {"url": "javascript:alert(1)"})
    assert rec["ok"] is False
    assert "scheme" in rec["result"]["error"]


def test_harness_chat_tool_defs_include_web(tmp_path):
    h = ToolHarness(workspace=str(tmp_path), web_tools=WebTools())
    names = {d["function"]["name"] for d in h.chat_tool_defs()}
    assert {"web_search", "news_search", "web_fetch"} <= names


def test_webtools_execute_news_dispatch(tmp_path):
    wt = WebTools()
    with patch("forge.web_primitives.urlopen", _fake_urlopen(_GNEWS_XML)):
        res = wt.execute("news_search", {"query": "top news"})
    assert "error" not in res
    assert res["results"][0]["source"] == "Example News"


def test_harness_read_only_keeps_web_tools(tmp_path):
    # web tools are read-only GET → must survive read-only mode
    h = ToolHarness(workspace=str(tmp_path), web_tools=WebTools(),
                    read_only=True)
    names = {d["function"]["name"] for d in h.tool_defs()}
    assert "web_search" in names
    assert "web_fetch" in names
