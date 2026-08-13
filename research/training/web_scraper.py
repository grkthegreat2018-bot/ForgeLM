"""Web scraping backends for data generation.

Free-tier providers (round-robin to distribute load):
- DuckDuckGo (ddgs): No key, no limit. General web search.
- Jina Reader (r.jina.ai): No key. Reads any URL → markdown.
- Wikipedia API: No key. Encyclopedia articles.
- Tavily: Free tier (limited). Agent-native search.
- Exa: Free tier (limited). Semantic/neural search.

Usage:
    from research.training.web_scraper import WebScraper
    scraper = WebScraper(provider="ddg")
    results = scraper.search("Python async best practices", n=5)
"""
import os
import random as _r
from dataclasses import dataclass
from typing import Optional

# API keys from docs/WebScrappers.md
TAVILY_KEY = "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"
EXA_KEY = "33d6c6e0-7f69-4da3-a96c-a63cb3c0348f"
FIRECRAWL_KEY = "fc-1f766c3e4367479ab8c01ba0e2cb573e"

# All available providers, ordered by preference (free first)
ALL_PROVIDERS = ["semantic_scholar", "arxiv", "ar5iv", "ddg", "wikipedia", "jina_read",
                 "stackexchange", "github", "tavily", "exa"]
# Free providers (no key, no limit) — use these most
FREE_PROVIDERS = ["semantic_scholar", "arxiv", "ar5iv", "ddg", "wikipedia",
                  "jina_read", "stackexchange", "github"]
# Paid/free-tier providers (have keys, rate limited)
KEYED_PROVIDERS = ["tavily", "exa"]


@dataclass
class ScrapedResult:
    """Normalized result from any scraper."""
    title: str
    url: str
    content: str
    source: str  # "tavily" | "exa" | "firecrawl"


class WebScraper:
    """Unified web scraper with multiple backends."""

    def __init__(self, provider: str = "tavily"):
        self.provider = provider
        self._client = None
        self._init_client()

    def _init_client(self):
        if self.provider == "tavily":
            from tavily import TavilyClient
            self._client = TavilyClient(TAVILY_KEY)
        elif self.provider == "exa":
            from exa_py import Exa
            self._client = Exa(api_key=EXA_KEY)
        elif self.provider == "firecrawl":
            from firecrawl import FirecrawlApp
            self._client = FirecrawlApp(api_key=FIRECRAWL_KEY)
        elif self.provider in ("ddg", "wikipedia", "jina_read", "stackexchange", "github",
                                "semantic_scholar", "arxiv", "ar5iv"):
            self._client = None  # no persistent client needed
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def search(self, query: str, n: int = 5, search_depth: str = "advanced") -> list[ScrapedResult]:
        """Search the web and return normalized results."""
        if self.provider == "tavily":
            return self._search_tavily(query, n, search_depth)
        elif self.provider == "exa":
            return self._search_exa(query, n)
        elif self.provider == "firecrawl":
            return self._search_firecrawl(query, n)
        elif self.provider == "ddg":
            return self._search_ddg(query, n)
        elif self.provider == "wikipedia":
            return self._search_wikipedia(query, n)
        elif self.provider == "jina_read":
            return self._search_jina_read(query, n)
        elif self.provider == "stackexchange":
            return self._search_stackexchange(query, n)
        elif self.provider == "github":
            return self._search_github(query, n)
        elif self.provider == "semantic_scholar":
            return self._search_semantic_scholar(query, n)
        elif self.provider == "arxiv":
            return self._search_arxiv(query, n)
        elif self.provider == "ar5iv":
            return self._search_ar5iv(query, n)
        return []

    def _search_ddg(self, query: str, n: int) -> list[ScrapedResult]:
        """DuckDuckGo search — free, no key, no limit."""
        from ddgs import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=n):
                results.append(ScrapedResult(
                    title=r.get("title", ""),
                    url=r.get("href", ""),
                    content=r.get("body", ""),
                    source="ddg"))
        return results

    def _search_wikipedia(self, query: str, n: int) -> list[ScrapedResult]:
        """Wikipedia API — free, no key. Returns article snippets."""
        import httpx
        r = httpx.get("https://en.wikipedia.org/w/api.php", params={
            "action": "query", "list": "search", "srsearch": query,
            "format": "json", "srlimit": n,
        }, timeout=10, headers={"User-Agent": "ForgeAI/1.0"})
        results = []
        if r.status_code == 200:
            data = r.json()
            for item in data.get("query", {}).get("search", []):
                # Fetch full extract for top results
                title = item.get("title", "")
                snippet = item.get("snippet", "")
                # Clean HTML from snippet
                import re
                snippet = re.sub(r'<[^>]+>', '', snippet)
                results.append(ScrapedResult(
                    title=title,
                    url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                    content=snippet,
                    source="wikipedia"))
        return results

    def _search_jina_read(self, query: str, n: int) -> list[ScrapedResult]:
        """Jina AI Reader (r.jina.ai) — free, no key. Reads any URL to markdown.

        First searches via DDG for URLs, then reads them with Jina.
        """
        import httpx
        # Step 1: Get URLs from DDG
        from ddgs import DDGS
        urls = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=min(n, 2)):  # fewer URLs, more content
                urls.append(r.get("href", ""))
        # Step 2: Read each URL with Jina
        results = []
        for url in urls[:2]:
            if not url:
                continue
            try:
                r = httpx.get(f"https://r.jina.ai/{url}", timeout=20,
                              headers={"User-Agent": "ForgeAI/1.0"})
                if r.status_code == 200 and len(r.text) > 100:
                    results.append(ScrapedResult(
                        title=url.split("/")[-1][:60],
                        url=url,
                        content=r.text[:5000],
                        source="jina_read"))
            except Exception:
                continue
        return results

    def _search_stackexchange(self, query: str, n: int) -> list[ScrapedResult]:
        """StackExchange API — free, no key. Returns Q&A from StackOverflow etc.

        Great for coding topics — returns actual question bodies and answers.
        """
        import httpx
        import re
        # Search StackOverflow
        r = httpx.get("https://api.stackexchange.com/2.3/search/advanced", params={
            "order": "desc", "sort": "relevance", "q": query,
            "site": "stackoverflow", "pagesize": min(n, 5),
            "filter": "withbody",
        }, timeout=10)
        results = []
        if r.status_code == 200:
            for item in r.json().get("items", []):
                title = item.get("title", "")
                body = item.get("body", "") or ""
                # Strip HTML tags
                body = re.sub(r'<[^>]+>', '', body)
                # Get top answer if available
                answers = item.get("answers", [])
                answer_text = ""
                if answers:
                    top = max(answers, key=lambda a: a.get("score", 0))
                    answer_text = re.sub(r'<[^>]+>', '', top.get("body", ""))
                content = body[:500]
                if answer_text:
                    content += "\n\nAnswer: " + answer_text[:2000]
                results.append(ScrapedResult(
                    title=title,
                    url=item.get("link", ""),
                    content=content,
                    source="stackexchange"))
        return results

    def _search_github(self, query: str, n: int) -> list[ScrapedResult]:
        """GitHub Search API — free, no key (60 req/hr). Returns repos + READMEs.

        Great for finding code examples and open-source projects.
        """
        import httpx
        r = httpx.get("https://api.github.com/search/repositories", params={
            "q": query, "per_page": min(n, 5), "sort": "stars",
        }, timeout=10, headers={
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "ForgeAI/1.0",
        })
        results = []
        if r.status_code == 200:
            for item in r.json().get("items", [])[:n]:
                name = item.get("full_name", "")
                desc = item.get("description", "") or ""
                topics = ", ".join(item.get("topics", [])[:5])
                readme_url = item.get("html_url", "") + "/blob/main/README.md"
                content = f"Repository: {name}\nDescription: {desc}\nTopics: {topics}\nStars: {item.get('stargazers_count', 0)}"
                # Try to fetch README content
                try:
                    readme_r = httpx.get(
                        f"https://raw.githubusercontent.com/{name}/main/README.md",
                        timeout=8, headers={"User-Agent": "ForgeAI/1.0"})
                    if readme_r.status_code == 200:
                        content += "\n\nREADME:\n" + readme_r.text[:3000]
                except Exception:
                    pass
                results.append(ScrapedResult(
                    title=name,
                    url=item.get("html_url", ""),
                    content=content,
                    source="github"))
        return results

    def _search_semantic_scholar(self, query: str, n: int) -> list[ScrapedResult]:
        """Semantic Scholar API — 214M papers, free, no key (100 req/5min shared).

        Returns full abstracts + TLDRs + citation counts + open access PDF links.
        Best source for academic/research content. May 429 under load.
        """
        import httpx
        try:
            r = httpx.get("https://api.semanticscholar.org/graph/v1/paper/search", params={
                "query": query,
                "limit": min(n, 10),
                "fields": "title,abstract,authors,year,citationCount,tldr,externalIds,openAccessPdf,url",
            }, timeout=15, follow_redirects=True, headers={"User-Agent": "ForgeAI/1.0"})
        except Exception:
            return []
        if r.status_code == 429:
            return []  # rate limited, skip gracefully
        results = []
        if r.status_code == 200:
            for paper in r.json().get("data", [])[:n]:
                title = paper.get("title", "")
                abstract = paper.get("abstract", "") or ""
                tldr = (paper.get("tldr") or {}).get("text", "") or ""
                authors = [a.get("name", "") for a in paper.get("authors", [])[:5]]
                year = paper.get("year", "")
                citations = paper.get("citationCount", 0)
                ext_ids = paper.get("externalIds") or {}
                arxiv_id = ext_ids.get("ArXiv", "")
                doi = ext_ids.get("DOI", "")
                oa_pdf = paper.get("openAccessPdf") or {}
                pdf_url = oa_pdf.get("url", "") if oa_pdf else ""

                # Build rich content: TLDR + abstract + metadata
                content_parts = []
                if tldr:
                    content_parts.append(f"TL;DR: {tldr}")
                if abstract:
                    content_parts.append(f"Abstract: {abstract}")
                content_parts.append(f"Authors: {', '.join(authors)}")
                content_parts.append(f"Year: {year} | Citations: {citations}")
                if arxiv_id:
                    content_parts.append(f"arXiv: {arxiv_id}")
                    content_parts.append(f"Full text: https://ar5iv.org/abs/{arxiv_id}")
                if pdf_url:
                    content_parts.append(f"PDF: {pdf_url}")

                url = paper.get("url", "") or (f"https://doi.org/{doi}" if doi else "")
                results.append(ScrapedResult(
                    title=f"{title} ({year}, {citations} citations)",
                    url=url,
                    content="\n".join(content_parts),
                    source="semantic_scholar"))
        return results

    def _search_arxiv(self, query: str, n: int) -> list[ScrapedResult]:
        """arXiv API — free, no key. Returns full abstracts + metadata.

        Uses the Atom XML feed. Best for CS/ML/Physics papers.
        """
        import httpx
        import xml.etree.ElementTree as ET
        try:
            r = httpx.get("http://export.arxiv.org/api/query", params={
                "search_query": f"all:{query}",
                "max_results": min(n, 5),
                "sortBy": "relevance",
                "sortOrder": "descending",
            }, timeout=15, follow_redirects=True, headers={"User-Agent": "ForgeAI/1.0"})
        except Exception:
            return []
        results = []
        if r.status_code == 200:
            try:
                root = ET.fromstring(r.text)
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                for entry in root.findall("atom:entry", ns)[:n]:
                    title = entry.find("atom:title", ns)
                    title_text = title.text.strip().replace("\n", " ") if title is not None else ""
                    summary = entry.find("atom:summary", ns)
                    abstract = summary.text.strip().replace("\n", " ") if summary is not None else ""
                    published = entry.find("atom:published", ns)
                    pub_date = published.text[:10] if published is not None else ""
                    id_elem = entry.find("atom:id", ns)
                    arxiv_url = id_elem.text if id_elem is not None else ""
                    arxiv_id = arxiv_url.split("/abs/")[-1] if "/abs/" in arxiv_url else ""
                    authors = [a.find("atom:name", ns).text
                               for a in entry.findall("atom:author", ns)
                               if a.find("atom:name", ns) is not None]

                    content = (
                        f"Title: {title_text}\n"
                        f"Authors: {', '.join(authors[:5])}\n"
                        f"Published: {pub_date}\n"
                        f"Abstract: {abstract}\n"
                        f"arXiv ID: {arxiv_id}\n"
                        f"Full HTML: https://ar5iv.org/abs/{arxiv_id}\n"
                        f"PDF: https://arxiv.org/pdf/{arxiv_id}.pdf"
                    )
                    results.append(ScrapedResult(
                        title=f"{title_text[:80]} ({pub_date[:4]})",
                        url=arxiv_url,
                        content=content,
                        source="arxiv"))
            except ET.ParseError:
                pass
        return results

    def _search_ar5iv(self, query: str, n: int) -> list[ScrapedResult]:
        """ar5iv — full HTML text of arXiv papers, read via Jina Reader.

        Two-step: (1) search arXiv for paper IDs, (2) fetch full text via ar5iv+Jina.
        Returns MUCH richer content than abstracts alone — full paper text.
        """
        import httpx
        import xml.etree.ElementTree as ET

        # Step 1: Search arXiv for paper IDs
        try:
            r = httpx.get("http://export.arxiv.org/api/query", params={
                "search_query": f"all:{query}",
                "max_results": min(n, 2),  # fewer papers, but full text
                "sortBy": "relevance",
            }, timeout=15, follow_redirects=True, headers={"User-Agent": "ForgeAI/1.0"})
        except Exception:
            return []
        if r.status_code != 200:
            return []

        arxiv_ids = []
        try:
            root = ET.fromstring(r.text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            for entry in root.findall("atom:entry", ns)[:2]:
                id_elem = entry.find("atom:id", ns)
                if id_elem is not None:
                    arxiv_url = id_elem.text
                    arxiv_id = arxiv_url.split("/abs/")[-1] if "/abs/" in arxiv_url else ""
                    if arxiv_id:
                        title_elem = entry.find("atom:title", ns)
                        title = title_elem.text.strip().replace("\n", " ") if title_elem is not None else ""
                        arxiv_ids.append((arxiv_id, title))
        except ET.ParseError:
            pass

        # Step 2: Fetch full paper text from ar5iv directly (HTML → stripped text)
        import re
        results = []
        for arxiv_id, title in arxiv_ids:
            try:
                r2 = httpx.get(f"https://ar5iv.org/abs/{arxiv_id}", timeout=30,
                               follow_redirects=True,
                               headers={"User-Agent": "ForgeAI/1.0"})
                if r2.status_code == 200 and len(r2.text) > 1000:
                    # Strip HTML tags and collapse whitespace
                    text = re.sub(r'<script[^>]*>.*?</script>', '', r2.text, flags=re.DOTALL)
                    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
                    text = re.sub(r'<[^>]+>', ' ', text)
                    text = re.sub(r'\s+', ' ', text).strip()
                    # Find the abstract section and take first 8000 chars from there
                    abs_idx = text.lower().find("abstract")
                    if abs_idx > 0:
                        text = text[abs_idx:]
                    full_text = text[:8000]
                    results.append(ScrapedResult(
                        title=f"{title[:60]} [FULL PAPER]",
                        url=f"https://ar5iv.org/abs/{arxiv_id}",
                        content=full_text,
                        source="ar5iv"))
            except Exception:
                continue
        return results

    def _search_tavily(self, query: str, n: int, depth: str) -> list[ScrapedResult]:
        resp = self._client.search(query=query, max_results=n, search_depth=depth)
        results = []
        if "answer" in resp and resp["answer"]:
            results.append(ScrapedResult(
                title="Tavily Answer", url="", content=resp["answer"], source="tavily"))
        for r in resp.get("results", []):
            results.append(ScrapedResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                content=r.get("content", ""),
                source="tavily"))
        return results

    def _search_exa(self, query: str, n: int) -> list[ScrapedResult]:
        resp = self._client.search(
            query, type="auto", num_results=n, contents={"highlights": True})
        results = []
        for r in resp.results:
            highlights = " ".join(r.highlights) if hasattr(r, "highlights") and r.highlights else ""
            results.append(ScrapedResult(
                title=r.title or "",
                url=r.url or "",
                content=highlights,
                source="exa"))
        return results

    def _search_firecrawl(self, query: str, n: int) -> list[ScrapedResult]:
        # Firecrawl doesn't have a direct search API; use map + scrape
        # Fall back to Tavily for search, Firecrawl for crawling
        from tavily import TavilyClient
        t = TavilyClient(TAVILY_KEY)
        resp = t.search(query=query, max_results=n, search_depth="basic")
        results = []
        for r in resp.get("results", []):
            results.append(ScrapedResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                content=r.get("content", ""),
                source="firecrawl"))
        return results

    def scrape_url(self, url: str) -> str:
        """Scrape a single URL for full content."""
        if self.provider == "tavily":
            resp = self._client.extract(urls=[url])
            if resp.get("results"):
                return resp["results"][0].get("raw_content", "")
        elif self.provider == "exa":
            resp = self._client.get_contents([url], text={"max_characters": 20000})
            if resp.results:
                return resp.results[0].text or ""
        elif self.provider == "firecrawl":
            resp = self._client.scrape_url(url, params={"formats": ["markdown"]})
            return resp.get("markdown", "")
        return ""


def search_all(query: str, n: int = 3) -> list[ScrapedResult]:
    """Search using round-robin across all providers.

    70% of calls go to free providers (ddg, wikipedia, jina_read).
    30% go to keyed providers (tavily, exa) — only when free ones fail.

    This distributes load so we never hit any single API's rate limit.
    """
    # Pick a free provider 70% of the time
    if _r.random() < 0.7:
        provider = _r.choice(FREE_PROVIDERS)
    else:
        provider = _r.choice(KEYED_PROVIDERS)

    try:
        s = WebScraper(provider=provider)
        results = s.search(query, n=n)
        if results:
            return results
    except Exception:
        pass

    # Fallback: try other free providers first, then keyed
    fallback_order = [p for p in FREE_PROVIDERS if p != provider] + \
                     [p for p in KEYED_PROVIDERS if p != provider]
    for fb in fallback_order:
        try:
            s = WebScraper(provider=fb)
            results = s.search(query, n=n)
            if results:
                return results
        except Exception:
            continue
    return []


def search_all_parallel(query: str, n: int = 3) -> list[ScrapedResult]:
    """Search all free providers in parallel using threads. Max throughput.

    Fires ddg + wikipedia + jina_read simultaneously and merges.
    Never hits paid API limits unless all free providers fail.
    """
    import concurrent.futures

    def _search(prov):
        try:
            return WebScraper(provider=prov).search(query, n=n)
        except Exception:
            return []

    # Try all free providers in parallel first
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(FREE_PROVIDERS)) as pool:
        results = list(pool.map(_search, FREE_PROVIDERS))

    merged = []
    for r in results:
        merged.extend(r)

    # If free providers returned nothing, fall back to keyed
    if not merged:
        for prov in KEYED_PROVIDERS:
            try:
                merged.extend(WebScraper(provider=prov).search(query, n=n))
                if merged:
                    break
            except Exception:
                continue

    return merged
