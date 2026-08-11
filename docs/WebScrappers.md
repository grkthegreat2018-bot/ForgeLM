Query Generation: Have your LLM formulate a search query based on the user's task. 
Agentic Retrieval: Send this query to Tavily or Parallel AI. These APIs return content, not just links. 
Reasoning: Feed the returned text chunks directly back into your LLM's context window.
Iterative Browsing: If the initial search is insufficient, program your agent to trigger a second search with refined keywords (a feature native to Devin's "Cascade/Devin Local" logic).

1. Tavily API (Closest to Devin's "RAG" flow)
Tavily is widely considered the standard for "agent-native" search. Like Devin, it performs the search, scrapes the content, filters for relevance, and returns a clean text block ready for your LLM.

Best for: Real-time grounding and RAG pipelines where you need the agent to "read" the page instantly. 
Key Feature: include_answer and include_raw_content flags mimic Devin's ability to synthesize info on the fly.
2. Exa API (Best for Semantic Discovery)
Devin often needs to find "conceptually similar" code or documentation, not just keyword matches. Exa uses a neural index (embeddings) to find pages based on meaning.

Best for: When your agent needs to find obscure documentation, GitHub issues, or technical forums that keyword search misses. 
Key Feature: "Find similar" endpoint allows you to pass a snippet of code or text and find relevant web pages.
3. Firecrawl (Best for Full Control)
If you want to replicate Devin's ability to "browse" deeply (clicking links, mapping sites), Firecrawl allows you to search and then recursively crawl specific domains.

Best for: Agents that need to ingest entire documentation sites or perform deep research on a specific domain. 

> Set up Tavily's MCP server so I get live web access (search, extract, crawl, and map) in your coding agent.

Add Tavily to your coding agent's MCP config. This form works with most MCP clients (needs Node.js 18+ on your PATH):
```json
{
  "mcpServers": {
    "tavily": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "https://mcp.tavily.com/mcp/?tavilyApiKey=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"]
    }
  }
}
```

Restart your coding agent, then reply once the `tavily_*` tools show up ✅.

If the `tavily_*` tools don't appear, call Tavily directly (also confirms the key):
```bash
curl -s https://api.tavily.com/search -H "Authorization: Bearer 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" -H "Content-Type: application/json" -d '{"query":"latest AI news"}'
```
```python
from tavily import TavilyClient; print(TavilyClient("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").search(query="latest AI news"))
```
```typescript
import { tavily } from "@tavily/core"; console.log(await tavily({ apiKey: "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" }).search("latest AI news"));
```

Install the Tavily CLI (search, extract, crawl, map, research from the terminal):
```bash
curl -fsSL https://cli.tavily.com/install.sh | bash
tvly login --api-key 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
```

Add Tavily's Agent Skills (best-practice patterns), then restart:
```bash
npx skills add tavily-ai/skills --all
```

More: https://docs.tavily.com/agents.md (setup + defaults), https://docs.tavily.com/llms.txt (index; append `.md` to any docs URL).
# To install: pip install tavily-python
from tavily import TavilyClient
client = TavilyClient("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")
response = client.search(
    query="",
    search_depth="advanced"
)
print(response)

> **Canonical reference:** https://docs.exa.ai/reference/search-api-guide-for-coding-agents
>
> If anything below looks outdated or contradicts real API behavior, fetch that URL — it is the source of truth for search types, parameters, and response shape. Report staleness back to the user.

---

# Exa API Setup Guide

## Your Configuration

| Setting | Value |
|---------|-------|
| Coding Tool | Devin |
| Integration | Python |
| Use Case | Coding agent |
| Search Type | Auto - Balanced relevance and speed (default) |
| Content | Highlights |

**Project Description:** R&D of my own LLM model. needs APIs for web tool calling soon. current scope is small, hobby level. but might be published if it meets expectations.

---

## API Key Setup

### Environment Variable

```bash
export EXA_API_KEY="YOUR_API_KEY"
```

### .env File

```env
EXA_API_KEY=YOUR_API_KEY
```

### Usage in Code

```python
import os
from exa_py import Exa

exa = Exa(api_key=os.environ.get("EXA_API_KEY"))
```

---

## Quick Start (Python)

```bash
pip install exa-py==2.14.0
```

```python
from exa_py import Exa

exa = Exa(api_key="YOUR_API_KEY")

results = exa.search(
    "Next.js route handler authentication example",
    type="auto",
    num_results=10,
    contents={"highlights": True}
)

for result in results.results:
    print(result.title, result.url)
```

---

## Pick Your Search Pattern

Start with one of these two patterns:

### 1. Raw retrieval for your own agent

Use this when your app should inspect `results` directly, pass `highlights` into your own LLM, or expose Exa as a tool inside an existing agent loop.

```json
{
  "query": "Next.js route handler authentication example",
  "type": "auto",
  "numResults": 10,
  "contents": {
    "highlights": true
  }
}
```

### 2. Synthesized search when you want grounded output

Use this when you want Exa to synthesize a grounded answer or structured payload for you. `systemPrompt` sets behavior and source preferences; `outputSchema` sets the shape of `output.content`.

```json
{
  "query": "Next.js route handler authentication example",
  "type": "deep",
  "systemPrompt": "Prefer official sources, collapse duplicate reporting, and keep the output grounded.",
  "outputSchema": {
    "type": "object",
    "properties": {
      "summary": {
        "type": "string",
        "description": "A grounded summary of the most important findings"
      }
    },
    "required": [
      "summary"
    ]
  },
  "contents": {
    "highlights": true
  }
}
```

### Deep search notes

- Use `deep` when you need harder comparisons, structured synthesis, or multi-step reasoning across many sources.
- Use `additionalQueries` only on `deep-lite`, `deep`, and `deep-reasoning` when you want to force a few explicit query angles instead of relying entirely on automatic query expansion.
- If you only need raw search results and excerpts, stay on `results` + `highlights` and skip `outputSchema`.

---

## Search Type Reference

| Type | Best For | Approx Latency | Depth |
|------|----------|----------------|-------|
| `auto` | Most queries — balanced relevance and speed | ~1 second | Smart | ← your selection
| `fast` | Latency-sensitive queries that still need good relevance | ~450 ms | Basic |
| `instant` | Chat, voice, autocomplete, quick lookups | ~250 ms | Basic |
| `deep-lite` | Cheaper synthesis when full deep search is overkill | 4 seconds | Deep |
| `deep` | Research, enrichment, thorough results | 4-15 seconds | Deep |
| `deep-reasoning` | Complex research, multi-step reasoning, hard synthesis tasks | 12-40 seconds | Deepest |

Latency numbers are ballpark — synthesis (`outputSchema`) and forced livecrawls (`contents.maxAgeHours: 0`) stack on top of the base `type`. See the Latency Characteristics section for details.

**Tip:** `type="auto"` works well for most queries. `outputSchema` works on every search type, so you can request structured, grounded output regardless of which type you pick.

---

## Optional: Structured Outputs (outputSchema)

Raw `results` + `highlights` should still be your default starting point for many agent workflows. Add `outputSchema` only when you want Exa to synthesize grounded JSON or a structured answer for you.

`outputSchema` works on **every** search type. Pass a JSON schema and Exa returns the synthesized answer as structured JSON in `output.content`, with field-level citations in `output.grounding`. Deep variants (`deep-lite`, `deep`, `deep-reasoning`) give higher-quality synthesis for complex queries, but the response shape is the same.

**Use `systemPrompt` and `outputSchema` together:** `systemPrompt` controls source preferences, dedupe behavior, and synthesis rules; `outputSchema` controls the exact shape of `output.content`.

**Schema controls:** `type`, `description`, `required`, `properties`, `items`. Max nesting depth 2, max total properties 10. Do NOT add citation or confidence fields to the schema — `/search` returns grounding data automatically.

```python
from exa_py import Exa

exa = Exa(api_key="YOUR_API_KEY")

results = exa.search(
    "articles about GPUs",
    type="auto",
    system_prompt="Prefer official sources, collapse duplicate reporting, and keep the output grounded.",
    output_schema={
        "type": "object",
        "description": "Companies mentioned in articles",
        "required": ["companies"],
        "properties": {
            "companies": {
                "type": "array",
                "description": "List of companies mentioned",
                "items": {
                    "type": "object",
                    "required": ["name"],
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name of the company"
                        },
                        "description": {
                            "type": "string",
                            "description": "Short description of what the company does"
                        }
                    }
                }
            }
        }
    },
    contents={"highlights": True}
)

# Access structured output
print(results.output.content)   # {"companies": [{"name": "Nvidia", "description": "..."}]}
print(results.output.grounding) # Field-level citations
```

### Response Shape

Responses with `outputSchema` include:
- `output.content` — structured JSON matching your schema (or a string for `{"type": "text"}` schemas)
- `output.grounding` — array of `{field, citations, confidence}` entries with source URLs

```json
{
  "output": {
    "content": {
      "companies": [
        {"name": "Nvidia", "description": "GPU and AI chip manufacturer"},
        {"name": "AMD", "description": "Semiconductor company producing GPUs and CPUs"}
      ]
    },
    "grounding": [
      {
        "field": "companies[0].name",
        "citations": [{"url": "https://...", "title": "Source"}],
        "confidence": "high"
      }
    ]
  }
}
```

### When to Use Structured Outputs

- **Enrichment workflows** — extract specific fields (company info, people data, product details)
- **Data pipelines** — get structured data directly instead of parsing free text
- **Grounded answers** — prefer `outputSchema` on `/search` for new structured search flows
- Prefer a deep variant (`deep-lite`/`deep`/`deep-reasoning`) when you need multi-step reasoning or synthesis across many sources

---

## Content Configuration

The generated examples request highlights by default:

```json
"contents": {
  "highlights": true
}
```

Highlights return query-relevant excerpts, which are usually the right content mode for LLM workflows because they keep token usage predictable.

Content is controlled via the `contents` object on `/search` (or top-level fields on `/contents`). Pick one of `text`, `highlights`, or `summary` by default. You can combine them, but it is usually an antipattern to do so at the start of a project.

| Mode | Config | Best For |
|------|--------|----------|
| Highlights | `"highlights": true` | Token-efficient excerpts |
| Text | `"text": {"maxCharacters": 20000}` | Full content extraction, RAG |
| Summary | `"summary": {"query": "your question"}` or `"summary": true` | LLM-written summary per result |

### Tuning knobs

- **`highlights`** — pass `true` to return query-relevant highlights for each result.
- **`summary`** — pass `true` for a generic summary, or `{"query": "..."}` to bias the summary toward a specific question. Supports a `schema` field for per-result structured output. Summary has no `verbosity` setting — verbosity lives on `text` (below).
- **`text.verbosity`** — `"compact" | "full"` (default `"compact"`). Compact returns only the main content of the page, excluding navbars, banners, footers etc.
- **`text.includeHtmlTags`** — boolean (default `false`). When `true`, preserves HTML structure (useful for code blocks, tables).
- **`text.maxCharacters`** — hard cap on extracted text length. Always set this to control token cost when requesting text.

**Case conventions:** JavaScript SDK and raw JSON use camelCase (`maxCharacters`). Python SDK uses snake_case (`max_characters`) — this applies inside nested dicts too.

**Token usage:** `text: true` with no cap can blow up context. Prefer `highlights: true` for most agent workflows, and add `text` only when downstream reasoning truly needs broad page context.

---

## Domain Filtering (Optional)

Usually not needed - Exa's neural search finds relevant results without domain restrictions.

**When to use:**
- Targeting specific authoritative sources
- Excluding low-quality domains from results

**Example:**

```json
{
  "includeDomains": ["arxiv.org", "github.com"],
  "excludeDomains": ["pinterest.com"]
}
```

**Note:** `includeDomains` and `excludeDomains` can be used together to include a broad domain while excluding specific subdomains (e.g., `"includeDomains": ["vercel.com"], "excludeDomains": ["community.vercel.com"]`).

---

## Coding Agent

```json
{
  "query": "Next.js route handler authentication example",
  "numResults": 10,
  "contents": {
    "highlights": true
  }
}
```

**Tips:**
- Use `type: "auto"` with `contents: { highlights: true }` for most queries
- Great for documentation lookup, API references, code examples

---

## Content Freshness (maxAgeHours)

`maxAgeHours` sets the maximum acceptable age (in hours) for cached content. If the cached version is older than this threshold, Exa will livecrawl the page to get fresh content.

| Value | Behavior | Best For |
|-------|----------|----------|
| 24 | Use cache if less than 24 hours old, otherwise livecrawl | Daily-fresh content |
| 1 | Use cache if less than 1 hour old, otherwise livecrawl | Near real-time data |
| 0 | Always livecrawl (ignore cache entirely) | Real-time data where cached content is unusable |
| -1 | Never livecrawl (cache only) | Maximum speed, historical/static content |
| *(omit)* | Default behavior (livecrawl as fallback if no cache exists) | **Recommended** — balanced speed and freshness |

**When LiveCrawl Isn't Necessary:**
Cached data is sufficient for many queries, especially for historical topics or educational content. These subjects rarely change, so reliable cached results can provide accurate information quickly.

See [maxAgeHours docs](https://exa.ai/docs/reference/livecrawling-contents#maxAgeHours) for more details.

---

## Other Endpoints

Beyond `/search`, the next two endpoints to know are `/contents` and `/answer`:

| Endpoint | Description | Docs |
|----------|-------------|------|
| `/contents` | Get clean, parsed content for URLs you already have | [Docs](https://exa.ai/docs/reference/get-contents) |
| `/answer` | Get a grounded answer with citations when the UI is question-first | [Docs](https://exa.ai/docs/reference/answer) |

> For new structured search flows, prefer `/search` + `outputSchema` when you want both retrieval control and grounded output. Keep `/answer` for question-first UIs where you do not need to inspect raw search results.

### /contents — Get Contents for Known URLs

Use `/contents` when you already have URLs and need their content. Unlike `/search` (which finds and optionally retrieves content), `/contents` is purely for content extraction from known URLs.

**When to use `/contents` vs `/search`:**
- URLs from another source (database, user input, RSS feeds) → `/contents`
- Need to refresh stale content for URLs you already have → `/contents` with `maxAgeHours`
- Need to find AND get content in one call → `/search` with `contents`

```python
from exa_py import Exa

exa = Exa(api_key="YOUR_API_KEY")

results = exa.get_contents(
    ["https://example.com/article", "https://example.com/blog-post"],
    highlights=True
)

for result in results.results:
    print(result.title, result.url)
    print(result.highlights)
```

**Content retrieval options** (choose one per request):

| Option | Config | Best For |
|--------|--------|----------|
| Highlights | `"highlights": true` | Key excerpts, lower token usage |
| Text | `"text": {"max_characters": 20000}` | Full content extraction, RAG |

**Highlights example:**

```json
{
  "urls": ["https://example.com/article"],
  "highlights": true
}
```

**Freshness control:** Add `maxAgeHours` to ensure content is fresh:
- `24` — livecrawl if cached content is older than 24 hours
- `0` — always livecrawl (ignore cache)
- Omit — use cache when available, livecrawl as fallback

---

## Troubleshooting

**⚠️ COMMON PARAMETER MISTAKES — avoid these:**
- `useAutoprompt` → **deprecated**, remove it entirely
- `includeUrls` / `excludeUrls` → **do not exist**. Use `includeDomains` / `excludeDomains`
- `text`, `summary`, `highlights` at the top level of `/search` → **must be nested** inside `contents` (e.g. `"contents": {"highlights": true}`). On `/contents` they ARE top-level — don't confuse the two.
- `numSentences`, `highlightsPerUrl` → **deprecated** highlights params. Use `highlights: true` instead
- `tokensNum` → **does not exist**. Use `contents.text.maxCharacters` to limit text length
- `livecrawl: "always"` → **deprecated**. Use `contents.maxAgeHours: 0` instead
- `excludeDomains` + `category: "company" | "people"` → **400 error**. Those categories don't support `excludeDomains` or any date filters.

> **`stream: true`** switches `/search` to SSE mode (OpenAI-compatible chat-completion chunks). It's supported — just expect streaming chunks instead of one JSON response.

**Results not relevant?**
1. Try `type: "auto"` - most balanced option
2. Try `type: "deep"` - runs multiple query variations and ranks the combined results
3. Refine query - use singular form, be specific
4. Check category matches your use case

**Need structured data from search?**
1. Pass `outputSchema` on any search type — `auto` works, `deep`/`deep-reasoning` gives higher-quality synthesis
2. Define the fields you need in the schema, then add `systemPrompt` for source preferences and dedupe rules

**Results too slow?**
1. Use `type: "fast"` or `type: "instant"`
2. Reduce `numResults`
3. Skip contents if you only need URLs

**No results?**
1. Remove filters (date, domain restrictions)
2. Simplify query
3. Try `type: "auto"` - has fallback mechanisms

---

## Resources

- Docs: https://exa.ai/docs
- Dashboard: https://dashboard.exa.ai
- API Status: https://status.exa.ai
- API key: 33d6c6e0-7f69-4da3-a96c-a63cb3c0348f


firecrawl; fc-1f766c3e4367479ab8c01ba0e2cb573e