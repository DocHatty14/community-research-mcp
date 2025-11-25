# Community Research MCP

<div align="center">

![download](https://github.com/user-attachments/assets/20f7470f-ae0c-4010-8bdf-e07da6a3f769)

**Where the official documentation ends and real solutions begin.**

*A Model Context Protocol server that aggregates developer community knowledge — Stack Overflow answers, GitHub issue fixes, Reddit discussions, and forum wisdom — into structured, LLM-ready output.*

[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square)](https://www.python.org/)
[![MCP](https://img.shields.io/badge/MCP-Compatible-green?style=flat-square)](https://modelcontextprotocol.io/)

</div>

---

## Overview

Community Research MCP is a specialized search aggregator for developer communities. It queries multiple sources in parallel, deduplicates and scores results, and returns structured findings optimized for LLM consumption via the Model Context Protocol.

**What it does:**
- Parallel async queries across 10+ sources (Stack Overflow, GitHub Issues, HN, Reddit, Discourse, Lobsters, + web search APIs)
- Circuit breakers and exponential backoff for reliability
- URL/title-based deduplication (~25-30% reduction)
- Weighted quality scoring (authority, validation, recency, evidence)
- Structured JSON/Markdown output with code snippets and source links

**What it doesn't do:**
- Semantic/vector search (keyword + scoring only)
- Real-time streaming (results return after all sources complete)
- Proprietary data access (public APIs only)

---

## Performance

Benchmarked November 25, 2025:

| Metric | Value |
|--------|-------|
| Cold search (10 sources parallel) | ~4.5 seconds |
| Cached search | <1ms |
| Average results per search | 40-60 |
| Deduplication rate | ~25-30% |

### Source Response Times

| Source | Avg Response | Notes |
|--------|--------------|-------|
| Hacker News | ~418ms | Algolia API |
| Discourse | ~438ms | May 404 on some domains |
| Stack Overflow | ~665ms | 300 req/day without key |
| Lobsters | ~778ms | No auth required |
| Serper (Google) | ~858ms | Requires API key |
| GitHub Issues | ~1,053ms | 60 req/hr without key |
| Brave Search | ~1,088ms | Requires API key |
| Tavily | ~1,222ms | Requires API key |
| Firecrawl | ~1,248ms | Requires API key |

### Reliability

- **Circuit Breakers** — 5-failure threshold, 5-minute cooldown
- **Exponential Backoff** — 1s → 2s → 4s retry delays
- **Graceful Degradation** — Returns partial results when sources fail
- **24-hour Cache** — Reduces API load (caveat: may return stale results for fast-moving topics)

### Smart Query Distribution

To avoid rate limits while maximizing result diversity, multi-query searches are distributed across API groups:

| Query | API Group | Sources |
|-------|-----------|---------|
| Primary | All sources | SO, GitHub, HN, + all configured APIs |
| Secondary | Web search | Brave, Tavily, Serper (if configured) |
| Tertiary | Supplementary | Reddit, Lobsters, Discourse, Firecrawl |

Each API is called once per search — different query variations go to different groups.

---

## Quick Start

```bash
git clone https://github.com/DocHatty/community-research-mcp.git
cd community-research-mcp

# Windows
initialize.bat

# Linux/Mac
chmod +x setup.sh && ./setup.sh

# Or manually
pip install -e .
cp .env.example .env
```

### API Keys

**Required:** None — core functionality works with free public APIs

**Optional (enhanced coverage):**

```env
BRAVE_SEARCH_API_KEY=       # https://brave.com/search/api/
SERPER_API_KEY=             # https://serper.dev/
TAVILY_API_KEY=             # https://tavily.com/
FIRECRAWL_API_KEY=          # https://firecrawl.dev/

# Enhanced Reddit (optional)
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=
```

The server auto-detects configured APIs and adjusts query distribution accordingly.

---

## MCP Configuration

Add to your MCP client (e.g., Claude Desktop):

```json
{
  "mcpServers": {
    "community-research": {
      "command": "python",
      "args": ["-m", "community_research_mcp"],
      "cwd": "/path/to/community-research-mcp"
    }
  }
}
```

---

## Tools

### `get_server_context`

Returns server capabilities, detected workspace context, and tool schemas. **Call this first** — it helps LLMs understand available parameters and avoid common mistakes.

### `community_search`

Primary search tool.

```python
community_search(
    language="Python",                              # Required
    topic="FastAPI background tasks Celery Redis",  # Required, min 10 chars
    goal="Process tasks without blocking",          # Optional, improves relevance
    current_setup="FastAPI + SQLAlchemy on Docker", # Optional, adds context
    response_format="markdown"                      # "json" or "markdown"
)
```

### `deep_community_search`

Multi-phase research for complex problems. Runs multiple searches with different angles.

### `plan_research`

Creates a research plan before searching. Useful for architecture decisions or comparing approaches.

---

## Output Format

Results are returned with quality scores (0-100) and tiered presentation:

```markdown
# Community Research: Python async error handling

| | |
|:--|:--|
| **Language** | Python |
| **Evidence** | 52 results · 8 sources · ✓ Strong |

## ⭐ Best Matches

### 1. asyncio exception handling patterns
`████████░░` **87** · stackoverflow · 82% relevant

**Issue:** Unhandled exceptions in asyncio tasks silently fail

**Solution:** Use `asyncio.gather(..., return_exceptions=True)` or wrap in try/except

[View Code] [View Source]

---

## More Results
...
```

All results are returned — quality tiers are informational, not filtered. The consuming LLM decides what's relevant.

---

## Quality Scoring

| Signal | Weight | Description |
|--------|--------|-------------|
| Authority | ~22% | Maintainer replies, accepted answers |
| Community Validation | ~23% | Upvotes, stars, answer counts |
| Recency | ~20% | Newer solutions preferred |
| Specificity | ~20% | Step-by-step fixes over generic advice |
| Evidence | ~15% | Code snippets, reproduction steps |

---

## Architecture

```
community-research-mcp/
├── community_research_mcp.py   # MCP server, tools, orchestration
├── api/                        # Source integrations
│   ├── stackoverflow.py
│   ├── github.py
│   ├── hackernews.py
│   ├── lobsters.py
│   ├── discourse.py
│   ├── brave.py
│   ├── serper.py
│   ├── tavily.py
│   └── firecrawl.py
├── enhanced_mcp_utilities.py   # Circuit breakers, caching, dedup
└── models/                     # Pydantic schemas
```

### Data Flow

1. **Query Enrichment** — Expands query with variations for broader coverage
2. **Distributed Search** — Queries distributed across API groups (avoids rate limits)
3. **Parallel Execution** — All sources in each group queried via asyncio
4. **Normalization** — Results standardized to common schema
5. **Deduplication** — URL/title matching removes duplicates
6. **Scoring** — Weighted quality scoring
7. **Output** — Structured JSON or Markdown

---

## Known Limitations

| Limitation | Details |
|------------|---------|
| **Keyword search only** | No semantic/vector search — relies on keyword matching + quality scoring |
| **No streaming** | Results return after all sources complete (~4.5s cold) |
| **Rate limits** | Free tier limits apply without API keys (SO: 300/day, GH: 60/hr) |
| **Cache staleness** | 24-hour TTL may return outdated results for rapidly evolving topics |
| **Discourse 404s** | Some language-specific Discourse URLs don't exist |
| **Reddit limits** | Basic access without credentials |

---

## Alternatives

This tool is specialized for developer community aggregation. For other use cases:

- **General web search**: Tavily, Exa, Perplexity
- **Semantic search**: Build a RAG pipeline with embeddings
- **Stack Overflow corpus**: Download and index locally

---

## Roadmap

Potential improvements (contributions welcome):

- [ ] Semantic search via embeddings
- [ ] Streaming results as sources complete
- [ ] Vector store for search history
- [ ] Additional community sources

---

## Contributing

PRs welcome. Please:
- Keep changes focused
- Don't break existing functionality
- Add tests for new features

---

## License

MIT License — see [LICENSE](LICENSE)

---

<div align="center">

**Aggregating developer community knowledge for LLM tools.**

</div>
