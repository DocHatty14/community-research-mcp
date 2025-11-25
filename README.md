# Community Research MCP

<div align="center">

![download](https://github.com/user-attachments/assets/20f7470f-ae0c-4010-8bdf-e07da6a3f769)

**Where the official documentation ends and actual street-smart solutions begin.**

*A Model Context Protocol server that finds real fixes from real developers — the workarounds, hacks, and "this finally worked for me" solutions from Stack Overflow, GitHub Issues, Reddit, and forums.*

[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square)](https://www.python.org/)
[![MCP](https://img.shields.io/badge/MCP-Compatible-green?style=flat-square)](https://modelcontextprotocol.io/)

</div>

---

## What This Does

Most AI tools give you textbook answers. Community Research MCP finds what actually works in production:

- **Stack Overflow** — Accepted answers AND the real fix buried in comment #3
- **GitHub Issues** — Closed issues with workarounds, maintainer-approved fixes
- **Reddit** — "Don't use X, use Y instead" discussions
- **Hacker News** — Architecture critiques from experienced developers
- **Discourse Forums** — Framework-specific community wisdom
- **Lobsters** — Technical deep-dives
- **Web Search APIs** — Brave, Google (Serper), Tavily, Firecrawl for broader coverage

**The Mission:** Find the messy workarounds, the battle-tested hacks, the "after 6 hours I finally figured out" solutions that people actually use.

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

### Reliability Engineering

- **Circuit Breakers** — 5-failure threshold, 5-minute cooldown prevents cascade failures
- **Exponential Backoff** — 1s → 2s → 4s retry delays
- **Graceful Degradation** — Returns partial results when sources fail
- **24-hour Cache** — Cold: ~4.5s → Cached: <1ms

### Smart Query Distribution

To avoid rate limits while maximizing result diversity, multi-query searches are distributed across API groups:

| Query | API Group | Sources |
|-------|-----------|---------|
| Primary | All sources | SO, GitHub, HN + all configured APIs |
| Secondary | Web search | Brave, Tavily, Serper (if configured) |
| Tertiary | Community | Reddit, Lobsters, Discourse, Firecrawl |

Each API is called once per search — different query variations go to different groups, avoiding rate limits while still getting diverse results from multiple phrasings.

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

**Required:** None — works with free public APIs

**Optional (for enhanced results):**

```env
# Web Search APIs (all optional, add any/all)
BRAVE_SEARCH_API_KEY=your_key      # https://brave.com/search/api/
SERPER_API_KEY=your_key            # https://serper.dev/
TAVILY_API_KEY=your_key            # https://tavily.com/
FIRECRAWL_API_KEY=your_key         # https://firecrawl.dev/

# Enhanced Reddit access (optional)
REDDIT_CLIENT_ID=your_id
REDDIT_CLIENT_SECRET=your_secret
```

The server auto-detects which APIs are configured and adjusts query distribution accordingly.

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

**Call this first.** Returns server capabilities, detected workspace context, and LLM-friendly tool schemas.

### `community_search`

Primary search tool for finding street-smart solutions.

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

Multi-phase deep research for complex problems. Runs multiple searches with different angles.

### `plan_research`

Creates a strategic research plan before searching — useful for architecture decisions or comparing approaches.

---

## Example Output

**Query:** "Rust wgpu PipelineCompilationOptions removed"

```markdown
# Community Research: Rust wgpu PipelineCompilationOptions removed

| | |
|:--|:--|
| **Language** | Rust |
| **Evidence** | 12 results · 8 sources · ✓ Strong |

---

## ⭐ Best Matches

### 1. API cleanup deprecated PipelineCompilationOptions

`████████░░` **92** · github · 89% relevant

**Issue:** API cleanup deprecated PipelineCompilationOptions in wgpu 0.19

**Solution:** Replace with `ShaderSource::Wgsl`; shader modules now only take label/source

<details><summary>📄 View Code</summary>

```rust
let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
    label: Some("main"),
    source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
});
```

</details>

🔗 [View Source](https://github.com/gfx-rs/wgpu/issues/4528)
```

**Output Philosophy:** All results returned with relevance scores. Quality tiers (⭐ Best Matches vs More Results) are informational — nothing filtered out. The consuming LLM decides what's useful.

---

## Quality Scoring

Results are scored 0-100 based on real signals:

| Signal | Weight | Description |
|--------|--------|-------------|
| Authority | ~22% | Maintainer replies, accepted answers, reputable sources |
| Community Validation | ~23% | Upvotes, stars, answer counts |
| Recency | ~20% | Newer solutions preferred |
| Specificity | ~20% | Step-by-step fixes beat generic advice |
| Evidence | ~15% | Code snippets, benchmarks, reproduction steps |

---

## Source Weights

Community sources weighted higher than web search:

| Source | Weight | Rationale |
|--------|--------|-----------|
| Stack Overflow | 10 | Accepted answers with real fixes |
| GitHub Issues | 9 | Real bugs, real solutions |
| Discourse | 8 | Framework-specific wisdom |
| Lobsters | 7 | Technical depth |
| Hacker News | 6 | Industry experience |
| Reddit | 6 | Honest community discussions |
| Brave/Serper/Tavily | 4 | Broader coverage |
| Firecrawl | 3 | Web scraping fallback |

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

1. **Query Enrichment** — Adds street-smart keywords, generates variations
2. **Distributed Search** — Queries spread across API groups to avoid rate limits
3. **Parallel Execution** — All sources queried simultaneously via asyncio
4. **Normalization** — Results standardized to common schema
5. **Deduplication** — URL/title matching removes duplicates (~25-30%)
6. **Quality Scoring** — Ranked by authority, validation, recency, evidence
7. **Structured Output** — JSON or Markdown with findings, code snippets, source links

---

## LLM Integration Tips

1. **Call `get_server_context` first** — Returns tool schemas and parameter hints
2. **Use `topic`, not `query`** — Common mistake: the parameter is `topic`
3. **Be specific** — "Django ORM N+1 query optimization" beats "performance"
4. **Include `goal` and `current_setup`** — Context improves results

---

## Known Limitations

| Limitation | Details |
|------------|---------|
| **Keyword search** | No semantic/vector search — relies on keyword matching + quality scoring |
| **No streaming** | Results return after all sources complete (~4.5s cold) |
| **Rate limits** | Free tier limits apply without API keys (SO: 300/day, GH: 60/hr) |
| **Cache staleness** | 24-hour TTL may return outdated results for fast-moving topics |
| **Discourse 404s** | Some language-specific Discourse URLs don't exist |

---

## Roadmap

Potential improvements (contributions welcome):

- [ ] Semantic search via embeddings
- [ ] Streaming results as sources complete
- [ ] Vector store for search history
- [ ] Additional community sources

---

## Contributing

PRs welcome. Keep it simple, don't break existing functionality.

---

## License

MIT License — see [LICENSE](LICENSE)

---

<div align="center">

**Built for developers who know the real answer is in the comments.**

</div>
