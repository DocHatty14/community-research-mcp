# Community Research MCP

<div align="center">

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
- **Web Search APIs** — Brave, Google (Serper), Tavily, Firecrawl for broader coverage

**The Mission:** Find the messy workarounds, the battle-tested hacks, the "after 6 hours I finally figured out" solutions that people actually use.

---

## Performance

Benchmarked on November 25, 2025:

### Individual Source Response Times

| Source | Response Time | Typical Results |
|--------|---------------|-----------------|
| Stack Overflow | ~665ms | 3-15 results |
| GitHub Issues | ~1,053ms | 15 results |
| Hacker News | ~418ms | 0-10 results |
| Lobsters | ~778ms | 0-10 results |
| Discourse | ~438ms | 2-10 results |
| Brave Search | ~1,088ms | 10 results |
| Serper (Google) | ~858ms | 10 results |
| Tavily | ~1,222ms | 10 results |
| Firecrawl | ~1,248ms | 10 results |

### Aggregated Search Performance

| Metric | Value |
|--------|-------|
| Cold search (10 sources parallel) | ~4.5 seconds |
| Cached search | <1ms (instant) |
| Average results per search | 40-60 |
| Deduplication rate | ~25-30% |
| Average source response | ~863ms |

### Reliability Features

- **Circuit Breakers** — Prevents cascade failures (5-failure threshold, 5min cooldown)
- **Exponential Backoff** — 1s → 2s → 4s retry delays
- **Graceful Degradation** — Returns partial results when sources fail
- **24-hour Cache TTL** — Reduces API load significantly

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

The server auto-detects which APIs are configured and uses them automatically.

---

## Usage

### MCP Client Configuration

Add to your MCP client config (e.g., Claude Desktop):

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

### Available Tools

#### `get_server_context`
**Always call this first.** Returns server capabilities, detected workspace context, and LLM-friendly tool schemas.

```python
# Returns: detected languages, tool parameter schemas, usage tips
await get_server_context()
```

#### `community_search`
Primary search tool for finding street-smart solutions.

```python
community_search(
    language="Python",                    # Required
    topic="FastAPI background tasks with Celery Redis queue",  # Required, min 10 chars
    goal="Process long-running tasks without blocking",        # Optional but recommended
    current_setup="FastAPI with SQLAlchemy on Docker",         # Optional but recommended
    response_format="json"                # "json" or "markdown"
)
```

#### `deep_community_search`
Multi-phase deep research for complex problems.

```python
deep_community_search(
    language="Python",
    topic="Microservices event-driven architecture with Kafka",
    goal="Design scalable async system"
)
```

#### `plan_research`
Create a strategic research plan before searching (for architecture decisions, comparing approaches).

```python
plan_research(
    language="JavaScript",
    topic="State management React 2024",
    goal="Choose between Redux, Zustand, Jotai"
)
```

### Example Output

**Query:** "Rust wgpu PipelineCompilationOptions removed"

```markdown
# Community Research: Rust wgpu PipelineCompilationOptions removed

| | |
|:--|:--|
| **Language** | Rust |
| **Sources** | 8 searched |
| **Results** | 12 findings |

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

---

### 2. compilation_options removed from ShaderModuleDescriptor

`████████░░` **88** · stackoverflow · 85% relevant

**Issue:** `compilation_options` field removed from `ShaderModuleDescriptor` in 0.19

**Solution:** Use `ShaderSource::Wgsl` directly in `ShaderModuleDescriptor`

🔗 [View Source](https://stackoverflow.com/questions/...)

---

## More Results

### 3. wgpu 0.19 migration guide

`███████░░░` **71** · github · 72% relevant

**Issue:** Breaking changes in wgpu 0.19 release

**Solution:** Follow official migration guide for shader module updates

🔗 [View Source](https://github.com/gfx-rs/wgpu/releases)

---

## Quick Apply

```rust
let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
    label: Some("main"),
    source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
});
```

## Verification

- `cargo build` passes without `compilation_options` errors
```

**Output Philosophy:** The server returns all results with relevance scores. Quality tiers (⭐ Best Matches vs More Results) are informational — no results are filtered out. The consuming LLM decides what's useful.

---

## Architecture

```
community-research-mcp/
├── community_research_mcp.py   # Main MCP server and tool definitions
├── api/                        # External API integrations
│   ├── stackoverflow.py        # Stack Exchange API
│   ├── github.py               # GitHub Issues API
│   ├── hackernews.py           # Algolia HN API
│   ├── lobsters.py             # Lobsters search
│   ├── discourse.py            # Discourse forums
│   ├── brave.py                # Brave Search API
│   ├── serper.py               # Google Search via Serper
│   ├── tavily.py               # Tavily AI search
│   └── firecrawl.py            # Firecrawl web search
├── models/                     # Pydantic input/output models
├── enhanced_mcp_utilities.py   # Circuit breakers, caching, dedup
├── streaming_capabilities.py   # Result classification
└── docs/                       # Additional documentation
```

### Data Flow

1. **Query Enrichment** — Adds street-smart keywords ("workaround", "fix", "solved")
2. **Parallel Search** — All sources queried simultaneously via asyncio
3. **Normalization** — Results standardized to common schema
4. **Deduplication** — URL/title matching removes duplicates (~25-30%)
5. **Quality Scoring** — Ranked by authority, validation, recency, evidence
6. **Structured Output** — JSON or Markdown with findings, conflicts, recommendations

---

## Quality Scoring

Results are scored 0-100 based on:

| Signal | Weight | Description |
|--------|--------|-------------|
| Authority | ~22% | Maintainer replies, accepted answers, reputable sources |
| Community Validation | ~23% | Upvotes, stars, answer counts |
| Recency | ~20% | Newer solutions preferred |
| Specificity | ~20% | Step-by-step fixes beat generic advice |
| Evidence | ~15% | Code snippets, benchmarks, reproduction steps |

See [`docs/quality_scoring.md`](docs/quality_scoring.md) for detailed scoring rubric.

---

## Source Weights

Community sources are weighted higher than web search APIs:

| Source | Weight | Rationale |
|--------|--------|-----------|
| Stack Overflow | 10 | Accepted answers = gold standard |
| GitHub Issues | 9 | Real bugs and real fixes |
| Discourse | 8 | Framework-specific wisdom |
| Lobsters | 7 | Technical depth |
| Hacker News | 6 | Industry experience |
| Reddit | 6 | Honest community discussions |
| Brave/Serper/Tavily | 4 | Broader coverage but may include docs |
| Firecrawl | 3 | Web scraping fallback |

---

## Rate Limits

Without API keys:
- Stack Overflow: 300 requests/day
- GitHub: 60 requests/hour
- Reddit: Limited access

With API keys: 10-100x higher limits depending on plan.

**Built-in protections:**
- Circuit breakers prevent quota exhaustion
- Caching reduces repeat requests
- Graceful degradation on failures

---

## LLM Integration Tips

The server is designed to be LLM-friendly:

1. **Call `get_server_context` first** — Returns tool schemas and parameter hints
2. **Use `topic`, not `query`** — Common mistake: the parameter is `topic`
3. **Be specific** — "Django ORM N+1 query optimization" not "performance"
4. **Include `goal` and `current_setup`** — Better results with context

Example schema returned by `get_server_context`:

```json
{
  "tool_schemas": {
    "community_search": {
      "parameters": {
        "language": {"type": "string", "required": true},
        "topic": {"type": "string", "required": true, "min_length": 10},
        "goal": {"type": "string", "required": false},
        "current_setup": {"type": "string", "required": false}
      }
    }
  },
  "llm_tips": {
    "common_mistakes": [
      "DON'T use 'query' - use 'topic' instead",
      "DON'T use 'max_results' - not a valid parameter"
    ]
  }
}
```

---

## Known Limitations

- **Discourse 404s** — Some language-specific Discourse URLs don't exist (e.g., `discuss.js.org`)
- **Rate limits** — Heavy use may hit API limits; add keys for production use
- **Stale cache** — 24-hour TTL may return outdated results for fast-moving topics
- **No streaming** — Results returned after all sources complete

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
