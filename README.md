# Community Research MCP

![download](https://github.com/user-attachments/assets/0cf397e9-cc69-4ee7-8c45-6c0ad7a3b676)

> **Find real solutions from developers who've solved your problem before.**

An MCP server that searches Stack Overflow, GitHub Issues, Hacker News, and other developer communities to find battle-tested solutions, workarounds, and implementation patterns.

```
┌─────────────────────────────────────────────────────────────────────┐
│  "How do I handle FastAPI background tasks with Redis?"            │
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │
│  │ Stack       │  │ GitHub      │  │ Hacker      │  │ Discourse  │ │
│  │ Overflow    │  │ Issues      │  │ News        │  │ Forums     │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └─────┬──────┘ │
│         │                │                │               │        │
│         └────────────────┴────────────────┴───────────────┘        │
│                                   │                                 │
│                          ┌────────▼────────┐                       │
│                          │   Deduplicate   │                       │
│                          │   & Score       │                       │
│                          └────────┬────────┘                       │
│                                   │                                 │
│                          ┌────────▼────────┐                       │
│                          │  Ranked Results │                       │
│                          │  with Quality   │                       │
│                          │  Scores 0-100   │                       │
│                          └─────────────────┘                       │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/community-research-mcp.git
cd community-research-mcp

# Install dependencies
pip install -e .

# Or with uv
uv pip install -e .
```

### Configure Claude Desktop

Add to your `claude_desktop_config.json`:

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

### Basic Usage

Ask Claude:

> "Search for Python async database connection pooling patterns"

The MCP will search across developer communities and return ranked solutions with quality scores.

---

## Features

### Multi-Source Search

Searches 9 developer communities simultaneously:

| Source | Type | API Key Required |
|--------|------|------------------|
| **Stack Overflow** | Q&A | No |
| **GitHub Issues** | Bug reports, discussions | No (optional for higher limits) |
| **Hacker News** | Tech discussions | No |
| **Lobsters** | Technical articles | No |
| **Discourse** | Language-specific forums | No |
| **Serper** | Google Search | Yes |
| **Tavily** | AI-optimized search | Yes |
| **Brave** | Privacy-focused search | Yes |
| **Firecrawl** | Web scraping | Yes |

### Quality Scoring

Every result gets a quality score (0-100) based on:

- **Source Authority** — Stack Overflow answers weighted higher than random blog posts
- **Community Validation** — Upvotes, accepted answers, reactions
- **Recency** — Recent solutions for evolving technologies
- **Specificity** — Code examples and detailed explanations
- **Evidence** — Benchmarks, reproduction steps, real metrics

### Reliability Features

- **Circuit Breaker** — Prevents cascading failures when APIs are down
- **Automatic Retry** — Exponential backoff for transient failures
- **Deduplication** — Removes duplicate results across sources (~20% reduction)
- **Caching** — 1-hour TTL to reduce API calls

---

## Configuration

### Environment Variables

Create a `.env` file:

```bash
# Optional: Web Search APIs (enable more sources)
SERPER_API_KEY=your_key        # https://serper.dev
TAVILY_API_KEY=your_key        # https://tavily.com
BRAVE_SEARCH_API_KEY=your_key  # https://brave.com/search/api
FIRECRAWL_API_KEY=your_key     # https://firecrawl.dev

# Optional: Higher rate limits
GITHUB_TOKEN=your_token        # https://github.com/settings/tokens
STACKEXCHANGE_API_KEY=your_key # https://stackapps.com

# Optional: Reddit (requires app registration)
REDDIT_CLIENT_ID=your_id
REDDIT_CLIENT_SECRET=your_secret
REDDIT_REFRESH_TOKEN=your_token
```

### Source Weights

Configure source priorities in `config.json`:

```json
{
  "sources": {
    "stackoverflow": {"enabled": true, "weight": 10},
    "github": {"enabled": true, "weight": 9},
    "hackernews": {"enabled": true, "weight": 6},
    "serper": {"enabled": true, "weight": 4}
  }
}
```

Higher weights = more trusted for "street-smart" solutions.

---

## Architecture

```
community-research-mcp/
├── community_research_mcp.py   # Main MCP server & tools
├── api/                        # Source integrations
│   ├── stackexchange.py        # Stack Overflow + 19 SE sites
│   ├── github.py               # GitHub Issues
│   ├── hackernews.py           # HN via Algolia
│   ├── lobsters.py             # Lobsters.rs
│   ├── discourse.py            # Discourse forums
│   ├── serper.py               # Google Search
│   ├── tavily.py               # Tavily API
│   ├── brave.py                # Brave Search
│   └── firecrawl.py            # Web scraping
├── core/                       # Reliability & quality
│   ├── reliability.py          # Circuit breaker, retry logic
│   ├── quality.py              # Quality scoring
│   ├── dedup.py                # Deduplication
│   └── metrics.py              # Performance monitoring
├── models/                     # Pydantic models
├── utils/                      # Cache, rate limiting
└── tests/                      # Test cases
```

---

## MCP Tools

### `community_search`

Primary search tool. Searches all enabled sources and returns ranked results.

```
language: "Python"
topic: "async SQLAlchemy connection pooling with FastAPI"
goal: "Handle 1000 concurrent database connections"
current_setup: "FastAPI + PostgreSQL + SQLAlchemy 2.0"
```

### `get_source_status`

Check health of all sources — which are enabled, have API keys, circuit breaker state.

### `get_rate_limit_status`

View rate limit quotas and usage across all APIs.

### `clear_cache`

Clear the search result cache.

---

## Rate Limits

| Source | Free Tier | With API Key |
|--------|-----------|--------------|
| Stack Exchange | 300/day | 10,000/day |
| GitHub | 10/min | 30/min |
| Hacker News | 1000/hour | — |
| Serper | — | 2,500/month |
| Tavily | — | 1,000/month |
| Brave | — | 2,000/month |

---

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

```bash
# Format
black .
isort .

# Lint
flake8
mypy .
```

### Adding a New Source

1. Create `api/yoursource.py` with an async `search()` function
2. Export from `api/__init__.py`
3. Add to source config in `community_research_mcp.py`
4. Add rate limit info

---

## Troubleshooting

### "No results found"

- Check that your topic is specific enough (not just "performance" or "settings")
- Verify API keys are set for web search sources
- Check `get_source_status` for circuit breaker state

### "Rate limit exceeded"

- Wait for the rate limit window to reset
- Add API keys for higher limits
- Use caching to reduce repeated searches

### "Source is failing"

- Check `get_source_status` for circuit breaker state
- Circuit breaker opens after 5 failures, resets after 5 minutes
- Some sources may be temporarily unavailable

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

---

<p align="center">
  <i>Built for developers who want real solutions, not documentation.</i>
</p>
