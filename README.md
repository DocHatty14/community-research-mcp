# Community Research MCP

<img width="1170" height="1170" alt="image" src="https://github.com/user-attachments/assets/fe3d1f19-b60a-4d95-bad4-bf8039873fed" />

> **Street-smart tips, hacks, and workarounds from devs who've been there.**

An MCP server that digs through Stack Overflow, GitHub Issues, Hacker News, and other developer watering holes to find battle-tested solutions, clever workarounds, and real-world patterns that actually work.

```
┌────────────────────────────────────────────────────────────────────────┐
│  "How do I handle FastAPI background tasks with Redis?"                │
│                                                                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  │
│  │ Stack        │  │ GitHub       │  │ Hacker       │  │ Discourse  │  │
│  │ Overflow     │  │ Issues       │  │ News         │  │ Forums     │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └─────┬──────┘  │
│         │                 │                 │                │         │
│         └─────────────────┴─────────────────┴────────────────┘         │
│                                  │                                     │
│                         ┌────────▼────────┐                            │
│                         │   Deduplicate   │                            │
│                         │    & Score      │                            │
│                         └────────┬────────┘                            │
│                                  │                                     │
│                         ┌────────▼────────┐                            │
│                         │  Ranked Results │                            │
│                         │  with Quality   │                            │
│                         │  Scores 0-100   │                            │
│                         └─────────────────┘                            │
└────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Installation

```bash
# Clone the repo
git clone https://github.com/your-repo/community-research-mcp.git
cd community-research-mcp

# Install deps
pip install -e .

# Or with uv (faster)
uv pip install -e .
```

### Configure Claude Desktop

Drop this into your `claude_desktop_config.json`:

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

Just ask Claude:

> "Search for Python async database connection pooling patterns"

The MCP taps into developer communities and surfaces ranked solutions with quality scores.

---

## What It Does

### Multi-Source Search

Hits 9 developer communities at once:

| Source | What You Get | API Key? |
|--------|--------------|----------|
| **Stack Overflow** | Q&A gold | Nope |
| **GitHub Issues** | Bug reports, workarounds | Optional (higher limits) |
| **Hacker News** | Tech war stories | Nope |
| **Lobsters** | Deep technical dives | Nope |
| **Discourse** | Language-specific forums | Nope |
| **Serper** | Google Search results | Yes |
| **Tavily** | AI-optimized search | Yes |
| **Brave** | Privacy-focused search | Yes |
| **Firecrawl** | Web scraping | Yes |

### Quality Scoring

Every result gets a street-cred score (0-100) based on:

- **Source Authority** — Stack Overflow accepted answers beat random blog posts
- **Community Validation** — Upvotes, reactions, the crowd has spoken
- **Recency** — Fresh fixes for fast-moving tech
- **Specificity** — Actual code, not hand-wavy explanations
- **Evidence** — Benchmarks, repro steps, real numbers

### Built-In Reliability

- **Circuit Breaker** — Stops hammering dead APIs
- **Auto-Retry** — Exponential backoff for flaky connections
- **Deduplication** — Kills duplicate results (~20% noise reduction)
- **Caching** — 1-hour TTL so you don't burn through rate limits

---

## Configuration

### Environment Variables

Create a `.env` file:

```bash
# Optional: Web Search APIs (unlock more sources)
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

Tweak source priorities in `config.json`:

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

Higher weight = more trusted for street-smart solutions.

---

## Project Structure

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

The main workhorse. Searches all enabled sources and returns ranked results.

```
language: "Python"
topic: "async SQLAlchemy connection pooling with FastAPI"
goal: "Handle 1000 concurrent database connections"
current_setup: "FastAPI + PostgreSQL + SQLAlchemy 2.0"
```

### `get_source_status`

Check what's up with all sources — enabled, has API keys, circuit breaker tripped, etc.

### `get_rate_limit_status`

See how much quota you've burned through across all APIs.

### `clear_cache`

Nuke the search cache when you need fresh results.

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

- Make your query more specific (not just "performance" or "settings")
- Check that API keys are set for web search sources
- Run `get_source_status` to see if circuit breakers tripped

### "Rate limit exceeded"

- Chill and wait for the window to reset
- Add API keys for higher limits
- Rely on caching to avoid repeat searches

### "Source is failing"

- Check `get_source_status` for circuit breaker state
- Breaker trips after 5 failures, resets after 5 minutes
- Sometimes sources just go down — it happens

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Contributing

1. Fork the repo
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a PR

---

<p align="center">
  <i>Built for devs who want real solutions, not documentation fluff.</i>
</p>
