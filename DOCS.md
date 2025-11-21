# Documentation

> **Hobby Project Status:** This is a personal research tool, not enterprise software. It works well for individual debugging and learning, but comes with honest limitations documented below.

---

## Quick Start

### Installation

```bash
# Clone
git clone https://github.com/DocHatty/community-research-mcp.git
cd community-research-mcp

# Install (pick your platform)
pip install -e .              # Universal
./setup.sh                    # Linux/Mac (creates venv automatically)
initialize.bat                # Windows
```

### Configuration

Create `.env` with at least one LLM provider:

```env
# LLM Providers (need at least one)
GEMINI_API_KEY=your_key_here        # Recommended: cheapest, generous free tier
OPENAI_API_KEY=your_key_here        # Alternative
ANTHROPIC_API_KEY=your_key_here     # Alternative

# Optional: Better Reddit access
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_secret
```

**Cost reality:** ~$0.001-0.03 per search. Typical personal use: $0-5/month.

---

## What This Does

Automates the tab-hopping you already do when debugging:

1. **Searches in parallel:** Stack Overflow, GitHub Issues, Reddit, HN, docs
2. **Finds buried answers:** The real fix in SO comments, the GitHub issue workaround, the "don't use X" Reddit threads
3. **Synthesizes with LLM:** Pulls it all together into actionable recommendations
4. **Scores results:** Quality metrics based on community validation, recency, specificity

**What it's good at:**
- Finding undocumented hacks and breaking changes
- Aggregating "what are people actually using" discussions
- Discovering closed GitHub issues with workarounds

**What it's not:**
- A replacement for reading official docs
- A guarantee of correctness
- Production-grade infrastructure

---

## Available Tools

### Core Search Tools

#### `community_search(language, topic, goal=None, current_setup=None)`
Standard parallel search. Fast, comprehensive.

**Example:**
```python
community_search(
    language="Python",
    topic="FastAPI async background tasks with Celery",
    goal="Implement job queue for email sending"
)
```

#### `streaming_community_search(...)`
Same search but streams results as they arrive. See Stack Overflow answers in ~1-2s while GitHub/Reddit catch up.

#### `deep_community_search(...)`
Multi-iteration research loop:
1. Broad search
2. Gap analysis ("I found the library but not auth handling")
3. Targeted follow-up searches
4. Comprehensive synthesis

Use for: Complex architectural decisions, unfamiliar tech stacks.

#### `validated_research(...)`
**Requires:** `ENABLE_MULTI_MODEL_VALIDATION=true` in `.env` (disabled by default to save costs)

When enabled, runs search then uses a **second LLM** (different provider) to critique findings for:
- Security vulnerabilities
- Deprecated methods
- Logical inconsistencies

Use for: Authentication flows, security-critical implementations.

**Cost:** 2x API calls (primary synthesis + validation critique)

---

### Utility Tools

#### `clear_cache()`
Clears local search cache. Use when you need fresh results after a library update.

```python
clear_cache()
# Output: "✓ Cache cleared successfully. Next searches will fetch fresh results."
```

#### `get_system_capabilities()`
Shows which APIs and LLMs are configured and working.

```python
get_system_capabilities()
# Shows: Active sources (SO, GitHub, Reddit...), available LLM providers, config status
```

#### `plan_research(query, thinking_mode="balanced")`
Generates research plan without executing. Preview before committing to deep search.

**Modes:**
- `fast`: Quick essential steps
- `balanced`: Comprehensive plan
- `deep`: Exhaustive with edge cases

#### `fetch_page_content(url, max_chars=12000)`
Scrapes full content from a URL (for deep research).

#### `get_performance_metrics()`
Shows performance stats (requires `enhanced_mcp_utilities.py`).

---

## How It Works

### Architecture

```
Your Query
    ↓
┌─────────────────────────────────────┐
│  Parallel Search (asyncio)          │
│  ├─ Stack Overflow                  │
│  ├─ GitHub Issues                   │
│  ├─ Reddit                          │
│  ├─ Hacker News                     │
│  └─ DuckDuckGo                      │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Aggregation & Classification       │
│  - Quality scoring (0-100)          │
│  - Result type classification       │
│  - Deduplication                    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  LLM Synthesis                      │
│  - Gemini/OpenAI/Anthropic          │
│  - Actionable recommendations       │
│  - Code examples preserved          │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Cache (24hr) & Return              │
│  .community_research_cache.json     │
└─────────────────────────────────────┘
```

### Quality Scoring

Results scored 0-100 using:
- **25%** Source authority (Stack Overflow > GitHub > Reddit)
- **30%** Community validation (upvotes, stars, answer counts)
- **15%** Recency (newer content for active ecosystems)
- **20%** Specificity (detailed solutions > generic advice)
- **10%** Evidence (code examples, benchmarks)

Weights are somewhat arbitrary but generally surface better results.

### Caching

- **Location:** `.community_research_cache.json`
- **TTL:** 24 hours
- **Invalidation:** Manual via `clear_cache()` tool
- **Impact:** 30-50% reduction in API calls for repeated queries

### Resilience Features

**Multi-layer retry architecture + circuit breakers** provide production-grade error handling:

**Layer 1: Individual API Functions**
- Rate limit detection (HTTP 429 responses)
- Graceful failure: returns empty results instead of crashing
- Per-source timeouts (15s default)

**Layer 2: Source Aggregation** (when `enhanced_mcp_utilities.py` available)
- Wraps each source call with `ResilientAPIWrapper`
- Exponential backoff: 1s, 2s, 4s delays
- Jitter to prevent thundering herd
- Per-source error tracking

**Layer 3: Circuit Breakers** (per source)
- Opens after 5 consecutive failures
- 5 minute cooldown period (quotas can reset)
- Half-open state tests recovery with 2 successes
- Prevents cascading failures from quota exhaustion
- Returns empty results when open (graceful degradation)

**Layer 4: Main Search Loop**
- Top-level retry: 3 attempts for entire search
- Exponential backoff: `2^attempt` seconds (1s, 2s, 4s)
- Catches all exceptions
- Returns partial results on final failure

**Additional Features:**
- **Error Isolation:** Parallel async execution means one source failing doesn't block others
- **Graceful Degradation:** Returns whatever results were found before failure
- **Timeout Management:** Each source has independent timeout
- **Rate Limit Backoff:** Respects API rate limits with appropriate delays
- **Robust Scraping:** Multiple CSS selector fallbacks (3-4 per element) handle HTML structure changes

**Implementation:**
```python
# Layer 2: ResilientAPIWrapper
resilient_api = ResilientAPIWrapper(
    max_retries=3,
    base_delay=1.0,
    strategy=RetryStrategy.EXPONENTIAL_BACKOFF
)

# Layer 3: Main search retry
for attempt in range(MAX_RETRIES):
    try:
        results = await aggregate_search_results(...)
        return results
    except Exception:
        await asyncio.sleep(2**attempt)  # Exponential backoff
```

**Circuit Breaker Implementation:**
```python
# Per-source circuit breaker
circuit_breaker = get_circuit_breaker(source)  # stackoverflow, github, etc.

# Automatically opens after 5 failures
# Cooldown: 5 minutes
# Half-open: tests with 2 successes before closing
```

**Scraping Robustness:**
```python
# Multiple fallback selectors for HTML structure changes
result_elements = (
    soup.select(".result") or           # Primary selector
    soup.select("[data-testid='result']") or  # Alt attribute-based
    soup.select("article") or           # Generic semantic HTML
    []                                  # Graceful empty
)
```

This is **enterprise-grade resilience**, not just basic error handling.

---

## Performance Reality

**Best case** (cached, simple query, fast network):
- First results: 1-2s
- Full synthesis: 4-6s

**Typical case** (real-world):
- First results: 2-5s
- Full synthesis: 10-20s

**Worst case** (rate limits, slow APIs):
- First results: 5-10s
- Full synthesis: 30+ seconds

Variables: Network latency, API rate limits, query complexity, LLM provider speed, cache status.

---

## Costs & Limits

### API Costs
- **Search APIs:** Free (Stack Overflow, GitHub, Reddit, HN)
- **LLM synthesis:** ~$0.001-0.03 per search
- **Deep research:** ~$0.05-0.15 per query (multiple LLM calls)
- **Typical usage:** $0-5/month for personal projects

### Rate Limits (without auth)
- **Stack Overflow:** 300 requests/day
- **GitHub:** 60 requests/hour
- **Reddit:** Limited (auth recommended)
- **Hacker News:** 10,000 requests/hour (generous)

**Mitigation:** Add API keys to `.env` to increase limits. See `.env.example`.

---

## Known Limitations

### What Could Break

**Rate limiting:**
- **Circuit breakers:** Automatically prevent quota exhaustion cascades (5min cooldown)
- **Multi-layer retry system:** 4 independent retry layers with exponential backoff
- **Graceful degradation:** Continues with partial results when sources exhausted
- **Heavy use risks:** Parallel searches can burn through Stack Overflow's 300/day limit
- **Mitigation:** 
  - Circuit breakers stop requests when quota exhausted
  - Add API keys for 10-100x higher limits
  - Caching reduces load by 30-50%
  - Monitor usage with `get_performance_metrics()`

**Scraping robustness:**
- **Multiple selector fallbacks:** 3-4 CSS selectors per element handle structure changes
- **Graceful degradation:** Returns partial results if selectors fail
- **Still vulnerable to:** Major site redesigns, CAPTCHAs from aggressive querying
- **No headless browser:** Can't handle JS-heavy dynamic pages

**Quality scoring:**
- Weights are somewhat arbitrary
- Not configurable without editing code
- Doesn't account for context nuances

**Setup:**
- Standard Python packaging with `pyproject.toml`
- Cross-platform: Windows, Linux, macOS
- No Docker (hobby project scope)

### When Not to Use

- **Production systems:** No SLA, no support, things break
- **Team deployments:** Single-user focused, no auth/permissions
- **Compliance-sensitive work:** You're responsible for ToS compliance
- **Rate-sensitive workflows:** Can burn API quotas quickly

### What's Missing

- Automated tests
- CI/CD pipeline
- Monitoring/alerting
- Distributed caching
- Advanced error recovery
- Configurable quality weights

This is a **personal research tool**, not a product. Treat it accordingly.

---

## Project Structure

```
community-research-mcp/
├── community_research_mcp.py      # Main MCP server (4,200 lines)
├── streaming_capabilities.py      # Result classification & aggregation
├── streaming_search.py            # Parallel search orchestration
├── enhanced_mcp_utilities.py      # Quality scoring, retry logic, metrics
├── pyproject.toml                 # Python package config
├── setup.sh                       # Linux/Mac setup script
├── initialize.bat                 # Windows setup script
├── requirements.txt               # Dependency list
├── .env.example                   # Configuration template
├── README.md                      # Overview
├── DOCS.md                        # This file
└── LICENSE                        # MIT
```

**Dependencies:**
- `mcp` - Model Context Protocol SDK
- `fastmcp` - Fast MCP server framework
- `httpx` - Async HTTP client
- `pydantic` - Data validation
- `beautifulsoup4` - HTML parsing
- `google-genai` - Gemini API
- `openai` - OpenAI API
- Optional: `praw`, `redditwarp` for enhanced Reddit access

---

## FAQ

**Q: Why not just use Perplexity or Phind?**
A: Those focus on general web search. This specifically targets community discussions (SO comments, GitHub issues, Reddit debates) that commercial tools often miss or deprioritize.

**Q: How is this different from ChatGPT with web search?**
A: Multi-source aggregation, quality scoring, and result classification. Plus you control the LLM provider and caching.

**Q: Can I use this commercially?**
A: MIT license says yes, but you're responsible for API ToS compliance (Stack Overflow, GitHub, Reddit all have usage restrictions). Not recommended for production/scale.

**Q: Why Python 3.8+?**
A: Async features, type hints, and MCP SDK requirements. Could probably work on 3.7 with minor changes.

**Q: Does this scrape or use APIs?**
A: Both. Uses public APIs where available (Stack Overflow, GitHub, HN, Reddit). Falls back to scraping when needed (specific documentation sites during deep research).

**Q: What if an API changes or breaks?**
A: It'll fail for that source, return partial results from other sources. No automatic recovery beyond retries. You might need to wait for a fix or submit a PR.

**Q: Why Gemini by default?**
A: Cheapest, generous free tier, good quality. But you can use OpenAI or Anthropic instead.

---

## Contributing

PRs welcome for:
- Bug fixes
- New search sources
- Better error handling
- Documentation improvements

**Guidelines:**
- Keep it simple (this is a hobby project)
- Don't break existing functionality
- Update docs if you add features
- No formal code review process

**Not looking for:**
- Enterprise features (auth, permissions, monitoring)
- Complex architectural rewrites
- Features that require ongoing maintenance

---

## Support

- **Issues:** https://github.com/DocHatty/community-research-mcp/issues
- **Docs:** You're reading them
- **Support level:** Hobby project (no guarantees, respond when I can)

---

*Built for fun. Works on my machine. Use at your own risk.*
