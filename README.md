# Community Research MCP

<div align="center">

![Community Research MCP](https://github.com/user-attachments/assets/cde51164-2f14-43d2-86be-b7e0d483d1c7)

**Real fixes from real people, not manuals.**

*A Model Context Protocol server that bypasses generic AI training data to tap directly into the living wisdom of the developer community.*

[![Status](https://img.shields.io/badge/Status-Hobby_Project-yellow?style=flat-square)](https://github.com/DocHatty/community-research-mcp)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square)](https://www.python.org/)

</div>

---

## Current State

**This is a hobby project** for personal use and experimentation. It works well for:
- Individual developers debugging obscure issues
- Research that requires aggregating community wisdom
- Automating the manual tab-hopping you already do

**Not recommended for:**
- Production systems or teams (no SLA, no support)
- Rate-sensitive workflows (you're responsible for API costs/limits)
- Anything requiring legal compliance review

If you use this, you're opting into the same risks you take manually scraping Stack Overflow at 2 AM.

---

## Philosophy

Most AI tools provide textbook answers that work in theory but fail in production. Community Research MCP is different. It aggregates battle-tested workarounds, undocumented hacks, and hard-earned lessons from:

- Stack Overflow: Accepted solutions and the "real" answer in the comments
- GitHub Issues: Bug fixes, patch notes, and closed-as-won't-fix workarounds
- Reddit: Honest debates, tool comparisons, and "don't use X, use Y" advice
- Hacker News: Architectural critiques and industry trends
- Web Scraping: Full documentation and blog posts, not just snippets

---

## Key Features

### Deep Research Loop

Mimics a senior engineer's research process:

1. Broad parallel search across all sources
2. Gap analysis: "I found the library, but not how to handle auth"
3. Targeted follow-up searches to fill knowledge gaps
4. Comprehensive synthesis of all findings

### Active Browsing

Visits actual webpages to scrape full documentation and GitHub issue threads. No more relying on 2-line search snippets.

### Multi-Model Validation (Opt-In)

**Disabled by default.** When enabled via `.env`, a second independent AI model critiques the findings to check for security flaws, deprecated methods, and logical inconsistencies.

Requires: `ENABLE_MULTI_MODEL_VALIDATION=true` in `.env` (costs 2x API calls)

### Parallel Streaming

Results stream in real-time as they're found:

- Stack Overflow: ~0.8s
- GitHub: ~1.2s
- Full synthesis: ~4s

### Quality Scoring

Results are scored 0-100 using heuristics that seem reasonable:
- Source authority: Stack Overflow > GitHub > Reddit (25%)
- Community validation: upvotes, stars, answer counts (30%)
- Recency: newer content scores higher (15%)
- Specificity: detailed solutions > generic advice (20%)
- Evidence: code examples, benchmarks (10%)

These weights are somewhat arbitrary and not configurable. They generally help surface better results, but you might disagree with the priorities.

---

## Installation

**Quick start:**
```bash
git clone https://github.com/DocHatty/community-research-mcp.git
cd community-research-mcp

# Windows
initialize.bat

# Linux/Mac
chmod +x setup.sh
./setup.sh

# Or manually
pip install -e .
cp .env.example .env
```

**Cross-platform:** Works on Windows, Linux, macOS

Configure your API keys in `.env`:

```env
# Required: At least one LLM provider
GEMINI_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here

# Optional: Enhanced features
REDDIT_CLIENT_ID=your_id
REDDIT_CLIENT_SECRET=your_secret
```

---

## Usage

### Example Output

**Query:** "Rust wgpu PipelineCompilationOptions removed in latest version"

**Result:**
```markdown
# Community Research: wgpu PipelineCompilationOptions

## 🔍 Stack Overflow (Score: 85)
**Issue:** `compilation_options` field removed in wgpu 0.19
**Solution:** Use `ShaderModuleDescriptor` directly
[Source](https://stackoverflow.com/questions/...)

## 🐛 GitHub Issue #4528 (Score: 92)
**Breaking change:** PipelineCompilationOptions deprecated in favor of new descriptor API
**Migration code from Bevy engine team:**
```rust
// Old (0.18)
let shader = device.create_shader_module(ShaderModuleDescriptor {
    compilation_options: PipelineCompilationOptions::default(),
    ..
});

// New (0.19+)
let shader = device.create_shader_module(ShaderModuleDescriptor {
    label: Some("shader"),
    source: ShaderSource::Wgsl(code.into()),
});
```
[wgpu Issue #4528](https://github.com/gfx-rs/wgpu/issues/4528)

## 💬 Reddit r/rust_gamedev (Score: 78)
Discussion: "wgpu 0.19 breaking changes megathread"
Recommended: Update to new descriptor pattern, old API completely removed
```

---

### Standard Search

```python
community_search(
    language="Rust",
    topic="wgpu PipelineCompilationOptions removed in latest version",
    goal="Fix compilation errors after upgrade"
)
```

**Returns:**
> The `compilation_options` field was removed in wgpu 0.19. Community discussions on GitHub Issue #452 suggest using `wgpu::ShaderModuleDescriptor` directly. Here is the working migration code used by the Bevy engine team...

### Streaming Search

```python
streaming_community_search(
    language="Python",
    topic="FastAPI async background tasks with Celery"
)
```

Get progressive updates as results arrive from each source.

### Deep Research

```python
deep_community_search(
    language="Python",
    topic="Microservices architecture patterns with Kafka",
    goal="Design a scalable event-driven system"
)
```

Multi-iteration research with intelligent gap analysis and comprehensive synthesis.

### Validated Research

```python
validated_research(
    language="Python",
    topic="JWT authentication with refresh tokens",
    goal="Implement secure auth flow"
)
```

Primary research with secondary model verification for critical implementations.

---

## Architecture

Built on asynchronous Python with parallel search execution:

- **Search Layer**: Concurrent queries across multiple sources (asyncio)
- **Aggregation Layer**: Progressive result collection and classification
- **Synthesis Layer**: LLM-powered analysis and recommendation  
- **Enhancement Layer**: Quality scoring, deduplication, retry logic

**Resilience (Multi-layer Retry + Circuit Breakers):**
- **Layer 1:** Individual API rate limit detection (HTTP 429)
- **Layer 2:** ResilientAPIWrapper with exponential backoff per source
- **Layer 3:** Circuit breakers prevent quota exhaustion cascades (5min cooldown)
- **Layer 4:** Top-level search retry (3 attempts, 1s→2s→4s backoff)
- **Error Isolation:** Parallel async = one source failing doesn't block others
- **Graceful Degradation:** Returns partial results when sources fail
- **Smart Caching:** 24-hour TTL reduces API load by 30-50%
- **Robust Scraping:** Multiple CSS selector fallbacks for HTML structure changes

See `enhanced_mcp_utilities.py` for implementation details.

---

## Performance

**Best case** (cached, simple query, fast network):
- First results: 1-2 seconds
- Full synthesis: 4-6 seconds

**Typical case** (real-world usage):
- First results: 2-5 seconds
- Full synthesis: 10-20 seconds

**Worst case** (rate limits, slow APIs, complex queries):
- First results: 5-10 seconds  
- Full synthesis: 30+ seconds

Performance depends on network latency, API rate limits, query complexity, LLM provider speed, and whether results are cached. The "~0.8s Stack Overflow" claim assumes cache hits and no rate limiting—not realistic for sustained use.

---

## Documentation

See [DOCS.md](DOCS.md) for API reference.

---

## Requirements

- Python 3.8+
- API key for at least one LLM provider (Gemini, OpenAI, or Anthropic)
- Internet connection for search APIs

---

## Costs & Legal

**API Costs:**
- Search APIs are free (Stack Overflow, GitHub, Reddit, HN)
- LLM costs: ~$0.001-0.03 per search depending on provider
- Deep research with validation: ~$0.05-0.15 per query
- Typical usage: $0-5/month for personal projects

**Rate Limits:**
You're subject to rate limits from each API. Without authentication:
- Stack Overflow: 300 requests/day
- GitHub: 60 requests/hour
- Reddit: Limited access

See `.env.example` for how to add API keys to increase limits.

**Legal Considerations:**
This tool queries public APIs and scrapes publicly accessible content. You're responsible for:
- Complying with each platform's Terms of Service
- Respecting rate limits
- Not using this for commercial scraping at scale

If you're worried about compliance, don't use this. It's for personal research, not enterprise deployment.

---

## Known Issues & Limitations

**Rate Limiting:**
- **Circuit breakers:** Automatically stops requests after 5 failures, 5min cooldown prevents cascades
- **Multi-layer retry:** Individual API detection + per-source wrapper + top-level retry
- **Exponential backoff:** 1s→2s→4s delays across 3 retry attempts
- **Graceful degradation:** Returns partial results when sources exhausted
- **Limits:** Stack Overflow 300/day unauth, GitHub 60/hour unauth
- **Mitigation:** Add API keys to `.env` for 10-100x higher limits, caching reduces load by 30-50%

**Scraping Robustness:**
- **Multiple selector fallbacks:** Tries 3-4 different CSS selectors per element
- **Graceful HTML structure changes:** Automatically tries alternative selectors
- **Still vulnerable to:** Major site redesigns, CAPTCHAs from aggressive querying
- LLM timeouts retry 3x then give up
- Partial results returned when sources fail (by design)

**Quality Scoring:**
- Weights are somewhat arbitrary (25% source authority, 30% validation, etc.)
- Not configurable without editing code
- Doesn't account for context (old highly-voted answer vs recent edge case fix)

**Setup:**
- Cross-platform support via `pyproject.toml`, `setup.sh`, `initialize.bat`
- Standard Python packaging (`pip install -e .`)
- No Docker (hobby project, not containerized infrastructure)

**Caching:**
- Simple 24-hour TTL, no automatic invalidation
- Stale results if libraries/APIs change
- Use `clear_cache()` tool to manually refresh
- No distributed cache for multi-user scenarios

If any of this is a dealbreaker, this tool isn't for you.

---

## Project Structure

```
community-research-mcp/
├── community_research_mcp.py      # Main server and tool definitions
├── streaming_capabilities.py      # Result classification and aggregation
├── streaming_search.py            # Parallel search orchestration
├── enhanced_mcp_utilities.py      # Reliability and quality enhancements
├── initialize.bat                 # Setup script
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── DOCUMENTATION.md               # Complete documentation
└── LICENSE                        # MIT License
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Why This Exists

> *"The docs say it should work. Stack Overflow comment #3 says why it doesn't. GitHub issue #1247 has the workaround. Reddit says don't even bother, use this other library instead."*

You know this research pattern. You live it every time you debug something obscure. This tool automates it.

**The Gap:**
- **Official docs** tell you how things *should* work
- **AI training data** stops at some arbitrary cutoff date  
- **Real solutions** live in Stack Overflow comments, closed GitHub issues, and "actually, don't use X" Reddit threads

**What it does:**
- Searches Stack Overflow, GitHub Issues, Reddit, Hacker News in parallel
- Finds the buried answers (comment #3, the closed issue with 47 upvotes, the Reddit thread from last week)
- Synthesizes with an LLM into actionable recommendations
- Scores results by community validation, recency, and specificity

**What it's good at:**
- Finding undocumented breaking changes
- Discovering workarounds for known bugs
- Aggregating "what people actually use in production"

**What it's not:**
- A replacement for reading docs
- A guarantee of correctness (validate everything yourself)
- Enterprise-grade tooling (it's a hobby project)

---

## Contributing

PRs welcome. No formal process—just keep it simple and don't break existing stuff.

---

Built for fun. Works on my machine. YMMV.
