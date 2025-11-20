# Documentation

## Table of Contents

1. [Installation](#installation)
2. [Configuration](#configuration)
3. [Core Concepts](#core-concepts)
4. [API Reference](#api-reference)
5. [Architecture](#architecture)
6. [Advanced Features](#advanced-features)

---

## Installation

### Prerequisites

- Python 3.8+
- API key for at least one LLM provider (Gemini, OpenAI, or Anthropic)

### Setup

```bash
git clone https://github.com/your-repo/community-research-mcp.git
cd community-research-mcp
initialize.bat  # Windows
```

### Environment Configuration

Create a `.env` file in the project root:

```env
# Required: At least one LLM provider
GEMINI_API_KEY=your_gemini_key
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Optional: Enhanced Reddit access
REDDIT_CLIENT_ID=your_reddit_id
REDDIT_CLIENT_SECRET=your_reddit_secret
REDDIT_REFRESH_TOKEN=your_refresh_token

# Optional: Premium search APIs
BRAVE_SEARCH_API_KEY=your_brave_key
SERPER_API_KEY=your_serper_key
```

---

## Configuration

### LLM Provider Selection

The system automatically detects available LLM providers and uses them in priority order:

1. Gemini (primary)
2. OpenAI (fallback)
3. Anthropic (fallback)

### Search Source Configuration

By default, all free search sources are enabled:

- Stack Overflow (no configuration required)
- GitHub (no configuration required)
- Reddit (public API, enhanced with authentication)
- Hacker News (no configuration required)
- DuckDuckGo (no configuration required)

Premium sources (Brave, Serper) require API keys.

---

## Core Concepts

### Search Philosophy

Community Research MCP prioritizes real-world, battle-tested solutions over theoretical documentation. It aggregates wisdom from:

- **Stack Overflow**: Validated solutions with community consensus
- **GitHub Issues**: Bug fixes, workarounds, and implementation details
- **Reddit**: Honest discussions and tool comparisons
- **Hacker News**: Architectural insights and industry perspectives
- **Web Content**: Active browsing of documentation and tutorials

### Result Quality

Every result is scored based on multiple factors:

- Source authority (official docs > community discussions)
- Community validation (upvotes, accepted answers)
- Recency (newer solutions for active ecosystems)
- Specificity (targeted answers > generic advice)
- Evidence quality (code examples, detailed explanations)

### Progressive Enhancement

The system operates in layers:

1. **Basic Search**: Parallel search across all sources
2. **Streaming Search**: Real-time results as they arrive
3. **Deep Research**: Iterative search with gap analysis
4. **Validated Research**: Multi-model verification

---

## API Reference

### Standard Search

```python
community_search(
    language: str,
    topic: str,
    goal: Optional[str] = None,
    current_setup: Optional[str] = None
) -> str
```

Performs parallel search across all sources and returns synthesized results.

**Parameters:**
- `language`: Programming language or framework context
- `topic`: Search query describing the problem or topic
- `goal`: Optional specific objective to guide synthesis
- `current_setup`: Optional description of current environment

**Returns:** Markdown-formatted synthesis with key findings

---

### Streaming Search

```python
streaming_community_search(
    language: str,
    topic: str,
    goal: Optional[str] = None,
    current_setup: Optional[str] = None
) -> str
```

Real-time search with progressive result updates.

**Parameters:** Same as `community_search`

**Returns:** Progressive markdown updates followed by final synthesis

**Benefits:**
- See results as they arrive (0.8s to first result)
- Early insights from fast sources while slower sources complete
- Transparent progress tracking

---

### Deep Research

```python
deep_community_search(
    language: str,
    topic: str,
    goal: Optional[str] = None,
    current_setup: Optional[str] = None
) -> str
```

Multi-iteration research loop with intelligent gap analysis.

**Parameters:** Same as `community_search`

**Process:**
1. Initial broad search across all sources
2. LLM analyzes results to identify knowledge gaps
3. Targeted follow-up searches for missing information
4. Active browsing of top URLs for full context
5. Comprehensive synthesis of all findings

**Use Cases:**
- Complex architectural decisions
- Unfamiliar technologies requiring deep understanding
- Problems with multiple potential solutions requiring comparison

---

### Research Planning

```python
plan_research(
    query: str,
    thinking_mode: Literal["fast", "balanced", "deep"] = "balanced"
) -> str
```

Generates a structured research plan without executing it.

**Parameters:**
- `query`: Research topic or question
- `thinking_mode`: Depth of planning
  - `fast`: Quick plan with essential steps
  - `balanced`: Comprehensive plan with alternatives
  - `deep`: Exhaustive plan with edge cases

**Returns:** Structured research plan in markdown

---

### Validated Research

```python
validated_research(
    language: str,
    topic: str,
    goal: Optional[str] = None,
    current_setup: Optional[str] = None
) -> str
```

Performs search with independent secondary verification.

**Process:**
1. Primary LLM conducts research and generates findings
2. Secondary LLM (different provider) critiques results
3. Verification checks for:
   - Technical accuracy
   - Security vulnerabilities
   - Deprecated methods
   - Logical inconsistencies

**Returns:** Primary findings with validation report

---

### System Capabilities

```python
get_system_capabilities() -> str
```

Returns detailed report of available search sources and LLM providers.

**Returns:** Markdown-formatted capability report showing:
- Active/inactive search APIs
- Available LLM providers
- Configuration status

---

### Content Fetching

```python
fetch_page_content(
    url: str,
    max_chars: int = 12000
) -> str
```

Fetches and extracts main content from a URL.

**Parameters:**
- `url`: Target URL to fetch
- `max_chars`: Maximum content length

**Returns:** Extracted text content with code blocks preserved

---

### Performance Metrics

```python
get_performance_metrics() -> str
```

Returns real-time performance and reliability metrics.

**Requires:** Enhanced MCP utilities module

**Returns:** Detailed metrics including:
- System uptime and search counts
- Average search/synthesis times
- Cache hit rates
- API reliability and error distribution
- Quality score statistics

---

## Architecture

### System Design

```
┌─────────────────────────────────────────────────────────────┐
│                        MCP Server                            │
│  ┌────────────┐  ┌────────────┐  ┌──────────────────────┐  │
│  │  Tool API  │  │   Cache    │  │  Capability Detect   │  │
│  └────────────┘  └────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼────────┐  ┌────────▼───────┐  ┌────────▼────────┐
│  Search Layer  │  │  Synthesis     │  │  Enhancement    │
│  ├─ Stack OF   │  │  ├─ Primary    │  │  ├─ Quality     │
│  ├─ GitHub     │  │  │   LLM       │  │  │   Scoring    │
│  ├─ Reddit     │  │  ├─ Secondary  │  │  ├─ Dedup       │
│  ├─ HN         │  │  │   LLM       │  │  ├─ Retry       │
│  └─ DDG        │  │  │   (verify)  │  │  └─ Monitoring  │
└────────────────┘  └────────────────┘  └─────────────────┘
```

### Data Flow

1. **Request Reception**: Tool receives query with context
2. **Capability Detection**: System identifies available search sources and LLM providers
3. **Parallel Dispatch**: Async tasks launched for each search source
4. **Result Aggregation**: Progressive aggregation with type classification
5. **Content Enhancement**: Top URLs fetched for full context
6. **Synthesis**: LLM processes aggregated results into coherent findings
7. **Quality Enhancement**: Results scored, deduplicated, and formatted
8. **Response Delivery**: Markdown-formatted synthesis returned to client

### Concurrency Model

Built on `asyncio` for efficient parallel execution:

- **Search Tasks**: Independent async tasks per source
- **Result Queue**: FIFO queue for progressive aggregation
- **Timeout Management**: Per-source timeouts with graceful degradation
- **Error Isolation**: Failures in one source don't affect others

### Caching Strategy

**File**: `.community_research_cache.json`

**Cache Key**: `hash(query + language + tool_name)`

**TTL**: 24 hours (configurable)

**Behavior**:
- Read-through cache (check on every request)
- Write-through cache (save on every successful search)
- Disk persistence (survives application restarts)
- Automatic eviction (expired entries removed on startup)

### Module Organization

**Core Modules:**
- `community_research_mcp.py`: Main server, tool definitions, search functions
- `streaming_capabilities.py`: Result classification, progressive aggregation
- `streaming_search.py`: Parallel search orchestration
- `enhanced_mcp_utilities.py`: Reliability and quality enhancements

**Dependencies:**
- `mcp`: Model Context Protocol SDK
- `httpx`: Async HTTP client
- `beautifulsoup4`: HTML parsing
- `python-dotenv`: Environment configuration

---

## Advanced Features

### Result Classification

Results are automatically classified by content type:

- **Quick Fix**: Accepted answers with code examples
- **Code Example**: GitHub code samples and snippets
- **Discussion**: Community debates and comparisons
- **Warning**: Known issues, gotchas, and pitfalls
- **Tutorial**: Step-by-step guides and walkthroughs
- **Official Docs**: Documentation and reference material

### Smart Clustering

Results are semantically organized into clusters:

- Identifies common themes across sources
- Groups similar solutions together
- Highlights alternative approaches
- Separates official guidance from community workarounds

### Active Browsing

For deep research, the system:

1. Identifies high-value URLs from search results
2. Fetches full page content (not just snippets)
3. Extracts main text and code blocks
4. Preserves context for LLM synthesis
5. Ensures comprehensive understanding before generating answers

### Multi-Model Validation

When using `validated_research`:

1. **Primary Research**: First LLM conducts comprehensive search
2. **Critique Generation**: Second LLM (different provider) reviews findings
3. **Validation Checks**:
   - Code correctness
   - Security vulnerabilities
   - Deprecated APIs
   - Missing context
   - Alternative approaches
4. **Consolidated Report**: Primary findings with validation notes

### Resilient API Calls

Enhanced utilities module provides:

- **Automatic Retry**: 3 attempts with exponential backoff
- **Strategy Options**: Exponential, linear, or constant backoff
- **Jitter**: Prevents thundering herd on retries
- **Error Tracking**: Per-type error distribution monitoring
- **Graceful Degradation**: Continues with partial results on failure

### Performance Monitoring

Comprehensive metrics tracking:

- **Search Metrics**: Time per search, results per source
- **Synthesis Metrics**: LLM response time and token usage
- **Cache Metrics**: Hit rate, miss rate, total entries
- **API Metrics**: Success rate, retry count, error distribution
- **Quality Metrics**: Average confidence scores, deduplication savings

---

## Troubleshooting

### Common Issues

**"Streaming capabilities not available"**
- Ensure `streaming_capabilities.py` and `streaming_search.py` are in the project directory
- Verify file permissions

**"No LLM provider available"**
- Check `.env` file exists
- Verify at least one API key is configured: `GEMINI_API_KEY`, `OPENAI_API_KEY`, or `ANTHROPIC_API_KEY`
- Ensure API keys are valid

**"Rate limit exceeded"**
- Wait 60 seconds and retry
- Consider upgrading API tier
- Use caching to reduce API calls

**"No results found"**
- Broaden search terms
- Check network connectivity
- Verify search sources are not blocked by firewall
- Try `get_system_capabilities()` to check source availability

### Performance Optimization

**Slow searches:**
- Enable caching (enabled by default)
- Use `streaming_community_search` for faster initial results
- Reduce search scope with specific language/framework

**High API costs:**
- Enable result caching
- Use `plan_research` to preview before executing
- Prefer `community_search` over `deep_community_search` for simple queries

**Memory issues:**
- Reduce `max_chars` in `fetch_page_content`
- Clear cache file periodically: `rm .community_research_cache.json`
- Limit concurrent searches in custom implementations

---

## Best Practices

### Query Formulation

**Effective queries:**
- "FastAPI async background tasks with Redis queue"
- "Next.js server components data fetching patterns"
- "Rust async trait object lifetime issues"

**Ineffective queries:**
- "how to code" (too vague)
- "error" (missing context)
- "best practice" (no specific technology)

### Tool Selection

- **Standard search**: Most common problems with clear solutions
- **Streaming search**: When you want early insights
- **Deep research**: Complex problems requiring comprehensive understanding
- **Validated research**: Critical systems where accuracy is paramount
- **Plan research**: Exploring unfamiliar territory before committing

### Context Enrichment

Always provide:
- **Language**: Specific version if relevant (e.g., "Python 3.11", "Node.js 18")
- **Goal**: What you're trying to achieve
- **Current setup**: Relevant environment details

**Example:**
```python
community_search(
    language="Python 3.11 with FastAPI 0.104",
    topic="async background tasks that survive server restart",
    goal="Implement reliable job queue for email sending",
    current_setup="Docker container with Redis available"
)
```

---

*For additional support, consult the source code or open an issue in the repository.*
