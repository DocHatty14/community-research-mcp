# 🚀 Streaming Search Features

## Overview

The Community Research MCP now supports **real-time streaming search** with automatic capability detection, parallel execution, and progressive result aggregation!

## Key Features

### 1. ⚡ Parallel Execution Across ALL Sources

All search capabilities fire **simultaneously** when you make a query:

- **Stack Overflow** - Quick fixes & accepted answers
- **GitHub** - Real code examples & implementations  
- **Reddit** - Community discussions & gotchas
- **Hacker News** - High-quality technical discussions
- **DuckDuckGo** - Broader web search

**No waiting for sequential searches!** Results stream in as each source completes.

### 2. 🔍 Automatic Capability Detection

The system automatically detects what API keys and search capabilities you have configured:

```python
# Use the get_system_capabilities tool
{
  "search_apis": {
    "stackoverflow": true,       # ✓ Always available
    "github": true,             # ✓ Always available
    "reddit": true,             # ✓ Public API fallback
    "reddit_authenticated": false,  # ✗ Needs credentials
    "hackernews": true,         # ✓ Always available
    "duckduckgo": true,         # ✓ Always available
    "brave": false,             # ✗ Needs API key
    "serper": false             # ✗ Needs API key
  },
  "llm_providers": {
    "gemini": true,             # ✓ Configured
    "openai": false,            # ✗ Not configured
    "anthropic": true,          # ✓ Configured
    "openrouter": false,
    "perplexity": false
  }
}
```

### 3. 📊 Real-Time Progress Updates

As searches execute, you get live updates via MCP context reporting:

```
🚀 Starting parallel search across 4 sources...
✓ stackoverflow: 5 results (0.8s)
✓ github: 5 results (1.2s)  
✓ reddit: 8 results (1.5s)
✓ hackernews: 3 results (2.1s)
✨ Search complete! 21 total results
```

### 4. 🔄 Progressive Reorganization

Results are **automatically reorganized** as they arrive:

#### First Result (0.8s)
```markdown
# 🔍 Search Progress: 1/4 sources
**Results:** 5 | **Elapsed:** 0.8s

⏳ Waiting for: github, reddit, hackernews

## 📊 Results by Type

### ✅ Quick Fixes (5)
- **FastAPI background tasks with Redis** (Stack Overflow)
```

#### Second Result (1.2s)
```markdown
# 🔍 Search Progress: 2/4 sources
**Results:** 10 | **Elapsed:** 1.2s

⏳ Waiting for: reddit, hackernews

## 📊 Results by Type

### ✅ Quick Fixes (5)
### 💻 Code Examples (5)
- **fastapi-redis-queue** (GitHub)
```

#### Final Result (2.1s)
```markdown
# 🎯 Community Research Results
**Total Results:** 21
**Sources:** stackoverflow, github, reddit, hackernews
**Search Time:** 2.10s

## 📋 Key Findings
[LLM synthesis of all results]
```

### 5. 🎯 Adaptive Content Formatting

Results are classified and organized by type:

- **Quick Fixes** ✅ - Accepted answers with working code
- **Code Examples** 💻 - GitHub repos with real implementations
- **Discussions** 💬 - Community threads and experiences
- **Warnings** ⚠️ - Known issues and gotchas
- **Tutorials** 📚 - Step-by-step guides
- **Official Docs** 📖 - Documentation links

This smart categorization helps you find exactly what you need faster.

### 6. 🤖 LLM Synthesis After Streaming

Once all results arrive, the system synthesizes them using your configured LLM:

```markdown
## 📋 Key Findings

### 1. Use Celery with Redis for Production
**Difficulty:** Medium
**Community Score:** 92/100

**Problem:** FastAPI's background tasks are in-process and don't scale...
**Solution:** [Detailed solution with code]
**Gotchas:** ⚠️ Celery requires separate worker processes...
```

## New Tools

### `get_system_capabilities`

Auto-detect all available capabilities:

```python
capabilities = get_system_capabilities()
# Returns formatted report of active/inactive features
```

### `streaming_community_search`

Main streaming search tool with real-time updates:

```python
result = streaming_community_search(
    language="Python",
    topic="FastAPI background task queue with Redis for email processing",
    goal="Send emails asynchronously without blocking API requests",
    current_setup="Currently using FastAPI with SQLite"
)
```

**Returns:** Progressive markdown updates as results stream in!

### `parallel_multi_source_search`

Advanced tool with fine-grained source control:

```python
result = parallel_multi_source_search(
    query="async/await error handling best practices",
    language="JavaScript",
    sources="stackoverflow,github,reddit"  # Skip HackerNews
)
```

**Returns:** JSON with results organized by source and type.

## Architecture

### Streaming Pipeline

```
User Query
    ↓
[Capability Detection] ← Auto-detect available APIs
    ↓
[Parallel Search Launch]
    ├─→ Stack Overflow ──┐
    ├─→ GitHub ──────────┤
    ├─→ Reddit ──────────┤→ [Result Queue]
    └─→ Hacker News ─────┘      ↓
                        [Progressive Aggregator]
                                ↓
                        [Real-time Updates] → User
                                ↓
                        [LLM Synthesis]
                                ↓
                        [Final Report] → User
```

### Progressive Aggregation

The `ProgressiveAggregator` class maintains state as results arrive:

```python
class AggregatedState:
    results_by_source: Dict[str, List]  # Raw results
    results_by_type: Dict[str, List]    # Categorized results
    sources_completed: List[str]         # Which sources are done
    total_results: int                   # Running count
    start_time: datetime                 # When search started
    last_update: datetime                # Last result received
```

Each new result triggers:
1. **Add to source results**
2. **Reclassify by content type**
3. **Update counters and timestamps**
4. **Generate smart summary**
5. **Format for display**

### Adaptive Formatting

Results are classified using heuristics:

```python
def classify_result(result, source):
    if source == "stackoverflow":
        if has_accepted_answer:
            return ResultType.QUICK_FIX
        return ResultType.DISCUSSION
    
    if source == "github":
        return ResultType.CODE_EXAMPLE
    
    if source == "reddit":
        if "warning" or "issue" in title:
            return ResultType.WARNING
        if "tutorial" in title:
            return ResultType.TUTORIAL
        return ResultType.DISCUSSION
```

## Performance Benefits

### Before (Sequential)
```
Stack Overflow: 0-3s
→ GitHub: 3-6s  
→ Reddit: 6-9s
→ Hacker News: 9-12s
Total: ~12 seconds + synthesis
```

### After (Parallel + Streaming)
```
All sources: 0-3s (concurrent)
First results visible: ~0.8s
All results visible: ~2.1s
Total: ~2-3 seconds + synthesis
```

**~4-5x faster** with real-time feedback!

## Error Handling

Streaming is resilient to failures:

- **Source timeout:** 35s max per source
- **Source failure:** Other sources continue
- **Partial results:** Display what's available
- **Graceful degradation:** Falls back to standard search if streaming unavailable

## Usage Examples

### Example 1: Quick Search with Live Updates

```python
# MCP Tool Call
streaming_community_search(
    language="Python",
    topic="Django ORM N+1 query problem solutions"
)

# Output (progressive):
# → 0.7s: Stack Overflow results appear
# → 1.1s: GitHub examples added
# → 1.6s: Reddit discussions added
# → 2.3s: Hacker News threads added
# → 3.5s: LLM synthesis complete
```

### Example 2: Capability Check First

```python
# Check what's available
capabilities = get_system_capabilities()

# Then search with full knowledge
streaming_community_search(
    language="JavaScript", 
    topic="React Server Components data fetching patterns"
)
```

### Example 3: Custom Source Selection

```python
# Only search technical sources
parallel_multi_source_search(
    query="Rust async/await memory overhead",
    language="Rust",
    sources="stackoverflow,github"  # Skip Reddit/HN
)
```

## Module Files

### `streaming_capabilities.py`
- Capability auto-detection (`detect_all_capabilities`)
- Result classification (`classify_result`, `ResultType`)
- Progressive aggregation (`ProgressiveAggregator`)
- Adaptive formatting (`format_streaming_update`, `format_final_results`)

### `streaming_search.py`
- Parallel execution (`parallel_streaming_search`)
- Streaming wrapper (`search_with_streaming`)
- Synthesis integration (`streaming_search_with_synthesis`)
- Convenience functions (`get_all_search_results_streaming`)

### `community_research_mcp.py` (Enhanced)
- New tool: `get_system_capabilities`
- New tool: `streaming_community_search`  
- New tool: `parallel_multi_source_search`
- Import statements and fallback handling

## Configuration

No additional configuration needed! The system automatically:

1. ✅ Detects available API keys from environment
2. ✅ Enables streaming if modules are present
3. ✅ Falls back to standard search if not
4. ✅ Reports capabilities on demand

## Backward Compatibility

All existing tools continue to work:

- `community_search` - Standard search (no streaming)
- `plan_research` - Research planning
- `comparative_search` - Multi-model comparison
- `validated_research` - With validation

The new streaming tools are **additive** and **optional**.

## Future Enhancements

Potential additions:

- [ ] WebSocket support for true push updates
- [ ] Search result caching across streaming calls
- [ ] Dynamic source prioritization based on success rate
- [ ] Parallel LLM synthesis (multiple models simultaneously)
- [ ] User-defined result classifiers
- [ ] Streaming to file for very large result sets

## Troubleshooting

### "Streaming capabilities not available"

**Cause:** `streaming_capabilities.py` or `streaming_search.py` not found

**Fix:** Ensure both files are in the same directory as `community_research_mcp.py`

### No progress updates appearing

**Cause:** MCP client doesn't support context reporting

**Fix:** Use a client that supports MCP context (Claude Desktop, compatible clients)

### Slow streaming performance

**Cause:** Network latency or API rate limits

**Check:**
- Internet connection speed
- API rate limit status
- Concurrent request limits

## Summary

The streaming search features provide:

✅ **4-5x faster** results via parallel execution  
✅ **Real-time feedback** as sources complete  
✅ **Smart organization** by content type  
✅ **Auto-detection** of capabilities  
✅ **Progressive updates** for better UX  
✅ **Backward compatible** with existing tools  

Try `streaming_community_search` for your next query! 🚀
