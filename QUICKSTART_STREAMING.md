# Quick Start: Streaming Search

## 5-Minute Setup

### 1. Verify Files

Ensure these files are present:
```
community-research-mcp/
├── community_research_mcp.py
├── streaming_capabilities.py    ← NEW
├── streaming_search.py          ← NEW
└── .env
```

### 2. Check Capabilities

First, see what's available on your system:

```python
# MCP Tool Call
get_system_capabilities()
```

**Output:**
```markdown
# 🔍 System Capabilities

## Search APIs
**Active (5):**
  ✓ stackoverflow
  ✓ github
  ✓ reddit
  ✓ hackernews
  ✓ duckduckgo

## LLM Providers
**Active (1):**
  ✓ gemini

**Total Active Capabilities:** 6
```

### 3. Run Your First Streaming Search

```python
# MCP Tool Call
streaming_community_search(
    language="Python",
    topic="FastAPI async database queries with SQLAlchemy"
)
```

**What Happens:**

```
🚀 Starting parallel search across 4 sources...
✓ stackoverflow: 5 results
✓ github: 5 results  
✓ reddit: 7 results
✓ hackernews: 3 results
✨ Search complete! 20 total results
🤖 Synthesizing results with LLM...
```

**You Get:**

1. **Progressive Updates** - See results as they arrive
2. **Organized by Type** - Quick fixes, code examples, discussions
3. **LLM Synthesis** - Smart summary of all findings
4. **~2-3 seconds** - Instead of 12+ seconds sequential

## Common Use Cases

### Use Case 1: Quick Problem Solving

**Scenario:** Need to fix something NOW

```python
streaming_community_search(
    language="JavaScript",
    topic="React useEffect infinite loop causes and fixes"
)
```

**Benefits:**
- Get Stack Overflow accepted answers first (~0.8s)
- See GitHub code examples next (~1.2s)
- Review warnings from Reddit (~1.5s)
- Complete synthesis in ~3s total

### Use Case 2: Learning Best Practices

**Scenario:** Want to learn the right way to do something

```python
streaming_community_search(
    language="Go",
    topic="Go error handling idiomatic patterns",
    goal="Write production-ready error handling code"
)
```

**Benefits:**
- See official patterns (quick fixes)
- Review real implementations (GitHub)
- Learn gotchas (Reddit warnings)
- Compare approaches (synthesis)

### Use Case 3: Technology Evaluation

**Scenario:** Deciding between solutions

```python
streaming_community_search(
    language="Python",
    topic="Django vs FastAPI for REST API with complex business logic",
    goal="Choose framework for new microservice"
)
```

**Benefits:**
- See performance discussions (HackerNews)
- Review production code (GitHub)
- Learn pain points (Reddit)
- Get balanced recommendation (synthesis)

### Use Case 4: Debugging

**Scenario:** Stuck on cryptic error

```python
streaming_community_search(
    language="Rust",
    topic="Rust borrow checker error cannot move out of borrowed content"
)
```

**Benefits:**
- Find exact error on Stack Overflow
- See working code fixes
- Learn underlying concepts
- Avoid common mistakes

## Advanced Usage

### Custom Source Selection

Only search specific sources:

```python
parallel_multi_source_search(
    query="TypeScript generic constraints",
    language="TypeScript",
    sources="stackoverflow,github"  # Skip Reddit/HN
)
```

**When to Use:**
- You only trust certain sources
- Faster results (fewer sources)
- Avoid discussion noise
- Focus on code examples

### Check Before Searching

Verify capabilities first:

```python
# 1. Check what's available
capabilities = get_system_capabilities()

# 2. Configure based on results
if "gemini" in capabilities:
    use_advanced_synthesis = True

# 3. Search
streaming_community_search(language="Python", topic="...")
```

## Comparison: Standard vs Streaming

### Standard Search (`community_search`)

```python
community_search(
    language="Python",
    topic="Django ORM optimization"
)
```

- ⏱️ **Time:** 12-15 seconds
- 📊 **Updates:** None until complete
- 🔀 **Execution:** Sequential
- 📦 **Output:** All at once
- ✅ **Best for:** Simple queries, caching

### Streaming Search (`streaming_community_search`)

```python
streaming_community_search(
    language="Python",
    topic="Django ORM optimization"
)
```

- ⏱️ **Time:** 2-3 seconds
- 📊 **Updates:** Real-time progress
- 🔀 **Execution:** Parallel
- 📦 **Output:** Progressive
- ✅ **Best for:** Interactive use, urgent queries

## Tips & Tricks

### 1. Use Specific Topics

❌ **Vague:**
```python
topic="performance"  # Too vague, will be rejected
```

✅ **Specific:**
```python
topic="React performance optimization for large lists with virtualization"
```

### 2. Include Context

```python
streaming_community_search(
    language="Python",
    topic="FastAPI rate limiting per user",
    current_setup="Using Redis, 10k req/day",
    goal="Implement fair rate limiting with burst allowance"
)
```

More context = Better synthesis!

### 3. Check Capabilities First

```python
# Morning routine:
caps = get_system_capabilities()

# Know what you're working with
# Adjust workflow based on available LLMs
```

### 4. Combine with Other Tools

```python
# 1. Plan research
plan = plan_research(
    query="Microservices architecture",
    language="Go"
)

# 2. Execute with streaming
results = streaming_community_search(
    language="Go",
    topic="Go microservices service discovery patterns"
)

# 3. Validate if critical
validated = validated_research(
    language="Go",
    topic="Go microservices security best practices"
)
```

## Troubleshooting

### Issue: "Streaming capabilities not available"

**Solution:** Copy `streaming_capabilities.py` and `streaming_search.py` to your directory.

### Issue: No progress updates

**Check:**
- Using MCP-compatible client (Claude Desktop)
- Context parameter is being passed
- Client supports progress reporting

**Workaround:** Results still work, just no real-time updates.

### Issue: Slow performance

**Check:**
- Internet connection speed
- API rate limits (check with `get_system_capabilities`)
- Time of day (Stack Overflow API can be slower during peak)

**Solution:** Use `sources` parameter to search fewer sources.

### Issue: Empty results

**Causes:**
- Topic too specific/niche
- Spelling errors in query
- Very new technology (no community content yet)

**Solution:** 
- Broaden the topic slightly
- Check spelling
- Try `duckduckgo_search` for newer topics

## Real-World Examples

### Example 1: Production Bug

```python
# 2 AM, production is down
streaming_community_search(
    language="PostgreSQL",
    topic="PostgreSQL connection pool exhausted high load",
    current_setup="Django, pgbouncer, 100 connections",
    goal="Fix production connection pool issue NOW"
)

# Results in 2-3 seconds:
# - Stack Overflow: Pool configuration fixes
# - GitHub: Production pool configs
# - Reddit: "We had this, here's what worked"
# - Synthesis: "Increase pool size + add connection timeout"
```

### Example 2: Code Review

```python
# Reviewing PR, suspicious code
streaming_community_search(
    language="JavaScript",
    topic="JavaScript promise constructor anti-pattern new Promise executor",
    goal="Understand if this pattern is problematic"
)

# Quick answers:
# - Stack Overflow: "Don't use new Promise() unnecessarily"
# - GitHub: Examples of better patterns
# - Warnings: "Can cause unhandled rejections"
```

### Example 3: Learning New Framework

```python
# Day 1 with FastAPI
streaming_community_search(
    language="Python",
    topic="FastAPI project structure best practices for large applications",
    goal="Set up project structure correctly from start"
)

# Comprehensive results:
# - Stack Overflow: Structure Q&As
# - GitHub: Real production apps
# - Reddit: Lessons learned
# - Synthesis: Recommended structure
```

## Next Steps

1. ✅ **Try it:** Run `get_system_capabilities()` now
2. ✅ **Search:** Test `streaming_community_search()` with your topic
3. ✅ **Compare:** Time standard vs streaming search
4. ✅ **Explore:** Try `parallel_multi_source_search()` with custom sources
5. ✅ **Read:** Check out `STREAMING_FEATURES.md` for architecture details

## Questions?

### "Which tool should I use?"

- **Quick answers:** `streaming_community_search`
- **Complex research:** `plan_research` → `streaming_community_search`
- **Critical decisions:** `validated_research`
- **Custom sources:** `parallel_multi_source_search`
- **Standard search:** `community_search` (still works!)

### "Is streaming always faster?"

**Yes** for total time (parallel execution)
**Even better** for perceived time (progressive updates)

Only exception: Cached results (standard search returns instantly from cache)

### "Can I use this in production?"

**Yes!** The implementation includes:
- Error handling and timeouts
- Rate limiting
- Graceful degradation
- Fallback to standard search
- Character limits

### "What if I don't have an LLM API key?"

Streaming still works! You'll get:
- ✅ All search results
- ✅ Organized by type
- ✅ Real-time updates
- ❌ No LLM synthesis

Set `GEMINI_API_KEY`, `OPENAI_API_KEY`, or `ANTHROPIC_API_KEY` for synthesis.

---

**Happy Streaming!** 🚀
