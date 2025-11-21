# Documentation

## Setup

```bash
git clone https://github.com/DocHatty/community-research-mcp.git
cd community-research-mcp
initialize.bat
```

Create `.env` file:
```env
# Need at least one of these
GEMINI_API_KEY=your_key
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key

# Optional for better Reddit results
REDDIT_CLIENT_ID=your_id
REDDIT_CLIENT_SECRET=your_secret
```

## Available Tools

### `community_search(language, topic, goal=None, current_setup=None)`
Basic parallel search across Stack Overflow, GitHub, Reddit, HN, DuckDuckGo.

### `streaming_community_search(...)`
Same as above but streams results as they arrive.

### `deep_community_search(...)`
Does multiple iterations to fill knowledge gaps.

### `validated_research(...)`
Uses a second LLM to double-check the results.

### `plan_research(query, thinking_mode="balanced")`
Plans out research without executing. Modes: `fast`, `balanced`, `deep`.

### `get_system_capabilities()`
Shows what's working (which APIs/LLMs are configured).

### `fetch_page_content(url, max_chars=12000)`
Grabs content from a URL.

### `get_performance_metrics()`
Stats (requires `enhanced_mcp_utilities.py`).

## How It Works

1. Query hits the server
2. Searches run in parallel (asyncio)
3. Results come back and get aggregated
4. LLM synthesizes everything into markdown
5. Gets cached for 24 hours in `.community_research_cache.json`

## Files

- `community_research_mcp.py` - Main server
- `streaming_capabilities.py` - Result classification
- `streaming_search.py` - Parallel search orchestration
- `enhanced_mcp_utilities.py` - Retry logic, quality scoring

## Notes

- Uses Gemini by default, falls back to OpenAI/Anthropic
- Results get classified (quick fix, code example, discussion, etc.)
- Has retry logic with exponential backoff
- Free APIs for search (Stack Overflow, GitHub, Reddit, HN)
- LLM costs ~$0.001-0.03 per search depending on provider
