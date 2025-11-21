# Architecture

## Structure

```
community-research-mcp/
├── models/           # Data structures & validation
│   ├── config.py     # ThinkingMode, ResponseFormat enums
│   ├── search.py     # CommunitySearchInput, DeepAnalyzeInput
│   └── __init__.py
├── utils/            # Pure utility functions
│   ├── cache.py      # Caching (get/set/clear)
│   ├── rate_limit.py # Rate limiting
│   ├── helpers.py    # Helper functions
│   └── __init__.py
├── core/             # Business logic
│   ├── llm_clients.py      # LLM API calls (Gemini, OpenAI, etc.)
│   ├── orchestrator.py     # ModelOrchestrator class
│   └── __init__.py
├── api/              # External API integrations
│   ├── stackoverflow.py    # Stack Overflow search
│   ├── github.py           # GitHub Issues search
│   ├── hackernews.py       # Hacker News search
│   └── __init__.py
├── enhanced_mcp_utilities.py  # Circuit breakers, quality scoring
├── streaming_capabilities.py   # Streaming support
├── streaming_search.py         # Streaming search logic
└── community_research_mcp.py   # Main MCP server (tools, synthesis, aggregation)
```

## Key Components

**models/** - Zero dependencies, pure validation  
**utils/** - Stateless functions, fully testable  
**core/** - LLM integration and orchestration  
**api/** - External API wrappers  

## Import Pattern

```python
from models import CommunitySearchInput, ThinkingMode
from utils import get_cached_result, check_rate_limit
from core import call_gemini, ModelOrchestrator
from api import search_stackoverflow, search_github
```

## File Sizes

- models: ~150 lines total
- utils: ~150 lines total
- core: ~300 lines total  
- api: ~200 lines total
- Main file: ~3,600 lines (MCP tools + synthesis + aggregation)

**Total extracted: ~800 lines into organized modules**
