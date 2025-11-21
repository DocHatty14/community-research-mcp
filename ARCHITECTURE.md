# Architecture

## Module Structure

```
community-research-mcp/
├── models/                    # Data models & validation
│   ├── config.py             # ThinkingMode, ResponseFormat enums
│   ├── search.py             # CommunitySearchInput, DeepAnalyzeInput  
│   └── __init__.py
│
├── utils/                     # Stateless utilities
│   ├── cache.py              # Disk-persisted caching (get/set/clear)
│   ├── rate_limit.py         # Sliding window rate limiter
│   ├── helpers.py            # Query normalization, result counting
│   └── __init__.py
│
├── core/                      # Business logic
│   ├── llm_clients.py        # Gemini, OpenAI, Anthropic, etc.
│   ├── orchestrator.py       # ModelOrchestrator - intelligent provider selection
│   └── __init__.py
│
├── api/                       # External API wrappers
│   ├── stackoverflow.py      # Stack Exchange API + language tags
│   ├── github.py             # GitHub Issues search
│   ├── hackernews.py         # Hacker News (Algolia)
│   └── __init__.py
│
├── enhanced_mcp_utilities.py # Circuit breakers, quality scoring
├── streaming_capabilities.py  # Streaming support
├── streaming_search.py        # Streaming aggregation
└── community_research_mcp.py  # MCP server (tools, synthesis, aggregation)
```

## Design Principles

**1. Separation of Concerns**
- `models/` - Pure data validation (zero business logic)
- `utils/` - Stateless functions (easily testable)
- `core/` - Business logic (LLM integration, orchestration)
- `api/` - External service wrappers (isolated failures)

**2. Minimal Dependencies**
- Each module imports only what it needs
- `models/` and `utils/` have no cross-dependencies
- `core/` depends on `models/`
- `api/` has no dependencies on other modules

**3. Clear Interfaces**
- All `__init__.py` files document exports
- Module docstrings explain purpose
- Function signatures are explicit

## Import Examples

```python
# Configuration and models
from models import ThinkingMode, ResponseFormat, CommunitySearchInput

# Utilities
from utils import get_cached_result, check_rate_limit, clear_cache

# Core logic
from core import call_gemini, ModelOrchestrator, get_available_llm_provider

# API integrations
from api import search_stackoverflow, search_github, search_hackernews
```

## Module Stats

| Module | Files | Lines | Purpose |
|--------|-------|-------|---------|
| models | 3 | 152 | Data validation |
| utils | 4 | 151 | Helpers & caching |
| core | 3 | 306 | LLM integration |
| api | 4 | 200 | External APIs |
| **Total** | **14** | **809** | **Extracted modules** |

**Main file:** 4,377 lines (MCP tools + synthesis + aggregation)

## Benefits

✅ **Testability** - Each module can be tested independently  
✅ **Reusability** - Import `api.stackoverflow` in other projects  
✅ **Maintainability** - Find functions in seconds, not minutes  
✅ **Clarity** - New contributors understand structure immediately  
✅ **Reliability** - Isolated failures don't cascade
