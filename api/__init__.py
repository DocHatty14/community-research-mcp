"""
Community Research API Integrations.

This module provides async search functions for developer communities,
Q&A sites, and web search APIs. All functions follow a consistent pattern:

    async def search(query: str, language: str = None, **options) -> list[dict]

Each result dict contains at minimum: title, url, snippet, source

Available Sources:
─────────────────────────────────────────────────────────────────────────────
COMMUNITY SOURCES (Free, no API key required)
    stackexchange    Stack Overflow + 19 Stack Exchange sites
    github           GitHub Issues & Discussions
    hackernews       Hacker News (via Algolia)
    lobsters         Lobsters.rs technical community
    discourse        Language-specific Discourse forums

WEB SEARCH APIs (Require API keys)
    serper           Google Search via Serper.dev
    tavily           AI-optimized web search
    brave            Privacy-focused Brave Search
    firecrawl        Web scraping + search

Configuration:
─────────────────────────────────────────────────────────────────────────────
Set API keys in environment variables or .env file:

    SERPER_API_KEY        https://serper.dev
    TAVILY_API_KEY        https://tavily.com
    BRAVE_SEARCH_API_KEY  https://brave.com/search/api
    FIRECRAWL_API_KEY     https://firecrawl.dev
    GITHUB_TOKEN          https://github.com/settings/tokens (optional)
    STACKEXCHANGE_API_KEY https://stackapps.com (optional, higher limits)
"""

# ══════════════════════════════════════════════════════════════════════════════
# Environment Setup
# ══════════════════════════════════════════════════════════════════════════════

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # Use system environment variables

# ══════════════════════════════════════════════════════════════════════════════
# Community Sources (Free)
# ══════════════════════════════════════════════════════════════════════════════

from api.brave import (
    search as search_brave_web,
)
from api.brave import (
    search_brave,
)
from api.brave import (
    search_news as search_brave_news,
)
from api.discourse import (
    FORUMS as DISCOURSE_FORUMS,
)
from api.discourse import (
    search as search_discourse_forums,
)
from api.discourse import (
    search_discourse,
)
from api.firecrawl import (
    map_firecrawl,
    scrape_firecrawl,
    search_firecrawl,
)
from api.firecrawl import (
    map_site as map_firecrawl_site,
)
from api.firecrawl import (
    scrape as scrape_firecrawl_page,
)
from api.firecrawl import (
    search as search_firecrawl_web,
)
from api.github import (
    search as search_github_issues,
)
from api.github import (
    search_github,
)
from api.hackernews import (
    search as search_hackernews_posts,
)
from api.hackernews import (
    search_hackernews,
)
from api.lobsters import (
    search as search_lobsters_posts,
)
from api.lobsters import (
    search_lobsters,
)
from api.serper import (
    get_related as get_serper_related,
)
from api.serper import (
    get_serper_related_searches,
    search_serper,
)

# ══════════════════════════════════════════════════════════════════════════════
# Web Search APIs (Require Keys)
# ══════════════════════════════════════════════════════════════════════════════
from api.serper import (
    search as search_serper_web,
)
from api.serper import (
    search_news as search_serper_news,
)
from api.stackexchange import (
    SITES as STACKEXCHANGE_SITES,
)
from api.stackexchange import (
    enrich_with_answers as enrich_stackexchange_answers,
)
from api.stackexchange import (
    fetch_accepted_answer,
    search_stackoverflow,
)
from api.stackexchange import (
    search as search_stackexchange,
)
from api.stackexchange import (
    search_multi as search_stackexchange_multi,
)
from api.tavily import (
    extract as extract_tavily_content,
)
from api.tavily import (
    extract_tavily,
    search_tavily,
)
from api.tavily import (
    search as search_tavily_web,
)
from api.tavily import (
    search_news as search_tavily_news,
)

# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Stack Exchange
    "search_stackexchange",
    "search_stackexchange_multi",
    # Lobsters
    "search_lobsters_posts",
    "search_lobsters",
    # Discourse
    "search_discourse_forums",
    "search_discourse",
    "DISCOURSE_FORUMS",
    # Serper (Google)
    "search_serper_web",
    "search_serper_news",
    "get_serper_related",
    "search_serper",
    "get_serper_related_searches",
    # Tavily
    "search_tavily_web",
    "search_tavily_news",
    "extract_tavily_content",
    "search_tavily",
    "extract_tavily",
    # Brave
    "search_brave_web",
    "search_brave_news",
    "search_brave",
    "search_brave_news",
    "search_brave",
    "search_hackernews",

    # Lobsters
    "search_lobsters_posts",
    "search_lobsters",

    # Discourse
    "search_discourse_forums",
    "search_discourse",
    "DISCOURSE_FORUMS",

    # Serper (Google)
    "search_serper_web",
    "search_serper_news",
    "get_serper_related",
    "search_serper",
    "get_serper_related_searches",

    # Tavily
    "search_tavily_web",
    "search_tavily_news",
    "extract_tavily_content",
    "search_tavily",
    "extract_tavily",

    # Brave
    "search_brave_web",
    "search_brave_news",
    "search_brave",

    # Firecrawl
    "search_firecrawl_web",
    "scrape_firecrawl_page",
    "map_firecrawl_site",
    "search_firecrawl",
    "scrape_firecrawl",
    "map_firecrawl",
]
