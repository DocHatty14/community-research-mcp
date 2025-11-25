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

from api.stackexchange import (
    search as search_stackexchange,
    search_multi as search_stackexchange_multi,
    search_stackoverflow,
    SITES as STACKEXCHANGE_SITES,
)

from api.github import (
    search as search_github_issues,
    search_github,
)

from api.hackernews import (
    search as search_hackernews_posts,
    search_hackernews,
)

from api.lobsters import (
    search as search_lobsters_posts,
    search_lobsters,
)

from api.discourse import (
    search as search_discourse_forums,
    search_discourse,
    FORUMS as DISCOURSE_FORUMS,
)

# ══════════════════════════════════════════════════════════════════════════════
# Web Search APIs (Require Keys)
# ══════════════════════════════════════════════════════════════════════════════

from api.serper import (
    search as search_serper_web,
    search_news as search_serper_news,
    get_related as get_serper_related,
    search_serper,
    get_serper_related_searches,
)

from api.tavily import (
    search as search_tavily_web,
    search_news as search_tavily_news,
    extract as extract_tavily_content,
    search_tavily,
    extract_tavily,
)

from api.brave import (
    search as search_brave_web,
    search_news as search_brave_news,
    search_brave,
)

from api.firecrawl import (
    search as search_firecrawl_web,
    scrape as scrape_firecrawl_page,
    map_site as map_firecrawl_site,
    search_firecrawl,
    scrape_firecrawl,
    map_firecrawl,
)

# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Stack Exchange
    "search_stackexchange",
    "search_stackexchange_multi",
    "search_stackoverflow",
    "STACKEXCHANGE_SITES",
    # GitHub
    "search_github_issues",
    "search_github",
    # Hacker News
    "search_hackernews_posts",
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
