"""
External API integrations for community search sources.

This module provides async functions for searching Stack Overflow,
GitHub Issues, Hacker News, and other developer communities.
"""

from api.brave import search_brave, search_brave_news, search_brave_web
from api.discourse import search_discourse
from api.firecrawl import (
    map_firecrawl,
    scrape_firecrawl,
    scrape_multiple_firecrawl,
    search_firecrawl,
)
from api.github import search_github
from api.hackernews import search_hackernews
from api.lobsters import search_lobsters
from api.serper import (
    get_serper_related_searches,
    search_serper,
    search_serper_images,
    search_serper_news,
)
from api.stackoverflow import search_stackoverflow
from api.tavily import (
    extract_tavily,
    search_tavily,
    search_tavily_news,
    search_tavily_with_extract,
)

__all__ = [
    # Community sources
    "search_stackoverflow",
    "search_github",
    "search_hackernews",
    "search_lobsters",
    "search_discourse",
    # Tavily
    "search_tavily",
    "search_tavily_news",
    "extract_tavily",
    "search_tavily_with_extract",
    # Brave Search
    "search_brave",
    "search_brave_web",
    "search_brave_news",
    # Serper (Google Search)
    "search_serper",
    "search_serper_news",
    "search_serper_images",
    "get_serper_related_searches",
    # Firecrawl
    "search_firecrawl",
    "scrape_firecrawl",
    "scrape_multiple_firecrawl",
    "map_firecrawl",
]
