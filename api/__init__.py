"""
External API integrations for community search sources.

This module provides async functions for searching Stack Overflow,
GitHub Issues, Hacker News, and other developer communities.
"""

from api.discourse import search_discourse
from api.firecrawl import search_firecrawl
from api.github import search_github
from api.hackernews import search_hackernews
from api.lobsters import search_lobsters
from api.stackoverflow import search_stackoverflow
from api.tavily import search_tavily

__all__ = [
    "search_stackoverflow",
    "search_github",
    "search_hackernews",
    "search_firecrawl",
    "search_tavily",
    "search_lobsters",
    "search_discourse",
]
