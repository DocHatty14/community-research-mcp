"""
Brave Search API.

Privacy-focused web search with news, filtering,
freshness controls, and extra snippet extraction.

API: https://brave.com/search/api/
Rate Limits: 2,000/month (free tier)
"""

import logging
import os
from typing import Any, Optional

import httpx

__all__ = ["search", "search_news"]

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

API_BASE = "https://api.search.brave.com/res/v1"
API_TIMEOUT = 30.0
API_KEY = os.getenv("BRAVE_SEARCH_API_KEY")

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# Search Functions
# ══════════════════════════════════════════════════════════════════════════════


async def search(
    query: str,
    language: Optional[str] = None,
    *,
    max_results: int = 10,
    freshness: Optional[str] = None,
    country: str = "us",
) -> list[dict[str, Any]]:
    """
    Search the web via Brave Search API.

    Args:
        query: Search query string (max 400 chars)
        language: Programming language context (prepended to query)
        max_results: Maximum results (1-20)
        freshness: Filter by age - 'pd' (24h), 'pw' (week), 'pm' (month), 'py' (year)
        country: Country code for results

    Returns:
        List of results with title, url, snippet

    Example:
        >>> results = await search("websocket tutorial", language="python")
    """
    if not API_KEY:
        logger.debug("Skipped: BRAVE_SEARCH_API_KEY not set")
        return []

    full_query = f"{language} {query}".strip() if language else query
    full_query = full_query[:400]  # API limit

    params: dict[str, Any] = {
        "q": full_query,
        "count": min(max(max_results, 1), 20),
        "country": country,
        "extra_snippets": "true",
    }

    if freshness:
        params["freshness"] = freshness

    try:
        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            response = await client.get(
                f"{API_BASE}/web/search",
                headers={
                    "Accept": "application/json",
                    "X-Subscription-Token": API_KEY,
                },
                params=params,
            )
            response.raise_for_status()
            data = response.json()

        results = []

        # Web results
        for item in data.get("web", {}).get("results", []):
            if url := item.get("url"):
                # Combine main snippet with extras
                snippets = [item.get("description", "")]
                snippets.extend(item.get("extra_snippets", []))

                results.append(
                    {
                        "title": item.get("title", ""),
                        "url": url,
                        "snippet": " ".join(filter(None, snippets)),
                        "age": item.get("age", ""),
                        "source": "brave",
                    }
                )

        # Include news if present
        for item in data.get("news", {}).get("results", []):
            if url := item.get("url"):
                results.append(
                    {
                        "title": item.get("title", ""),
                        "url": url,
                        "snippet": item.get("description", ""),
                        "age": item.get("age", ""),
                        "type": "news",
                        "source": "brave",
                    }
                )

        return results

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            logger.error("Invalid API key")
        elif e.response.status_code == 429:
            logger.warning("Rate limit exceeded")
        else:
            logger.warning(f"HTTP {e.response.status_code}")
        return []
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []


async def search_news(
    query: str,
    language: Optional[str] = None,
    *,
    max_results: int = 10,
    freshness: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    Search Brave News.

    Args:
        query: Search query string
        language: Programming language context
        max_results: Maximum results (1-20)
        freshness: Filter by age - 'pd', 'pw', 'pm'

    Returns:
        List of news articles
    """
    if not API_KEY:
        return []

    full_query = f"{language} {query}".strip() if language else query

    params: dict[str, Any] = {
        "q": full_query[:400],
        "count": min(max(max_results, 1), 20),
    }

    if freshness:
        params["freshness"] = freshness

    try:
        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            response = await client.get(
                f"{API_BASE}/news/search",
                headers={
                    "Accept": "application/json",
                    "X-Subscription-Token": API_KEY,
                },
                params=params,
            )
            response.raise_for_status()
            data = response.json()

        return [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("description", ""),
                "age": item.get("age", ""),
                "publisher": item.get("meta_url", {}).get("hostname", ""),
                "source": "brave:news",
            }
            for item in data.get("results", [])
            if item.get("url")
        ]

    except Exception as e:
        logger.warning(f"News search failed: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# Backward Compatibility
# ══════════════════════════════════════════════════════════════════════════════

search_brave = search
search_brave_web = search
search_brave_news = search_news
