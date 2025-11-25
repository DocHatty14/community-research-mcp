"""
Serper API (Google Search).

Real-time Google Search results including web, news, images,
knowledge graph, and "People Also Ask" questions.

API: https://serper.dev/
Rate Limits: 2,500/month (free tier)
"""

import logging
import os
from typing import Any, Optional

import httpx

__all__ = ["search", "search_news", "get_related"]

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

API_BASE = "https://google.serper.dev"
API_TIMEOUT = 30.0
API_KEY = os.getenv("SERPER_API_KEY")

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# Search Functions
# ══════════════════════════════════════════════════════════════════════════════


async def search(
    query: str,
    language: Optional[str] = None,
    *,
    max_results: int = 10,
    country: str = "us",
    locale: str = "en",
) -> list[dict[str, Any]]:
    """
    Search Google via Serper API.

    Args:
        query: Search query string
        language: Programming language context (prepended to query)
        max_results: Maximum results to return
        country: Country code (e.g., 'us', 'uk')
        locale: Language code (e.g., 'en', 'es')

    Returns:
        List of results with title, url, snippet, and metadata

    Example:
        >>> results = await search("async patterns", language="python")
    """
    if not API_KEY:
        logger.debug("Skipped: SERPER_API_KEY not set")
        return []

    full_query = f"{language} {query}".strip() if language else query

    try:
        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            response = await client.post(
                f"{API_BASE}/search",
                headers={"X-API-KEY": API_KEY, "Content-Type": "application/json"},
                json={
                    "q": full_query,
                    "num": min(max_results, 100),
                    "gl": country,
                    "hl": locale,
                },
            )
            response.raise_for_status()
            data = response.json()

        results = []

        # Knowledge graph
        if kg := data.get("knowledgeGraph"):
            results.append(
                {
                    "title": kg.get("title", "Knowledge Graph"),
                    "url": kg.get("website", ""),
                    "snippet": kg.get("description", ""),
                    "type": "knowledge_graph",
                    "source": "serper",
                }
            )

        # Organic results
        for item in data.get("organic", []):
            if url := item.get("link"):
                results.append(
                    {
                        "title": item.get("title", ""),
                        "url": url,
                        "snippet": item.get("snippet", ""),
                        "position": item.get("position", 0),
                        "source": "serper",
                    }
                )

        # People Also Ask
        for item in data.get("peopleAlsoAsk", []):
            if url := item.get("link"):
                results.append(
                    {
                        "title": item.get("question", ""),
                        "url": url,
                        "snippet": item.get("snippet", ""),
                        "type": "question",
                        "source": "serper",
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
) -> list[dict[str, Any]]:
    """
    Search Google News via Serper.

    Args:
        query: Search query string
        language: Programming language context
        max_results: Maximum results to return

    Returns:
        List of news articles with title, url, date, source
    """
    if not API_KEY:
        return []

    full_query = f"{language} {query}".strip() if language else query

    try:
        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            response = await client.post(
                f"{API_BASE}/news",
                headers={"X-API-KEY": API_KEY, "Content-Type": "application/json"},
                json={"q": full_query, "num": min(max_results, 100)},
            )
            response.raise_for_status()
            data = response.json()

        return [
            {
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "date": item.get("date", ""),
                "publisher": item.get("source", ""),
                "source": "serper:news",
            }
            for item in data.get("news", [])
            if item.get("link")
        ]

    except Exception as e:
        logger.warning(f"News search failed: {e}")
        return []


async def get_related(query: str) -> list[str]:
    """
    Get related search suggestions.

    Args:
        query: Original search query

    Returns:
        List of related query strings
    """
    if not API_KEY:
        return []

    try:
        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            response = await client.post(
                f"{API_BASE}/search",
                headers={"X-API-KEY": API_KEY, "Content-Type": "application/json"},
                json={"q": query},
            )
            response.raise_for_status()
            data = response.json()

        return [
            item.get("query", "")
            for item in data.get("relatedSearches", [])
            if item.get("query")
        ]

    except Exception as e:
        logger.warning(f"Related searches failed: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# Backward Compatibility
# ══════════════════════════════════════════════════════════════════════════════

search_serper = search
search_serper_news = search_news
get_serper_related_searches = get_related
