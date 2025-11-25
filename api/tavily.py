"""
Tavily Search API.

AI-optimized web search with optional content extraction
and answer generation capabilities.

API: https://tavily.com/
Rate Limits: 1,000/month (free tier)
"""

import logging
import os
from typing import Any, Optional

import httpx

__all__ = ["search", "search_news", "extract"]

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

API_BASE = "https://api.tavily.com"
API_TIMEOUT = 10.0  # Reduced from 30s to fail fast
EXTRACT_TIMEOUT = 30.0
API_KEY = os.getenv("TAVILY_API_KEY")

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# Search Functions
# ══════════════════════════════════════════════════════════════════════════════


async def search(
    query: str,
    language: Optional[str] = None,
    *,
    max_results: int = 10,
    search_depth: str = "basic",
    include_answer: bool = False,
    include_domains: Optional[list[str]] = None,
    exclude_domains: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """
    Search the web via Tavily API.

    Args:
        query: Search query string
        language: Programming language context (prepended to query)
        max_results: Maximum results to return
        search_depth: 'basic' (fast) or 'advanced' (thorough)
        include_answer: Include AI-generated answer summary
        include_domains: Only search these domains
        exclude_domains: Exclude these domains

    Returns:
        List of results with title, url, snippet, score

    Example:
        >>> results = await search("FastAPI authentication", language="python")
    """
    if not API_KEY:
        logger.debug("Skipped: TAVILY_API_KEY not set")
        return []

    full_query = f"{language} {query}".strip() if language else query

    payload: dict[str, Any] = {
        "api_key": API_KEY,
        "query": full_query,
        "max_results": max_results,
        "search_depth": search_depth,
        "include_answer": include_answer,
    }

    if include_domains:
        payload["include_domains"] = include_domains
    if exclude_domains:
        payload["exclude_domains"] = exclude_domains

    try:
        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            response = await client.post(f"{API_BASE}/search", json=payload)
            response.raise_for_status()
            data = response.json()

        results = []

        # AI answer (if requested)
        if include_answer and (answer := data.get("answer")):
            results.append(
                {
                    "title": "AI Summary",
                    "url": "",
                    "snippet": answer,
                    "type": "answer",
                    "source": "tavily",
                }
            )

        # Search results
        for item in data.get("results", []):
            if url := item.get("url"):
                results.append(
                    {
                        "title": item.get("title") or url,
                        "url": url,
                        "snippet": item.get("content", ""),
                        "score": item.get("score", 0),
                        "source": "tavily",
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
    days: int = 7,
) -> list[dict[str, Any]]:
    """
    Search for recent news articles.

    Args:
        query: Search query string
        language: Programming language context
        max_results: Maximum results to return
        days: Limit to last N days

    Returns:
        List of news results
    """
    if not API_KEY:
        return []

    full_query = f"{language} {query}".strip() if language else query

    try:
        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            response = await client.post(
                f"{API_BASE}/search",
                json={
                    "api_key": API_KEY,
                    "query": full_query,
                    "max_results": max_results,
                    "topic": "news",
                    "days": days,
                },
            )
            response.raise_for_status()
            data = response.json()

        return [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", ""),
                "score": item.get("score", 0),
                "source": "tavily:news",
            }
            for item in data.get("results", [])
            if item.get("url")
        ]

    except Exception as e:
        logger.warning(f"News search failed: {e}")
        return []


async def extract(urls: list[str]) -> list[dict[str, Any]]:
    """
    Extract content from web pages.

    Args:
        urls: List of URLs to extract (max 20)

    Returns:
        List of extracted content with url, content, success status
    """
    if not API_KEY:
        return []

    if not urls:
        return []

    urls = urls[:20]  # API limit

    try:
        async with httpx.AsyncClient(timeout=EXTRACT_TIMEOUT) as client:
            response = await client.post(
                f"{API_BASE}/extract",
                json={"api_key": API_KEY, "urls": urls},
            )
            response.raise_for_status()
            data = response.json()

        results = []

        for item in data.get("results", []):
            results.append(
                {
                    "url": item.get("url", ""),
                    "content": item.get("raw_content") or item.get("content", ""),
                    "success": True,
                    "source": "tavily:extract",
                }
            )

        for url in data.get("failed_results", []):
            results.append(
                {
                    "url": url,
                    "content": "",
                    "success": False,
                    "source": "tavily:extract",
                }
            )

        return results

    except Exception as e:
        logger.warning(f"Extract failed: {e}")
        return [{"url": url, "content": "", "success": False} for url in urls]


# ══════════════════════════════════════════════════════════════════════════════
# Backward Compatibility
# ══════════════════════════════════════════════════════════════════════════════

search_tavily = search
search_tavily_news = search_news
extract_tavily = extract
