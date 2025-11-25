"""Tavily API integration.

Implements Tavily endpoints:
- Search: Real-time web search with filtering and context
- Extract: Intelligent content extraction from web pages
"""

import logging
import os
from typing import Any, Dict, List, Optional

import httpx

DEFAULT_TIMEOUT = 30.0
BASE_URL = "https://api.tavily.com"


def _get_api_key() -> Optional[str]:
    """Get Tavily API key from environment."""
    return os.getenv("TAVILY_API_KEY")


async def search_tavily(
    query: str,
    language: Optional[str] = None,
    max_results: int = 10,
    search_depth: str = "basic",
    topic: str = "general",
    days: Optional[int] = None,
    include_answer: bool = False,
    include_raw_content: bool = False,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
    community_focus: bool = True,
) -> List[Dict[str, Any]]:
    """Search Tavily for relevant web results.

    Optimized to find community solutions, workarounds, and real-world fixes.

    Args:
        query: Search query
        language: Optional programming language context (prepended to query)
        max_results: Maximum results to return (default 10)
        search_depth: "basic" (fast) or "advanced" (thorough)
        topic: "general" or "news" for news-focused results
        days: Limit results to last N days (news topic recommended)
        include_answer: Include AI-generated answer summary
        include_raw_content: Include raw HTML content
        include_domains: Only search these domains
        exclude_domains: Exclude these domains from results
        community_focus: If True, enrich query to target community solutions

    Returns:
        List of normalized search results with title, url, snippet, and content.
    """
    api_key = _get_api_key()

    if not api_key:
        logging.debug("Tavily skipped: TAVILY_API_KEY not set")
        return []

    base_query = f"{language} {query}".strip() if language else query.strip()

    if community_focus:
        # Enrich query to find community solutions
        query_lower = query.lower()
        if any(
            word in query_lower
            for word in ["error", "exception", "fails", "not working"]
        ):
            enriched_query = f"{base_query} solution workaround"
        elif any(word in query_lower for word in ["how to", "how do", "implement"]):
            enriched_query = f"{base_query} example working"
        else:
            enriched_query = f"{base_query} community solution"
    else:
        enriched_query = base_query

    payload: Dict[str, Any] = {
        "api_key": api_key,
        "query": enriched_query,
        "max_results": max_results,
        "search_depth": search_depth,
        "topic": topic,
        "include_answer": include_answer,
        "include_raw_content": include_raw_content,
    }

    if days is not None:
        payload["days"] = days

    if include_domains:
        payload["include_domains"] = include_domains

    if exclude_domains:
        payload["exclude_domains"] = exclude_domains

    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            response = await client.post(
                f"{BASE_URL}/search",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        items = data.get("results", [])
        results: List[Dict[str, Any]] = []

        for item in items:
            url = item.get("url")
            if not url:
                continue

            result = {
                "title": item.get("title") or url,
                "url": url,
                "snippet": item.get("content") or item.get("snippet") or "",
                "content": item.get("content") or "",
                "score": item.get("score", 0),
                "source": "tavily",
            }

            # Include raw content if requested
            if include_raw_content and item.get("raw_content"):
                result["raw_content"] = item.get("raw_content")

            results.append(result)

        # Include answer if available
        if include_answer and data.get("answer"):
            results.insert(
                0,
                {
                    "title": "Tavily AI Answer",
                    "url": "",
                    "snippet": data.get("answer"),
                    "content": data.get("answer"),
                    "source": "tavily_answer",
                    "is_answer": True,
                },
            )

        return results

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            logging.error("Tavily: Invalid API key")
        elif e.response.status_code == 429:
            logging.warning("Tavily: Rate limit exceeded")
        else:
            logging.exception("Tavily HTTP error during search")
        return []
    except httpx.HTTPError:
        logging.exception("Tavily HTTP error during search")
        return []
    except Exception:
        logging.exception("Tavily unexpected error during search")
        return []


async def search_tavily_news(
    query: str,
    language: Optional[str] = None,
    max_results: int = 10,
    days: int = 7,
) -> List[Dict[str, Any]]:
    """Search Tavily for recent news articles.

    Args:
        query: Search query
        language: Optional language context
        max_results: Maximum results (default 10)
        days: Limit to last N days (default 7)

    Returns:
        List of news results.
    """
    return await search_tavily(
        query=query,
        language=language,
        max_results=max_results,
        topic="news",
        days=days,
    )


async def extract_tavily(
    urls: List[str],
) -> List[Dict[str, Any]]:
    """Extract content from web pages using Tavily Extract.

    Args:
        urls: List of URLs to extract content from (max 20)

    Returns:
        List of extracted content with url, markdown content, and metadata.
    """
    api_key = _get_api_key()

    if not api_key:
        logging.debug("Tavily extract skipped: TAVILY_API_KEY not set")
        return []

    if not urls:
        return []

    # Tavily extract supports up to 20 URLs
    urls = urls[:20]

    payload = {
        "api_key": api_key,
        "urls": urls,
    }

    try:
        async with httpx.AsyncClient(
            timeout=60.0
        ) as client:  # Longer timeout for extraction
            response = await client.post(
                f"{BASE_URL}/extract",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        results: List[Dict[str, Any]] = []

        for item in data.get("results", []):
            url = item.get("url")
            if not url:
                continue

            results.append(
                {
                    "url": url,
                    "content": item.get("raw_content") or item.get("content") or "",
                    "success": True,
                    "source": "tavily_extract",
                }
            )

        # Handle failed extractions
        for failed_url in data.get("failed_results", []):
            results.append(
                {
                    "url": failed_url,
                    "content": "",
                    "success": False,
                    "error": "Extraction failed",
                    "source": "tavily_extract",
                }
            )

        return results

    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP {e.response.status_code}"
        if e.response.status_code == 401:
            error_msg = "Invalid API key"
        elif e.response.status_code == 429:
            error_msg = "Rate limit exceeded"
        return [{"url": url, "success": False, "error": error_msg} for url in urls]
    except httpx.HTTPError as e:
        logging.exception("Tavily extract HTTP error")
        return [{"url": url, "success": False, "error": str(e)} for url in urls]
    except Exception as e:
        logging.exception("Tavily extract unexpected error")
        return [{"url": url, "success": False, "error": str(e)} for url in urls]


async def search_tavily_with_extract(
    query: str,
    language: Optional[str] = None,
    max_results: int = 5,
    extract_top_n: int = 3,
) -> Dict[str, Any]:
    """Search Tavily and extract full content from top results.

    This combines search + extract for comprehensive results.

    Args:
        query: Search query
        language: Optional language context
        max_results: Number of search results
        extract_top_n: Number of top results to extract full content from

    Returns:
        Dict with search results and extracted content.
    """
    # First, search
    search_results = await search_tavily(
        query=query,
        language=language,
        max_results=max_results,
    )

    if not search_results:
        return {
            "search_results": [],
            "extracted_content": [],
            "source": "tavily_combined",
        }

    # Extract content from top results
    top_urls = [r["url"] for r in search_results[:extract_top_n] if r.get("url")]
    extracted = await extract_tavily(top_urls) if top_urls else []

    return {
        "search_results": search_results,
        "extracted_content": extracted,
        "source": "tavily_combined",
    }
