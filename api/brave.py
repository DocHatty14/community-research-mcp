"""Brave Search API integration.

Implements Brave Search endpoints:
- Web search with filtering, freshness, and pagination
- News search for recent articles
- Extra snippets for additional context
"""

import logging
import os
from typing import Any, Dict, List, Optional

import httpx

DEFAULT_TIMEOUT = 30.0
BASE_URL = "https://api.search.brave.com/res/v1"


async def search_brave_web(
    query: str,
    count: int = 10,
    offset: int = 0,
    freshness: Optional[str] = None,
    extra_snippets: bool = True,
    result_filter: Optional[str] = None,
    country: str = "us",
    search_lang: str = "en",
) -> List[Dict[str, Any]]:
    """Search Brave Web Search API.

    Args:
        query: Search query (max 400 chars, 50 words)
        count: Number of results (1-20, default 10)
        offset: Pagination offset (0-9, default 0)
        freshness: Filter by age - pd (24h), pw (week), pm (month), py (year), or YYYY-MM-DDtoYYYY-MM-DD
        extra_snippets: Include additional snippets from the page
        result_filter: Filter types - web, news, images, videos (comma-separated)
        country: Country code for results (default: us)
        search_lang: Language code (default: en)

    Returns:
        List of normalized search results with title, url, snippet, and metadata.
    """
    api_key = os.getenv("BRAVE_SEARCH_API_KEY")

    if not api_key:
        logging.debug("Brave Search skipped: BRAVE_SEARCH_API_KEY not set")
        return []

    params: Dict[str, Any] = {
        "q": query[:400],  # Max 400 chars
        "count": min(max(count, 1), 20),  # Clamp to 1-20
        "offset": min(max(offset, 0), 9),  # Clamp to 0-9
        "country": country,
        "search_lang": search_lang,
        "extra_snippets": str(extra_snippets).lower(),
    }

    if freshness:
        params["freshness"] = freshness

    if result_filter:
        params["result_filter"] = result_filter

    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            response = await client.get(
                f"{BASE_URL}/web/search",
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": api_key,
                },
                params=params,
            )
            response.raise_for_status()
            data = response.json()

        results: List[Dict[str, Any]] = []

        # Process web results
        web_results = data.get("web", {}).get("results", [])
        for item in web_results:
            url = item.get("url")
            if not url:
                continue

            # Combine description with extra snippets if available
            snippet_parts = [item.get("description", "")]
            extra = item.get("extra_snippets", [])
            if extra:
                snippet_parts.extend(extra)

            results.append(
                {
                    "title": item.get("title", "Untitled"),
                    "url": url,
                    "snippet": " ".join(filter(None, snippet_parts)),
                    "age": item.get("age", ""),
                    "language": item.get("language", ""),
                    "family_friendly": item.get("family_friendly", True),
                    "source": "brave_web",
                }
            )

        # Process news results if present
        news_results = data.get("news", {}).get("results", [])
        for item in news_results:
            url = item.get("url")
            if not url:
                continue

            results.append(
                {
                    "title": item.get("title", "Untitled"),
                    "url": url,
                    "snippet": item.get("description", ""),
                    "age": item.get("age", ""),
                    "source_name": item.get("meta_url", {}).get("hostname", ""),
                    "source": "brave_news",
                }
            )

        return results

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            logging.error("Brave Search: Invalid API key")
        elif e.response.status_code == 429:
            logging.warning("Brave Search: Rate limit exceeded")
        else:
            logging.exception("Brave Search HTTP error")
        return []
    except httpx.HTTPError:
        logging.exception("Brave Search HTTP error during search")
        return []
    except Exception:
        logging.exception("Brave Search unexpected error")
        return []


async def search_brave_news(
    query: str,
    count: int = 10,
    offset: int = 0,
    freshness: Optional[str] = None,
    country: str = "us",
    search_lang: str = "en",
) -> List[Dict[str, Any]]:
    """Search Brave News Search API for recent articles.

    Args:
        query: Search query
        count: Number of results (1-20)
        offset: Pagination offset (0-9)
        freshness: Filter by age - pd (24h), pw (week), pm (month)
        country: Country code
        search_lang: Language code

    Returns:
        List of news results with title, url, snippet, and metadata.
    """
    api_key = os.getenv("BRAVE_SEARCH_API_KEY")

    if not api_key:
        logging.debug("Brave News skipped: BRAVE_SEARCH_API_KEY not set")
        return []

    params: Dict[str, Any] = {
        "q": query[:400],
        "count": min(max(count, 1), 20),
        "offset": min(max(offset, 0), 9),
        "country": country,
        "search_lang": search_lang,
    }

    if freshness:
        params["freshness"] = freshness

    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            response = await client.get(
                f"{BASE_URL}/news/search",
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": api_key,
                },
                params=params,
            )
            response.raise_for_status()
            data = response.json()

        results: List[Dict[str, Any]] = []
        news_results = data.get("results", [])

        for item in news_results:
            url = item.get("url")
            if not url:
                continue

            results.append(
                {
                    "title": item.get("title", "Untitled"),
                    "url": url,
                    "snippet": item.get("description", ""),
                    "age": item.get("age", ""),
                    "source_name": item.get("meta_url", {}).get("hostname", ""),
                    "thumbnail": item.get("thumbnail", {}).get("src", ""),
                    "source": "brave_news",
                }
            )

        return results

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            logging.warning("Brave News: Rate limit exceeded")
        else:
            logging.exception("Brave News HTTP error")
        return []
    except httpx.HTTPError:
        logging.exception("Brave News HTTP error during search")
        return []
    except Exception:
        logging.exception("Brave News unexpected error")
        return []


async def search_brave(
    query: str,
    language: Optional[str] = None,
    count: int = 10,
    freshness: Optional[str] = None,
    community_focus: bool = True,
) -> List[Dict[str, Any]]:
    """Unified Brave search function matching other API signatures.

    This is the primary function used by the search aggregator.
    Optimized to find community solutions, workarounds, and real-world fixes.

    Args:
        query: Search query
        language: Optional programming language context (prepended to query)
        count: Number of results
        freshness: Optional freshness filter
        community_focus: If True, enrich query to target community solutions

    Returns:
        List of normalized search results.
    """
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

    return await search_brave_web(
        query=enriched_query,
        count=count,
        extra_snippets=True,
        freshness=freshness,
    )
