"""
Firecrawl API.

Web scraping and search with full page content extraction,
markdown conversion, and site mapping capabilities.

API: https://firecrawl.dev/
Rate Limits: Credit-based (check dashboard)
"""

import asyncio
import logging
import os
from typing import Any, Optional

import httpx

__all__ = ["search", "scrape", "map_site"]

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

API_BASE = "https://api.firecrawl.dev/v1"
API_TIMEOUT = 30.0
SCRAPE_TIMEOUT = 60.0
API_KEY = os.getenv("FIRECRAWL_API_KEY")

logger = logging.getLogger(__name__)


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


# ══════════════════════════════════════════════════════════════════════════════
# Search Function
# ══════════════════════════════════════════════════════════════════════════════


async def search(
    query: str,
    language: Optional[str] = None,
    *,
    max_results: int = 10,
) -> list[dict[str, Any]]:
    """
    Search the web via Firecrawl.

    Args:
        query: Search query string
        language: Programming language context (prepended to query)
        max_results: Maximum results to return

    Returns:
        List of results with title, url, snippet, content

    Example:
        >>> results = await search("GraphQL best practices", language="typescript")
    """
    if not API_KEY:
        logger.debug("Skipped: FIRECRAWL_API_KEY not set")
        return []

    full_query = f"{language} {query}".strip() if language else query

    try:
        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            response = await client.post(
                f"{API_BASE}/search",
                headers=_headers(),
                json={"query": full_query, "limit": max_results},
            )
            response.raise_for_status()
            data = response.json()

        # Handle various response formats
        items = data.get("data") or data.get("results") or []
        if isinstance(items, dict):
            items = items.get("web", []) + items.get("news", [])

        return [
            {
                "title": item.get("title") or item.get("heading") or "",
                "url": item.get("url") or item.get("link", ""),
                "snippet": item.get("description") or item.get("content", "")[:500],
                "content": item.get("markdown") or item.get("content", ""),
                "source": "firecrawl",
            }
            for item in items
            if item.get("url") or item.get("link")
        ]

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            logger.error("Invalid API key")
        elif e.response.status_code == 402:
            logger.warning("Payment required - check credits")
        elif e.response.status_code == 429:
            logger.warning("Rate limit exceeded")
        else:
            logger.warning(f"HTTP {e.response.status_code}")
        return []
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# Scrape Function
# ══════════════════════════════════════════════════════════════════════════════


async def scrape(
    url: str,
    *,
    formats: Optional[list[str]] = None,
    main_content_only: bool = True,
) -> dict[str, Any]:
    """
    Scrape a URL and extract content as markdown.

    Args:
        url: URL to scrape
        formats: Output formats - 'markdown', 'html', 'links', 'screenshot'
        main_content_only: Exclude navigation/footers

    Returns:
        Dict with markdown, html, links, metadata, success status

    Example:
        >>> result = await scrape("https://docs.python.org/3/tutorial/")
        >>> print(result["markdown"][:500])
    """
    if not API_KEY:
        return {"success": False, "error": "FIRECRAWL_API_KEY not set", "url": url}

    try:
        async with httpx.AsyncClient(timeout=SCRAPE_TIMEOUT) as client:
            response = await client.post(
                f"{API_BASE}/scrape",
                headers=_headers(),
                json={
                    "url": url,
                    "formats": formats or ["markdown"],
                    "onlyMainContent": main_content_only,
                },
            )
            response.raise_for_status()
            data = response.json()

        if not data.get("success"):
            return {"success": False, "error": data.get("error", "Unknown"), "url": url}

        result = data.get("data", {})
        return {
            "success": True,
            "url": url,
            "markdown": result.get("markdown", ""),
            "html": result.get("html", ""),
            "links": result.get("links", []),
            "metadata": result.get("metadata", {}),
            "source": "firecrawl:scrape",
        }

    except httpx.HTTPStatusError as e:
        error = f"HTTP {e.response.status_code}"
        if e.response.status_code == 402:
            error = "Payment required"
        elif e.response.status_code == 429:
            error = "Rate limit exceeded"
        return {"success": False, "error": error, "url": url}
    except Exception as e:
        return {"success": False, "error": str(e), "url": url}


async def scrape_many(
    urls: list[str],
    *,
    main_content_only: bool = True,
) -> list[dict[str, Any]]:
    """
    Scrape multiple URLs concurrently.

    Args:
        urls: List of URLs to scrape
        main_content_only: Exclude navigation/footers

    Returns:
        List of scrape results
    """
    tasks = [scrape(url, main_content_only=main_content_only) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    return [
        r if isinstance(r, dict) else {"success": False, "error": str(r), "url": url}
        for url, r in zip(urls, results)
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Map Function
# ══════════════════════════════════════════════════════════════════════════════


async def map_site(
    url: str,
    *,
    search_filter: Optional[str] = None,
    include_subdomains: bool = False,
    max_urls: int = 100,
) -> dict[str, Any]:
    """
    Discover all URLs on a website.

    Args:
        url: Base URL to map
        search_filter: Optional term to filter URLs
        include_subdomains: Include subdomain links
        max_urls: Maximum URLs to return

    Returns:
        Dict with links list, count, success status

    Example:
        >>> result = await map_site("https://fastapi.tiangolo.com")
        >>> print(f"Found {result['count']} pages")
    """
    if not API_KEY:
        return {"success": False, "error": "FIRECRAWL_API_KEY not set", "url": url}

    payload: dict[str, Any] = {
        "url": url,
        "includeSubdomains": include_subdomains,
        "limit": max_urls,
    }

    if search_filter:
        payload["search"] = search_filter

    try:
        async with httpx.AsyncClient(timeout=SCRAPE_TIMEOUT) as client:
            response = await client.post(
                f"{API_BASE}/map",
                headers=_headers(),
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        if not data.get("success"):
            return {"success": False, "error": data.get("error", "Unknown"), "url": url}

        links = data.get("links", [])
        return {
            "success": True,
            "url": url,
            "links": links,
            "count": len(links),
            "source": "firecrawl:map",
        }

    except httpx.HTTPStatusError as e:
        error = f"HTTP {e.response.status_code}"
        if e.response.status_code == 402:
            error = "Payment required"
        return {"success": False, "error": error, "url": url}
    except Exception as e:
        return {"success": False, "error": str(e), "url": url}


# ══════════════════════════════════════════════════════════════════════════════
# Backward Compatibility
# ══════════════════════════════════════════════════════════════════════════════

search_firecrawl = search
scrape_firecrawl = scrape
scrape_multiple_firecrawl = scrape_many
map_firecrawl = map_site
