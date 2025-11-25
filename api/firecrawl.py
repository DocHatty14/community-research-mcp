"""Firecrawl API integration.

Implements Firecrawl endpoints:
- Search: Find relevant pages based on query
- Scrape: Get full markdown content from a URL
- Map: Discover all URLs on a website
"""

import logging
import os
from typing import Any, Dict, List, Optional

import httpx

DEFAULT_TIMEOUT = 30.0
BASE_URL = "https://api.firecrawl.dev/v1"


def _get_api_key() -> Optional[str]:
    """Get Firecrawl API key from environment."""
    return os.getenv("FIRECRAWL_API_KEY")


def _get_headers(api_key: str) -> Dict[str, str]:
    """Build authorization headers."""
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


async def search_firecrawl(
    query: str,
    language: Optional[str] = None,
    limit: int = 10,
    scrape_options: Optional[Dict[str, Any]] = None,
    community_focus: bool = True,
) -> List[Dict[str, Any]]:
    """Search Firecrawl for relevant pages.

    Optimized to find community solutions, workarounds, and real-world fixes.

    Args:
        query: User search query
        language: Optional programming language context
        limit: Maximum number of results (default 10)
        scrape_options: Optional scrape configuration for results
        community_focus: If True, enrich query to target community solutions

    Returns:
        List of normalized search results with title, url, snippet, and content.
    """
    api_key = _get_api_key()

    if not api_key:
        logging.debug("Firecrawl skipped: FIRECRAWL_API_KEY not set")
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
        "query": enriched_query,
        "limit": limit,
    }

    if scrape_options:
        payload["scrapeOptions"] = scrape_options

    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            response = await client.post(
                f"{BASE_URL}/search",
                headers=_get_headers(api_key),
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        raw_items = data.get("data") or data.get("results") or []
        items: List[Dict[str, Any]] = []

        if isinstance(raw_items, dict):
            for bucket in ("web", "news", "images"):
                bucket_items = raw_items.get(bucket) or []
                if isinstance(bucket_items, list):
                    items.extend(bucket_items)
        elif isinstance(raw_items, list):
            items = raw_items

        results: List[Dict[str, Any]] = []

        for item in items:
            url = item.get("url") or item.get("link")
            if not url:
                continue

            results.append(
                {
                    "title": item.get("title")
                    or item.get("heading")
                    or item.get("url")
                    or "Untitled",
                    "url": url,
                    "snippet": item.get("description")
                    or item.get("content")
                    or item.get("markdown")
                    or "",
                    "content": item.get("content") or item.get("markdown") or "",
                    "source": "firecrawl",
                }
            )

        return results

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            logging.error("Firecrawl: Invalid API key")
        elif e.response.status_code == 402:
            logging.warning("Firecrawl: Payment required - check credits")
        elif e.response.status_code == 429:
            logging.warning("Firecrawl: Rate limit exceeded")
        else:
            logging.exception("Firecrawl HTTP error during search")
        return []
    except httpx.HTTPError:
        logging.exception("Firecrawl HTTP error during search")
        return []
    except Exception:
        logging.exception("Firecrawl unexpected error during search")
        return []


async def scrape_firecrawl(
    url: str,
    formats: Optional[List[str]] = None,
    only_main_content: bool = True,
    include_tags: Optional[List[str]] = None,
    exclude_tags: Optional[List[str]] = None,
    wait_for: int = 0,
) -> Dict[str, Any]:
    """Scrape a single URL and get its content.

    Args:
        url: The URL to scrape
        formats: Output formats - markdown, html, rawHtml, links, screenshot, etc.
        only_main_content: Extract only main content, excluding navs/footers (default True)
        include_tags: Only include content from these HTML tags
        exclude_tags: Exclude content from these HTML tags
        wait_for: Wait time in ms for dynamic content (default 0)

    Returns:
        Dict with scraped content including markdown, metadata, and optional extras.
    """
    api_key = _get_api_key()

    if not api_key:
        logging.debug("Firecrawl scrape skipped: FIRECRAWL_API_KEY not set")
        return {"success": False, "error": "API key not configured"}

    payload: Dict[str, Any] = {
        "url": url,
        "formats": formats or ["markdown"],
        "onlyMainContent": only_main_content,
    }

    if include_tags:
        payload["includeTags"] = include_tags
    if exclude_tags:
        payload["excludeTags"] = exclude_tags
    if wait_for > 0:
        payload["waitFor"] = wait_for

    try:
        async with httpx.AsyncClient(
            timeout=60.0
        ) as client:  # Longer timeout for scraping
            response = await client.post(
                f"{BASE_URL}/scrape",
                headers=_get_headers(api_key),
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        if not data.get("success"):
            return {
                "success": False,
                "error": data.get("error", "Unknown error"),
                "url": url,
            }

        result_data = data.get("data", {})
        return {
            "success": True,
            "url": url,
            "markdown": result_data.get("markdown", ""),
            "html": result_data.get("html", ""),
            "links": result_data.get("links", []),
            "metadata": result_data.get("metadata", {}),
            "source": "firecrawl_scrape",
        }

    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP {e.response.status_code}"
        if e.response.status_code == 402:
            error_msg = "Payment required - check credits"
        elif e.response.status_code == 429:
            error_msg = "Rate limit exceeded"
        elif e.response.status_code == 408:
            error_msg = "Request timeout - page may be slow"
        return {"success": False, "error": error_msg, "url": url}
    except httpx.HTTPError as e:
        logging.exception("Firecrawl scrape HTTP error")
        return {"success": False, "error": str(e), "url": url}
    except Exception as e:
        logging.exception("Firecrawl scrape unexpected error")
        return {"success": False, "error": str(e), "url": url}


async def map_firecrawl(
    url: str,
    search: Optional[str] = None,
    ignore_sitemap: bool = False,
    include_subdomains: bool = False,
    limit: int = 100,
) -> Dict[str, Any]:
    """Map all URLs on a website.

    Args:
        url: Base URL to map from
        search: Optional search term to filter URLs
        ignore_sitemap: Skip sitemap.xml and crawl manually
        include_subdomains: Include links to subdomains
        limit: Maximum URLs to return (default 100)

    Returns:
        Dict with list of discovered URLs and metadata.
    """
    api_key = _get_api_key()

    if not api_key:
        logging.debug("Firecrawl map skipped: FIRECRAWL_API_KEY not set")
        return {"success": False, "error": "API key not configured"}

    payload: Dict[str, Any] = {
        "url": url,
        "ignoreSitemap": ignore_sitemap,
        "includeSubdomains": include_subdomains,
        "limit": limit,
    }

    if search:
        payload["search"] = search

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{BASE_URL}/map",
                headers=_get_headers(api_key),
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        if not data.get("success"):
            return {
                "success": False,
                "error": data.get("error", "Unknown error"),
                "url": url,
            }

        return {
            "success": True,
            "url": url,
            "links": data.get("links", []),
            "count": len(data.get("links", [])),
            "source": "firecrawl_map",
        }

    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP {e.response.status_code}"
        if e.response.status_code == 402:
            error_msg = "Payment required - check credits"
        elif e.response.status_code == 429:
            error_msg = "Rate limit exceeded"
        return {"success": False, "error": error_msg, "url": url}
    except httpx.HTTPError as e:
        logging.exception("Firecrawl map HTTP error")
        return {"success": False, "error": str(e), "url": url}
    except Exception as e:
        logging.exception("Firecrawl map unexpected error")
        return {"success": False, "error": str(e), "url": url}


async def scrape_multiple_firecrawl(
    urls: List[str],
    formats: Optional[List[str]] = None,
    only_main_content: bool = True,
) -> List[Dict[str, Any]]:
    """Scrape multiple URLs concurrently.

    Args:
        urls: List of URLs to scrape
        formats: Output formats for each scrape
        only_main_content: Extract only main content

    Returns:
        List of scrape results for each URL.
    """
    import asyncio

    tasks = [
        scrape_firecrawl(url, formats=formats, only_main_content=only_main_content)
        for url in urls
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    processed: List[Dict[str, Any]] = []
    for url, result in zip(urls, results):
        if isinstance(result, Exception):
            processed.append(
                {
                    "success": False,
                    "error": str(result),
                    "url": url,
                }
            )
        else:
            processed.append(result)

    return processed
