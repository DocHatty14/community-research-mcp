"""
Hacker News Search (via Algolia).

Search HN for high-quality tech discussions, filtering for
substantive posts with significant community engagement.

API: https://hn.algolia.com/api
Rate Limits: Generous (Algolia hosted)
"""

import logging
from typing import Any, Optional

import httpx

__all__ = ["search"]

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

API_BASE = "https://hn.algolia.com/api/v1"
API_TIMEOUT = 30.0

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# Search Function
# ══════════════════════════════════════════════════════════════════════════════


async def search(
    query: str,
    *,
    min_points: int = 50,
    max_results: int = 10,
    search_type: str = "story",
) -> list[dict[str, Any]]:
    """
    Search Hacker News via Algolia.

    Args:
        query: Search query string
        min_points: Minimum points/upvotes filter (default: 50)
        max_results: Maximum results to return
        search_type: Type of content - 'story', 'comment', or 'all'

    Returns:
        List of posts with title, url, points, comments, snippet

    Example:
        >>> results = await search("rust async", min_points=100)
    """
    params = {
        "query": query,
        "hitsPerPage": min(max_results, 50),
    }

    if search_type != "all":
        params["tags"] = search_type

    if min_points > 0:
        params["numericFilters"] = f"points>{min_points}"

    try:
        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            response = await client.get(f"{API_BASE}/search", params=params)
            response.raise_for_status()
            data = response.json()

            return [
                {
                    "title": hit.get("title", ""),
                    "url": hit.get("url")
                    or f"https://news.ycombinator.com/item?id={hit.get('objectID', '')}",
                    "points": hit.get("points", 0),
                    "comments": hit.get("num_comments", 0),
                    "author": hit.get("author", ""),
                    "snippet": (hit.get("story_text") or "")[:500],
                    "source": "hackernews",
                }
                for hit in data.get("hits", [])
            ]

    except Exception as e:
        logger.warning(f"Search failed: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# Backward Compatibility
# ══════════════════════════════════════════════════════════════════════════════

search_hackernews = search
