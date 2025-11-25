"""
GitHub Issues & Discussions Search.

Search GitHub's issue tracker sorted by community reactions
to find the most relevant discussions and solutions.

API: https://docs.github.com/en/rest/search
Rate Limits: 10/min (anonymous), 30/min (authenticated)
"""

import logging
import os
from typing import Any, Optional

import httpx

__all__ = ["search"]

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

API_BASE = "https://api.github.com"
API_TIMEOUT = 30.0
API_TOKEN = os.getenv("GITHUB_TOKEN")

logger = logging.getLogger(__name__)

# Words to filter from queries (too generic)
STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "how",
        "what",
        "where",
        "when",
        "why",
        "solution",
        "fix",
        "help",
        "error",
        "issue",
        "problem",
        "code",
        "example",
    }
)

# ══════════════════════════════════════════════════════════════════════════════
# Search Function
# ══════════════════════════════════════════════════════════════════════════════


async def search(
    query: str,
    language: Optional[str] = None,
    *,
    max_results: int = 15,
) -> list[dict[str, Any]]:
    """
    Search GitHub issues and discussions.

    Args:
        query: Search query string
        language: Programming language filter
        max_results: Maximum results to return

    Returns:
        List of issues with title, url, state, comments, snippet

    Example:
        >>> results = await search("memory leak", language="python")
    """
    # Build simplified query (GitHub has 256 char limit)
    search_query = _build_query(query, language)

    headers = {"Accept": "application/vnd.github.v3+json"}
    if API_TOKEN:
        headers["Authorization"] = f"token {API_TOKEN}"

    params = {
        "q": search_query,
        "sort": "reactions",
        "order": "desc",
        "per_page": min(max_results, 100),
    }

    try:
        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            response = await client.get(
                f"{API_BASE}/search/issues",
                params=params,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()

            return [
                {
                    "title": item.get("title", ""),
                    "url": item.get("html_url", ""),
                    "state": item.get("state", ""),
                    "comments": item.get("comments", 0),
                    "reactions": item.get("reactions", {}).get("total_count", 0),
                    "snippet": (item.get("body") or "")[:1000],
                    "source": "github",
                }
                for item in data.get("items", [])
            ]

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 422:
            logger.warning(f"Query too complex: {query[:50]}...")
        else:
            logger.warning(f"HTTP {e.response.status_code}")
        return []
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []


def _build_query(query: str, language: Optional[str], max_length: int = 256) -> str:
    """Build a simplified GitHub search query within length limits."""
    suffix = f"is:issue{f' language:{language}' if language else ''}"
    available = max_length - len(suffix) - 1

    # Extract important keywords
    words = []
    for word in query.split()[:10]:
        clean = word.strip().lower()
        is_technical = (
            len(clean) > 2 and clean not in STOPWORDS or any(c.isdigit() for c in clean)
        )
        if is_technical:
            words.append(word)

    # Build query within limit
    result = ""
    for word in words:
        test = f"{result} {word}".strip()
        if len(test) <= available:
            result = test
        else:
            break

    return f"{result} {suffix}".strip()


# ══════════════════════════════════════════════════════════════════════════════
# Backward Compatibility
# ══════════════════════════════════════════════════════════════════════════════

search_github = search
