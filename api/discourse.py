"""
Discourse Forums Search.

Search language-specific Discourse communities for technical discussions.
Uses public JSON endpoints available on all Discourse instances.

Supported Forums: Rust, Elixir, Swift, Julia, Python, and more
Rate Limits: Varies by forum
"""

import logging
from typing import Any, Optional

import httpx

__all__ = ["search", "FORUMS"]

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

API_TIMEOUT = 30.0

logger = logging.getLogger(__name__)

# Language-specific Discourse forums
FORUMS: dict[str, str] = {
    "rust": "https://users.rust-lang.org",
    "elixir": "https://elixirforum.com",
    "swift": "https://forums.swift.org",
    "julia": "https://discourse.julialang.org",
    "python": "https://discuss.python.org",
    "ruby": "https://discuss.rubyonrails.org",
    "ember": "https://discuss.emberjs.com",
    "kubernetes": "https://discuss.kubernetes.io",
    "pytorch": "https://discuss.pytorch.org",
    "terraform": "https://discuss.hashicorp.com",
}

DEFAULT_FORUM = "https://meta.discourse.org"

# ══════════════════════════════════════════════════════════════════════════════
# Search Function
# ══════════════════════════════════════════════════════════════════════════════


async def search(
    query: str,
    language: Optional[str] = None,
    *,
    forum_url: Optional[str] = None,
    max_results: int = 10,
) -> list[dict[str, Any]]:
    """
    Search Discourse forums for discussions.

    Args:
        query: Search query string
        language: Programming language to select forum
        forum_url: Direct forum URL (overrides language selection)
        max_results: Maximum results to return

    Returns:
        List of topics with title, url, views, replies, likes

    Example:
        >>> results = await search("async await", language="rust")
        >>> results = await search("deployment", forum_url="https://discuss.kubernetes.io")
    """
    # Select forum
    if forum_url:
        base_url = forum_url.rstrip("/")
    elif language:
        base_url = FORUMS.get(language.lower(), DEFAULT_FORUM)
    else:
        base_url = DEFAULT_FORUM

    try:
        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            response = await client.get(
                f"{base_url}/search.json",
                params={"q": query},
            )
            response.raise_for_status()
            data = response.json()

            return [
                {
                    "title": topic.get("title", ""),
                    "url": f"{base_url}/t/{topic.get('slug', '')}/{topic.get('id', '')}",
                    "views": topic.get("views", 0),
                    "replies": max(0, topic.get("posts_count", 1) - 1),
                    "likes": topic.get("like_count", 0),
                    "solved": topic.get("has_accepted_answer", False),
                    "snippet": (topic.get("blurb") or "")[:500],
                    "source": f"discourse:{base_url.split('//')[1].split('/')[0]}",
                }
                for topic in data.get("topics", [])[:max_results]
            ]

    except Exception as e:
        logger.warning(f"Search failed on {base_url}: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# Backward Compatibility
# ══════════════════════════════════════════════════════════════════════════════

search_discourse = search
