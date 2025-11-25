"""
Discourse forums search integration.

Searches popular Discourse forums for technical discussions.
Uses public JSON endpoints (.json suffix on URLs).
"""

import logging
from typing import Any, Dict, List

import httpx

API_TIMEOUT = 30.0

# Popular Discourse forums for different tech stacks
DISCOURSE_FORUMS = {
    "rust": "https://users.rust-lang.org",
    "elixir": "https://elixirforum.com",
    "swift": "https://forums.swift.org",
    "julia": "https://discourse.julialang.org",
    "python": "https://discuss.python.org",
    "javascript": "https://discuss.js.org",
    "general": "https://meta.discourse.org",
}


async def search_discourse(query: str, language: str) -> List[Dict[str, Any]]:
    """Search Discourse forums for language-specific discussions."""
    try:
        # Pick forum based on language
        lang_lower = language.lower()
        forum_url = DISCOURSE_FORUMS.get(lang_lower, DISCOURSE_FORUMS["general"])

        # Discourse search endpoint
        url = f"{forum_url}/search.json"
        params = {
            "q": query,
        }

        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            results = []
            for topic in data.get("topics", [])[:10]:
                results.append(
                    {
                        "title": topic.get("title", ""),
                        "url": f"{forum_url}/t/{topic.get('slug', '')}/{topic.get('id', '')}",
                        "views": topic.get("views", 0),
                        "replies": topic.get("posts_count", 0) - 1,
                        "likes": topic.get("like_count", 0),
                        "snippet": topic.get("blurb", "")[:500]
                        if topic.get("blurb")
                        else "",
                        "solved": topic.get("has_accepted_answer", False),
                    }
                )

            return results
    except Exception as e:
        logging.warning(f"Discourse search failed: {e}")
        return []
