"""Hacker News search API integration."""

import logging
from typing import Any, Dict, List

import httpx

API_TIMEOUT = 30.0


async def search_hackernews(query: str) -> List[Dict[str, Any]]:
    """Search Hacker News for high-quality tech discussions."""
    try:
        url = "https://hn.algolia.com/api/v1/search"
        params = {
            "query": query,
            "tags": "story",
            "numericFilters": "points>100",
        }

        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("hits", [])[:10]:
                results.append(
                    {
                        "title": item.get("title", ""),
                        "url": item.get(
                            "url",
                            f"https://news.ycombinator.com/item?id={item.get('objectID', '')}",
                        ),
                        "points": item.get("points", 0),
                        "comments": item.get("num_comments", 0),
                        "snippet": item.get("story_text", "")[:500],
                    }
                )
            return results
    except Exception as e:
        return []
