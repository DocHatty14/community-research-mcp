"""
GitHub Issues and Discussions search.

Searches GitHub's issue tracker API sorted by reactions
to find the most relevant community discussions.
"""

import logging
from typing import Any, Dict, List

import httpx

API_TIMEOUT = 30.0


async def search_github(query: str, language: str) -> List[Dict[str, Any]]:
    """Search GitHub issues and discussions."""
    try:
        url = "https://api.github.com/search/issues"
        params = {
            "q": f"{query} language:{language} is:issue",
            "sort": "reactions",
            "order": "desc",
            "per_page": 15,
        }

        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("items", []):
                results.append(
                    {
                        "title": item.get("title", ""),
                        "url": item.get("html_url", ""),
                        "state": item.get("state", ""),
                        "comments": item.get("comments", 0),
                        "snippet": (item.get("body", "") or "")[:1000],
                    }
                )
            return results
    except Exception as e:
        return []
