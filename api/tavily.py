"""Tavily search integration."""

import logging
import os
from typing import Any, Dict, List, Optional

import httpx

DEFAULT_TIMEOUT = 30.0


def _build_payload(query: str, language: Optional[str], max_results: int) -> Dict[str, Any]:
    enriched_query = f"{language} {query}" if language else query
    return {
        "api_key": os.getenv("TAVILY_API_KEY"),
        "query": enriched_query,
        "max_results": max_results,
        "include_answer": False,
    }


async def search_tavily(
    query: str, language: Optional[str] = None, max_results: int = 10
) -> List[Dict[str, Any]]:
    """Search Tavily for concise, relevant snippets."""

    api_key = os.getenv("TAVILY_API_KEY")
    api_url = os.getenv("TAVILY_API_URL", "https://api.tavily.com/search")

    if not api_key:
        logging.debug("Tavily skipped: TAVILY_API_KEY not set")
        return []

    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            response = await client.post(
                api_url, json=_build_payload(query, language, max_results)
            )
            response.raise_for_status()
            data = response.json()

        items = data.get("results", [])
        results: List[Dict[str, Any]] = []

        for item in items:
            url = item.get("url")
            if not url:
                continue

            results.append(
                {
                    "title": item.get("title") or url,
                    "url": url,
                    "snippet": item.get("content") or item.get("snippet") or "",
                    "content": item.get("content") or "",
                    "source": "tavily",
                }
            )

        return results
    except Exception as exc:  # pragma: no cover - network guarded
        logging.error(f"Tavily search failed: {exc}")
        return []
