"""Firecrawl search integration."""

import logging
import os
from typing import Any, Dict, List, Optional

import httpx

DEFAULT_TIMEOUT = 30.0


def _build_payload(query: str, language: Optional[str]) -> Dict[str, Any]:
    if language:
        return {"q": f"{language} {query}"}
    return {"q": query}


async def search_firecrawl(query: str, language: Optional[str] = None) -> List[Dict[str, Any]]:
    """Search Firecrawl for relevant pages.

    Args:
        query: User search query
        language: Optional programming language context

    Returns:
        List of normalized search results with title, url, snippet, and content.
    """

    api_key = os.getenv("FIRECRAWL_API_KEY")
    api_url = os.getenv("FIRECRAWL_API_URL", "https://api.firecrawl.dev/v1/search")

    if not api_key:
        logging.debug("Firecrawl skipped: FIRECRAWL_API_KEY not set")
        return []

    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            response = await client.post(
                api_url,
                headers={"Authorization": f"Bearer {api_key}"},
                json=_build_payload(query, language),
            )
            response.raise_for_status()
            data = response.json()

        items = data.get("data") or data.get("results") or []
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
                    or "",
                    "content": item.get("content") or "",
                    "source": "firecrawl",
                }
            )

        return results
    except Exception as exc:  # pragma: no cover - network guarded
        logging.error(f"Firecrawl search failed: {exc}")
        return []
