"""Serper (Google Search) API integration.

Implements Serper endpoints for Google Search results:
- Web search with knowledge graph, organic results, people also ask
- News search
- Image search
"""

import logging
import os
from typing import Any, Dict, List, Optional

import httpx

DEFAULT_TIMEOUT = 30.0
BASE_URL = "https://google.serper.dev"


def _get_api_key() -> Optional[str]:
    """Get Serper API key from environment."""
    return os.getenv("SERPER_API_KEY")


async def search_serper(
    query: str,
    language: Optional[str] = None,
    num_results: int = 10,
    country: str = "us",
    locale: str = "en",
    autocorrect: bool = True,
    search_type: str = "search",
    community_focus: bool = True,
) -> List[Dict[str, Any]]:
    """Search using Serper (Google Search API).

    Optimized to find community solutions, workarounds, and real-world fixes.

    Args:
        query: Search query
        language: Optional programming language context (prepended to query)
        num_results: Number of results (default 10, max 100)
        country: Country code for results (e.g., "us", "uk")
        locale: Language code (e.g., "en", "es")
        autocorrect: Enable autocorrect (default True)
        search_type: Type of search - "search", "news", "images"
        community_focus: If True, enrich query to target community solutions

    Returns:
        List of normalized search results with title, url, snippet, and metadata.
    """
    api_key = _get_api_key()

    if not api_key:
        logging.debug("Serper skipped: SERPER_API_KEY not set")
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
        "q": enriched_query,
        "num": min(max(num_results, 1), 100),
        "gl": country,
        "hl": locale,
        "autocorrect": autocorrect,
    }

    # Determine endpoint based on search type
    endpoint = f"/{search_type}"
    if search_type not in ["search", "news", "images"]:
        endpoint = "/search"

    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            response = await client.post(
                f"{BASE_URL}{endpoint}",
                headers={
                    "X-API-KEY": api_key,
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        results: List[Dict[str, Any]] = []

        # Extract knowledge graph if present
        knowledge_graph = data.get("knowledgeGraph")
        if knowledge_graph:
            kg_result = {
                "title": knowledge_graph.get("title", "Knowledge Graph"),
                "url": knowledge_graph.get("website", ""),
                "snippet": knowledge_graph.get("description", ""),
                "type": knowledge_graph.get("type", ""),
                "attributes": knowledge_graph.get("attributes", {}),
                "source": "serper_knowledge_graph",
                "is_knowledge_graph": True,
            }
            results.append(kg_result)

        # Extract organic results
        organic_results = data.get("organic", [])
        for item in organic_results:
            url = item.get("link")
            if not url:
                continue

            results.append(
                {
                    "title": item.get("title", "Untitled"),
                    "url": url,
                    "snippet": item.get("snippet", ""),
                    "position": item.get("position", 0),
                    "source": "serper",
                }
            )

        # Extract "People Also Ask" questions
        people_also_ask = data.get("peopleAlsoAsk", [])
        for item in people_also_ask:
            if item.get("link"):
                results.append(
                    {
                        "title": item.get("question", "Related Question"),
                        "url": item.get("link", ""),
                        "snippet": item.get("snippet", ""),
                        "source": "serper_people_also_ask",
                        "is_question": True,
                    }
                )

        # Extract news results if present
        news_results = data.get("news", [])
        for item in news_results:
            url = item.get("link")
            if not url:
                continue

            results.append(
                {
                    "title": item.get("title", "Untitled"),
                    "url": url,
                    "snippet": item.get("snippet", ""),
                    "date": item.get("date", ""),
                    "source_name": item.get("source", ""),
                    "source": "serper_news",
                }
            )

        return results

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            logging.error("Serper: Invalid API key")
        elif e.response.status_code == 429:
            logging.warning("Serper: Rate limit exceeded")
        else:
            logging.exception("Serper HTTP error during search")
        return []
    except httpx.HTTPError:
        logging.exception("Serper HTTP error during search")
        return []
    except Exception:
        logging.exception("Serper unexpected error during search")
        return []


async def search_serper_news(
    query: str,
    language: Optional[str] = None,
    num_results: int = 10,
    country: str = "us",
    locale: str = "en",
) -> List[Dict[str, Any]]:
    """Search Serper for news articles.

    Args:
        query: Search query
        language: Optional language context
        num_results: Number of results (default 10)
        country: Country code
        locale: Language code

    Returns:
        List of news results.
    """
    api_key = _get_api_key()

    if not api_key:
        logging.debug("Serper News skipped: SERPER_API_KEY not set")
        return []

    enriched_query = f"{language} {query}" if language else query

    payload = {
        "q": enriched_query,
        "num": min(max(num_results, 1), 100),
        "gl": country,
        "hl": locale,
    }

    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            response = await client.post(
                f"{BASE_URL}/news",
                headers={
                    "X-API-KEY": api_key,
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        results: List[Dict[str, Any]] = []
        news_items = data.get("news", [])

        for item in news_items:
            url = item.get("link")
            if not url:
                continue

            results.append(
                {
                    "title": item.get("title", "Untitled"),
                    "url": url,
                    "snippet": item.get("snippet", ""),
                    "date": item.get("date", ""),
                    "source_name": item.get("source", ""),
                    "image_url": item.get("imageUrl", ""),
                    "source": "serper_news",
                }
            )

        return results

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            logging.warning("Serper News: Rate limit exceeded")
        else:
            logging.exception("Serper News HTTP error")
        return []
    except httpx.HTTPError:
        logging.exception("Serper News HTTP error during search")
        return []
    except Exception:
        logging.exception("Serper News unexpected error")
        return []


async def search_serper_images(
    query: str,
    language: Optional[str] = None,
    num_results: int = 10,
    country: str = "us",
    locale: str = "en",
) -> List[Dict[str, Any]]:
    """Search Serper for images.

    Args:
        query: Search query
        language: Optional language context
        num_results: Number of results (default 10)
        country: Country code
        locale: Language code

    Returns:
        List of image results.
    """
    api_key = _get_api_key()

    if not api_key:
        logging.debug("Serper Images skipped: SERPER_API_KEY not set")
        return []

    enriched_query = f"{language} {query}" if language else query

    payload = {
        "q": enriched_query,
        "num": min(max(num_results, 1), 100),
        "gl": country,
        "hl": locale,
    }

    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            response = await client.post(
                f"{BASE_URL}/images",
                headers={
                    "X-API-KEY": api_key,
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        results: List[Dict[str, Any]] = []
        image_items = data.get("images", [])

        for item in image_items:
            url = item.get("link")
            if not url:
                continue

            results.append(
                {
                    "title": item.get("title", "Untitled"),
                    "url": url,
                    "image_url": item.get("imageUrl", ""),
                    "thumbnail_url": item.get("thumbnailUrl", ""),
                    "source_name": item.get("source", ""),
                    "source": "serper_images",
                }
            )

        return results

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            logging.warning("Serper Images: Rate limit exceeded")
        else:
            logging.exception("Serper Images HTTP error")
        return []
    except httpx.HTTPError:
        logging.exception("Serper Images HTTP error during search")
        return []
    except Exception:
        logging.exception("Serper Images unexpected error")
        return []


async def get_serper_related_searches(
    query: str,
    country: str = "us",
    locale: str = "en",
) -> List[str]:
    """Get related search queries from Serper.

    Args:
        query: Original search query
        country: Country code
        locale: Language code

    Returns:
        List of related search query strings.
    """
    api_key = _get_api_key()

    if not api_key:
        return []

    payload = {
        "q": query,
        "gl": country,
        "hl": locale,
    }

    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            response = await client.post(
                f"{BASE_URL}/search",
                headers={
                    "X-API-KEY": api_key,
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        related = data.get("relatedSearches", [])
        return [item.get("query", "") for item in related if item.get("query")]

    except Exception:
        logging.exception("Serper related searches error")
        return []
