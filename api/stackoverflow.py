"""Stack Overflow search API integration."""

import asyncio
import logging
from typing import Any, Dict, List

import httpx

API_TIMEOUT = 30.0

SO_LANGUAGE_TAGS = {
    "python": "python",
    "python3": "python-3.x",
    "python2": "python-2.7",
    "javascript": "javascript",
    "js": "javascript",
    "typescript": "typescript",
    "node": "node.js",
    "react": "reactjs",
    "nodejs": "node.js",
    "java": "java",
    "csharp": "c#",
    "cpp": "c++",
    "c++": "c++",
    "rust": "rust",
    "go": "go",
    "ruby": "ruby",
    "php": "php",
    "sql": "sql",
    "database": "database",
    "rest": "rest",
    "api": "api",
}


async def search_stackoverflow(query: str, language: str) -> List[Dict[str, Any]]:
    """Search Stack Overflow using the Stack Exchange API with proper error handling."""
    try:
        # Map language to SO tags
        so_tag = SO_LANGUAGE_TAGS.get(language.lower(), language.lower())

        url = "https://api.stackexchange.com/2.3/search/advanced"
        params = {
            "order": "desc",
            "sort": "relevance",
            "q": query,
            "tagged": so_tag,
            "site": "stackoverflow",
            "filter": "withbody",
        }

        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            response = await client.get(url, params=params)

            if response.status_code == 429:
                logging.warning(f"SO API rate limited. Backoff required.")
                await asyncio.sleep(2)
                return []

            response.raise_for_status()
            data = response.json()

            if "error_id" in data:
                logging.warning(
                    f"SO API error: {data.get('error_message', 'Unknown error')}"
                )
                return []

            items = data.get("items", [])
            if not items:
                logging.debug(f"SO: No results for query '{query}' with tag '{so_tag}'")
                return []

            results = []
            for item in items[:15]:
                results.append(
                    {
                        "title": item.get("title", ""),
                        "url": item.get("link", ""),
                        "score": item.get("score", 0),
                        "answer_count": item.get("answer_count", 0),
                        "snippet": item.get("body", "")[:1000],
                    }
                )

            logging.info(f"SO: Found {len(results)} results for '{query}'")
            return results

    except httpx.TimeoutException:
        logging.warning(f"SO API timeout for query '{query}'")
        return []
    except httpx.HTTPStatusError as e:
        logging.warning(f"SO API HTTP error {e.response.status_code}: {str(e)}")
        return []
    except Exception as e:
        logging.error(f"SO API error: {str(e)}")
        return []
