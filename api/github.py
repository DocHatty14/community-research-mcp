"""
GitHub Issues and Discussions search.

Searches GitHub's issue tracker API sorted by reactions
to find the most relevant community discussions.
"""

import logging
from typing import Any, Dict, List

import httpx

API_TIMEOUT = 30.0


def _simplify_github_query(query: str, language: str, max_length: int = 256) -> str:
    """
    Simplify and truncate a GitHub search query to avoid 422 errors.

    GitHub API has query length limits. This function extracts the most
    important keywords and ensures the query stays within limits.

    Args:
        query: The original search query
        language: Programming language
        max_length: Maximum query length (default 256 chars)

    Returns:
        Simplified query string
    """
    # Build base query with language filter
    base = f"language:{language} is:issue"
    remaining = max_length - len(base) - 1  # -1 for space

    if remaining <= 0:
        return base

    # Extract important keywords (remove common words)
    stopwords = {
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

    # Tokenize and filter
    words = query.split()
    important_words = []

    for word in words:
        clean_word = word.strip().lower()
        # Keep technical terms, version numbers, and capitalized words
        if (
            (len(clean_word) > 2 and clean_word not in stopwords)
            or any(c.isdigit() for c in clean_word)
            or word[0].isupper()
        ):
            important_words.append(word)

    # Join words until we hit the length limit
    simplified_query = ""
    for word in important_words[:10]:  # Limit to 10 most important words
        test_query = f"{simplified_query} {word}".strip()
        if len(test_query) <= remaining:
            simplified_query = test_query
        else:
            break

    return f"{simplified_query} {base}".strip()


async def search_github(query: str, language: str) -> List[Dict[str, Any]]:
    """
    Search GitHub issues and discussions.

    This function automatically simplifies overly complex queries to avoid
    GitHub API 422 errors (query too long).
    """
    try:
        url = "https://api.github.com/search/issues"

        # Simplify query to avoid 422 errors
        simplified_query = _simplify_github_query(query, language, max_length=256)

        params = {
            "q": simplified_query,
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
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 422:
            logging.error(f"GitHub query too complex (422): {query[:100]}...")
            # Return empty results with a logged error
            return []
        logging.error(f"GitHub HTTP error {e.response.status_code}: {str(e)}")
        return []
    except Exception as e:
        logging.error(f"GitHub search failed: {str(e)}")
        return []
