"""
Lobsters Community Search.

Search Lobsters.rs for high-quality technical discussions.
Uses HTML scraping since no public JSON API exists.

Site: https://lobste.rs
Rate Limits: Be respectful - small community site
"""

import logging
from typing import Any
from urllib.parse import quote_plus

import httpx
from bs4 import BeautifulSoup

__all__ = ["search"]

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

API_BASE = "https://lobste.rs"
API_TIMEOUT = 30.0
USER_AGENT = "Mozilla/5.0 (compatible; CommunityResearchBot/1.0)"

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# Search Function
# ══════════════════════════════════════════════════════════════════════════════


async def search(
    query: str,
    *,
    max_results: int = 10,
) -> list[dict[str, Any]]:
    """
    Search Lobsters for technical discussions.

    Args:
        query: Search query string
        max_results: Maximum results to return

    Returns:
        List of stories with title, url, points, comments, tags

    Example:
        >>> results = await search("distributed systems")
    """
    url = f"{API_BASE}/search?q={quote_plus(query)}&what=stories&order=relevance"

    try:
        async with httpx.AsyncClient(
            timeout=API_TIMEOUT, follow_redirects=True
        ) as client:
            response = await client.get(url, headers={"User-Agent": USER_AGENT})
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            results = []

            for story in soup.find_all("li", class_="story")[:max_results]:
                try:
                    result = _parse_story(story)
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.debug(f"Parse error: {e}")
                    continue

            return results

    except Exception as e:
        logger.warning(f"Search failed: {e}")
        return []


def _parse_story(story) -> dict[str, Any] | None:
    """Parse a single story element."""
    title_elem = story.find("span", class_="link")
    if not title_elem:
        return None

    link = title_elem.find("a")
    if not link:
        return None

    title = link.get_text(strip=True)
    url = link.get("href", "")
    if not url.startswith("http"):
        url = f"{API_BASE}{url}"

    # Extract score
    points = 0
    score_elem = story.find("div", class_="score")
    if score_elem:
        try:
            points = int(score_elem.get_text(strip=True).split()[0])
        except (ValueError, IndexError):
            pass

    # Extract comments
    comments = 0
    comments_elem = story.find("span", class_="comments_label")
    if comments_elem:
        comments_link = comments_elem.find("a")
        if comments_link:
            try:
                comments = int(comments_link.get_text(strip=True).split()[0])
            except (ValueError, IndexError):
                pass

    # Extract tags
    tags = [tag.get_text(strip=True) for tag in story.find_all("a", class_="tag")]

    return {
        "title": title,
        "url": url,
        "points": points,
        "comments": comments,
        "tags": tags,
        "source": "lobsters",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Backward Compatibility
# ══════════════════════════════════════════════════════════════════════════════

search_lobsters = search
