"""
Lobsters search integration.

Lobsters is a tech community focused on high-quality discussions.
Scrapes their HTML search since no JSON API exists for search.
"""

import logging
from typing import Any, Dict, List
from urllib.parse import quote_plus

import httpx
from bs4 import BeautifulSoup

API_TIMEOUT = 30.0


async def search_lobsters(query: str) -> List[Dict[str, Any]]:
    """Search Lobsters for technical discussions."""
    try:
        # Lobsters search URL (HTML only, no JSON API)
        search_url = f"https://lobste.rs/search?q={quote_plus(query)}&what=stories&order=relevance"

        async with httpx.AsyncClient(
            timeout=API_TIMEOUT, follow_redirects=True
        ) as client:
            response = await client.get(
                search_url,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; CommunityResearchBot/1.0)"
                },
            )
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            results = []

            # Find story items
            stories = soup.find_all("li", class_="story")

            for story in stories[:10]:
                try:
                    title_elem = story.find("span", class_="link")
                    if not title_elem:
                        continue

                    link = title_elem.find("a")
                    if not link:
                        continue

                    title = link.get_text(strip=True)
                    url = link.get("href", "")

                    # Get points
                    score_elem = story.find("div", class_="score")
                    points = 0
                    if score_elem:
                        score_text = score_elem.get_text(strip=True)
                        try:
                            points = int(score_text.split()[0])
                        except:
                            pass

                    # Get comment count
                    comments_elem = story.find("span", class_="comments_label")
                    comments = 0
                    if comments_elem:
                        comments_link = comments_elem.find("a")
                        if comments_link:
                            comment_text = comments_link.get_text(strip=True)
                            try:
                                comments = int(comment_text.split()[0])
                            except:
                                pass

                    # Get tags
                    tags = []
                    tag_elems = story.find_all("a", class_="tag")
                    for tag in tag_elems:
                        tags.append(tag.get_text(strip=True))

                    # Get description/snippet
                    desc_elem = story.find("div", class_="byline")
                    snippet = ""
                    if desc_elem:
                        snippet = desc_elem.get_text(strip=True)[:300]

                    results.append(
                        {
                            "title": title,
                            "url": url
                            if url.startswith("http")
                            else f"https://lobste.rs{url}",
                            "points": points,
                            "comments": comments,
                            "snippet": snippet,
                            "tags": tags,
                            "source": "lobsters",
                        }
                    )

                except Exception as item_error:
                    logging.debug(f"Error parsing Lobsters item: {item_error}")
                    continue

            return results

    except Exception as e:
        logging.warning(f"Lobsters search failed: {e}")
        return []
