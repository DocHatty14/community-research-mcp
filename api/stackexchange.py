"""
Stack Exchange API Integration.

Unified search across 19+ Stack Exchange network sites including
Stack Overflow, Server Fault, Super User, and specialized communities.

API: https://api.stackexchange.com/docs
Rate Limits: 300/day (anonymous), 10,000/day (with API key)
"""

import asyncio
import logging
import os
from typing import Any, Optional

import httpx

__all__ = [
    "search",
    "search_multi",
    "SITES",
    "LANGUAGE_TAGS",
]

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

API_BASE = "https://api.stackexchange.com/2.3"
API_TIMEOUT = 30.0
API_KEY = os.getenv("STACKEXCHANGE_API_KEY")

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# Site Registry
# ══════════════════════════════════════════════════════════════════════════════

SITES: dict[str, dict[str, str]] = {
    # Programming & Development
    "stackoverflow": {"name": "Stack Overflow", "focus": "Programming Q&A"},
    "codereview": {"name": "Code Review", "focus": "Code improvement"},
    "softwareengineering": {
        "name": "Software Engineering",
        "focus": "Design & architecture",
    },
    # DevOps & Infrastructure
    "serverfault": {"name": "Server Fault", "focus": "System administration"},
    "devops": {"name": "DevOps", "focus": "CI/CD & automation"},
    "unix": {"name": "Unix & Linux", "focus": "POSIX systems"},
    "askubuntu": {"name": "Ask Ubuntu", "focus": "Ubuntu Linux"},
    # Databases
    "dba": {"name": "Database Administrators", "focus": "Database optimization"},
    # Security
    "security": {"name": "Information Security", "focus": "Security & cryptography"},
    # Data & AI
    "datascience": {"name": "Data Science", "focus": "ML & data analysis"},
    "ai": {"name": "Artificial Intelligence", "focus": "AI theory & applications"},
    "stats": {"name": "Cross Validated", "focus": "Statistics & ML theory"},
    # Hardware & IoT
    "superuser": {"name": "Super User", "focus": "Computer hardware/software"},
    "raspberrypi": {"name": "Raspberry Pi", "focus": "Pi hardware & software"},
    "arduino": {"name": "Arduino", "focus": "Arduino & electronics"},
    "iot": {"name": "Internet of Things", "focus": "IoT devices & protocols"},
    # Platforms
    "apple": {"name": "Ask Different", "focus": "Apple products"},
    "android": {"name": "Android Enthusiasts", "focus": "Android devices"},
    "webmasters": {"name": "Webmasters", "focus": "Website operation & SEO"},
}

# ══════════════════════════════════════════════════════════════════════════════
# Language Tag Mapping
# ══════════════════════════════════════════════════════════════════════════════

LANGUAGE_TAGS: dict[str, str] = {
    # Languages
    "python": "python",
    "python3": "python-3.x",
    "python2": "python-2.7",
    "javascript": "javascript",
    "js": "javascript",
    "typescript": "typescript",
    "java": "java",
    "csharp": "c#",
    "cpp": "c++",
    "c++": "c++",
    "rust": "rust",
    "go": "go",
    "golang": "go",
    "ruby": "ruby",
    "php": "php",
    # Frameworks
    "node": "node.js",
    "nodejs": "node.js",
    "react": "reactjs",
    # Databases
    "sql": "sql",
    "mysql": "mysql",
    "postgresql": "postgresql",
    "mongodb": "mongodb",
    "redis": "redis",
    "database": "database",
    # DevOps
    "docker": "docker",
    "kubernetes": "kubernetes",
    "k8s": "kubernetes",
    "aws": "amazon-web-services",
    "azure": "azure",
    "gcp": "google-cloud-platform",
    # System
    "linux": "linux",
    "bash": "bash",
    "shell": "shell",
    "git": "git",
    "nginx": "nginx",
    "apache": "apache",
    # APIs
    "rest": "rest",
    "api": "api",
    "graphql": "graphql",
}

# ══════════════════════════════════════════════════════════════════════════════
# Search Functions
# ══════════════════════════════════════════════════════════════════════════════


async def search(
    query: str,
    language: Optional[str] = None,
    site: str = "stackoverflow",
    *,
    sort: str = "relevance",
    accepted_only: bool = False,
    min_answers: int = 0,
    max_results: int = 15,
) -> list[dict[str, Any]]:
    """
    Search a Stack Exchange site.

    Args:
        query: Search query string
        language: Programming language/technology to filter by
        site: Site key (default: stackoverflow). See SITES for options.
        sort: Sort by 'relevance', 'votes', 'creation', or 'activity'
        accepted_only: Only questions with accepted answers
        min_answers: Minimum answer count required
        max_results: Maximum results to return (max 100)

    Returns:
        List of question results with title, url, score, answers, snippet

    Example:
        >>> results = await search("async await", language="python")
        >>> results = await search("docker compose", site="devops")
    """
    # Validate site
    if site not in SITES:
        logger.warning(f"Unknown site '{site}', using stackoverflow")
        site = "stackoverflow"

    # Build request
    params: dict[str, Any] = {
        "order": "desc",
        "sort": sort,
        "q": query,
        "site": site,
        "filter": "withbody",
        "pagesize": min(max_results, 100),
    }

    if language:
        params["tagged"] = LANGUAGE_TAGS.get(language.lower(), language.lower())
    if accepted_only:
        params["accepted"] = "True"
    if min_answers > 0:
        params["answers"] = min_answers
    if API_KEY:
        params["key"] = API_KEY

    try:
        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            response = await client.get(f"{API_BASE}/search/advanced", params=params)

            if response.status_code == 429:
                logger.warning(f"Rate limited on {site}")
                await asyncio.sleep(2)
                return []

            response.raise_for_status()
            data = response.json()

            if "error_id" in data:
                logger.warning(f"API error: {data.get('error_message')}")
                return []

            logger.debug(f"Quota remaining: {data.get('quota_remaining')}")

            return [
                {
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "score": item.get("score", 0),
                    "answer_count": item.get("answer_count", 0),
                    "is_answered": item.get("is_answered", False),
                    "view_count": item.get("view_count", 0),
                    "tags": item.get("tags", []),
                    "snippet": (item.get("body") or "")[:1000],
                    "source": site
                    if site == "stackoverflow"
                    else f"stackexchange:{site}",
                }
                for item in data.get("items", [])[:max_results]
            ]

    except httpx.TimeoutException:
        logger.warning(f"Timeout searching {site}")
        return []
    except httpx.HTTPStatusError as e:
        logger.warning(f"HTTP {e.response.status_code} from {site}")
        return []
    except Exception as e:
        logger.error(f"Error searching {site}: {e}")
        return []


async def search_multi(
    query: str,
    language: Optional[str] = None,
    sites: Optional[list[str]] = None,
    *,
    max_per_site: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    """
    Search multiple Stack Exchange sites concurrently.

    Args:
        query: Search query string
        language: Programming language/technology to filter by
        sites: Site keys to search (default: core dev sites)
        max_per_site: Maximum results per site

    Returns:
        Dict mapping site keys to their results

    Example:
        >>> results = await search_multi("memory leak", language="python")
        >>> so_results = results["stackoverflow"]
    """
    if sites is None:
        sites = ["stackoverflow", "unix", "serverfault", "devops", "dba"]

    valid_sites = [s for s in sites if s in SITES]
    if not valid_sites:
        logger.warning("No valid sites provided")
        return {}

    tasks = [
        search(query, language=language, site=site, max_results=max_per_site)
        for site in valid_sites
    ]

    results_list = await asyncio.gather(*tasks, return_exceptions=True)

    results = {}
    for site, site_results in zip(valid_sites, results_list):
        if isinstance(site_results, Exception):
            logger.error(f"Error on {site}: {site_results}")
            results[site] = []
        else:
            results[site] = site_results

    total = sum(len(r) for r in results.values())
    logger.info(f"Found {total} results across {len(valid_sites)} sites")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Answer Fetching
# ══════════════════════════════════════════════════════════════════════════════


async def fetch_accepted_answer(
    question_id: int,
    site: str = "stackoverflow",
) -> Optional[dict[str, Any]]:
    """
    Fetch the accepted answer for a Stack Overflow question.

    Args:
        question_id: The question ID from the URL
        site: Stack Exchange site (default: stackoverflow)

    Returns:
        Answer dict with body, score, is_accepted, or None if no accepted answer
    """
    params: dict[str, Any] = {
        "order": "desc",
        "sort": "votes",
        "site": site,
        "filter": "withbody",  # Include answer body
    }
    if API_KEY:
        params["key"] = API_KEY

    try:
        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            response = await client.get(
                f"{API_BASE}/questions/{question_id}/answers",
                params=params,
            )

            if response.status_code == 429:
                logger.warning("Rate limited fetching answers")
                return None

            response.raise_for_status()
            data = response.json()

            answers = data.get("items", [])
            if not answers:
                return None

            # Prioritize accepted answer, then highest voted
            accepted = next((a for a in answers if a.get("is_accepted")), None)
            best = accepted or answers[0]

            return {
                "body": best.get("body", ""),
                "score": best.get("score", 0),
                "is_accepted": best.get("is_accepted", False),
                "answer_id": best.get("answer_id"),
            }

    except Exception as e:
        logger.debug(f"Failed to fetch answer for question {question_id}: {e}")
        return None


async def enrich_with_answers(
    results: list[dict[str, Any]],
    site: str = "stackoverflow",
    max_to_enrich: int = 5,
) -> list[dict[str, Any]]:
    """
    Enrich top Stack Overflow results with their accepted/best answer content.

    Only enriches results that have accepted answers to avoid wasted API calls.

    Args:
        results: List of search results
        site: Stack Exchange site
        max_to_enrich: Maximum number of results to enrich (to save API quota)

    Returns:
        Enriched results with answer_body field added
    """
    import re

    # Only enrich results that are answered
    to_enrich = [
        r
        for r in results[:max_to_enrich]
        if r.get("is_answered") or r.get("answer_count", 0) > 0
    ]

    async def enrich_one(result: dict[str, Any]) -> dict[str, Any]:
        url = result.get("url", "")
        # Extract question ID from URL like https://stackoverflow.com/questions/12345/title
        match = re.search(r"/questions/(\d+)", url)
        if not match:
            return result

        question_id = int(match.group(1))
        answer = await fetch_accepted_answer(question_id, site)

        if answer and answer.get("body"):
            # Clean HTML from answer body
            body = answer["body"]
            # Remove HTML tags but keep code blocks
            clean_body = re.sub(r"<code>", "```", body)
            clean_body = re.sub(r"</code>", "```", clean_body)
            clean_body = re.sub(r"<pre>", "\n", clean_body)
            clean_body = re.sub(r"</pre>", "\n", clean_body)
            clean_body = re.sub(r"<[^>]+>", " ", clean_body)
            clean_body = re.sub(r"&lt;", "<", clean_body)
            clean_body = re.sub(r"&gt;", ">", clean_body)
            clean_body = re.sub(r"&amp;", "&", clean_body)
            clean_body = re.sub(r"&quot;", '"', clean_body)
            clean_body = re.sub(r"&#39;", "'", clean_body)
            clean_body = re.sub(r"\s+", " ", clean_body).strip()

            result["answer_body"] = clean_body[:2000]  # Limit size
            result["answer_score"] = answer.get("score", 0)
            result["has_accepted_answer"] = answer.get("is_accepted", False)

            # Use answer as snippet if it's better
            if len(clean_body) > len(result.get("snippet", "")):
                result["snippet"] = clean_body[:1000]

        return result

    # Fetch answers concurrently
    enriched = await asyncio.gather(*[enrich_one(r) for r in to_enrich])

    # Merge enriched results back
    enriched_map = {r.get("url"): r for r in enriched}
    return [enriched_map.get(r.get("url"), r) for r in results]


# ══════════════════════════════════════════════════════════════════════════════
# Backward Compatibility
# ══════════════════════════════════════════════════════════════════════════════

# Alias for legacy imports
search_stackoverflow = search
search_stackexchange = search
search_multiple_stackexchange_sites = search_multi
