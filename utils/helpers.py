"""Helper utilities for Community Research MCP."""

from typing import Any, Dict, List


def normalize_query_for_policy(query: str) -> str:
    """Normalize query whitespace for guardrail checks."""
    return " ".join(query.split()).strip()


def result_only_sources(results: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Filter out non-list metadata keys from aggregated results."""
    return {k: v for k, v in results.items() if isinstance(v, list)}


def total_result_count(results: Dict[str, Any]) -> int:
    """Compute total result count, ignoring metadata keys."""
    return sum(len(v) for v in results.values() if isinstance(v, list))
