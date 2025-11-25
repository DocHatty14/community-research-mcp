"""
Utility functions for caching, rate limiting, and data helpers.

All utilities are stateless and lightweight with minimal dependencies.
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

__all__ = [
    # Cache
    "get_cache_key",
    "get_cached_result",
    "set_cached_result",
    "clear_cache",
    "CACHE_TTL_SECONDS",
    # Rate limiting
    "check_rate_limit",
    "RATE_LIMIT_WINDOW",
    "RATE_LIMIT_MAX_CALLS",
    # Helpers
    "normalize_query",
    "count_results",
]

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# Cache Configuration
# ══════════════════════════════════════════════════════════════════════════════

CACHE_FILE = Path(".cache.json")
CACHE_TTL_SECONDS = 3600  # 1 hour

_cache: dict[str, dict[str, Any]] = {}

# ══════════════════════════════════════════════════════════════════════════════
# Cache Functions
# ══════════════════════════════════════════════════════════════════════════════


def get_cache_key(tool_name: str, **params) -> str:
    """Generate cache key from tool name and parameters."""
    param_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(f"{tool_name}:{param_str}".encode()).hexdigest()


def get_cached_result(key: str) -> Optional[str]:
    """Retrieve cached result if not expired."""
    if key in _cache:
        entry = _cache[key]
        if time.time() - entry["ts"] < CACHE_TTL_SECONDS:
            return entry["data"]
        del _cache[key]
    return None


def set_cached_result(key: str, result: str) -> None:
    """Store result in cache."""
    _cache[key] = {"data": result, "ts": time.time()}
    _save_cache()


def clear_cache() -> bool:
    """Clear all cached results."""
    global _cache
    _cache = {}
    try:
        if CACHE_FILE.exists():
            CACHE_FILE.unlink()
        return True
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        return False


def _load_cache() -> dict[str, Any]:
    """Load cache from disk."""
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_cache() -> None:
    """Save cache to disk."""
    try:
        CACHE_FILE.write_text(json.dumps(_cache), encoding="utf-8")
    except Exception as e:
        logger.warning(f"Cache save failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Rate Limiting
# ══════════════════════════════════════════════════════════════════════════════

RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX_CALLS = 10

_rate_tracker: dict[str, list[float]] = {}


def check_rate_limit(tool_name: str) -> bool:
    """
    Check if tool call is within rate limit.

    Returns True if allowed, False if rate limited.
    """
    now = time.time()

    if tool_name not in _rate_tracker:
        _rate_tracker[tool_name] = []

    # Remove old timestamps
    _rate_tracker[tool_name] = [
        ts for ts in _rate_tracker[tool_name] if now - ts < RATE_LIMIT_WINDOW
    ]

    # Check limit
    if len(_rate_tracker[tool_name]) >= RATE_LIMIT_MAX_CALLS:
        return False

    _rate_tracker[tool_name].append(now)
    return True


# ══════════════════════════════════════════════════════════════════════════════
# Helper Functions
# ══════════════════════════════════════════════════════════════════════════════


def normalize_query(query: str) -> str:
    """Normalize query whitespace."""
    return " ".join(query.split()).strip()


def count_results(results: dict[str, Any]) -> int:
    """Count total results across all sources."""
    return sum(len(v) for v in results.values() if isinstance(v, list))


# ══════════════════════════════════════════════════════════════════════════════
# Initialize
# ══════════════════════════════════════════════════════════════════════════════

_cache = _load_cache()
