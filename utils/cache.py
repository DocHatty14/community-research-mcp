"""Caching utilities for Community Research MCP."""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Cache configuration
CACHE_FILE = Path(".community_research_cache.json")
CACHE_TTL_SECONDS = 3600  # 1 hour

# In-memory cache (loaded from disk on startup)
_cache: Dict[str, Dict[str, Any]] = {}


def get_cache_key(tool_name: str, **params) -> str:
    """Generate cache key from tool name and parameters."""
    param_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(f"{tool_name}:{param_str}".encode()).hexdigest()


def load_cache() -> Dict[str, Any]:
    """Load cache from disk."""
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            logging.warning(f"Failed to load cache: {e}")
    return {}


def save_cache() -> None:
    """Save cache to disk."""
    try:
        CACHE_FILE.write_text(json.dumps(_cache, indent=2), encoding="utf-8")
    except Exception as e:
        logging.warning(f"Failed to save cache: {e}")


def get_cached_result(cache_key: str) -> Optional[str]:
    """Retrieve cached result if not expired."""
    if cache_key in _cache:
        cached = _cache[cache_key]
        if time.time() - cached["timestamp"] < CACHE_TTL_SECONDS:
            return cached["result"]
        else:
            del _cache[cache_key]
            save_cache()  # Clean up expired
    return None


def set_cached_result(cache_key: str, result: str) -> None:
    """Store result in cache with timestamp."""
    _cache[cache_key] = {"result": result, "timestamp": time.time()}
    save_cache()


def clear_cache() -> bool:
    """Clear all cached results."""
    global _cache
    _cache = {}
    try:
        if CACHE_FILE.exists():
            CACHE_FILE.unlink()
        return True
    except Exception as e:
        logging.error(f"Failed to clear cache: {e}")
        return False


# Initialize cache from disk on module import
_cache = load_cache()
