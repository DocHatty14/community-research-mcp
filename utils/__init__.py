"""
Utility functions for caching, rate limiting, and helpers.

All utilities are stateless and have minimal dependencies,
making them easy to test independently.
"""

from utils.cache import (
    CACHE_TTL_SECONDS,
    clear_cache,
    get_cache_key,
    get_cached_result,
    set_cached_result,
)
from utils.helpers import (
    normalize_query_for_policy,
    result_only_sources,
    total_result_count,
)
from utils.rate_limit import (
    RATE_LIMIT_MAX_CALLS,
    RATE_LIMIT_WINDOW,
    check_rate_limit,
)

__all__ = [
    # Cache functions
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
    "normalize_query_for_policy",
    "result_only_sources",
    "total_result_count",
]
