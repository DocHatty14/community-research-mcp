"""Rate limiting utilities for Community Research MCP."""

import time
from typing import Dict, List

# Rate limit configuration
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX_CALLS = 10  # max calls per window

# Rate limit tracker (tool_name -> list of timestamps)
_rate_limit_tracker: Dict[str, List[float]] = {}


def check_rate_limit(tool_name: str) -> bool:
    """
    Check if tool call is within rate limit.
    Returns True if allowed, False if rate limited.
    """
    now = time.time()
    if tool_name not in _rate_limit_tracker:
        _rate_limit_tracker[tool_name] = []

    # Remove old timestamps outside the window
    _rate_limit_tracker[tool_name] = [
        ts for ts in _rate_limit_tracker[tool_name] if now - ts < RATE_LIMIT_WINDOW
    ]

    # Check if under limit
    if len(_rate_limit_tracker[tool_name]) >= RATE_LIMIT_MAX_CALLS:
        return False

    # Add current timestamp
    _rate_limit_tracker[tool_name].append(now)
    return True
