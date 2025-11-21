"""Configuration enums and constants for Community Research MCP."""

from enum import Enum


class ResponseFormat(str, Enum):
    """Output format for tool responses."""

    MARKDOWN = "markdown"
    JSON = "json"


class ThinkingMode(str, Enum):
    """Analysis depth modes affecting cost vs insight trade-offs."""

    QUICK = "quick"  # Fast responses, lower cost, basic analysis
    BALANCED = "balanced"  # Default mode, good balance
    DEEP = "deep"  # Thorough analysis, higher cost, maximum insight
