"""
Data models for the Community Research MCP.

Provides Pydantic models for request validation and
configuration enums for controlling analysis behavior.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "ResponseFormat",
    "ThinkingMode",
    "SearchInput",
    "AnalyzeInput",
    # Legacy aliases
    "CommunitySearchInput",
    "DeepAnalyzeInput",
]

# ══════════════════════════════════════════════════════════════════════════════
# Configuration Enums
# ══════════════════════════════════════════════════════════════════════════════


class ResponseFormat(str, Enum):
    """Output format for tool responses."""

    MARKDOWN = "markdown"
    JSON = "json"
    RAW = "raw"  # Full unprocessed results for LLM synthesis


class ThinkingMode(str, Enum):
    """Analysis depth modes."""

    QUICK = "quick"  # Fast, basic analysis
    BALANCED = "balanced"  # Default, good balance
    DEEP = "deep"  # Thorough, slower


# ══════════════════════════════════════════════════════════════════════════════
# Input Models
# ══════════════════════════════════════════════════════════════════════════════


class SearchInput(BaseModel):
    """Input model for community search."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    language: str = Field(
        ...,
        description="Programming language (e.g., 'Python', 'JavaScript')",
        min_length=2,
        max_length=50,
    )

    topic: str = Field(
        ...,
        description=(
            "Specific topic to search for. Be detailed! "
            "Example: 'FastAPI background tasks with Redis queue'"
        ),
        min_length=10,
        max_length=500,
    )

    goal: Optional[str] = Field(
        default=None,
        description="What you want to achieve",
        max_length=500,
    )

    current_setup: Optional[str] = Field(
        default=None,
        description="Your current tech stack (highly recommended)",
        max_length=1000,
    )

    response_format: ResponseFormat = Field(
        default=ResponseFormat.RAW,
        description="Output format: 'raw' (default - full data for LLM synthesis), 'markdown', or 'json'",
    )

    thinking_mode: ThinkingMode = Field(
        default=ThinkingMode.BALANCED,
        description="Analysis depth: 'quick', 'balanced', or 'deep'",
    )

    expanded_mode: bool = Field(
        default=False,
        description="Enable expanded result limits",
    )

    @field_validator("topic")
    @classmethod
    def validate_topic(cls, v: str) -> str:
        """Ensure topic is specific enough."""
        v = v.strip()

        vague_terms = {
            "settings",
            "config",
            "setup",
            "performance",
            "best practices",
            "tutorial",
            "basics",
            "help",
            "issue",
            "problem",
            "error",
            "install",
        }

        words = v.lower().split()
        if len(words) <= 2 and any(t in v.lower() for t in vague_terms):
            raise ValueError(
                f"Topic '{v}' is too vague. Be more specific! "
                "Include technologies, libraries, or patterns."
            )

        return v


class AnalyzeInput(BaseModel):
    """Input model for workspace analysis."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    query: str = Field(
        ...,
        description="What you want to understand about your codebase",
        min_length=10,
        max_length=1000,
    )

    workspace_path: Optional[str] = Field(
        default=None,
        description="Path to analyze (defaults to current directory)",
    )

    language: Optional[str] = Field(
        default=None,
        description="Language to focus on",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Legacy Aliases (backward compatibility)
# ══════════════════════════════════════════════════════════════════════════════

CommunitySearchInput = SearchInput
DeepAnalyzeInput = AnalyzeInput
