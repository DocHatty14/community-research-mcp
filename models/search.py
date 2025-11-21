"""Search input/output models for Community Research MCP."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from models.config import ResponseFormat


class CommunitySearchInput(BaseModel):
    """Input model for community search."""

    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )

    language: str = Field(
        ...,
        description="Programming language (e.g., 'Python', 'JavaScript', 'Rust')",
        min_length=2,
        max_length=50,
    )
    topic: str = Field(
        ...,
        description=(
            "Specific, detailed topic. Be VERY specific - not just 'settings' or 'performance'. "
            "Examples: 'FastAPI background task queue with Redis', "
            "'React custom hooks for form validation', "
            "'Docker multi-stage builds to reduce image size'"
        ),
        min_length=10,
        max_length=500,
    )
    goal: Optional[str] = Field(
        default=None,
        description="What you want to achieve (e.g., 'async task processing without blocking requests')",
        max_length=500,
    )
    current_setup: Optional[str] = Field(
        default=None,
        description=(
            "Your current tech stack and setup. HIGHLY RECOMMENDED for implementation questions. "
            "Example: 'FastAPI app with SQLAlchemy, need queue for long-running jobs'"
        ),
        max_length=1000,
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' (default, human-readable) or 'json' (machine-readable)",
    )
    use_planning: bool = Field(
        default=False,
        description="Enable research planning mode for complex queries. Uses AI to create a structured research strategy before executing the search.",
    )
    thinking_mode: str = Field(
        default="balanced",
        description="Analysis depth mode: 'quick' (fast, basic), 'balanced' (default), or 'deep' (thorough, slower)",
    )
    expanded_mode: bool = Field(
        default=False,
        description="If true, allow expanded result budgets per source (higher caps, still read-only).",
    )
    use_fixtures: bool = Field(
        default=False,
        description="If true, return deterministic fixture data without hitting live sources (for health checks).",
    )

    @field_validator("topic")
    @classmethod
    def validate_topic_specificity(cls, v: str) -> str:
        """Ensure topic is specific enough to get useful results."""
        v = v.strip()

        # List of vague terms that indicate a non-specific query
        vague_terms = [
            "settings",
            "configuration",
            "config",
            "setup",
            "performance",
            "optimization",
            "best practices",
            "how to",
            "tutorial",
            "getting started",
            "basics",
            "help",
            "issue",
            "problem",
            "error",
            "debugging",
            "install",
            "installation",
        ]

        # Check if topic is just one or two vague words
        words = v.lower().split()
        if len(words) <= 2 and any(term in v.lower() for term in vague_terms):
            raise ValueError(
                f"Topic '{v}' is too vague. Be more specific! "
                f"Instead of 'settings', say 'GUI settings dialog with tabs and save/load buttons'. "
                f"Instead of 'performance', say 'reduce Docker image size with multi-stage builds'. "
                f"Include specific technologies, libraries, or patterns you're interested in."
            )

        return v


class DeepAnalyzeInput(BaseModel):
    """Input model for deep workspace analysis."""

    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )

    user_query: str = Field(
        ...,
        description="What you want to understand or improve about your codebase",
        min_length=10,
        max_length=1000,
    )
    workspace_path: Optional[str] = Field(
        default=None,
        description="Path to workspace to analyze (defaults to current directory)",
    )
    target_language: Optional[str] = Field(
        default=None,
        description="Specific language to focus on (e.g., 'Python', 'JavaScript')",
    )
