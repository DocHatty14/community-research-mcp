#!/usr/bin/env python3
"""
Streaming Capabilities Module

Provides real-time streaming search with progressive result aggregation,
capability auto-detection, and adaptive formatting.
"""

import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional

# ============================================================================
# Capability Detection
# ============================================================================


class SearchCapability(Enum):
    """Available search capabilities."""

    STACKOVERFLOW = "stackoverflow"
    GITHUB = "github"
    REDDIT = "reddit"
    REDDIT_AUTHENTICATED = "reddit_authenticated"
    HACKERNEWS = "hackernews"
    DUCKDUCKGO = "duckduckgo"
    BRAVE = "brave"
    SERPER = "serper"
    WEB_SCRAPING = "web_scraping"


class LLMCapability(Enum):
    """Available LLM providers."""

    GEMINI = "gemini"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"
    PERPLEXITY = "perplexity"


@dataclass
class SystemCapabilities:
    """Complete system capability profile."""

    search_apis: Dict[str, bool]
    llm_providers: Dict[str, bool]
    workspace_context: Optional[Dict[str, Any]]
    timestamp: datetime


def detect_all_capabilities() -> SystemCapabilities:
    """
    Auto-detect all available API keys and system capabilities.

    Returns complete profile of what the system can do right now.
    """
    # Detect search API capabilities
    search_capabilities = {
        SearchCapability.STACKOVERFLOW.value: True,  # No key needed
        SearchCapability.GITHUB.value: True,  # No key needed
        SearchCapability.HACKERNEWS.value: True,  # No key needed
        SearchCapability.DUCKDUCKGO.value: True,  # No key needed
        SearchCapability.WEB_SCRAPING.value: True,  # Built-in
        # Authenticated Reddit
        SearchCapability.REDDIT_AUTHENTICATED.value: all(
            [
                os.getenv("REDDIT_CLIENT_ID"),
                os.getenv("REDDIT_CLIENT_SECRET"),
                os.getenv("REDDIT_REFRESH_TOKEN"),
            ]
        ),
        # Reddit fallback (public API)
        SearchCapability.REDDIT.value: True,
        # Premium search APIs (if keys available)
        SearchCapability.BRAVE.value: bool(os.getenv("BRAVE_SEARCH_API_KEY")),
        SearchCapability.SERPER.value: bool(os.getenv("SERPER_API_KEY")),
    }

    # Detect LLM providers
    llm_capabilities = {
        LLMCapability.GEMINI.value: bool(os.getenv("GEMINI_API_KEY")),
        LLMCapability.OPENAI.value: bool(os.getenv("OPENAI_API_KEY")),
        LLMCapability.ANTHROPIC.value: bool(os.getenv("ANTHROPIC_API_KEY")),
        LLMCapability.OPENROUTER.value: bool(os.getenv("OPENROUTER_API_KEY")),
        LLMCapability.PERPLEXITY.value: bool(os.getenv("PERPLEXITY_API_KEY")),
    }

    return SystemCapabilities(
        search_apis=search_capabilities,
        llm_providers=llm_capabilities,
        workspace_context=None,  # Will be populated by workspace detection
        timestamp=datetime.now(),
    )


def format_capabilities_report(capabilities: SystemCapabilities) -> str:
    """Format capabilities as a user-friendly report."""
    report = ["# System Capabilities\n"]

    # Search APIs
    report.append("## Search APIs")
    active_search = [k for k, v in capabilities.search_apis.items() if v]
    inactive_search = [k for k, v in capabilities.search_apis.items() if not v]

    report.append(f"**Active ({len(active_search)}):**")
    for api in active_search:
        report.append(f"  ✓ {api}")

    if inactive_search:
        report.append(f"\n**Inactive ({len(inactive_search)}):**")
        for api in inactive_search:
            report.append(f"  ✗ {api} (API key not configured)")

    # LLM Providers
    report.append("\n## LLM Providers")
    active_llm = [k for k, v in capabilities.llm_providers.items() if v]
    inactive_llm = [k for k, v in capabilities.llm_providers.items() if not v]

    report.append(f"**Active ({len(active_llm)}):**")
    for provider in active_llm:
        report.append(f"  ✓ {provider}")

    if inactive_llm:
        report.append(f"\n**Inactive ({len(inactive_llm)}):**")
        for provider in inactive_llm:
            report.append(f"  ✗ {provider} (API key not configured)")

    # Total capability count
    total_active = len(active_search) + len(active_llm)
    report.append(f"\n**Total Active Capabilities:** {total_active}")
    report.append(
        f"**Detection Time:** {capabilities.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
    )

    return "\n".join(report)


# ============================================================================
# Result Content Type Classification
# ============================================================================


class ResultType(Enum):
    """Types of search results for adaptive formatting."""

    QUICK_FIX = "quick_fix"  # Accepted answer with code
    DISCUSSION = "discussion"  # Community discussion
    OFFICIAL_DOCS = "official_docs"  # Documentation/guides
    CODE_EXAMPLE = "code_example"  # GitHub code samples
    WARNING = "warning"  # Gotchas/issues
    TUTORIAL = "tutorial"  # Step-by-step guides


def classify_result(result: Dict[str, Any], source: str) -> ResultType:
    """Classify a single result by content type."""

    # Stack Overflow - check for accepted answers
    if source == "stackoverflow":
        if result.get("is_answered") or result.get("accepted_answer_id"):
            return ResultType.QUICK_FIX
        return ResultType.DISCUSSION

    # GitHub - code examples
    if source == "github":
        return ResultType.CODE_EXAMPLE

    # Reddit - discussions
    if source == "reddit":
        title_lower = result.get("title", "").lower()
        if any(word in title_lower for word in ["warning", "issue", "problem", "bug"]):
            return ResultType.WARNING
        if any(word in title_lower for word in ["tutorial", "guide", "how to"]):
            return ResultType.TUTORIAL
        return ResultType.DISCUSSION

    # Hacker News - discussions
    if source == "hackernews":
        return ResultType.DISCUSSION

    # Default
    return ResultType.DISCUSSION


def organize_by_type(
    results: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Organize search results by content type for adaptive presentation.

    Returns categorized results optimized for different use cases.
    """
    categorized = {
        ResultType.QUICK_FIX.value: [],
        ResultType.CODE_EXAMPLE.value: [],
        ResultType.DISCUSSION.value: [],
        ResultType.WARNING.value: [],
        ResultType.TUTORIAL.value: [],
        ResultType.OFFICIAL_DOCS.value: [],
    }

    for source, items in results.items():
        if not isinstance(items, list):
            continue

        for item in items:
            result_type = classify_result(item, source)
            categorized[result_type.value].append(
                {**item, "source": source, "classified_as": result_type.value}
            )

    # Remove empty categories
    return {k: v for k, v in categorized.items() if v}


# ============================================================================
# Progressive Result Aggregation
# ============================================================================


@dataclass
class StreamingResult:
    """Container for a streaming search result."""

    source: str
    data: List[Dict[str, Any]]
    timestamp: datetime
    is_final: bool = False
    error: Optional[str] = None


@dataclass
class AggregatedState:
    """State of aggregated results as they stream in."""

    results_by_source: Dict[str, List[Dict[str, Any]]]
    results_by_type: Dict[str, List[Dict[str, Any]]]
    sources_completed: List[str]
    total_results: int
    start_time: datetime
    last_update: datetime


class ProgressiveAggregator:
    """
    Aggregates and reorganizes results as they stream in.

    Provides smart reorganization with each new result batch.
    """

    def __init__(self):
        self.state = AggregatedState(
            results_by_source={},
            results_by_type={},
            sources_completed=[],
            total_results=0,
            start_time=datetime.now(),
            last_update=datetime.now(),
        )

    def add_result(self, result: StreamingResult) -> AggregatedState:
        """
        Add a new streaming result and reorganize.

        Returns updated aggregated state.
        """
        # Update source results
        if result.data:
            self.state.results_by_source[result.source] = result.data
            self.state.total_results += len(result.data)

        # Mark source as completed
        if result.is_final and result.source not in self.state.sources_completed:
            self.state.sources_completed.append(result.source)

        # Handle errors
        if result.error:
            self.state.sources_completed.append(result.source)

        # Reorganize by type
        self.state.results_by_type = organize_by_type(self.state.results_by_source)

        # Update timestamp
        self.state.last_update = datetime.now()

        return self.state

    def get_smart_summary(self) -> Dict[str, Any]:
        """Generate smart summary of current state."""
        elapsed = (self.state.last_update - self.state.start_time).total_seconds()

        return {
            "total_results": self.state.total_results,
            "sources_completed": len(self.state.sources_completed),
            "sources_pending": self._get_pending_sources(),
            "elapsed_seconds": round(elapsed, 2),
            "results_by_source": {
                k: len(v) for k, v in self.state.results_by_source.items()
            },
            "results_by_type": {
                k: len(v) for k, v in self.state.results_by_type.items()
            },
            "is_complete": len(self.state.sources_completed)
            >= self._get_expected_source_count(),
        }

    def _get_pending_sources(self) -> List[str]:
        """Get list of sources still pending."""
        all_sources = ["stackoverflow", "github", "reddit", "hackernews"]
        return [s for s in all_sources if s not in self.state.sources_completed]

    def _get_expected_source_count(self) -> int:
        """Get expected number of sources."""
        return 4  # stackoverflow, github, reddit, hackernews


# ============================================================================
# Adaptive Result Formatting
# ============================================================================


def format_streaming_update(
    state: AggregatedState, summary: Dict[str, Any], format_style: str = "progressive"
) -> str:
    """
    Format current state for streaming output.

    Args:
        state: Current aggregated state
        summary: Smart summary from aggregator
        format_style: "progressive" (show all), "incremental" (show new only)
    """
    output = []

    # Header with progress
    output.append(f"# Search Progress: {summary['sources_completed']}/4 Sources")
    output.append(
        f"**Results:** {summary['total_results']} | **Elapsed:** {summary['elapsed_seconds']}s\n"
    )

    # Show pending sources
    if summary["sources_pending"]:
        output.append(f"*Waiting for: {', '.join(summary['sources_pending'])}*\n")

    # Organize by type (adaptive formatting)
    if state.results_by_type:
        output.append("## Results by Type\n")

        # Quick fixes first (most valuable)
        if ResultType.QUICK_FIX.value in state.results_by_type:
            fixes = state.results_by_type[ResultType.QUICK_FIX.value]
            output.append(f"### Quick Fixes ({len(fixes)})")
            for fix in fixes[:3]:  # Show top 3
                output.append(f"- **{fix.get('title', 'Untitled')}**")
                output.append(
                    f"  Source: {fix.get('source')} | Score: {fix.get('score', 0)}"
                )
            output.append("")

        # Code examples
        if ResultType.CODE_EXAMPLE.value in state.results_by_type:
            examples = state.results_by_type[ResultType.CODE_EXAMPLE.value]
            output.append(f"### Code Examples ({len(examples)})")
            for ex in examples[:3]:
                output.append(f"- **{ex.get('title', 'Untitled')}**")
                output.append(f"  {ex.get('url', '')}")
            output.append("")

        # Warnings (important!)
        if ResultType.WARNING.value in state.results_by_type:
            warnings = state.results_by_type[ResultType.WARNING.value]
            output.append(f"### Warnings & Issues ({len(warnings)})")
            for warn in warnings[:2]:
                output.append(f"- **{warn.get('title', 'Untitled')}**")
            output.append("")

        # Discussions
        if ResultType.DISCUSSION.value in state.results_by_type:
            discussions = state.results_by_type[ResultType.DISCUSSION.value]
            output.append(f"### Community Discussions ({len(discussions)})")
            output.append(f"  {len(discussions)} community discussions available")
            output.append("")

    # Completion message
    if summary["is_complete"]:
        output.append("\n**All sources completed!** Ready for synthesis.\n")

    return "\n".join(output)


def format_final_results(
    state: AggregatedState, synthesis: Optional[Dict[str, Any]] = None
) -> str:
    """Format final complete results with synthesis."""
    output = []

    output.append("# Community Research Results\n")
    output.append(f"**Total Results:** {state.total_results}")
    output.append(f"**Sources:** {', '.join(state.sources_completed)}")
    output.append(
        f"**Search Time:** {(state.last_update - state.start_time).total_seconds():.2f}s\n"
    )

    # Show synthesis if available
    if synthesis and synthesis.get("findings"):
        output.append("## Key Findings\n")
        for i, finding in enumerate(synthesis["findings"], 1):
            output.append(f"### {i}. {finding.get('title', 'Finding')}")
            output.append(f"**Difficulty:** {finding.get('difficulty', 'Unknown')}")
            output.append(
                f"**Community Score:** {finding.get('community_score', 0)}/100\n"
            )

            output.append(f"**Problem:** {finding.get('problem', '')}\n")
            output.append(f"**Solution:** {finding.get('solution', '')}\n")

            if finding.get("gotchas"):
                output.append(
                    f"**Important Considerations:** {finding.get('gotchas')}\n"
                )

            output.append("---\n")

    # Show categorized results
    output.append("## All Results by Category\n")
    for category, items in state.results_by_type.items():
        output.append(f"### {category.replace('_', ' ').title()} ({len(items)})")
        for item in items[:5]:  # Top 5 per category
            output.append(
                f"- [{item.get('title', 'Untitled')}]({item.get('url', '#')})"
            )
        if len(items) > 5:
            output.append(f"  *... and {len(items) - 5} more*")
        output.append("")

    return "\n".join(output)
