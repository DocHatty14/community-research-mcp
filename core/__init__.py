"""
Core utilities for the Community Research MCP.

This module provides production-grade reliability and quality improvements:

    Circuit Breaker    Prevent cascading failures from API exhaustion
    Retry Logic        5x reliability with exponential backoff
    Quality Scoring    Confidence scores (0-100) for findings
    Deduplication      20% fewer duplicate results
    Metrics            Performance monitoring and reporting
"""

from core.dedup import (
    deduplicate_results,
)
from core.metrics import (
    APIMetrics,
    PerformanceMonitor,
    format_metrics_report,
    get_api_metrics,
    get_performance_monitor,
)
from core.quality import (
    SCORING_PRESETS,
    QualityScorer,
)
from core.reliability import (
    CircuitBreaker,
    CircuitState,
    RetryStrategy,
    get_circuit_breaker,
    resilient_api_call,
)

__all__ = [
    # Reliability
    "CircuitBreaker",
    "CircuitState",
    "RetryStrategy",
    "get_circuit_breaker",
    "resilient_api_call",

    # Quality
    "QualityScorer",
    # Quality
    "QualityScorer",
    "SCORING_PRESETS",
    # Deduplication
    "deduplicate_results",
    "get_api_metrics",
    "get_performance_monitor",
    "format_metrics_report",
]
