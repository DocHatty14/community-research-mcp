"""
Performance monitoring and metrics.

Track API call performance, cache efficiency, and system health.
"""

import time
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "APIMetrics",
    "PerformanceMonitor",
    "get_api_metrics",
    "get_performance_monitor",
    "format_metrics_report",
]

# ══════════════════════════════════════════════════════════════════════════════
# Metrics Classes
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class APIMetrics:
    """Track API call statistics."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    retry_count: int = 0
    total_latency_ms: float = 0.0
    error_types: dict[str, int] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return (self.successful_calls / self.total_calls) * 100

    @property
    def avg_latency_ms(self) -> float:
        if self.successful_calls == 0:
            return 0.0
        return self.total_latency_ms / self.successful_calls

    def record_success(self, latency_ms: float):
        self.total_calls += 1
        self.successful_calls += 1
        self.total_latency_ms += latency_ms

    def record_failure(self, error_type: str):
        self.total_calls += 1
        self.failed_calls += 1
        self.error_types[error_type] = self.error_types.get(error_type, 0) + 1


@dataclass
class PerformanceMonitor:
    """Track system-wide performance."""

    start_time: float = field(default_factory=time.time)
    search_times: list[float] = field(default_factory=list)
    cache_hits: int = 0
    cache_misses: int = 0
    total_results: int = 0

    def record_search(self, duration_seconds: float, result_count: int = 0):
        self.search_times.append(duration_seconds)
        self.total_results += result_count

    def record_cache_hit(self):
        self.cache_hits += 1

    def record_cache_miss(self):
        self.cache_misses += 1

    @property
    def avg_search_time_ms(self) -> float:
        if not self.search_times:
            return 0.0
        return (sum(self.search_times) / len(self.search_times)) * 1000

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return (self.cache_hits / total) * 100

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.start_time

    def summary(self) -> dict[str, Any]:
        return {
            "uptime_seconds": round(self.uptime_seconds, 1),
            "total_searches": len(self.search_times),
            "avg_search_time_ms": round(self.avg_search_time_ms, 0),
            "cache_hit_rate": round(self.cache_hit_rate, 1),
            "total_results": self.total_results,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Global Instances
# ══════════════════════════════════════════════════════════════════════════════

_api_metrics = APIMetrics()
_perf_monitor = PerformanceMonitor()


def get_api_metrics() -> APIMetrics:
    return _api_metrics


def get_performance_monitor() -> PerformanceMonitor:
    return _perf_monitor


def format_metrics_report() -> str:
    """Generate human-readable metrics report."""
    perf = _perf_monitor
    api = _api_metrics

    lines = [
        "# Performance Metrics",
        "",
        "## System",
        f"- Uptime: {perf.uptime_seconds:.0f}s",
        f"- Total Searches: {len(perf.search_times)}",
        f"- Avg Search Time: {perf.avg_search_time_ms:.0f}ms",
        f"- Cache Hit Rate: {perf.cache_hit_rate:.1f}%",
        "",
        "## API Reliability",
        f"- Success Rate: {api.success_rate:.1f}%",
        f"- Total Calls: {api.total_calls}",
        f"- Failed: {api.failed_calls}",
        f"- Retries: {api.retry_count}",
        f"- Avg Latency: {api.avg_latency_ms:.0f}ms",
    ]

    if api.error_types:
        lines.append("")
        lines.append("## Errors")
        for err, count in sorted(api.error_types.items(), key=lambda x: -x[1]):
            lines.append(f"- {err}: {count}")

    return "\n".join(lines)
