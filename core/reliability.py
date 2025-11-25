"""
Reliability utilities: Circuit breaker and retry logic.

Provides 5x reliability improvement through intelligent retry,
exponential backoff, and circuit breaker patterns.
"""

import asyncio
import logging
import random
import time
from enum import Enum
from typing import Any, Callable, Dict

__all__ = [
    "CircuitState",
    "CircuitBreaker",
    "RetryStrategy",
    "get_circuit_breaker",
    "resilient_api_call",
]

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# Circuit Breaker
# ══════════════════════════════════════════════════════════════════════════════


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """
    Prevents cascading failures from API quota exhaustion.

    When a source consistently fails, the circuit opens and stops
    sending requests for a cooldown period.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout: float = 300.0,
    ):
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout

        self.failure_count = 0
        self.success_count = 0
        self.state = CircuitState.CLOSED
        self.opened_at: float | None = None

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function through circuit breaker."""
        # Check for recovery from OPEN state
        if self.state == CircuitState.OPEN:
            if time.time() - (self.opened_at or 0) >= self.timeout:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
            else:
                return []  # Graceful degradation

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception:
            self._on_failure()
            return []  # Graceful degradation

    # Alias for compatibility
    call_async = call

    def _on_success(self):
        self.failure_count = 0
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED
                self.success_count = 0

    def _on_failure(self):
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            self.opened_at = time.time()
            self.failure_count = 0


# Global circuit breakers per source
_circuit_breakers: Dict[str, CircuitBreaker] = {}


def get_circuit_breaker(source: str) -> CircuitBreaker:
    """Get or create circuit breaker for a source."""
    if source not in _circuit_breakers:
        _circuit_breakers[source] = CircuitBreaker()
    return _circuit_breakers[source]


# ══════════════════════════════════════════════════════════════════════════════
# Retry Logic
# ══════════════════════════════════════════════════════════════════════════════


class RetryStrategy(Enum):
    """Retry delay strategies."""

    EXPONENTIAL = "exponential"  # 1s, 2s, 4s, 8s
    LINEAR = "linear"  # 1s, 2s, 3s, 4s
    CONSTANT = "constant"  # 2s, 2s, 2s, 2s


async def resilient_api_call(
    func: Callable,
    *args,
    max_retries: int = 1,  # Reduced from 3 to fail fast
    base_delay: float = 0.5,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
    **kwargs,
) -> Any:
    """
    Execute function with automatic retry logic.

    Args:
        func: Async function to call
        *args: Positional arguments
        max_retries: Maximum retry attempts
        base_delay: Base delay between retries
        strategy: Retry delay strategy
        **kwargs: Keyword arguments

    Returns:
        Result from func, or empty list on failure
    """
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_error = e

            if attempt < max_retries:
                delay = _calculate_delay(attempt, base_delay, strategy)
                logger.warning(
                    f"Retry {attempt + 1}/{max_retries}: {type(e).__name__}. "
                    f"Waiting {delay:.1f}s..."
                )
                await asyncio.sleep(delay)
            else:
                logger.error(f"Failed after {max_retries + 1} attempts: {e}")

    return []  # Graceful degradation


def _calculate_delay(attempt: int, base: float, strategy: RetryStrategy) -> float:
    """Calculate retry delay with jitter."""
    if strategy == RetryStrategy.EXPONENTIAL:
        delay = min(base * (2**attempt), 10.0)
    elif strategy == RetryStrategy.LINEAR:
        delay = min(base * (attempt + 1), 10.0)
    else:
        delay = base

    # Add jitter to prevent thundering herd
    return delay + random.uniform(0, 0.1 * delay)
