"""
Core orchestration and LLM integration.

This module handles model selection, LLM API calls, and
intelligent orchestration of research tasks.
"""

from core.llm_clients import (
    call_anthropic,
    call_gemini,
    call_openai,
    call_openrouter,
    call_perplexity,
)
from core.orchestrator import ModelOrchestrator, get_available_llm_provider

__all__ = [
    # LLM API clients
    "call_gemini",
    "call_openai",
    "call_anthropic",
    "call_openrouter",
    "call_perplexity",
    # Orchestration
    "ModelOrchestrator",
    "get_available_llm_provider",
]
