"""
LLM API client functions.

Unified interface for calling Gemini, OpenAI, Anthropic,
OpenRouter, and Perplexity APIs with JSON response parsing.
"""

import json
from typing import Any, Dict, Iterable, Sequence, Union

import httpx

DEFAULT_TIMEOUT = 60.0


def _clean_json_text(raw_text: str) -> str:
    """Remove common markdown fences and whitespace from model output."""

    cleaned = raw_text
    for fence in ("```json\n", "```json", "```"):
        cleaned = cleaned.replace(fence, "")
    return cleaned.strip()


def _parse_json_text(raw_text: str, provider: str) -> Dict[str, Any]:
    """Convert provider text content into a JSON payload with clear errors."""

    cleaned = _clean_json_text(raw_text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive branch
        message = f"Provider {provider} returned invalid JSON: {exc.msg}"
        raise ValueError(message) from exc


def _extract_text_field(
    data: Dict[str, Any], path: Sequence[Union[str, int]], provider: str
) -> str:
    """Safely walk a nested provider response and return a text field."""

    current: Any = data
    for key in path:
        try:
            current = current[key]
        except (KeyError, IndexError, TypeError) as exc:
            message = f"Unexpected {provider} response structure; missing {key!r}"
            raise ValueError(message) from exc

    if not isinstance(current, str):
        message = (
            f"Expected {provider} response text at {list(path)} "
            f"but received {type(current).__name__}"
        )
        raise ValueError(message)

    return current


async def _post_json(
    url: str, payload: Dict[str, Any], headers: Iterable[tuple[str, str]] | None = None
) -> Dict[str, Any]:
    """Execute a JSON POST request and return the decoded body."""

    normalized_headers = dict(headers or [])
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        response = await client.post(url, headers=normalized_headers, json=payload)
        response.raise_for_status()
        return response.json()


async def call_gemini(api_key: str, prompt: str) -> Dict[str, Any]:
    """Call Google Gemini API."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={api_key}"

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 4096},
    }

    data = await _post_json(url, payload)
    text = _extract_text_field(
        data, ("candidates", 0, "content", "parts", 0, "text"), "gemini"
    )

    return _parse_json_text(text, "gemini")


async def call_openai(api_key: str, prompt: str) -> Dict[str, Any]:
    """Call OpenAI API."""
    url = "https://api.openai.com/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": "You are a technical research assistant. Always respond with valid JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 4096,
    }

    data = await _post_json(url, payload, headers=headers.items())
    text = _extract_text_field(data, ("choices", 0, "message", "content"), "openai")
    return _parse_json_text(text, "openai")


async def call_anthropic(api_key: str, prompt: str) -> Dict[str, Any]:
    """Call Anthropic Claude API."""
    url = "https://api.anthropic.com/v1/messages"

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "claude-3-5-haiku-20241022",
        "max_tokens": 4096,
        "temperature": 0.3,
        "messages": [{"role": "user", "content": prompt}],
    }

    data = await _post_json(url, payload, headers=headers.items())
    text = _extract_text_field(data, ("content", 0, "text"), "anthropic")
    return _parse_json_text(text, "anthropic")


async def call_openrouter(api_key: str, prompt: str) -> Dict[str, Any]:
    """Call OpenRouter API."""
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "google/gemini-2.0-flash-exp:free",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 4096,
    }

    data = await _post_json(url, payload, headers=headers.items())
    text = _extract_text_field(
        data, ("choices", 0, "message", "content"), "openrouter"
    )
    return _parse_json_text(text, "openrouter")


async def call_perplexity(api_key: str, prompt: str) -> Dict[str, Any]:
    """Call Perplexity API."""
    url = "https://api.perplexity.ai/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "llama-3.1-sonar-small-128k-online",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 4096,
    }

    data = await _post_json(url, payload, headers=headers.items())
    text = _extract_text_field(
        data, ("choices", 0, "message", "content"), "perplexity"
    )
    return _parse_json_text(text, "perplexity")
