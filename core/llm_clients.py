"""LLM API client functions for Community Research MCP."""

import json
from typing import Any, Dict

import httpx


async def call_gemini(api_key: str, prompt: str) -> Dict[str, Any]:
    """Call Google Gemini API."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={api_key}"

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 4096},
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

        text = data["candidates"][0]["content"]["parts"][0]["text"]

        # Clean up markdown code blocks if present
        text = (
            text.replace("```json\n", "")
            .replace("\n```", "")
            .replace("```", "")
            .strip()
        )

        return json.loads(text)


async def call_openai(api_key: str, prompt: str) -> Dict[str, Any]:
    """Call OpenAI API."""
    url = "https://api.openai.com/v1/chat/completions"

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

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

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

        text = data["choices"][0]["message"]["content"]
        text = (
            text.replace("```json\n", "")
            .replace("\n```", "")
            .replace("```", "")
            .strip()
        )

        return json.loads(text)


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

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

        text = data["content"][0]["text"]
        text = (
            text.replace("```json\n", "")
            .replace("\n```", "")
            .replace("```", "")
            .strip()
        )

        return json.loads(text)


async def call_openrouter(api_key: str, prompt: str) -> Dict[str, Any]:
    """Call OpenRouter API."""
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    payload = {
        "model": "google/gemini-2.0-flash-exp:free",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 4096,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

        text = data["choices"][0]["message"]["content"]
        text = (
            text.replace("```json\n", "")
            .replace("\n```", "")
            .replace("```", "")
            .strip()
        )

        return json.loads(text)


async def call_perplexity(api_key: str, prompt: str) -> Dict[str, Any]:
    """Call Perplexity API."""
    url = "https://api.perplexity.ai/chat/completions"

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    payload = {
        "model": "llama-3.1-sonar-small-128k-online",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 4096,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

        text = data["choices"][0]["message"]["content"]
        text = (
            text.replace("```json\n", "")
            .replace("\n```", "")
            .replace("```", "")
            .strip()
        )

        return json.loads(text)
