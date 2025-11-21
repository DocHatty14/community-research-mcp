"""Model orchestration and selection logic."""

import os
from typing import List, Optional, Tuple

from models.config import ThinkingMode


def get_available_llm_provider() -> Optional[Tuple[str, str]]:
    """Get any available LLM provider and API key."""
    providers = {
        "gemini": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }

    for provider, env_key in providers.items():
        api_key = os.getenv(env_key, "").strip()
        if api_key:
            return (provider, api_key)
    return None


class ModelOrchestrator:
    """Intelligent model selection and orchestration for research tasks."""

    def __init__(self):
        self.provider_priority = self._get_provider_priority()
        self.thinking_mode = self._get_thinking_mode()
        self.validation_enabled = self._get_validation_setting()
        self.validation_provider = self._get_validation_provider()

    def _get_provider_priority(self) -> List[str]:
        """Get provider priority from environment configuration."""
        priority_str = os.getenv(
            "PROVIDER_PRIORITY", "gemini,openai,anthropic,azure,ollama"
        )
        return [p.strip() for p in priority_str.split(",") if p.strip()]

    def _get_thinking_mode(self) -> ThinkingMode:
        """Get default thinking mode from environment."""
        mode_str = os.getenv("DEFAULT_THINKING_MODE", "balanced").lower()
        try:
            return ThinkingMode(mode_str)
        except ValueError:
            return ThinkingMode.BALANCED

    def _get_validation_setting(self) -> bool:
        """Check if multi-model validation is enabled."""
        return os.getenv("ENABLE_MULTI_MODEL_VALIDATION", "false").lower() == "true"

    def _get_validation_provider(self) -> str:
        """Get validation provider from environment."""
        return os.getenv("VALIDATION_PROVIDER", "gemini").lower()

    def select_model_for_task(
        self, task_type: str, complexity: str = "medium"
    ) -> Tuple[str, str]:
        """
        Select the best available model for a specific task.

        Args:
            task_type: Type of task ('synthesis', 'validation', 'planning', 'comparison')
            complexity: Task complexity ('low', 'medium', 'high')

        Returns:
            Tuple of (provider_name, api_key) or raises exception if none available
        """
        model_preferences = {
            "synthesis": {
                "high": ["openai", "anthropic", "gemini", "azure"],
                "medium": ["gemini", "openai", "anthropic", "azure"],
                "low": ["gemini", "openai", "azure", "anthropic"],
            },
            "validation": {
                "high": ["anthropic", "openai", "gemini", "azure"],
                "medium": ["gemini", "anthropic", "openai", "azure"],
                "low": ["gemini", "openai", "azure"],
            },
            "planning": {
                "high": ["anthropic", "openai", "gemini"],
                "medium": ["gemini", "anthropic", "openai"],
                "low": ["gemini", "openai"],
            },
            "comparison": {
                "high": ["openai", "anthropic", "gemini"],
                "medium": ["anthropic", "gemini", "openai"],
                "low": ["gemini", "openai"],
            },
        }

        preferred = model_preferences.get(task_type, {}).get(
            complexity, ["gemini", "openai", "anthropic"]
        )

        combined_priority = []
        for pref in preferred:
            if pref in self.provider_priority:
                combined_priority.append(pref)

        for user_pref in self.provider_priority:
            if user_pref not in combined_priority:
                combined_priority.append(user_pref)

        for provider in combined_priority:
            api_key = self._get_api_key_for_provider(provider)
            if api_key:
                return (provider, api_key)

        fallback_result = get_available_llm_provider()
        if fallback_result:
            return fallback_result

        raise Exception(
            "No LLM provider configured. Please set at least one API key in your .env file."
        )

    def _get_api_key_for_provider(self, provider: str) -> Optional[str]:
        """Get API key for a specific provider."""
        key_map = {
            "gemini": "GEMINI_API_KEY",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "azure": "AZURE_OPENAI_API_KEY",
            "ollama": "OLLAMA_ENDPOINT",
            "openrouter": "OPENROUTER_API_KEY",
            "perplexity": "PERPLEXITY_API_KEY",
        }

        env_key = key_map.get(provider)
        if env_key:
            value = os.getenv(env_key, "").strip()
            return value if value else None
        return None
