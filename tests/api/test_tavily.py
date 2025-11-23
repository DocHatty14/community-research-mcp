"""Comprehensive unit tests for api/tavily.py module."""

import logging
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import pytest
from api.tavily import _build_payload, search_tavily, DEFAULT_TIMEOUT


class TestBuildPayload:
    """Test suite for _build_payload function."""

    def test_build_payload_basic(self):
        """Test basic payload building."""
        result = _build_payload("test", None, 10, "key")
        assert result["api_key"] == "key"
        assert result["query"] == "test"
        assert result["max_results"] == 10
        assert result["include_answer"] is False

    def test_build_payload_with_language(self):
        """Test payload with language parameter."""
        result = _build_payload("query", "Python", 5, "key")
        assert result["query"] == "Python query"

    def test_build_payload_api_key_in_payload(self):
        """Test API key is in payload not header."""
        result = _build_payload("query", None, 10, "test_key")
        assert result["api_key"] == "test_key"


class TestSearchTavily:
    """Test suite for search_tavily async function."""

    @pytest.fixture
    def mock_env(self, monkeypatch):
        """Setup mock environment variables."""
        monkeypatch.setenv("TAVILY_API_KEY", "test_key")
        monkeypatch.setenv("TAVILY_API_URL", "https://api.tavily.com/search")

    @pytest.mark.asyncio
    async def test_search_without_api_key(self, monkeypatch, caplog):
        """Test search returns empty when API key not set."""
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        with caplog.at_level(logging.DEBUG):
            result = await search_tavily("test")
        assert result == []

    @pytest.mark.asyncio
    async def test_search_basic_success(self, mock_env):
        """Test successful search."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [{"title": "Test", "url": "https://example.com", "content": "Content"}]
        }
        mock_response.raise_for_status = MagicMock()
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
            result = await search_tavily("test")
        
        assert len(result) == 1
        assert result[0]["source"] == "tavily"

    @pytest.mark.asyncio
    async def test_search_skips_items_without_url(self, mock_env):
        """Test items without URL are skipped."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"title": "No URL"},
                {"title": "Has URL", "url": "https://example.com"}
            ]
        }
        mock_response.raise_for_status = MagicMock()
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
            result = await search_tavily("test")
        
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_search_snippet_prefers_content(self, mock_env):
        """Test snippet prefers content over snippet field."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [{"url": "https://example.com", "content": "Content", "snippet": "Snippet"}]
        }
        mock_response.raise_for_status = MagicMock()
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
            result = await search_tavily("test")
        
        assert result[0]["snippet"] == "Content"

    @pytest.mark.asyncio
    async def test_search_http_error(self, mock_env, caplog):
        """Test HTTP error handling."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.HTTPError("Error")
            )
            with caplog.at_level(logging.ERROR):
                result = await search_tavily("test")
        
        assert result == []

    @pytest.mark.asyncio
    async def test_search_passes_api_key_to_build_payload(self, mock_env):
        """Test API key passed to _build_payload."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = MagicMock()
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post
            await search_tavily("test")
            
            call_kwargs = mock_post.call_args[1]
            assert call_kwargs["json"]["api_key"] == "test_key"