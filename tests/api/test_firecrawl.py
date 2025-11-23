"""Comprehensive unit tests for api/firecrawl.py module."""

import logging
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import pytest
from api.firecrawl import _build_payload, search_firecrawl, DEFAULT_TIMEOUT


class TestBuildPayload:
    """Test suite for _build_payload function."""

    def test_build_payload_without_language(self):
        """Test payload building with query only."""
        result = _build_payload("test query", None)
        assert result == {"query": "test query"}

    def test_build_payload_with_language(self):
        """Test payload building with language prefix."""
        result = _build_payload("async await", "Python")
        assert result == {"query": "Python async await"}

    def test_build_payload_field_name_is_query(self):
        """Test that the field name is 'query' not 'q'."""
        result = _build_payload("test", None)
        assert "query" in result
        assert "q" not in result


class TestSearchFirecrawl:
    """Test suite for search_firecrawl async function."""

    @pytest.fixture
    def mock_env(self, monkeypatch):
        """Setup mock environment variables."""
        monkeypatch.setenv("FIRECRAWL_API_KEY", "test_key")
        monkeypatch.setenv("FIRECRAWL_API_URL", "https://api.firecrawl.dev/v1/search")

    @pytest.mark.asyncio
    async def test_search_without_api_key(self, monkeypatch, caplog):
        """Test search returns empty when API key not set."""
        monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
        with caplog.at_level(logging.DEBUG):
            result = await search_firecrawl("test")
        assert result == []

    @pytest.mark.asyncio
    async def test_search_basic_success(self, mock_env):
        """Test successful search with basic response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{"title": "Test", "url": "https://example.com"}]
        }
        mock_response.raise_for_status = MagicMock()
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
            result = await search_firecrawl("test")
        
        assert len(result) == 1
        assert result[0]["source"] == "firecrawl"

    @pytest.mark.asyncio
    async def test_search_bucketed_response(self, mock_env):
        """Test bucketed response structure (web/news/images)."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": {
                "web": [{"title": "Web", "url": "https://web.com"}],
                "news": [{"title": "News", "url": "https://news.com"}]
            }
        }
        mock_response.raise_for_status = MagicMock()
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
            result = await search_firecrawl("test")
        
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_search_skips_items_without_url(self, mock_env):
        """Test items without URL are skipped."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"title": "No URL"},
                {"title": "Has URL", "url": "https://example.com"}
            ]
        }
        mock_response.raise_for_status = MagicMock()
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
            result = await search_firecrawl("test")
        
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_search_markdown_fallback(self, mock_env):
        """Test markdown field is used as fallback."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{"url": "https://example.com", "markdown": "# Markdown"}]
        }
        mock_response.raise_for_status = MagicMock()
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
            result = await search_firecrawl("test")
        
        assert result[0]["snippet"] == "# Markdown"
        assert result[0]["content"] == "# Markdown"

    @pytest.mark.asyncio
    async def test_search_http_error(self, mock_env, caplog):
        """Test HTTP error handling."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.HTTPError("Error")
            )
            with caplog.at_level(logging.ERROR):
                result = await search_firecrawl("test")
        
        assert result == []