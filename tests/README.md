# Test Suite for Community Research MCP

Comprehensive unit tests for the files changed in the current branch.

## Changed Files
- `api/firecrawl.py` - Firecrawl search integration
- `api/tavily.py` - Tavily search integration

## Setup

Install test dependencies:
```bash
pip install -r requirements-test.txt
```

## Running Tests

Run all tests:
```bash
pytest
```

Run specific test file:
```bash
pytest tests/api/test_firecrawl.py
pytest tests/api/test_tavily.py
```

Run with coverage:
```bash
pytest --cov=api --cov-report=html --cov-report=term
```

## Test Coverage

### Firecrawl Tests (`test_firecrawl.py`)
Tests for changes in the current branch:
- ✅ Field name change: `q` → `query` in payload
- ✅ Bucketed response handling (web/news/images)
- ✅ Markdown field support in fallback chains
- ✅ Try-except-else flow structure

**Test Classes:**
- `TestBuildPayload` (3 tests)
  - Payload without language
  - Payload with language prefix
  - Verify field name is 'query' not 'q'

- `TestSearchFirecrawl` (7 tests)
  - Missing API key handling
  - Basic successful search
  - Bucketed response structure
  - Skip items without URL
  - Markdown field fallback
  - HTTP error handling
  - Result structure validation

### Tavily Tests (`test_tavily.py`)
Tests for changes in the current branch:
- ✅ API key passed as parameter to `_build_payload`
- ✅ Improved error handling with try-except-else
- ✅ Empty string fallback for missing API key

**Test Classes:**
- `TestBuildPayload` (3 tests)
  - Basic payload structure
  - Language parameter handling
  - API key in payload (not header)

- `TestSearchTavily` (7 tests)
  - Missing API key handling
  - Basic successful search
  - Skip items without URL
  - Snippet prefers content field
  - HTTP error handling
  - API key passed to _build_payload
  - Result structure validation

## Key Changes Tested

### Firecrawl Changes
1. **Payload field name**: Changed from `{"q": query}` to `{"query": query}`
2. **Bucketed responses**: Handles `data` as dict with web/news/images buckets
3. **Markdown support**: Added `markdown` field to fallback chains
4. **Control flow**: Try-except-else structure with results returned in else block

### Tavily Changes
1. **API key parameter**: `_build_payload` now accepts `api_key` parameter
2. **Environment handling**: `os.getenv("TAVILY_API_KEY") or ""` for safety
3. **Control flow**: Try-except-else structure with results returned in else block

## Testing Strategy

- **Mocking**: All external HTTP calls are mocked using `unittest.mock`
- **Async Testing**: Uses `pytest-asyncio` for async function testing
- **Environment Control**: Uses `monkeypatch` for environment variables
- **Error Scenarios**: Tests various failure modes (HTTP errors, missing data)
- **Edge Cases**: Tests None values, empty strings, missing fields

## Running Individual Tests

```bash
# Test specific function
pytest tests/api/test_firecrawl.py::TestBuildPayload::test_build_payload_field_name_is_query

# Test specific class
pytest tests/api/test_tavily.py::TestSearchTavily
```