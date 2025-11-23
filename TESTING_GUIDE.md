# Testing Guide for Community Research MCP

This guide covers the newly created test suite for the changes in branch `codex/fix-firecrawl-payload-field-name`.

## Quick Start

```bash
# 1. Install test dependencies
pip install -r requirements-test.txt

# 2. Run all tests
pytest

# 3. Run with verbose output
pytest -v

# 4. Run with coverage report
pytest --cov=api --cov-report=term-missing
```

## What Was Tested

### Modified Files (git diff main..HEAD)
- `api/firecrawl.py` (27 additions, 11 deletions)
- `api/tavily.py` (27 additions, 11 deletions)

### Test Files Created