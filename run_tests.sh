#!/bin/bash
# Quick test runner script

echo "=== Installing test dependencies ==="
pip install -q pytest pytest-asyncio pytest-mock pytest-cov

echo ""
echo "=== Running Firecrawl tests ==="
pytest tests/api/test_firecrawl.py -v

echo ""
echo "=== Running Tavily tests ==="
pytest tests/api/test_tavily.py -v

echo ""
echo "=== Running all tests with coverage ==="
pytest tests/ --cov=api --cov-report=term-missing