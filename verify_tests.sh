#!/bin/bash
echo "================================"
echo "Test Suite Verification"
echo "================================"
echo ""

echo "1. Checking test file structure..."
if [ -f "tests/api/test_firecrawl.py" ] && [ -f "tests/api/test_tavily.py" ]; then
    echo "   ✅ Test files exist"
else
    echo "   ❌ Test files missing"
    exit 1
fi

echo ""
echo "2. Checking configuration files..."
if [ -f "pytest.ini" ] && [ -f "requirements-test.txt" ]; then
    echo "   ✅ Configuration files exist"
else
    echo "   ❌ Configuration files missing"
    exit 1
fi

echo ""
echo "3. Checking Python syntax..."
python -m py_compile tests/api/test_firecrawl.py 2>/dev/null && echo "   ✅ test_firecrawl.py syntax OK" || echo "   ❌ test_firecrawl.py syntax error"
python -m py_compile tests/api/test_tavily.py 2>/dev/null && echo "   ✅ test_tavily.py syntax OK" || echo "   ❌ test_tavily.py syntax error"

echo ""
echo "4. File statistics..."
echo "   - Test files: $(find tests -name 'test_*.py' | wc -l)"
echo "   - Total test lines: $(cat tests/api/test_*.py | wc -l)"
echo "   - Documentation files: $(ls -1 *.md 2>/dev/null | wc -l)"

echo ""
echo "================================"
echo "✅ Verification Complete!"
echo "================================"
echo ""
echo "To run tests:"
echo "  1. pip install -r requirements-test.txt"
echo "  2. pytest -v"
echo ""