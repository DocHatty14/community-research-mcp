#!/usr/bin/env python3
"""
Test suite for streaming search functionality
"""

import asyncio
import os

from streaming_capabilities import (
    ProgressiveAggregator,
    ResultType,
    StreamingResult,
    classify_result,
    detect_all_capabilities,
    format_capabilities_report,
    organize_by_type,
)


def test_capability_detection():
    """Test automatic capability detection."""
    print("\n=== Test 1: Capability Detection ===")

    capabilities = detect_all_capabilities()

    print(
        f"Search APIs detected: {len([k for k, v in capabilities.search_apis.items() if v])}"
    )
    print(
        f"LLM providers detected: {len([k for k, v in capabilities.llm_providers.items() if v])}"
    )

    # Always available sources
    assert capabilities.search_apis["stackoverflow"] == True
    assert capabilities.search_apis["github"] == True
    assert capabilities.search_apis["reddit"] == True
    assert capabilities.search_apis["hackernews"] == True
    assert capabilities.search_apis["duckduckgo"] == True

    print("✓ Capability detection working")
    return capabilities


def test_capability_report(capabilities):
    """Test capability report formatting."""
    print("\n=== Test 2: Capability Report ===")

    report = format_capabilities_report(capabilities)

    assert "System Capabilities" in report
    assert "Search APIs" in report
    assert "LLM Providers" in report
    assert "stackoverflow" in report

    print("✓ Report formatting working")
    print(f"\nSample output:\n{report[:300]}...")


def test_result_classification():
    """Test result type classification."""
    print("\n=== Test 3: Result Classification ===")

    # Stack Overflow with accepted answer
    so_result = {"is_answered": True, "title": "How to use async/await"}
    assert classify_result(so_result, "stackoverflow") == ResultType.QUICK_FIX

    # GitHub result
    gh_result = {"title": "Example repo"}
    assert classify_result(gh_result, "github") == ResultType.CODE_EXAMPLE

    # Reddit warning
    reddit_warning = {"title": "Warning: this will break in production"}
    assert classify_result(reddit_warning, "reddit") == ResultType.WARNING

    # Reddit tutorial
    reddit_tutorial = {"title": "Tutorial: how to setup FastAPI"}
    assert classify_result(reddit_tutorial, "reddit") == ResultType.TUTORIAL

    print("✓ Result classification working")


def test_result_organization():
    """Test organizing results by type."""
    print("\n=== Test 4: Result Organization ===")

    test_results = {
        "stackoverflow": [
            {"title": "Quick fix", "is_answered": True, "url": "http://example.com/1"},
            {
                "title": "Discussion",
                "is_answered": False,
                "url": "http://example.com/2",
            },
        ],
        "github": [
            {"title": "Code example", "url": "http://example.com/3"},
        ],
        "reddit": [
            {"title": "Warning: gotcha ahead", "url": "http://example.com/4"},
            {"title": "Tutorial: step by step", "url": "http://example.com/5"},
        ],
    }

    organized = organize_by_type(test_results)

    assert ResultType.QUICK_FIX.value in organized
    assert ResultType.CODE_EXAMPLE.value in organized
    assert ResultType.WARNING.value in organized
    assert ResultType.TUTORIAL.value in organized

    assert len(organized[ResultType.QUICK_FIX.value]) == 1
    assert len(organized[ResultType.CODE_EXAMPLE.value]) == 1
    assert len(organized[ResultType.WARNING.value]) == 1
    assert len(organized[ResultType.TUTORIAL.value]) == 1

    print("✓ Result organization working")
    print(f"Categories created: {list(organized.keys())}")


def test_progressive_aggregator():
    """Test progressive result aggregation."""
    print("\n=== Test 5: Progressive Aggregator ===")

    aggregator = ProgressiveAggregator()

    # Simulate results arriving
    from datetime import datetime

    # First result: Stack Overflow
    result1 = StreamingResult(
        source="stackoverflow",
        data=[
            {"title": "Answer 1", "is_answered": True},
            {"title": "Answer 2", "is_answered": True},
        ],
        timestamp=datetime.now(),
        is_final=True,
    )

    state1 = aggregator.add_result(result1)
    summary1 = aggregator.get_smart_summary()

    assert summary1["total_results"] == 2
    assert summary1["sources_completed"] == 1
    assert len(summary1["sources_pending"]) == 3

    print(
        f"After 1 source: {summary1['total_results']} results, {summary1['sources_completed']}/4 complete"
    )

    # Second result: GitHub
    result2 = StreamingResult(
        source="github",
        data=[
            {"title": "Repo 1"},
            {"title": "Repo 2"},
            {"title": "Repo 3"},
        ],
        timestamp=datetime.now(),
        is_final=True,
    )

    state2 = aggregator.add_result(result2)
    summary2 = aggregator.get_smart_summary()

    assert summary2["total_results"] == 5
    assert summary2["sources_completed"] == 2

    print(
        f"After 2 sources: {summary2['total_results']} results, {summary2['sources_completed']}/4 complete"
    )

    # Third result: Reddit
    result3 = StreamingResult(
        source="reddit",
        data=[
            {"title": "Warning: gotcha"},
            {"title": "Discussion thread"},
        ],
        timestamp=datetime.now(),
        is_final=True,
    )

    state3 = aggregator.add_result(result3)
    summary3 = aggregator.get_smart_summary()

    assert summary3["total_results"] == 7
    assert summary3["sources_completed"] == 3
    assert summary3["is_complete"] == False

    print(
        f"After 3 sources: {summary3['total_results']} results, {summary3['sources_completed']}/4 complete"
    )

    # Fourth result: Hacker News
    result4 = StreamingResult(
        source="hackernews",
        data=[{"title": "HN discussion"}],
        timestamp=datetime.now(),
        is_final=True,
    )

    state4 = aggregator.add_result(result4)
    summary4 = aggregator.get_smart_summary()

    assert summary4["total_results"] == 8
    assert summary4["sources_completed"] == 4
    assert summary4["is_complete"] == True

    print(
        f"After 4 sources: {summary4['total_results']} results, {summary4['sources_completed']}/4 complete"
    )
    print("✓ Progressive aggregation working")

    # Test organization by type
    assert len(state4.results_by_type) > 0
    print(f"Result types: {list(state4.results_by_type.keys())}")


def test_error_handling():
    """Test error handling in streaming."""
    print("\n=== Test 6: Error Handling ===")

    aggregator = ProgressiveAggregator()
    from datetime import datetime

    # Simulate error result
    error_result = StreamingResult(
        source="stackoverflow",
        data=[],
        timestamp=datetime.now(),
        is_final=True,
        error="API rate limit exceeded",
    )

    state = aggregator.add_result(error_result)

    assert "stackoverflow" in aggregator.state.sources_completed
    assert state.total_results == 0

    print("✓ Error handling working")


async def test_streaming_formatting():
    """Test streaming output formatting."""
    print("\n=== Test 7: Streaming Format Output ===")

    from streaming_capabilities import format_final_results, format_streaming_update

    aggregator = ProgressiveAggregator()
    from datetime import datetime

    # Add some results
    result = StreamingResult(
        source="stackoverflow",
        data=[
            {
                "title": "FastAPI async best practices",
                "is_answered": True,
                "score": 42,
                "url": "http://example.com/1",
            },
            {
                "title": "Redis with FastAPI",
                "is_answered": True,
                "score": 38,
                "url": "http://example.com/2",
            },
        ],
        timestamp=datetime.now(),
        is_final=True,
    )

    state = aggregator.add_result(result)
    summary = aggregator.get_smart_summary()

    # Test progressive formatting
    formatted = format_streaming_update(state, summary)

    assert "Search Progress" in formatted
    assert "Results:" in formatted
    assert "Elapsed:" in formatted

    print("✓ Streaming format working")
    print(f"\nSample output:\n{formatted[:400]}...")

    # Test final formatting
    synthesis = {
        "findings": [
            {
                "title": "Use async/await for I/O operations",
                "difficulty": "Easy",
                "community_score": 85,
                "problem": "Blocking I/O slows down API",
                "solution": "Use async def and await",
                "gotchas": "Must await all async calls",
            }
        ]
    }

    final = format_final_results(state, synthesis)

    assert "Community Research Results" in final
    assert "Key Findings" in final
    assert "Use async/await" in final

    print("✓ Final format working")


def main():
    """Run all tests."""
    print("=" * 60)
    print("STREAMING SEARCH TEST SUITE")
    print("=" * 60)

    try:
        # Run synchronous tests
        capabilities = test_capability_detection()
        test_capability_report(capabilities)
        test_result_classification()
        test_result_organization()
        test_progressive_aggregator()
        test_error_handling()

        # Run async test
        asyncio.run(test_streaming_formatting())

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nStreaming search functionality is ready to use!")
        print("\nNext steps:")
        print("1. Start the MCP server: python community_research_mcp.py")
        print("2. Call get_system_capabilities() to see what's available")
        print("3. Try streaming_community_search() with your query")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise


if __name__ == "__main__":
    main()
