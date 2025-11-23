#!/usr/bin/env python3
"""
Streaming Search Module

Provides parallel search execution with real-time streaming results.
"""

import asyncio
from datetime import datetime
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

from streaming_capabilities import (
    ProgressiveAggregator,
    StreamingResult,
    summarize_content_shapes,
    format_final_results,
    format_streaming_update,
)

# ============================================================================
# Streaming Search Wrappers
# ============================================================================


async def search_with_streaming(
    search_func: Callable,
    source_name: str,
    result_queue: asyncio.Queue,
    *args,
    **kwargs,
) -> None:
    """
    Wrapper that executes a search function and streams results to queue.

    Args:
        search_func: The search function to execute
        source_name: Name of the search source (e.g., "stackoverflow")
        result_queue: Queue to send results to
        *args, **kwargs: Arguments to pass to search function
    """
    try:
        # Execute search
        results = await search_func(*args, **kwargs)

        # Send result to queue
        streaming_result = StreamingResult(
            source=source_name,
            data=results if isinstance(results, list) else [],
            timestamp=datetime.now(),
            is_final=True,
            error=None,
        )
        await result_queue.put(streaming_result)

    except Exception as e:
        # Send error to queue
        streaming_result = StreamingResult(
            source=source_name,
            data=[],
            timestamp=datetime.now(),
            is_final=True,
            error=str(e),
        )
        await result_queue.put(streaming_result)


async def parallel_streaming_search(
    search_functions: Dict[str, Callable],
    query: str,
    language: str,
    context: Optional[Any] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Execute multiple searches in parallel and yield results as they arrive.

    Args:
        search_functions: Dict of {source_name: search_function}
        query: Search query
        language: Programming language
        context: MCP context for progress reporting

    Yields:
        Progressive aggregation updates as results arrive
    """
    # Create queue for results
    result_queue = asyncio.Queue()

    # Create aggregator
    aggregator = ProgressiveAggregator(expected_sources=list(search_functions.keys()))

    # Report initial progress
    if context:
        await context.info(
            f"🚀 Starting parallel search across {len(search_functions)} sources..."
        )
        await context.report_progress(
            0, len(search_functions), "Initiating searches..."
        )

    # Launch all searches in parallel
    tasks = []
    for source_name, search_func in search_functions.items():
        # Determine arguments based on source
        if source_name == "hackernews":
            task = asyncio.create_task(
                search_with_streaming(search_func, source_name, result_queue, query)
            )
        elif source_name == "duckduckgo":
            task = asyncio.create_task(
                search_with_streaming(search_func, source_name, result_queue, query)
            )
        else:
            task = asyncio.create_task(
                search_with_streaming(
                    search_func, source_name, result_queue, query, language
                )
            )
        tasks.append(task)

    # Process results as they arrive
    completed_sources = 0
    total_sources = len(search_functions)

    while completed_sources < total_sources:
        # Wait for next result with timeout
        try:
            result = await asyncio.wait_for(result_queue.get(), timeout=35.0)
        except asyncio.TimeoutError:
            # Timeout - mark remaining as complete
            if context:
                await context.info("⏱️ Search timeout - using available results")
            break

        # Add to aggregator
        state = aggregator.add_result(result)
        completed_sources += 1

        # Report progress
        if context:
            summary = aggregator.get_smart_summary()
            if result.error:
                progress_msg = f"{result.source}: ❌ {result.error}"
                await context.info(f"⚠️ {progress_msg}")
            else:
                shape_stats = summarize_content_shapes({result.source: result.data})
                per = shape_stats.get("per_source", {}).get(result.source, {})
                shapes = per.get("shapes", {})
                shape_parts = [
                    f"{name}:{count}" for name, count in shapes.items() if count > 0
                ]
                shape_text = ", ".join(shape_parts) if shape_parts else "no text captured"
                label = per.get("label", result.source)
                progress_msg = f"{label}: {len(result.data)} items ({shape_text})"
                await context.info(f"✓ {progress_msg}")

            await context.report_progress(
                completed_sources,
                total_sources,
                f"Received {summary['total_results']} results from {completed_sources}/{total_sources} sources",
            )

        # Yield progressive update
        summary = aggregator.get_smart_summary()
        yield {
            "type": "progress",
            "state": state,
            "summary": summary,
            "formatted": format_streaming_update(state, summary),
        }

    # Wait for any remaining tasks to complete
    await asyncio.gather(*tasks, return_exceptions=True)

    # Final update
    final_summary = aggregator.get_smart_summary()

    if context:
        await context.info(
            f"✨ Search complete! {final_summary['total_results']} total results"
        )

    yield {
        "type": "complete",
        "state": aggregator.state,
        "summary": final_summary,
        "formatted": None,  # Will be formatted with synthesis
    }


# ============================================================================
# Streaming Search with Synthesis
# ============================================================================


async def streaming_search_with_synthesis(
    search_functions: Dict[str, Callable],
    synthesis_func: Callable,
    query: str,
    language: str,
    goal: Optional[str] = None,
    current_setup: Optional[str] = None,
    context: Optional[Any] = None,
) -> AsyncGenerator[str, None]:
    """
    Execute streaming search and synthesize results with LLM.

    Yields markdown-formatted updates as search progresses, then final synthesis.

    Args:
        search_functions: Dict of search functions
        synthesis_func: LLM synthesis function
        query: Search query
        language: Programming language
        goal: Optional user goal
        current_setup: Optional current setup description
        context: MCP context for progress reporting

    Yields:
        Markdown-formatted streaming updates
    """
    final_state = None

    # Stream search results
    async for update in parallel_streaming_search(
        search_functions, query, language, context
    ):
        if update["type"] == "progress":
            # Yield progressive update
            yield update["formatted"]
        elif update["type"] == "complete":
            # Store final state for synthesis
            final_state = update["state"]

    # Synthesize results
    if context:
        await context.info("🤖 Synthesizing results with LLM...")

    if final_state and final_state.total_results > 0:
        try:
            # Prepare results for synthesis
            synthesis_input = final_state.results_by_source

            # Call synthesis function
            synthesis = await synthesis_func(
                synthesis_input, query, language, goal, current_setup
            )

            # Format final results with synthesis
            final_output = format_final_results(final_state, synthesis)
            yield final_output

        except Exception as e:
            if context:
                await context.info(f"⚠️ Synthesis error: {str(e)}")

            # Yield results without synthesis
            final_output = format_final_results(final_state, None)
            yield final_output
    else:
        # No results found
        yield "\n## ❌ No Results Found\n\nTry refining your search query or using different keywords."


# ============================================================================
# Convenience Functions
# ============================================================================


async def get_all_search_results_streaming(
    search_stackoverflow_func,
    search_github_func,
    search_reddit_func,
    search_hackernews_func,
    search_duckduckgo_func,
    query: str,
    language: str,
    context: Optional[Any] = None,
    search_firecrawl_func=None,
    search_tavily_func=None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Convenience function to stream results from all default search sources.

    Yields progressive updates as results arrive.
    """
    search_functions = {
        name: func
        for name, func in {
            "stackoverflow": search_stackoverflow_func,
            "github": search_github_func,
            "reddit": search_reddit_func,
            "hackernews": search_hackernews_func,
            "duckduckgo": search_duckduckgo_func,
            "firecrawl": search_firecrawl_func,
            "tavily": search_tavily_func,
        }.items()
        if func is not None
    }

    async for update in parallel_streaming_search(
        search_functions, query, language, context
    ):
        yield update


async def collect_all_streaming_results(
    search_functions: Dict[str, Callable], query: str, language: str
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Collect all streaming results into final aggregated dict (non-streaming).

    Useful for backward compatibility with existing code.
    """
    final_results = {}

    async for update in parallel_streaming_search(search_functions, query, language):
        if update["type"] == "complete":
            final_results = update["state"].results_by_source
            break

    return final_results
