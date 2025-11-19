# Features

The Community Research MCP is packed with advanced features designed to provide "A+" quality research results.

## 1. 🔄 Deep Research Loop
**Tool**: `deep_community_search`

Unlike standard search which runs once, the Deep Research Loop mimics a human researcher:
1.  **Initial Search**: Broad parallel search across all sources.
2.  **Gap Analysis**: The AI analyzes the initial results to identify missing information.
3.  **Targeted Follow-up**: It generates specific queries to fill those gaps.
4.  **Synthesis**: Combines all findings into a comprehensive answer.

## 2. 🌐 Active Browsing
**Feature**: Full Content Fetching

The system doesn't just rely on search snippets. It **visits the top URLs** (via `fetch_page_content`) to scrape the full text, code blocks, and documentation. This ensures the LLM has the complete context, not just a 2-line summary.

## 3. 🧠 Smart Clustering
**Feature**: Semantic Organization

Instead of a flat list of links, results are organized into **Semantic Clusters**.
- **Example Clusters**: "Official Documentation", "Community Workarounds", "Third-Party Libraries".
- **Benefit**: Helps you quickly find the *type* of solution you need.

## 4. ⚡ Parallel Streaming
**Tool**: `streaming_community_search`

Get results as they arrive. The system searches Stack Overflow, GitHub, Reddit, Hacker News, and DuckDuckGo in parallel.
- **Time to First Result**: ~0.8s
- **Total Time**: ~3-4s (vs 15s+ for sequential search)

## 5. 🛡️ Multi-Model Validation
**Feature**: Auto-Critique

When `validated_research` is used, a secondary model (e.g., Claude if Gemini is primary) reviews the primary findings. It checks for:
- **Accuracy**: Are the code examples correct?
- **Relevance**: Do they actually answer the user's question?
- **Safety**: Are there security risks?

## 6. 💾 Persistent Caching
**Feature**: Disk-based Cache

Search results are cached to `.community_research_cache.json`.
- **Benefit**: Instant results for repeated queries.
- **Persistence**: Cache survives application restarts.

## 7. 🚀 Startup Validation
**Feature**: Environment Check

On startup, the system runs `validate_environment()` to check:
- Presence of API keys (`GEMINI_API_KEY`, etc.).
- Streaming capability availability.
- Reddit API configuration.
It provides clear feedback on what features are active.
