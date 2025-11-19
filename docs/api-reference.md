# API Reference

## Core Tools

### `community_search`
Standard parallel search. Returns a final synthesized answer.
- **Arguments**:
    - `language` (str): Programming language context.
    - `topic` (str): The search query.
    - `goal` (str, optional): Specific objective to guide synthesis.
    - `current_setup` (str, optional): User's current environment details.

### `streaming_community_search`
Real-time streaming search. Returns progressive updates followed by a final synthesis.
- **Arguments**: Same as `community_search`.

### `deep_community_search`
**[NEW]** Recursive research loop.
- **Arguments**: Same as `community_search`.
- **Behavior**:
    1. Initial Search.
    2. Gap Analysis.
    3. Follow-up Searches.
    4. Active Browsing.
    5. Final Synthesis.

### `plan_research`
Generates a structured research plan without executing it.
- **Arguments**:
    - `query` (str): The research topic.
    - `thinking_mode` (str): "fast", "balanced", or "deep".

### `validated_research`
Performs search and then uses a second model to critique the findings.
- **Arguments**: Same as `community_search`.

## Utility Tools

### `get_system_capabilities`
Returns a JSON report of active API keys and available search providers.

### `fetch_page_content`
**[Internal/Advanced]** Fetches and extracts main text from a URL.
- **Arguments**:
    - `url` (str): Target URL.
    - `max_chars` (int): Character limit (default 12000).

### `cluster_and_rerank_results`
**[Internal]** Organizes raw search results into semantic clusters.
- **Arguments**:
    - `search_results` (dict): Raw results from providers.
    - `query` (str): Original query.
