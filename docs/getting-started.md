# Getting Started

## 📥 Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-repo/community-research-mcp.git
    cd community-research-mcp
    ```

2.  **Initialize**:
    Run the initialization script to set up the environment and install dependencies.
    ```bash
    # Windows
    initialize.bat
    ```

3.  **Configure API Keys**:
    Create a `.env` file in the root directory (or use the one created by `initialize.bat`) and add your API keys:
    ```env
    GEMINI_API_KEY=your_gemini_key
    OPENAI_API_KEY=your_openai_key      # Optional
    ANTHROPIC_API_KEY=your_anthropic_key # Optional
    REDDIT_CLIENT_ID=your_reddit_id      # Optional (for authenticated Reddit search)
    REDDIT_CLIENT_SECRET=your_reddit_secret
    ```

## 🚦 Quick Start

Once installed, the MCP server will auto-detect your configuration.

### 1. Standard Search
Use `community_search` for a quick, parallel search across all sources.

```python
result = await community_search(
    language="Python",
    topic="FastAPI background tasks"
)
```

### 2. Streaming Search
Use `streaming_community_search` for real-time feedback as results arrive.

```python
result = await streaming_community_search(
    language="Python",
    topic="FastAPI background tasks"
)
```

### 3. Deep Research (Recommended for Complex Topics)
Use `deep_community_search` for a comprehensive, multi-step research session.

```python
result = await deep_community_search(
    language="Python",
    topic="Microservices architecture patterns with Kafka",
    goal="Design a scalable event-driven system"
)
```

## 🛠️ Troubleshooting

### "Streaming capabilities not available"
Ensure `streaming_capabilities.py` and `streaming_search.py` are in the same directory as `community_research_mcp.py`.

### Rate Limit Errors
If you hit rate limits (e.g., "429 Too Many Requests"), wait 60 seconds and try again. The system handles most rate limiting automatically.

### No Results Found
Try broadening your search terms. Instead of "FastAPI v0.99.1 specific error code 123", try "FastAPI background task error".
