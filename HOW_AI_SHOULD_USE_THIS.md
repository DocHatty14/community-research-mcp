# How AI Assistants Should Use Community Research MCP

This guide explains how AI assistants (like Claude, GPT, etc.) should effectively use the Community Research MCP server to help users solve problems.

---

## When To Use This Server

### ✅ USE when:
- User hits a problem you can't solve after 2-3 attempts
- User asks about specific implementation details
- User needs best practices for a specific use case
- You're uncertain about the correct approach
- User asks "what do developers actually use for X?"
- The problem involves specific libraries/frameworks
- You need real-world examples with code

### ❌ DON'T USE when:
- Simple conceptual questions (explain what REST is)
- Well-documented official features
- Math or logic problems
- Creative writing
- User just wants quick code without research
- The answer is in your training data with high confidence

---

## Workflow

### Step 1: Always Start With Context

```
get_server_context()
```

This tells you:
- What languages are in the user's workspace
- What frameworks are detected
- Which LLM providers are available
- Server status and capabilities

**Use this to automatically infer the language parameter.**

### Step 2: Analyze The Problem

Before searching, determine:
1. What specific technology/library is involved?
2. What is the exact goal?
3. What has the user already tried?
4. What is their current setup?

### Step 3: Craft A Specific Query

**BAD queries:**
- `topic="settings"` → Too vague
- `topic="performance"` → Too vague
- `topic="how to debug"` → Too vague

**GOOD queries:**
- `topic="FastAPI background task queue with Redis and Celery"`
- `topic="React custom hooks for form validation with Yup schema"`
- `topic="Docker multi-stage builds to reduce Python image size"`

**Include context:**
```python
community_search(
    language="Python",  # Auto-detected from workspace
    topic="FastAPI background task queue with Redis and Celery",
    goal="process long-running tasks without blocking API requests",
    current_setup="FastAPI 0.109, SQLAlchemy, need async task processing"
)
```

### Step 4: Present Results To User

Don't just dump the raw results. Instead:

1. **Summarize findings**: "I found 3 solutions from the community..."
2. **Highlight the best approach**: "The most recommended solution is..."
3. **Show evidence**: "This has 3,542 GitHub stars and 500+ Stack Overflow mentions"
4. **Include gotchas**: "But watch out for X..."
5. **Offer to implement**: "Would you like me to implement solution #2?"

---

## Example Conversation Flow

**User**: "My FastAPI app is slow. How do I make it faster?"

**You**: "That's a broad question. Let me search the community for specific FastAPI performance optimization techniques used in production."

```python
community_search(
    language="Python",
    topic="FastAPI performance optimization production best practices",
    goal="reduce API response time and increase throughput"
)
```

**Then**: Present the top 3 findings, explain each, and ask which approach the user wants to implement.

---

## Handling Query Rejections

If the server rejects your query as too vague:

1. **Read the error message carefully** - it tells you exactly what's wrong
2. **Extract specific terms** from the user's description
3. **Reformulate with more detail**
4. **Try again**

Example:
```
# First attempt (rejected):
topic="configuration"

# Error: "Topic 'configuration' is too vague..."

# Second attempt (accepted):
topic="environment variable configuration with python-dotenv and Pydantic settings"
```

---

## Best Practices

### DO:
- Always call `get_server_context()` first
- Be as specific as possible in queries
- Include `current_setup` when known
- Present findings in a structured, easy-to-understand way
- Offer to implement the solution
- Mention community validation (stars, votes)
- Include difficulty ratings in your summary

### DON'T:
- Search for things you already know with high confidence
- Use vague terms like "settings", "performance", "best practices"
- Search without any specific library or framework mentioned
- Present raw JSON results without interpretation
- Ignore the gotchas and edge cases
- Make more than one search per problem (results are cached)

---

## Response Formats

### Use Markdown (default) when:
- Presenting to the user
- User wants human-readable results
- You need to format code and explanations

### Use JSON when:
- You need to process data programmatically
- Comparing multiple solutions
- Building a structured response

---

## Error Handling

### Rate Limit Errors
**Cause**: More than 10 requests per minute
**Solution**: Wait 60 seconds, then try again. Tell the user there's a rate limit.

### No Results Found
**Cause**: Topic too niche or search terms too specific
**Solution**: Try broader terms or suggest the user check official docs.

### No API Key Configured
**Cause**: User hasn't set up any LLM provider
**Solution**: Tell them to run `initialize.bat` again and add an API key to `.env`

---

## Advanced Usage

### Comparing Multiple Solutions

If the user asks "what's the best way to...", make ONE search with a comprehensive query:

```python
community_search(
    language="Python",
    topic="FastAPI authentication JWT OAuth2 best practices production",
    goal="implement secure user authentication",
    current_setup="FastAPI REST API, PostgreSQL database"
)
```

The LLM will synthesize multiple approaches and you can present them as options.

### Following Up

If the user asks for more details on a specific finding:
1. Use the information from the previous search (it's cached)
2. Don't make another search unless it's a completely different topic
3. Offer to implement the solution with code

---

## Example: Full Interaction

**User**: "I'm building a FastAPI app and need to handle background tasks. What should I use?"

**You**: 
```
Let me search the community for production-ready solutions...
[calls get_server_context() - detects Python, FastAPI]
[calls community_search() with specific query]

I found 3 highly-recommended solutions:

1. **Celery with Redis** (Most Popular)
   - Used by major companies, 18k GitHub stars
   - Best for: Heavy workloads, distributed systems
   - Difficulty: Medium
   - Gotcha: Adds Redis dependency, needs worker processes

2. **FastAPI BackgroundTasks**
   - Built-in, no external dependencies
   - Best for: Simple, short-running tasks
   - Difficulty: Easy
   - Gotcha: Tasks die if app restarts

3. **Dramatiq**
   - Lighter than Celery, simpler API
   - Best for: Medium complexity workloads
   - Difficulty: Easy-Medium
   - Gotcha: Smaller community than Celery

Which approach fits your use case? I can help implement any of these.
```

---

## Common Mistakes To Avoid

1. **Searching too early**: Try to solve the problem yourself first
2. **Vague queries**: Always include specific libraries/frameworks
3. **Ignoring context**: Use `get_server_context()` to detect language
4. **Over-searching**: One good search is better than multiple vague ones
5. **Dumping raw data**: Always interpret and structure the findings
6. **Ignoring evidence**: Mention GitHub stars, SO votes, community adoption
7. **Skipping gotchas**: Always include warnings and edge cases

---

## Summary

The Community Research MCP is a research assistant, not a replacement for your problem-solving abilities. Use it when:
- You're uncertain about the best approach
- The user needs specific implementation details
- Real-world validation would help the user decide
- You want to save the user from the "try 5 libraries until one works" trap

Always be specific, use context, and present findings in a helpful, structured way.
