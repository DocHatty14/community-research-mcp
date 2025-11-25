# Quality Scoring

Community Research MCP ranks findings so that battle-tested, copy-paste-ready fixes rise to the top while lower-confidence results remain visible but clearly deprioritized.

## Scoring Rubric (0-100)

The default **balanced** preset weights each signal to prioritize street-smart, community-validated solutions:

| Signal | Weight | Description |
|--------|--------|-------------|
| **Source Authority** | 22% | Stack Overflow accepted answers, GitHub maintainer replies, and reputable sources outrank anonymous posts |
| **Community Validation** | 23% | Upvotes, stars, and answer counts — widely agreed solutions rise first |
| **Recency** | 20% | Newer fixes get a boost; stale workarounds are deprioritized |
| **Specificity** | 20% | Step-by-step fixes with code beat generic advice |
| **Evidence Quality** | 15% | Inline code, benchmarks, reproduction steps add proof |

### Source Authority Scores

| Source | Base Score | Rationale |
|--------|------------|-----------|
| Stack Overflow | 100 | Accepted answers, community voting, code examples |
| GitHub | 90 | Real bugs, maintainer involvement, actual fixes |
| Discourse | 88 | Framework-specific community wisdom |
| Hacker News | 85 | Experienced developer discussions |
| Lobsters | 83 | Technical depth, quality moderation |
| Reddit | 75 | Honest discussions, varies by subreddit |
| Unknown/Web | 50 | Neutral baseline |

## Scoring Presets

Set `QUALITY_SCORER_PRESET` environment variable or pass to `QualityScorer(preset="...")`:

### `balanced` (default)
General-purpose scoring for most queries.

```python
weights = {
    "source_authority": 0.22,
    "community_validation": 0.23,
    "recency": 0.20,
    "specificity": 0.20,
    "evidence_quality": 0.15,
}
```

### `bugfix-heavy`
For debugging and error resolution. Emphasizes concrete fixes and evidence.

```python
weights = {
    "source_authority": 0.20,
    "community_validation": 0.18,
    "recency": 0.17,
    "specificity": 0.25,      # Higher
    "evidence_quality": 0.20,  # Higher
}
source_bias = {"stackoverflow": 1.08, "github": 1.05}
```

### `perf-tuning`
For performance optimization queries. Prioritizes measured results.

```python
weights = {
    "source_authority": 0.18,
    "community_validation": 0.25,  # Higher (benchmarks get votes)
    "recency": 0.17,
    "specificity": 0.15,
    "evidence_quality": 0.25,      # Higher (needs numbers)
}
source_bias = {"github": 1.08, "hackernews": 1.05}
```

### `migration`
For library upgrades and breaking changes. Favors recent, authoritative guides.

```python
weights = {
    "source_authority": 0.25,  # Higher
    "community_validation": 0.18,
    "recency": 0.25,           # Higher
    "specificity": 0.17,
    "evidence_quality": 0.15,
}
```

## Scoring Details

### Community Validation Calculation

Uses logarithmic scaling to prevent outliers from dominating:

```python
vote_score = log1p(upvotes) * 25        # Diminishing returns on high votes
answers_score = log1p(answers) * 15     # Multiple answers = active discussion
comments_score = log1p(comments) * 10   # Comments often contain the real fix
validation_score = min(100, vote_score + answers_score + comments_score)
```

### Recency Calculation

```python
recency_score = max(0, 100 - (age_days * 0.5))  # Lose 1 point per 2 days
if age_days <= 14:
    recency_score += 10  # Fresh content boost
```

### Evidence Quality Signals

| Signal | Points | Detection |
|--------|--------|-----------|
| Has URL | 30 | Link to source |
| Has code | 45 | ``` or ` blocks |
| Has metrics | 25 | Patterns like "50% faster", "200ms" |
| Has repro steps | 30 | "Steps to reproduce", numbered lists |

**Penalties:**
- No code AND no repro steps: -12 points
- No URL: -5 points

## Deduplication

Results are deduplicated before scoring:

1. **URL Normalization** — Protocol stripped, `www` removed, query params/fragments removed
2. **Title Normalization** — Common suffixes removed (e.g., "- Stack Overflow")
3. **Highest Quality Retained** — When duplicates found, keep the highest-scored version

**Typical deduplication rate:** 25-30% reduction in results

## Score Interpretation

| Score Range | Interpretation |
|-------------|----------------|
| 90-100 | High confidence — maintainer-confirmed, recent, with working code |
| 70-89 | Good — community-validated with solid evidence |
| 50-69 | Moderate — useful but may need verification |
| 30-49 | Low — anecdotal or outdated, use with caution |
| 0-29 | Very low — minimal evidence, last resort |

## Usage

```python
from enhanced_mcp_utilities import QualityScorer

# Default balanced scoring
scorer = QualityScorer()

# Or with preset
scorer = QualityScorer(preset="bugfix-heavy")

# Score a single finding
score = scorer.score_finding({
    "source": "stackoverflow",
    "score": 42,  # upvotes
    "answer_count": 3,
    "snippet": "The fix is to use `async with` instead of `async for`...",
    "url": "https://stackoverflow.com/q/12345",
    "age_days": 30
})

# Score batch and sort
scored = scorer.score_findings_batch(findings)
# Returns findings with 'quality_score' field added, sorted descending
```
