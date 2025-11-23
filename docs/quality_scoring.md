# Quality scoring

Community Research MCP ranks findings from community sources so that high-quality, copy-paste-ready fixes float to the top while softer leads remain visible but clearly deprioritized.

## Rubric

Scores run 0–100 and are tuned to concentrate trustworthy answers near the top of the stream. The weights bias toward battle-tested fixes rather than raw popularity:

- **Authority (30%)** — Stack Overflow accepted answers, GitHub issues with maintainer replies, and reputable docs outrank anonymous posts.
- **Community validation (25%)** — Upvotes, stars, and answer counts are normalized so widely agreed solutions rise first.
- **Recency (20%)** — Newer fixes get a boost to avoid stale workarounds.
- **Specificity (15%)** — Step-by-step fixes and repro details beat generic advice.
- **Evidence (10%)** — Inline code, benchmarks, or patch snippets add proof.

Maintainer-confirmed fixes with recent activity and copy-pastable patches usually land in the 90–100 range. Lower-confidence answers stay listed for context but are intentionally demoted.

## Ongoing tuning

Planned improvements (tracked in issues) include:

- Per-source weighting presets (e.g., "bugfix-heavy", "perf-tuning", "migration")
- Automatic de-duplication across overlapping search results
- Stricter downranking of answers without repro steps or evidence

These adjustments aim to keep the default feed dominated by high-scoring, actionable items.
