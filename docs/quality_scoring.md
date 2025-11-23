# Quality scoring

Community Research MCP ranks findings from community sources so that high-quality, copy-paste-ready fixes float to the top while softer leads remain visible but clearly deprioritized.

## Rubric

Scores run 0–100 and are tuned to concentrate trustworthy answers near the top of the stream. The default **balanced** preset weights each signal so battle-tested fixes rise ahead of raw popularity:

- **Authority (~22%)** — Stack Overflow accepted answers, GitHub issues with maintainer replies, and reputable docs outrank anonymous posts.
- **Community validation (~23%)** — Upvotes, stars, and answer counts are normalized so widely agreed solutions rise first.
- **Recency (~20%)** — Newer fixes get a boost to avoid stale workarounds.
- **Specificity (~20%)** — Step-by-step fixes and repro details beat generic advice.
- **Evidence (~15%)** — Inline code, benchmarks, or patch snippets add proof.

Maintainer-confirmed fixes with recent activity and copy-pastable patches usually land in the 90–100 range. Lower-confidence answers stay listed for context but are intentionally demoted.

## Presets and deduplication

**Per-source weighting presets:** The scorer supports presets such as `bugfix-heavy`,
`perf-tuning`, and `migration`. Each preset redistributes the weight given to authority,
recency, specificity, and evidence, plus small source-specific biases (for example,
`bugfix-heavy` boosts Stack Overflow and GitHub slightly). Set `QUALITY_SCORER_PRESET`
in the environment to switch profiles; the default remains balanced.

**Automatic deduplication:** Results from all sources are deduplicated using normalized URLs
(protocol-stripped, query/fragments removed, `www` trimmed) and normalized titles that remove
common site suffixes. The highest-quality version of an overlapping result is retained and the
rest are dropped.

**Downranking low-evidence answers:** Findings without code blocks or reproduction details now
incur additional penalties. Evidence signals (links, code, benchmarks, repro steps) feed a
dedicated evidence score, and missing evidence can lower the overall quality score even after
other heuristics are applied.
