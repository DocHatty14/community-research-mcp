"""
Quality scoring for search findings.

Assigns confidence scores (0-100) based on source authority,
community validation, recency, specificity, and evidence quality.
"""

import math
import re
from typing import Any

__all__ = ["QualityScorer", "SCORING_PRESETS"]

# ══════════════════════════════════════════════════════════════════════════════
# Scoring Presets
# ══════════════════════════════════════════════════════════════════════════════

SCORING_PRESETS: dict[str, dict[str, Any]] = {
    "balanced": {
        "weights": {
            "source": 0.22,
            "community": 0.23,
            "recency": 0.20,
            "specificity": 0.20,
            "evidence": 0.15,
        },
        "source_boost": {},
    },
    "bugfix": {
        "weights": {
            "source": 0.20,
            "community": 0.18,
            "recency": 0.17,
            "specificity": 0.25,
            "evidence": 0.20,
        },
        "source_boost": {"stackoverflow": 1.08, "github": 1.05},
    },
    "performance": {
        "weights": {
            "source": 0.18,
            "community": 0.25,
            "recency": 0.17,
            "specificity": 0.15,
            "evidence": 0.25,
        },
        "source_boost": {"github": 1.08, "hackernews": 1.05},
    },
    "migration": {
        "weights": {
            "source": 0.25,
            "community": 0.18,
            "recency": 0.25,
            "specificity": 0.17,
            "evidence": 0.15,
        },
        "source_boost": {},
    },
}

# Source authority scores
SOURCE_AUTHORITY: dict[str, int] = {
    "stackoverflow": 100,
    "github": 90,
    "discourse": 88,
    "hackernews": 85,
    "lobsters": 83,
    "reddit": 75,
    "serper": 70,
    "tavily": 70,
    "brave": 70,
    "firecrawl": 65,
}

# ══════════════════════════════════════════════════════════════════════════════
# Quality Scorer
# ══════════════════════════════════════════════════════════════════════════════


class QualityScorer:
    """
    Score findings based on multiple quality signals.

    Example:
        >>> scorer = QualityScorer("bugfix")
        >>> score = scorer.score(finding)
        >>> findings = scorer.score_batch(findings_list)
    """

    def __init__(self, preset: str = "balanced"):
        config = SCORING_PRESETS.get(preset, SCORING_PRESETS["balanced"])
        self.weights = config["weights"]
        self.source_boost = config["source_boost"]

    def score(self, finding: dict[str, Any]) -> int:
        """Calculate quality score (0-100) for a finding."""
        total = 0.0

        # Source authority
        source = finding.get("source", "").split(":")[0].lower()
        authority = SOURCE_AUTHORITY.get(source, 50)
        total += (authority / 100) * self.weights["source"] * 100

        # Community validation (votes, answers, comments)
        votes = max(0, finding.get("score", 0))
        answers = max(0, finding.get("answer_count", 0))
        comments = max(0, finding.get("comments", 0))

        validation = min(
            100,
            math.log1p(votes) * 25
            + math.log1p(answers) * 15
            + math.log1p(comments) * 10,
        )
        total += (validation / 100) * self.weights["community"] * 100

        # Recency
        age_days = max(0, finding.get("age_days", 180))
        recency = max(0, 100 - (age_days * 0.5))
        if age_days <= 14:
            recency = min(100, recency + 10)
        total += (recency / 100) * self.weights["recency"] * 100

        # Specificity (length + code blocks)
        text = finding.get("snippet", "") + finding.get("solution", "")
        code_blocks = len(re.findall(r"```|`[^`]+`", text))
        specificity = min(100, (len(text) / 12) + (code_blocks * 22))
        total += (specificity / 100) * self.weights["specificity"] * 100

        # Evidence (links, code, numbers)
        has_link = bool(finding.get("url"))
        has_code = "```" in text or "`" in text
        has_metrics = bool(re.search(r"\d+%|\d+x faster|\d+ms", text))

        evidence = min(
            100,
            (30 if has_link else 0)
            + (45 if has_code else 0)
            + (25 if has_metrics else 0),
        )
        total += (evidence / 100) * self.weights["evidence"] * 100

        # Penalties
        if not has_code:
            total -= 8
        if not has_link:
            total -= 5

        # Source boost
        total *= self.source_boost.get(source, 1.0)

        return int(min(100, max(0, total)))

    def score_batch(self, findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Score multiple findings, adding 'quality_score' field."""
        for finding in findings:
            finding["quality_score"] = self.score(finding)
        return findings
