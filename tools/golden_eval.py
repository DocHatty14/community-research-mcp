"""
Golden-set lint for masterclass community_search outputs.

Usage:
    python tools/golden_eval.py path/to/output.md
    cat output.md | python tools/golden_eval.py

The linter checks for required sections, evidence links with quotes,
and presence of code/commands. It is intentionally lightweight so it
can run in CI without network access.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

REQUIRED_SECTIONS = [
    "## Findings",
    "## Conflicts",
    "## Recommended Path",
    "## Quick-apply Code/Commands",
    "## Verification",
    "## Search Stats",
]


def lint_masterclass_output(text: str) -> Dict[str, Any]:
    """Return lint findings for a masterclass-style markdown output."""
    issues: List[str] = []
    metrics: Dict[str, Any] = {}

    for section in REQUIRED_SECTIONS:
        if section not in text:
            issues.append(f"Missing section: {section}")

    evidence_links = re.findall(r"https?://\\S+", text)
    if len(evidence_links) < 2:
        issues.append("Evidence weak: fewer than 2 links detected.")
    metrics["evidence_links"] = len(evidence_links)

    quotes = re.findall(r"\"[^\"]{10,200}\"", text)
    if not quotes:
        issues.append("No quoted evidence snippets found.")
    metrics["quotes"] = len(quotes)

    code_blocks = text.count("```")
    if code_blocks < 2:
        issues.append("No code/command block detected.")
    metrics["code_blocks"] = code_blocks // 2

    findings_section = re.search(r"## Findings[\\s\\S]+?##", text)
    if findings_section:
        findings_count = len(re.findall(r"\\n\\d+\\)", findings_section.group(0)))
        metrics["findings_count"] = findings_count
        if findings_count == 0:
            issues.append("No enumerated findings detected.")

    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "metrics": metrics,
    }


def main() -> int:
    if len(sys.argv) > 1:
        content = Path(sys.argv[1]).read_text(encoding="utf-8")
    else:
        content = sys.stdin.read()

    report = lint_masterclass_output(content)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
