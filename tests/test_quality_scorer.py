import unittest

from enhanced_mcp_utilities import QualityScorer, deduplicate_results


class QualityScorerTests(unittest.TestCase):
    def setUp(self):
        self.scorer = QualityScorer()

    def test_recent_evidence_scores_higher(self):
        fresh = {
            "source": "stackoverflow",
            "score": 12,
            "answer_count": 3,
            "comments": 1,
            "age_days": 7,
            "snippet": "`example` use with benchmarks showing 20% speedup.",
            "solution": "```python\nprint('hello')\n```",
            "url": "https://example.com/so",
        }

        stale = {
            "source": "reddit",
            "score": 2,
            "answer_count": 0,
            "comments": 0,
            "age_days": 365,
            "snippet": "short note that this might work",
            "solution": "",
            "url": "https://example.com/reddit",
        }

        fresh_score = self.scorer.score_finding(fresh)
        stale_score = self.scorer.score_finding(stale)

        self.assertGreater(fresh_score, stale_score)
        self.assertGreaterEqual(fresh_score - stale_score, 15)

    def test_preset_can_shift_weights(self):
        balanced = QualityScorer()
        bugfix = QualityScorer(preset="bugfix-heavy")

        finding = {
            "source": "github",
            "score": 10,
            "answer_count": 2,
            "comments": 1,
            "age_days": 30,
            "snippet": "```python\nfix = True\n```",  # code heavy
            "solution": "",
            "url": "https://example.com/repo/issues/1",
        }

        self.assertGreater(bugfix.score_finding(finding), balanced.score_finding(finding))

    def test_downranks_when_missing_repro(self):
        evidence_rich = {
            "source": "stackoverflow",
            "score": 8,
            "answer_count": 1,
            "comments": 2,
            "age_days": 40,
            "snippet": "Steps to reproduce:\n1. Run script\n2. Observe crash\n```python\nprint('x')\n```",
            "solution": "",
            "url": "https://example.com/post",
        }

        vague = {
            **evidence_rich,
            "snippet": "It might work, try again later",
            "url": "https://example.com/post-2",
        }

        self.assertGreater(self.scorer.score_finding(evidence_rich), self.scorer.score_finding(vague))


class DeduplicationTests(unittest.TestCase):
    def test_deduplicate_prefers_high_quality(self):
        search_results = {
            "stackoverflow": [
                {
                    "title": "Fix widget issue",
                    "url": "https://example.com/post",
                    "score": 5,
                    "answer_count": 2,
                    "snippet": "`code` block with exact fix",
                    "age_days": 15,
                    "source": "stackoverflow",
                }
            ],
            "reddit": [
                {
                    "title": "Fix widget issue",
                    "url": "https://example.com/post",
                    "score": 1,
                    "answer_count": 0,
                    "snippet": "Maybe try restarting",
                    "age_days": 150,
                    "source": "reddit",
                }
            ],
        }

        deduped = deduplicate_results(search_results)

        self.assertEqual(len(deduped["stackoverflow"]), 1)
        self.assertEqual(len(deduped["reddit"]), 0)
        self.assertGreater(
            deduped["stackoverflow"][0].get("quality_score", 0), 0
        )

    def test_deduplicate_overlapping_titles(self):
        search_results = {
            "stackoverflow": [
                {
                    "title": "Fix widget issue - Stack Overflow",
                    "url": "https://example.com/post",
                    "score": 6,
                    "answer_count": 1,
                    "snippet": "Detailed fix with code",
                    "age_days": 10,
                    "source": "stackoverflow",
                }
            ],
            "github": [
                {
                    "title": "Fix widget issue",
                    "url": "https://example.com/post?ref=gh",
                    "score": 3,
                    "answer_count": 0,
                    "snippet": "Partial workaround",
                    "age_days": 20,
                    "source": "github",
                }
            ],
        }

        deduped = deduplicate_results(search_results)

        self.assertEqual(len(deduped["stackoverflow"]), 1)
        self.assertFalse(deduped.get("github"))


if __name__ == "__main__":
    unittest.main()
