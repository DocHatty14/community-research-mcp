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


if __name__ == "__main__":
    unittest.main()
