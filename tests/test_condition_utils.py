import math
import unittest

from condition_utils import existing_matches, match_conditions, rank_codes


class ConditionSimilarityTests(unittest.TestCase):
    def test_cosine_ranks_and_limits_matches(self):
        rows = [
            ("A", {"cough", "injury"}),
            ("B", {"cough"}),
            ("C", {"cough", "fever"}),
            ("D", {"cough", "pain"}),
            ("E", {"fever"}),
        ]
        ranked = rank_codes({"cough"}, rows)
        self.assertEqual([code for code, _ in ranked], ["B", "A", "C"])
        self.assertEqual(ranked[0][1], 1.0)
        self.assertAlmostEqual(ranked[1][1], 1 / math.sqrt(2))

    def test_no_overlap_returns_no_codes(self):
        rows = [("A", {"cough"}), ("B", set())]
        self.assertEqual(rank_codes(set(), rows), [])
        self.assertEqual(rank_codes({"fever"}, rows), [])

    def test_synonyms_and_unmatched_conditions(self):
        conditions = ["coughing", "fever", "cough"]
        response = {
            "mappings": [
                {"condition": "coughing", "column": "cough"},
                {"condition": "fever", "column": None},
                {"condition": "cough", "column": None},
            ]
        }
        mappings = match_conditions(conditions, ["cough"], "prompt", lambda *_: response)
        matched, unmatched = existing_matches(conditions, mappings, ["cough"])
        self.assertEqual(matched, {"cough"})
        self.assertEqual(unmatched, ["fever"])

    def test_rejects_hallucinated_column(self):
        response = {"mappings": [{"condition": "cough", "column": "invented"}]}
        with self.assertRaisesRegex(ValueError, "unknown existing column"):
            match_conditions(["cough"], ["cough"], "prompt", lambda *_: response)


if __name__ == "__main__":
    unittest.main()
