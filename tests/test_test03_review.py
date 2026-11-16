import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from test03_review import load_catalog, related_catalog_entries, review_note


class Test03ReviewTests(unittest.TestCase):
    def setUp(self):
        self.catalog = {
            code: [{"code": code, "title": f"Title {code}", "definition": "D", "text": "T"}]
            for code in ("A", "B", "C", "D")
        }
        self.valid_codes = {"A", "B", "C", "D", "E"}

    def review(self, doctor, judge, max_retries=3):
        return review_note(
            "A synthetic clinical note",
            self.catalog,
            self.valid_codes,
            "doctor prompt",
            "judge prompt",
            doctor,
            judge,
            max_retries,
        )

    def test_batch_approval_preserves_ranked_codes_and_excludes_gold(self):
        doctor = Mock(return_value={"codes": ["C", "A", "B"]})
        judge = Mock(return_value={"approved": True, "feedback": ""})
        result = self.review(doctor, judge)
        self.assertEqual(result["codes"], ["C", "A", "B"])
        self.assertTrue(result["approved"])
        self.assertTrue(result["complete"])
        self.assertEqual(len(result["attempts"]), 1)
        judge_input = json.loads(judge.call_args.args[1])
        self.assertEqual(judge_input["candidate_codes"], ["C", "A", "B"])
        self.assertEqual([family["stem_prefix"] for family in judge_input["icd_code_families"]], ["C", "A", "B"])
        self.assertNotIn("gold", judge_input)
        self.assertEqual(json.loads(doctor.call_args.args[1])["previous_attempts"], [])

    def test_batch_rejection_feedback_reaches_the_next_attempt(self):
        doctor = Mock(side_effect=[{"codes": ["A", "B", "C"]}, {"codes": ["B", "C", "A"]}])
        judge = Mock(side_effect=[
            {"approved": False, "feedback": "Reconsider the ranking."},
            {"approved": True, "feedback": ""},
        ])
        result = self.review(doctor, judge)
        previous = json.loads(doctor.call_args_list[1].args[1])["previous_attempts"]
        self.assertEqual(previous, [{"codes": ["A", "B", "C"], "feedback": "Reconsider the ranking."}])
        self.assertEqual(result["codes"], ["B", "C", "A"])
        self.assertTrue(result["approved"])
        self.assertEqual(judge.call_count, 2)

    def test_judge_gets_complete_families_once_and_validity_flags(self):
        catalog = {
            "1F40": [{"code": "1F40", "title": "Parent", "definition": "D", "text": "T"}],
            "1F40.0": [
                {"code": "1F40.0", "title": "Child 1", "definition": "D", "text": "T"},
                {"code": "1F40.0", "title": "Child 1 duplicate", "definition": "D", "text": "T"},
            ],
            "1F40.1": [{"code": "1F40.1", "title": "Child 2", "definition": "D", "text": "T"}],
            "1F40.Z": [{"code": "1F40.Z", "title": "Other", "definition": "D", "text": "T"}],
            "1F401": [{"code": "1F401", "title": "Same prefix", "definition": "D", "text": "T"}],
            "1F41.0": [{"code": "1F41.0", "title": "Other family", "definition": "D", "text": "T"}],
        }
        valid = {"1F40", "1F40.0", "1F40.1", "1F401", "1F41.0"}
        doctor = Mock(return_value={"codes": ["1F40.1", "1F40.0", "1F41.0"]})
        judge = Mock(return_value={"approved": True, "feedback": "Reviewed."})
        result = review_note("synthetic note", catalog, valid, "doctor", "judge", doctor, judge)
        judge_input = json.loads(judge.call_args.args[1])
        families = judge_input["icd_code_families"]
        self.assertEqual([family["stem_prefix"] for family in families], ["1F40", "1F41"])
        relatives = families[0]["entries"]
        self.assertCountEqual(
            [entry["normalized_code"] for entry in relatives],
            ["1F40", "1F40.0", "1F40.0", "1F40.1", "1F40.Z", "1F401"],
        )
        self.assertFalse(next(entry for entry in relatives if entry["normalized_code"] == "1F40.Z")["valid_mms"])
        self.assertTrue(all(entry["valid_mms"] for entry in relatives if entry["normalized_code"] != "1F40.Z"))
        self.assertEqual(result["codes"], ["1F40.1", "1F40.0", "1F41.0"])

    def test_short_stems_do_not_match_longer_unrelated_prefixes(self):
        catalog = {
            "1A": [{"code": "1A"}],
            "1A.0": [{"code": "1A.0"}],
            "1A00": [{"code": "1A00"}],
        }
        relatives = related_catalog_entries("1A", catalog, {"1A", "1A.0", "1A00"})
        self.assertEqual([entry["normalized_code"] for entry in relatives], ["1A.0"])

    def test_three_retries_keep_last_complete_batch_unapproved(self):
        doctor = Mock(return_value={"codes": ["A", "B", "C"]})
        judge = Mock(return_value={"approved": False, "feedback": "Not supported."})
        result = self.review(doctor, judge)
        self.assertEqual(result["codes"], ["A", "B", "C"])
        self.assertFalse(result["approved"])
        self.assertTrue(result["complete"])
        self.assertEqual(doctor.call_count, 4)
        self.assertEqual(judge.call_count, 4)

    def test_invalid_missing_and_duplicate_codes_retry_without_judge(self):
        doctor = Mock(side_effect=[
            {"codes": ["INVALID", "A", "B"]},
            {"codes": ["E", "A", "B"]},
            {"codes": ["A", "A", "B"]},
            {"codes": ["A", "B", "C"]},
        ])
        judge = Mock(return_value={"approved": True, "feedback": ""})
        result = self.review(doctor, judge)
        self.assertEqual([attempt["status"] for attempt in result["attempts"]],
                         ["invalid_candidates", "invalid_candidates", "invalid_candidates", "approved"])
        self.assertEqual(judge.call_count, 1)
        feedback = json.loads(doctor.call_args_list[3].args[1])["previous_attempts"]
        self.assertIn("MMS", feedback[0]["feedback"])
        self.assertIn("CSV", feedback[1]["feedback"])
        self.assertIn("duplicates", feedback[2]["feedback"])
        self.assertEqual(result["codes"], ["A", "B", "C"])

    def test_all_ineligible_codes_return_no_prediction(self):
        doctor = Mock(return_value={"codes": ["E", "INVALID", "INVALID"]})
        judge = Mock()
        result = self.review(doctor, judge)
        self.assertEqual(result["codes"], [])
        self.assertFalse(result["approved"])
        self.assertFalse(result["complete"])
        self.assertEqual(len(result["attempts"]), 4)
        judge.assert_not_called()

    def test_last_partial_batch_is_returned_as_incomplete(self):
        doctor = Mock(side_effect=[
            {"codes": ["A", "B", "C"]},
            {"codes": ["A", "E", "B"]},
            {"codes": ["INVALID", "INVALID", "INVALID"]},
            {"codes": ["B", "E", "INVALID"]},
        ])
        judge = Mock(return_value={"approved": False, "feedback": "Not supported."})
        result = self.review(doctor, judge)
        self.assertEqual(result["codes"], ["B"])
        self.assertFalse(result["approved"])
        self.assertFalse(result["complete"])
        self.assertEqual(judge.call_count, 1)

    def test_duplicates_are_checked_after_normalization(self):
        doctor = Mock(return_value={"codes": ["a", "A", "B"]})
        judge = Mock()
        result = self.review(doctor, judge, max_retries=0)
        self.assertEqual(result["codes"], ["A", "B"])
        self.assertFalse(result["complete"])
        self.assertIn("duplicates", result["attempts"][0]["feedback"])
        judge.assert_not_called()

    def test_fewer_than_three_codes_remain_available_after_retries(self):
        doctor = Mock(return_value={"codes": ["A", "B"]})
        judge = Mock()
        result = self.review(doctor, judge)
        self.assertEqual(result["codes"], ["A", "B"])
        self.assertFalse(result["approved"])
        self.assertFalse(result["complete"])
        self.assertEqual(doctor.call_count, 4)
        judge.assert_not_called()

    def test_malformed_response_and_empty_rejection_are_errors(self):
        with self.assertRaisesRegex(ValueError, "codes"):
            self.review(Mock(return_value={"codes": "A"}), Mock())
        doctor = Mock(return_value={"codes": ["A", "B", "C"]})
        judge = Mock(return_value={"approved": False, "feedback": ""})
        with self.assertRaisesRegex(ValueError, "feedback"):
            self.review(doctor, judge)

    def test_catalog_retains_duplicate_matches_and_ignores_blank_codes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "icd.csv"
            with path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["code", "title", "definition", "text"])
                writer.writerow(["A", "First", "Definition", "Text"])
                writer.writerow(["A", "Second", "Definition", "Text"])
                writer.writerow(["", "No code", "", ""])
            catalog = load_catalog(path)
        self.assertEqual([row["title"] for row in catalog["A"]], ["First", "Second"])
        self.assertEqual(len(catalog), 1)


if __name__ == "__main__":
    unittest.main()
