import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from test04_agents import (
    DoctorResult, JudgeResult, LocalIndex, create_agents, make_tools, review_note, run,
)


class FakeToolCallingModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def tool_call(name, args, call_id):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


class Test04AgentsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.mms_path = root / "mms.json"
        self.csv_path = root / "catalog.csv"
        self.training_path = root / "training.jsonl"
        self.notes_path = root / "notes.jsonl"
        self.predictions_path = root / "predictions.jsonl"
        self.reviews_path = root / "reviews.jsonl"
        self.mms_path.write_text(json.dumps([
            {"code": "1A00", "title": "Bacterial illness"},
            {"code": "1A00.0", "title": "Bacterial illness with cough"},
            {"code": "1A00.1", "title": "Bacterial illness without cough"},
            {"code": "2B00", "title": "Arm injury"},
        ]), encoding="utf-8")
        with self.csv_path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(["code", "title", "definition", "text"])
            writer.writerow(["1A00.0", "Cough illness", "Bacterial cough", "persistent cough"])
            writer.writerow(["1A00.X", "Not in MMS", "description", "text"])
            writer.writerow(["2B00", "Arm injury", "Arm trauma", "arm injury"])
        self.training = [
            {"id": "train-1", "text": "persistent cough after bacterial illness", "gold": ["1A00.0"]},
            {"id": "train-2", "text": "injury to arm", "gold": ["2B00"]},
        ]
        self.notes = [{"id": "test-1", "text": "persistent cough with bacterial illness"}]
        self.write_rows(self.training_path, self.training)
        self.write_rows(self.notes_path, self.notes)
        self.index = LocalIndex(self.mms_path, self.csv_path, self.training)

    @staticmethod
    def write_rows(path, rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def test_search_compare_and_mms_only_context(self):
        found = self.index.search("persistent cough", 3)
        self.assertEqual(found[0]["code"], "1A00.0")
        self.assertGreater(found[0]["similarity"], 0)
        compared = self.index.compare("persistent cough", ["1a00.0", "2B00"])
        self.assertGreater(compared[0]["similarity"], compared[1]["similarity"])
        self.assertEqual(self.index.search("zzzznonexistent"), [])
        with self.assertRaisesRegex(ValueError, "Not valid MMS"):
            self.index.compare("cough", ["NOT-A-CODE"])
        family = self.index.context("1A00.1")
        entries = {entry["code"]: entry for entry in family["entries"]}
        self.assertEqual(entries["1A00.1"]["csv_descriptions"], [])
        self.assertTrue(entries["1A00.1"]["valid_mms"])
        self.assertFalse(entries["1A00.X"]["valid_mms"])

    def test_training_examples_include_labels_but_exclude_identical_note(self):
        matches = self.index.examples("bacterial illness and persistent cough")
        self.assertEqual(matches[0]["gold_codes"], ["1A00.0"])
        identical = self.index.examples(self.training[0]["text"])
        self.assertNotIn("train-1", [row["id"] for row in identical])

    def test_real_langchain_agents_call_tools_and_return_structured_results(self):
        doctor_model = FakeToolCallingModel(messages=iter([
            tool_call("search_icd", {"query": "persistent cough"}, "search"),
            tool_call("DoctorResult", {"codes": ["1A00.0"]}, "answer"),
        ]))
        judge_model = FakeToolCallingModel(messages=iter([
            tool_call("icd_context", {"code": "1A00.0"}, "context"),
            tool_call("JudgeResult", {"approved": True, "feedback": "Supported."}, "verdict"),
        ]))
        with patch("test04_agents.ChatOpenAI", side_effect=[doctor_model, judge_model]):
            doctor, judge = create_agents(self.index, "gpt-6-luna", "gpt-6-sol")
        result = review_note("persistent cough", self.index, doctor, judge)
        self.assertEqual(result["codes"], ["1A00.0"])
        self.assertTrue(result["approved"])
        self.assertFalse(result["complete"])
        self.assertEqual(result["attempts"][0]["doctor_tools"], ["search_icd"])
        self.assertEqual(result["attempts"][0]["judge_tools"], ["icd_context"])

    def test_tool_errors_are_returned_to_agent(self):
        tools = {item.name: item for item in make_tools(self.index)}
        self.assertIn("Not valid MMS", tools["compare_similarity"].invoke(
            {"query": "cough", "codes": ["invalid"]}
        ))
        self.assertIn("validation error", tools["search_icd"].invoke(
            {"query": "cough", "limit": "invalid"}
        ).lower())

    def test_rejection_feedback_and_invalid_codes_trigger_retries(self):
        def result(response):
            return {"structured_response": response, "messages": []}

        doctor = Mock()
        doctor.invoke.side_effect = [
            result(DoctorResult(codes=["invalid", "1A00.0"])),
            result(DoctorResult(codes=["1A00.0"])),
            result(DoctorResult(codes=["1A00.1"])),
        ]
        judge = Mock()
        judge.invoke.side_effect = [
            result(JudgeResult(approved=False, feedback="Consider the sibling code.")),
            result(JudgeResult(approved=True, feedback="Supported.")),
        ]
        outcome = review_note("synthetic note", self.index, doctor, judge)
        self.assertEqual(outcome["codes"], ["1A00.1"])
        self.assertEqual([attempt["status"] for attempt in outcome["attempts"]],
                         ["invalid_candidates", "rejected", "approved"])
        third_payload = json.loads(doctor.invoke.call_args_list[2].args[0]["messages"][0]["content"])
        self.assertIn("Consider the sibling code.", third_payload["previous_attempts"][1]["feedback"])
        judge_payload = json.loads(judge.invoke.call_args_list[0].args[0]["messages"][0]["content"])
        self.assertEqual(judge_payload["candidate_codes"], ["1A00.0"])
        self.assertNotIn("gold", judge_payload)
        self.assertIn("1A00.1", [entry["code"] for entry in judge_payload["icd_code_families"][0]["entries"]])

    def test_no_key_explicitly_falls_back_and_outputs_scorer_shape(self):
        with patch.dict("os.environ", {}, clear=True), patch("test04_agents.load_dotenv"):
            run(self.notes_path, self.training_path, self.csv_path, self.mms_path,
                self.predictions_path, self.reviews_path)
        prediction = json.loads(self.predictions_path.read_text(encoding="utf-8"))
        review = json.loads(self.reviews_path.read_text(encoding="utf-8"))
        self.assertEqual(set(prediction), {"id", "codes"})
        self.assertEqual(prediction["id"], "test-1")
        self.assertTrue(set(prediction["codes"]) <= set(self.index.mms))
        self.assertEqual(review["mode"], "offline")
        self.assertIsNone(review["approved"])

    def test_train_test_overlap_is_rejected(self):
        self.write_rows(self.notes_path, [{"id": "test-1", "text": self.training[0]["text"]}])
        with self.assertRaisesRegex(ValueError, "texts overlap"):
            run(self.notes_path, self.training_path, self.csv_path, self.mms_path,
                self.predictions_path, self.reviews_path, offline=True)


if __name__ == "__main__":
    unittest.main()
