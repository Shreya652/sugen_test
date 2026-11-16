"""Local retrieval and LangChain doctor/judge workflow for Test04."""

import argparse
import json
import os
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import AIMessage
from langchain_core.tools import ToolException, tool
from langchain_openai import ChatOpenAI
from langsmith import tracing_context
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer

from llm import load_dotenv
from score import norm
from test03_review import load_catalog


ROOT = Path(__file__).resolve().parent


def read_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(row)
    return rows


def validate_notes(notes, path, *, training=False):
    ids = set()
    for row in notes:
        note_id = row.get("id")
        if not isinstance(note_id, str) or not note_id or note_id in ids:
            raise ValueError(f"{path}: every note must have a unique nonempty string id")
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError(f"{path}: note {note_id} must have nonempty text")
        if training and (
            not isinstance(row.get("gold"), list)
            or not row["gold"]
            or any(not isinstance(code, str) or not norm(code) for code in row["gold"])
        ):
            raise ValueError(f"{path}: training note {note_id} must have gold codes")
        ids.add(note_id)


class LocalIndex:
    """TF-IDF retrieval over the complete MMS list and separate training notes."""

    def __init__(self, mms_path, csv_path, training_notes):
        with mms_path.open(encoding="utf-8") as source:
            rows = json.load(source)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"{mms_path}: expected a nonempty MMS category list")
        self.mms = {}
        for row in rows:
            if not isinstance(row, dict) or not norm(row.get("code")) or not isinstance(row.get("title"), str):
                raise ValueError(f"{mms_path}: invalid MMS category")
            code = norm(row["code"])
            if code in self.mms:
                raise ValueError(f"{mms_path}: duplicate MMS category {code}")
            self.mms[code] = row
        self.catalog = load_catalog(csv_path)
        self.codes = list(self.mms)
        self.positions = {code: index for index, code in enumerate(self.codes)}
        documents = []
        for code in self.codes:
            title = self.mms[code]["title"]
            entries = self.catalog.get(code, [])
            descriptions = " ".join(
                f'{entry["title"]} {entry["definition"]} {entry["text"]}'
                for entry in entries
            )
            documents.append(f"{title} {title} {descriptions}")
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=100000)
        self.matrix = self.vectorizer.fit_transform(documents)
        if not training_notes:
            raise ValueError("Training notes must not be empty")
        self.training = training_notes
        self.training_vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.training_matrix = self.training_vectorizer.fit_transform(
            [row["text"] for row in training_notes]
        )
        self.families = {}
        for code in set(self.mms) | set(self.catalog):
            family = code.split(".", 1)[0][:4]
            self.families.setdefault(family, []).append(code)
        for codes in self.families.values():
            codes.sort()

    @staticmethod
    def _query(query):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("The search text must not be empty.")
        return query

    @staticmethod
    def _rank(matrix, vectorizer, query, limit):
        scores = (matrix @ vectorizer.transform([query]).T).tocoo()
        return sorted(
            ((int(row), float(score)) for row, score in zip(scores.row, scores.data) if score > 0),
            key=lambda item: (-item[1], item[0]),
        )[:limit]

    def search(self, query, limit=8):
        self._query(query)
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 20:
            raise ValueError("Search limit must be between 1 and 20.")
        return [
            {"code": self.codes[index], "title": self.mms[self.codes[index]]["title"], "similarity": score}
            for index, score in self._rank(self.matrix, self.vectorizer, query, limit)
        ]

    def compare(self, query, codes):
        self._query(query)
        if not isinstance(codes, list) or not 1 <= len(codes) <= 20:
            raise ValueError("Provide between 1 and 20 MMS codes to compare.")
        normalized = [norm(code) for code in codes]
        invalid = [str(code) for code, normalized_code in zip(codes, normalized) if normalized_code not in self.positions]
        if invalid:
            raise ValueError(f"Not valid MMS codes: {', '.join(invalid)}")
        vector = self.vectorizer.transform([query])
        indices = [self.positions[code] for code in normalized]
        scores = (self.matrix[indices] @ vector.T).toarray().ravel()
        return [
            {"code": code, "title": self.mms[code]["title"], "similarity": float(score)}
            for code, score in zip(normalized, scores)
        ]

    def context(self, code):
        code = norm(code)
        if code not in self.mms:
            raise ValueError(f"Not a valid MMS code: {code}")
        family = code.split(".", 1)[0][:4]
        entries = []
        for related in self.families[family]:
            descriptions = self.catalog.get(related, [])
            entries.append({
                "code": related,
                "valid_mms": related in self.mms,
                "mms_title": self.mms[related]["title"] if related in self.mms else None,
                "csv_descriptions": descriptions,
            })
        return {"requested_code": code, "family": family, "entries": entries}

    def examples(self, query, limit=3):
        self._query(query)
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 5:
            raise ValueError("Training example limit must be between 1 and 5.")
        # Do not return an identical training note even when called with its text.
        matches = self._rank(self.training_matrix, self.training_vectorizer, query, len(self.training))
        return [
            {"id": self.training[index]["id"], "text": self.training[index]["text"],
             "gold_codes": self.training[index]["gold"], "similarity": score}
            for index, score in matches
            if self.training[index]["text"].strip().casefold() != query.strip().casefold()
        ][:limit]


def make_tools(index):
    @tool
    def search_icd(query: str, limit: int = 8) -> str:
        """Search valid MMS categories using local ICD title and description similarity."""
        try:
            return json.dumps(index.search(query, limit), ensure_ascii=False)
        except ValueError as error:
            raise ToolException(str(error)) from error

    @tool
    def compare_similarity(query: str, codes: list[str]) -> str:
        """Compute TF-IDF cosine similarity between a note and up to 20 MMS codes."""
        try:
            return json.dumps(index.compare(query, codes), ensure_ascii=False)
        except ValueError as error:
            raise ToolException(str(error)) from error

    @tool
    def icd_context(code: str) -> str:
        """Get the full title, definition and text for a code and its four-character family."""
        try:
            return json.dumps(index.context(code), ensure_ascii=False)
        except ValueError as error:
            raise ToolException(str(error)) from error

    @tool
    def training_examples(query: str, limit: int = 3) -> str:
        """Find similar labeled training notes as examples, never as patient-specific evidence."""
        try:
            return json.dumps(index.examples(query, limit), ensure_ascii=False)
        except ValueError as error:
            raise ToolException(str(error)) from error

    tools = [search_icd, compare_similarity, icd_context, training_examples]
    for agent_tool in tools:
        agent_tool.handle_tool_error = True
        agent_tool.handle_validation_error = True
    return tools


class DoctorResult(BaseModel):
    codes: list[str] = Field(description="Up to three ranked ICD-11 MMS codes")


class JudgeResult(BaseModel):
    approved: bool
    feedback: str = Field(description="Rationale and actionable feedback when rejected")


def create_agents(index, doctor_model, judge_model, max_model_calls=5):
    tools = make_tools(index)
    doctor_prompt = (ROOT / "prompts/test04_doctor.txt").read_text(encoding="utf-8")
    judge_prompt = (ROOT / "prompts/test04_judge.txt").read_text(encoding="utf-8")

    def create(model, prompt, schema):
        return create_agent(
            model=ChatOpenAI(
                model=model, timeout=90, max_retries=2,
                reasoning_effort="none" if model in {"gpt-6-luna", "gpt-6-sol"} else None,
            ),
            tools=tools,
            system_prompt=prompt,
            middleware=[ModelCallLimitMiddleware(run_limit=max_model_calls, exit_behavior="error")],
            response_format=ToolStrategy(schema),
        )

    return create(doctor_model, doctor_prompt, DoctorResult), create(judge_model, judge_prompt, JudgeResult)


def invoke_agent(agent, payload):
    with tracing_context(enabled=False):
        result = agent.invoke(
            {"messages": [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]},
            {"recursion_limit": 30},
        )
    if "structured_response" not in result:
        raise RuntimeError("Agent returned no structured response.")
    tool_names = [
        call["name"] for message in result["messages"]
        for call in getattr(message, "tool_calls", [])
        if call["name"] not in {"DoctorResult", "JudgeResult"}
    ]
    model_calls = sum(isinstance(message, AIMessage) for message in result["messages"])
    return result["structured_response"], tool_names, model_calls


def review_note(note_text, index, doctor, judge, max_retries=3):
    if not isinstance(max_retries, int) or isinstance(max_retries, bool) or max_retries < 0:
        raise ValueError("max_retries must be a non-negative integer")
    attempts = []
    last_eligible = []
    for number in range(1, max_retries + 2):
        previous = [{"codes": item["codes"], "feedback": item["feedback"]} for item in attempts]
        print(f"  Attempt {number}: doctor agent", flush=True)
        response, doctor_tools, doctor_calls = invoke_agent(
            doctor, {"clinical_note": note_text, "previous_attempts": previous}
        )
        print(f"  Doctor: {doctor_calls} model calls; tools: {doctor_tools}", flush=True)
        if not isinstance(response, DoctorResult):
            raise ValueError("Doctor returned an unexpected structured response.")
        proposed = [norm(code) for code in response.codes]
        eligible = []
        issues = []
        if not proposed or len(proposed) > 3:
            issues.append("Provide one to three distinct ranked MMS codes.")
        for code in proposed[:3]:
            if code not in index.mms:
                issues.append(f"{code!r} is not a valid MMS code.")
            elif code in eligible:
                issues.append(f"{code} is duplicated.")
            else:
                eligible.append(code)
        if eligible:
            last_eligible = eligible
        if issues:
            attempts.append({"attempt": number, "codes": proposed, "status": "invalid_candidates",
                             "approved": False, "feedback": " ".join(issues), "doctor_tools": doctor_tools,
                             "doctor_model_calls": doctor_calls, "judge_tools": [], "judge_model_calls": 0})
            continue
        families = []
        seen_families = set()
        for code in eligible:
            family = code.split(".", 1)[0][:4]
            if family not in seen_families:
                families.append(index.context(code))
                seen_families.add(family)
        print(f"  Attempt {number}: judge agent", flush=True)
        judgement, judge_tools, judge_calls = invoke_agent(judge, {
            "clinical_note": note_text, "candidate_codes": eligible, "icd_code_families": families,
        })
        print(f"  Judge: {judge_calls} model calls; tools: {judge_tools}", flush=True)
        if not isinstance(judgement, JudgeResult):
            raise ValueError("Judge returned an unexpected structured response.")
        if not judgement.approved and not judgement.feedback.strip():
            raise ValueError("Judge must provide feedback when rejecting candidates.")
        attempts.append({"attempt": number, "codes": eligible, "approved": judgement.approved,
                         "status": "approved" if judgement.approved else "rejected",
                         "feedback": judgement.feedback, "doctor_tools": doctor_tools,
                         "doctor_model_calls": doctor_calls, "judge_tools": judge_tools,
                         "judge_model_calls": judge_calls})
        if judgement.approved:
            return {"codes": eligible, "approved": True, "complete": len(eligible) == 3, "attempts": attempts}
    return {"codes": last_eligible, "approved": False, "complete": len(last_eligible) == 3, "attempts": attempts}


def run(notes_path, training_path, csv_path, mms_path, predictions_path, reviews_path,
        *, offline=False, doctor_model=None, judge_model=None, max_retries=3, max_model_calls=5):
    notes = read_jsonl(notes_path)
    training = read_jsonl(training_path)
    validate_notes(notes, notes_path)
    validate_notes(training, training_path, training=True)
    if not notes:
        raise ValueError(f"{notes_path}: no notes")
    if {row["id"] for row in notes} & {row["id"] for row in training}:
        raise ValueError("Training and evaluation note IDs overlap.")
    if {row["text"].strip().casefold() for row in notes} & {row["text"].strip().casefold() for row in training}:
        raise ValueError("Training and evaluation note texts overlap.")
    index = LocalIndex(mms_path, csv_path, training)
    if not offline:
        load_dotenv(ROOT / ".env")
        if not os.environ.get("OPENAI_API_KEY"):
            print("OPENAI_API_KEY is missing; using the local-only baseline.")
            offline = True
    if offline:
        print("Offline baseline: no doctor or judge agent calls will be made.")
    else:
        doctor_model = doctor_model or os.environ.get("OPENAI_MODEL", "gpt-6-luna")
        judge_model = judge_model or doctor_model
        print(f"Agent mode: doctor={doctor_model}, judge={judge_model}; notes and tool results are sent to OpenAI.")
        doctor, judge = create_agents(index, doctor_model, judge_model, max_model_calls)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    reviews_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("w", encoding="utf-8") as predictions, reviews_path.open("w", encoding="utf-8") as reviews:
        for number, note in enumerate(notes, 1):
            print(f'[{number}/{len(notes)}] {note["id"]}', flush=True)
            if offline:
                candidates = index.search(note["text"], 3)
                result = {"codes": [row["code"] for row in candidates], "approved": None,
                          "complete": len(candidates) == 3, "attempts": [], "mode": "offline"}
                if not candidates:
                    print(f'  {note["id"]}: no lexical overlap; no prediction', flush=True)
            else:
                result = review_note(note["text"], index, doctor, judge, max_retries)
                result["mode"] = "agents"
            predictions.write(json.dumps({"id": note["id"], "codes": result["codes"]}, ensure_ascii=False) + "\n")
            predictions.flush()
            reviews.write(json.dumps({"id": note["id"], **result}, ensure_ascii=False) + "\n")
            reviews.flush()
    print(f"Predictions: {predictions_path}\nReviews: {reviews_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notes", type=Path, default=ROOT / "data/dev_notes_test.jsonl")
    parser.add_argument("--training", type=Path, default=ROOT / "data/dev_notes_train.jsonl")
    parser.add_argument("--csv", type=Path, default=ROOT / "data/final-icd-codes.csv")
    parser.add_argument("--mms", type=Path, default=ROOT / "data/mms-2026-01.json")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--offline", action="store_true", help="use local retrieval without model calls")
    parser.add_argument("--doctor-model")
    parser.add_argument("--judge-model")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-model-calls", type=int, default=5)
    args = parser.parse_args()
    if args.max_retries < 0 or args.max_model_calls < 1:
        parser.error("--max-retries must be non-negative and --max-model-calls must be positive")
    run(args.notes, args.training, args.csv, args.mms, args.predictions, args.reviews,
        offline=args.offline, doctor_model=args.doctor_model, judge_model=args.judge_model,
        max_retries=args.max_retries, max_model_calls=args.max_model_calls)


if __name__ == "__main__":
    main()
