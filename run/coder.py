#!/usr/bin/env python3
"""Production ICD-11 coder entrypoint for Docker grading (`BRIEF.md`).

Usage:
    python3 run/coder.py /in/notes.jsonl /out/predictions.jsonl
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unicodedata

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402

from llm import load_dotenv  # noqa: E402
from score import norm  # noqa: E402
from test04_agents import (  # noqa: E402
    LocalIndex,
    create_agents,
    read_jsonl,
    review_note,
    validate_notes,
)

DEFAULT_MODEL = "gpt-6-luna"
TRUTHY = {"1", "true", "yes", "on"}


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).casefold()


def resolve_openai_api_key(
    cli_api_key: str | None = None,
    *,
    dotenv_path: Path | None = None,
) -> str | None:
    """Resolve OPENAI_API_KEY: CLI arg -> non-empty env var -> .env file -> None."""
    if cli_api_key is not None and cli_api_key.strip():
        resolved = cli_api_key.strip()
        os.environ["OPENAI_API_KEY"] = resolved
        return resolved

    env_value = os.environ.get("OPENAI_API_KEY", "").strip()
    if env_value:
        os.environ["OPENAI_API_KEY"] = env_value
        return env_value

    # Clear whitespace/empty placeholder before reading .env fallback
    os.environ.pop("OPENAI_API_KEY", None)
    load_dotenv(dotenv_path if dotenv_path is not None else PROJECT_ROOT / ".env")

    dotenv_value = os.environ.get("OPENAI_API_KEY", "").strip()
    if dotenv_value:
        os.environ["OPENAI_API_KEY"] = dotenv_value
        return dotenv_value

    os.environ.pop("OPENAI_API_KEY", None)
    return None


def _sanitize_ssl_env() -> None:
    """Ensure CA bundle env vars are valid across macOS, Windows, and Linux containers."""
    for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        val = os.environ.get(var, "").strip()
        if not val or not Path(val).is_file():
            os.environ.pop(var, None)
    cert_dir = os.environ.get("SSL_CERT_DIR", "").strip()
    if not cert_dir or not Path(cert_dir).is_dir():
        os.environ.pop("SSL_CERT_DIR", None)

    if sys.platform.startswith("linux") and "SSL_CERT_FILE" not in os.environ and "SSL_CERT_DIR" not in os.environ:
        linux_ca_candidates = (
            "/etc/ssl/certs/ca-certificates.crt",
            "/etc/ssl/cert.pem",
            "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem",
            "/etc/pki/tls/cert.pem",
            "/etc/ssl/ca-bundle.pem",
        )
        if not any(Path(candidate).is_file() for candidate in linux_ca_candidates):
            try:
                import certifi

                os.environ["SSL_CERT_FILE"] = certifi.where()
            except ImportError:
                pass


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


class FallbackRetriever:
    """Word + accent-stripped character n-gram TF-IDF + valid MMS residual fallback."""

    def __init__(self, index: LocalIndex):
        self.index = index
        char_docs = []
        for code in index.codes:
            title = index.mms[code]["title"]
            entries = index.catalog.get(code, [])
            extra = " ".join(f'{e["title"]} {e["definition"]}' for e in entries)
            char_docs.append(_strip_accents(f"{title} {title} {extra}"))
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True,
            max_features=120000,
        )
        self.char_matrix = self.char_vectorizer.fit_transform(char_docs)
        self.default_code = self._pick_default_code(index)

    @staticmethod
    def _pick_default_code(index: LocalIndex) -> str:
        for preferred in ("MH2Y", "MG48", "ME86"):
            if preferred in index.mms:
                return preferred
        for code, row in index.mms.items():
            if row.get("chapter") == "21" and row.get("leaf") and row.get("residual"):
                return code
        for code, row in index.mms.items():
            if row.get("leaf") and row.get("residual"):
                return code
        return index.codes[0]

    def predict(self, text: str, limit: int = 3) -> list[str]:
        cleaned = (text or "").strip()
        codes: list[str] = []
        if cleaned:
            for item in self.index.search(cleaned, limit):
                code = item["code"]
                if code not in codes:
                    codes.append(code)
            if len(codes) < limit:
                stripped = _strip_accents(cleaned)
                if stripped:
                    scores = (self.char_matrix @ self.char_vectorizer.transform([stripped]).T).tocoo()
                    ranked = sorted(
                        ((int(row), float(score)) for row, score in zip(scores.row, scores.data) if score > 0),
                        key=lambda pair: (-pair[1], pair[0]),
                    )
                    for idx, _ in ranked:
                        code = self.index.codes[idx]
                        if code not in codes:
                            codes.append(code)
                        if len(codes) >= limit:
                            break
        if not codes:
            codes.append(self.default_code)
        return codes[:limit]


def _sanitize_codes(proposed: list[str], index: LocalIndex, fallback: FallbackRetriever, text: str) -> list[str]:
    valid: list[str] = []
    for raw in proposed:
        code = norm(raw)
        if code and code in index.mms and code not in valid:
            valid.append(code)
        if len(valid) == 3:
            break
    if not valid:
        return fallback.predict(text, limit=3)
    return valid


def _read_input_notes(notes_path: Path) -> tuple[list[dict], list[dict]]:
    rows = read_jsonl(notes_path)
    notes: list[dict] = []
    deduped_rows: list[dict] = []
    seen: set[str] = set()
    for line_no, row in enumerate(rows, 1):
        note_id = row.get("id")
        if not isinstance(note_id, str) or not note_id:
            raise ValueError(f"{notes_path}:{line_no}: every note must have a nonempty string id")
        if note_id in seen:
            continue
        seen.add(note_id)
        deduped_rows.append(row)
        raw_text = row.get("text")
        notes.append({"id": note_id, "text": raw_text if isinstance(raw_text, str) else ""})
    return notes, deduped_rows


def _publish_score_summary(
    raw_input_rows: list[dict],
    reference_notes: list[dict],
    predictions_path: Path,
    mms_path: Path,
    *,
    gold_path: Path | None = None,
    per_note_path: Path | None = None,
) -> None:
    """Run score.py and print its summary whenever gold labels are available."""
    score_script = PROJECT_ROOT / "score.py"
    if not score_script.is_file():
        return

    if gold_path is not None and gold_path.is_file():
        cmd = [
            sys.executable,
            str(score_script),
            "--gold",
            str(gold_path),
            "--pred",
            str(predictions_path),
            "--mms",
            str(mms_path),
        ]
        if per_note_path is not None:
            per_note_path.parent.mkdir(parents=True, exist_ok=True)
            cmd.extend(["--per-note", str(per_note_path)])
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)
        return

    ref_by_id = {
        row["id"]: row
        for row in reference_notes
        if isinstance(row.get("id"), str) and isinstance(row.get("gold"), list) and row["gold"]
    }
    gold_rows: list[dict] = []
    for row in raw_input_rows:
        note_id = row["id"]
        if isinstance(row.get("gold"), list) and row["gold"]:
            entry = {"id": note_id, "gold": row["gold"]}
            if "slice" in row:
                entry["slice"] = row["slice"]
            gold_rows.append(entry)
        elif note_id in ref_by_id:
            ref = ref_by_id[note_id]
            entry = {"id": note_id, "gold": ref["gold"]}
            if "slice" in ref:
                entry["slice"] = ref["slice"]
            gold_rows.append(entry)

    if not gold_rows or len(gold_rows) != len(raw_input_rows):
        return

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".jsonl", delete=False) as tmp_gold:
        tmp_path = Path(tmp_gold.name)
        for entry in gold_rows:
            tmp_gold.write(json.dumps(entry, ensure_ascii=False) + "\n")

    try:
        cmd = [
            sys.executable,
            str(score_script),
            "--gold",
            str(tmp_path),
            "--pred",
            str(predictions_path),
            "--mms",
            str(mms_path),
        ]
        if per_note_path is not None:
            per_note_path.parent.mkdir(parents=True, exist_ok=True)
            cmd.extend(["--per-note", str(per_note_path)])
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)
    finally:
        tmp_path.unlink(missing_ok=True)


def run_coder(
    notes_path: Path,
    predictions_path: Path,
    *,
    training_path: Path | None = None,
    csv_path: Path | None = None,
    mms_path: Path | None = None,
    reviews_path: Path | None = None,
    gold_path: Path | None = None,
    per_note_path: Path | None = None,
    dotenv_path: Path | None = None,
    openai_api_key: str | None = None,
    offline: bool = False,
    model: str | None = None,
    doctor_model: str | None = None,
    judge_model: str | None = None,
    max_retries: int | None = None,
    max_model_calls: int | None = None,
    workers: int | None = None,
    circuit_breaker_limit: int = 3,
) -> None:
    training_path = training_path or Path(os.environ.get("TRAINING_NOTES_PATH", PROJECT_ROOT / "data/dev_notes.jsonl"))
    csv_path = csv_path or Path(os.environ.get("ICD_CSV_PATH", PROJECT_ROOT / "data/final-icd-codes.csv"))
    mms_path = mms_path or Path(os.environ.get("MMS_PATH", PROJECT_ROOT / "data/mms-2026-01.json"))

    if reviews_path is None and os.environ.get("CODER_REVIEWS_PATH", "").strip():
        reviews_path = Path(os.environ["CODER_REVIEWS_PATH"].strip())
    if gold_path is None and os.environ.get("CODER_GOLD_PATH", "").strip():
        gold_path = Path(os.environ["CODER_GOLD_PATH"].strip())
    if per_note_path is None and os.environ.get("CODER_PER_NOTE_PATH", "").strip():
        per_note_path = Path(os.environ["CODER_PER_NOTE_PATH"].strip())

    notes, raw_input_rows = _read_input_notes(notes_path)
    all_reference_notes = read_jsonl(training_path)
    validate_notes(all_reference_notes, training_path, training=True)
    training = all_reference_notes

    input_ids = {row["id"] for row in notes}
    input_texts = {row["text"].strip().casefold() for row in notes if row["text"].strip()}
    non_overlapping = [
        row
        for row in training
        if row["id"] not in input_ids and row["text"].strip().casefold() not in input_texts
    ]
    if non_overlapping:
        training = non_overlapping

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    if reviews_path is not None:
        reviews_path.parent.mkdir(parents=True, exist_ok=True)

    if not notes:
        predictions_path.write_text("", encoding="utf-8")
        if reviews_path is not None:
            reviews_path.write_text("", encoding="utf-8")
        return

    index = LocalIndex(mms_path, csv_path, training)
    fallback = FallbackRetriever(index)

    env_offline = os.environ.get("CODER_OFFLINE", "").strip().lower() in TRUTHY
    use_offline = offline or env_offline

    if not use_offline:
        resolved_key = resolve_openai_api_key(openai_api_key, dotenv_path=dotenv_path)
        if not resolved_key:
            print("OPENAI_API_KEY not found in CLI, environment, or .env; using local TF-IDF fallback.", file=sys.stderr, flush=True)
            use_offline = True

    resolved_max_retries = max_retries if max_retries is not None else _env_int("MAX_RETRIES", 3)
    resolved_max_calls = max_model_calls if max_model_calls is not None else _env_int("MAX_MODEL_CALLS", 5)
    resolved_workers = workers if workers is not None else _env_int("CODER_WORKERS", 3)
    resolved_workers = max(1, resolved_workers)

    doctor = judge = None
    if not use_offline:
        _sanitize_ssl_env()
        base_model = (
            model
            or os.environ.get("OPENAI_MODEL", "").strip()
            or DEFAULT_MODEL
        )
        resolved_doctor = (
            doctor_model
            or os.environ.get("DOCTOR_MODEL", "").strip()
            or base_model
        )
        resolved_judge = (
            judge_model
            or os.environ.get("JUDGE_MODEL", "").strip()
            or resolved_doctor
        )
        print(
            f"Coder agent mode: doctor={resolved_doctor}, judge={resolved_judge}, "
            f"max_retries={resolved_max_retries}, max_model_calls={resolved_max_calls}, workers={resolved_workers}",
            file=sys.stderr,
            flush=True,
        )
        try:
            doctor, judge = create_agents(index, resolved_doctor, resolved_judge, resolved_max_calls)
        except Exception as error:  # noqa: BLE001
            print(
                f"Agent initialization failed ({error}); switching to local TF-IDF fallback.",
                file=sys.stderr,
                flush=True,
            )
            use_offline = True
    else:
        print("Coder offline mode: using local TF-IDF + char n-gram retrieval.", file=sys.stderr, flush=True)

    state_lock = threading.Lock()
    write_lock = threading.Lock()
    consecutive_failures = 0
    breaker_tripped = use_offline

    def process_one(pos: int, note: dict) -> tuple[dict, dict]:
        nonlocal consecutive_failures, breaker_tripped
        note_id = note["id"]
        text = note["text"]

        with state_lock:
            run_offline = breaker_tripped or not text.strip()

        if run_offline:
            codes = fallback.predict(text, limit=3)
            review = {
                "id": note_id,
                "codes": codes,
                "approved": None,
                "complete": len(codes) == 3,
                "attempts": [],
                "mode": "offline",
            }
            return {"id": note_id, "codes": codes}, review

        try:
            print(f"[{pos}/{len(notes)}] {note_id}", file=sys.stderr, flush=True)
            outcome = review_note(text, index, doctor, judge, resolved_max_retries)
            codes = _sanitize_codes(outcome.get("codes") or [], index, fallback, text)
            with state_lock:
                consecutive_failures = 0
            review = {
                "id": note_id,
                **outcome,
                "codes": codes,
                "mode": "agents",
            }
            return {"id": note_id, "codes": codes}, review
        except Exception as error:  # noqa: BLE001 — never drop a note during grading
            with state_lock:
                consecutive_failures += 1
                if circuit_breaker_limit > 0 and consecutive_failures >= circuit_breaker_limit and not breaker_tripped:
                    breaker_tripped = True
                    print(
                        f"Circuit breaker tripped after {consecutive_failures} consecutive API errors "
                        f"(last: {error}); switching remaining notes to offline retrieval.",
                        file=sys.stderr,
                        flush=True,
                    )
                else:
                    print(
                        f"  {note_id}: agent call failed ({error}); falling back to local retrieval.",
                        file=sys.stderr,
                        flush=True,
                    )
            codes = fallback.predict(text, limit=3)
            review = {
                "id": note_id,
                "codes": codes,
                "approved": None,
                "complete": len(codes) == 3,
                "attempts": [],
                "mode": "offline_fallback",
                "error": str(error),
            }
            return {"id": note_id, "codes": codes}, review

    rev_handle = reviews_path.open("w", encoding="utf-8") if reviews_path is not None else None
    try:
        with predictions_path.open("w", encoding="utf-8") as pred_handle:
            def write_row(pred_row: dict, rev_row: dict) -> None:
                with write_lock:
                    pred_handle.write(json.dumps(pred_row, ensure_ascii=False) + "\n")
                    pred_handle.flush()
                    if rev_handle is not None:
                        rev_handle.write(json.dumps(rev_row, ensure_ascii=False) + "\n")
                        rev_handle.flush()

            if use_offline or resolved_workers == 1:
                for pos, note in enumerate(notes, 1):
                    pred_row, rev_row = process_one(pos, note)
                    write_row(pred_row, rev_row)
            else:
                with ThreadPoolExecutor(max_workers=resolved_workers) as pool:
                    futures = [
                        pool.submit(process_one, pos, note)
                        for pos, note in enumerate(notes, 1)
                    ]
                    for future in as_completed(futures):
                        pred_row, rev_row = future.result()
                        write_row(pred_row, rev_row)
    finally:
        if rev_handle is not None:
            rev_handle.close()

    _publish_score_summary(
        raw_input_rows,
        all_reference_notes,
        predictions_path,
        mms_path,
        gold_path=gold_path,
        per_note_path=per_note_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ICD-11 consult note coder (Test04 doctor + judge with offline fallback).")
    parser.add_argument("input", type=Path, help="Input notes JSONL path (/in/notes.jsonl)")
    parser.add_argument("output", type=Path, help="Output predictions JSONL path (/out/predictions.jsonl)")
    parser.add_argument("--openai-api-key", dest="openai_api_key", default=None, help="OpenAI API key (overrides env and .env)")
    parser.add_argument("--model", default=None, help=f"Default model for doctor & judge (env OPENAI_MODEL, default {DEFAULT_MODEL})")
    parser.add_argument("--doctor-model", default=None, help="Doctor agent model override (env DOCTOR_MODEL)")
    parser.add_argument("--judge-model", default=None, help="Judge agent model override (env JUDGE_MODEL)")
    parser.add_argument("--max-retries", type=int, default=None, help="Max retries after initial attempt (env MAX_RETRIES, default 3)")
    parser.add_argument("--max-model-calls", type=int, default=None, help="Max model calls per agent invocation (env MAX_MODEL_CALLS, default 5)")
    parser.add_argument("--workers", type=int, default=None, help="Concurrent note workers in agent mode (env CODER_WORKERS, default 3)")
    parser.add_argument("--offline", action="store_true", help="Force local TF-IDF retrieval with no API calls")
    parser.add_argument("--training", type=Path, default=None, help="Labeled reference notes JSONL (default: data/dev_notes.jsonl)")
    parser.add_argument("--csv", type=Path, default=None, help="Enriched ICD-11 descriptions CSV (default: data/final-icd-codes.csv)")
    parser.add_argument("--mms", type=Path, default=None, help="Valid MMS categories JSON (default: data/mms-2026-01.json)")
    parser.add_argument("--reviews", type=Path, default=None, help="Optional output path for per-note review JSONL")
    parser.add_argument("--gold", type=Path, default=None, help="Optional gold JSONL path for score.py summary")
    parser.add_argument("--per-note", type=Path, default=None, help="Optional per-note score output JSONL path")
    parser.add_argument("--dotenv", type=Path, default=None, help="Optional .env path fallback (default: ./.env)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_retries is not None and args.max_retries < 0:
        parser.error("--max-retries must be non-negative")
    if args.max_model_calls is not None and args.max_model_calls < 1:
        parser.error("--max-model-calls must be positive")
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be positive")

    run_coder(
        args.input,
        args.output,
        training_path=args.training,
        csv_path=args.csv,
        mms_path=args.mms,
        reviews_path=args.reviews,
        gold_path=args.gold,
        per_note_path=args.per_note,
        dotenv_path=args.dotenv,
        openai_api_key=args.openai_api_key,
        offline=args.offline,
        model=args.model,
        doctor_model=args.doctor_model,
        judge_model=args.judge_model,
        max_retries=args.max_retries,
        max_model_calls=args.max_model_calls,
        workers=args.workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
