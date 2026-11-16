#!/usr/bin/env python3
"""CLI script exported from evaluate_dev_test04.ipynb to run and score Test04."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run.coder import DEFAULT_MODEL, _sanitize_ssl_env, resolve_openai_api_key  # noqa: E402
from test04_agents import run as run_test04  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("agents", "offline"), default="agents", help="Run mode (default: agents)")
    parser.add_argument("--openai-api-key", dest="openai_api_key", default=None, help="OpenAI API key (overrides env and .env)")
    parser.add_argument("--model", default=None, help=f"Default model (env OPENAI_MODEL, default {DEFAULT_MODEL})")
    parser.add_argument("--doctor-model", default=None, help="Doctor model override")
    parser.add_argument("--judge-model", default=None, help="Judge model override")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-model-calls", type=int, default=5)
    parser.add_argument("--notes", type=Path, default=PROJECT_ROOT / "data/dev_notes_test.jsonl")
    parser.add_argument("--training", type=Path, default=PROJECT_ROOT / "data/dev_notes_train.jsonl")
    parser.add_argument("--csv", type=Path, default=PROJECT_ROOT / "data/final-icd-codes.csv")
    parser.add_argument("--mms", type=Path, default=PROJECT_ROOT / "data/mms-2026-01.json")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "out")
    parser.add_argument("--dotenv", type=Path, default=PROJECT_ROOT / ".env")
    args = parser.parse_args(argv)

    offline = args.mode == "offline"
    if not offline:
        _sanitize_ssl_env()
        resolved_key = resolve_openai_api_key(args.openai_api_key, dotenv_path=args.dotenv)
        if not resolved_key:
            offline = True

    doctor_model = args.doctor_model or args.model
    judge_model = args.judge_model or args.model

    label = "offline_" if args.mode == "offline" else ""
    args.out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.out_dir / f"dev_predictions_{label}test04.jsonl"
    reviews_path = args.out_dir / f"dev_reviews_{label}test04.jsonl"
    per_note_path = args.out_dir / f"dev_per_note_{label}test04.jsonl"

    run_test04(
        args.notes,
        args.training,
        args.csv,
        args.mms,
        predictions_path,
        reviews_path,
        offline=offline,
        doctor_model=doctor_model,
        judge_model=judge_model,
        max_retries=args.max_retries,
        max_model_calls=args.max_model_calls,
    )

    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "score.py"),
            "--gold",
            str(args.notes),
            "--pred",
            str(predictions_path),
            "--mms",
            str(args.mms),
            "--per-note",
            str(per_note_path),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
