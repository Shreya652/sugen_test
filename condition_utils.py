"""Shared condition extraction and matching helpers."""

import json
import math
import subprocess


WORKER = """
import json
import os
import sys
from pathlib import Path
import llm

request = json.load(sys.stdin)
llm.load_dotenv(Path('.env'))
provider = request['provider']
key_name = f'{provider.upper()}_API_KEY'
api_key = os.environ.get(key_name)
if not api_key:
    raise SystemExit(f'{key_name} is not set in the environment or project .env')
model = request['model'] or os.environ.get(f'{provider.upper()}_MODEL', llm.DEFAULT_MODELS[provider])
result = llm.request_json(request['prompt'], request['input'], model, api_key, provider)
json.dump(result, sys.stdout)
"""


def call_model(prompt, input_text, *, root, provider, model, api_python):
    request = {
        "prompt": prompt,
        "input": input_text,
        "provider": provider,
        "model": model,
    }
    completed = subprocess.run(
        [api_python, "-c", WORKER],
        cwd=root,
        input=json.dumps(request),
        text=True,
        capture_output=True,
    )
    if completed.returncode:
        raise RuntimeError(f"Model request failed: {completed.stderr.strip()}")
    result = json.loads(completed.stdout)
    if not isinstance(result, dict):
        raise ValueError("Model response must be a JSON object.")
    return result


def extract_conditions(note_text, prompt, call):
    conditions = call(prompt, note_text).get("conditions")
    if not isinstance(conditions, list) or any(
        not isinstance(condition, str) or not condition.strip()
        for condition in conditions
    ):
        raise ValueError("Extraction response must contain a list of non-empty condition strings.")
    unique = {}
    for condition in conditions:
        condition = condition.strip().lower()
        unique.setdefault(condition.casefold(), condition)
    return list(unique.values())


def match_conditions(conditions, columns, prompt, call):
    response = call(
        prompt, json.dumps({"conditions": conditions, "existing_columns": columns})
    )
    mappings = response.get("mappings")
    if not isinstance(mappings, list):
        raise ValueError("Matching response must contain a mappings list.")
    matched = {}
    for mapping in mappings:
        if not isinstance(mapping, dict) or "condition" not in mapping or "column" not in mapping:
            raise ValueError("Each mapping must contain condition and column fields.")
        condition, column = mapping["condition"], mapping["column"]
        if not isinstance(condition, str) or condition not in conditions or condition in matched:
            raise ValueError("Matching response contains an unknown or duplicate condition.")
        if column is not None and (not isinstance(column, str) or column not in columns):
            raise ValueError("Matching response contains an unknown existing column.")
        matched[condition] = column
    if set(matched) != set(conditions):
        raise ValueError("Matching response must map every extracted condition exactly once.")
    return matched


def existing_matches(conditions, mappings, columns):
    exact = {column.casefold(): column for column in columns}
    matched = set()
    unmatched = []
    for condition in conditions:
        column = mappings[condition]
        if column is None:
            column = exact.get(condition.casefold())
        if column is None:
            unmatched.append(condition)
        else:
            matched.add(column)
    return matched, unmatched


def rank_codes(matched, code_features):
    if not matched:
        return []
    ranked = []
    for code, features in code_features:
        overlap = len(matched & features)
        if overlap:
            score = overlap / math.sqrt(len(matched) * len(features))
            ranked.append((code, score))
    return sorted(ranked, key=lambda item: (-item[1], item[0]))[:3]
