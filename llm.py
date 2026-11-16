#!/usr/bin/env python3
"""Generate ICD-11 predictions for JSONL notes with an LLM."""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API_URLS = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
}
DEFAULT_MODELS = {"openai": "gpt-6-luna", "openrouter": "openai/gpt-4o-mini"}
SYSTEM_PROMPT_PATH = Path(__file__).parent / "prompts" / "icd11_system.txt"
STEM = re.compile(r"[&/]")


def load_dotenv(path):
    path = Path(path)
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        if separator and key.strip():
            env_key = key.strip()
            env_val = value.strip().strip("\"'")
            if not os.environ.get(env_key, "").strip():
                os.environ[env_key] = env_val


def normalize_code(code):
    if not isinstance(code, str):
        return None
    return STEM.split(code.strip())[0].strip().upper() or None


def load_valid_codes(path):
    with path.open(encoding="utf-8") as file:
        rows = json.load(file)
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a JSON list of ICD-11 categories")
    return {
        normalize_code(row.get("code"))
        for row in rows
        if isinstance(row, dict) and normalize_code(row.get("code"))
    }


def request_json(system_prompt, user_text, model, api_key, provider="openrouter"):
    payload = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": user_text},
        ],
    }
    if provider == "openrouter":
        payload.update({"temperature": 0, "max_tokens": 200})
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if provider == "openrouter":
        headers["X-Title"] = "ICD-11 note coder"
        site_url = os.environ.get("OPENROUTER_SITE_URL")
        if site_url:
            headers["HTTP-Referer"] = site_url
    request = urllib.request.Request(
        API_URLS[provider],
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                result = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as error:
            retryable = error.code == 429 or 500 <= error.code < 600
            if not retryable or attempt == 2:
                raise RuntimeError(f"{provider} request failed with HTTP {error.code}") from error
            time.sleep(2**attempt)
        except urllib.error.URLError as error:
            if attempt == 2:
                raise RuntimeError(f"Could not reach {provider}: {error.reason}") from error
            time.sleep(2**attempt)
    else:
        raise RuntimeError(f"{provider} request failed after retries")

    try:
        content = result["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise TypeError("response content is not text")
        parsed = json.loads(content)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{provider} returned an invalid response") from error
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{provider} returned an invalid response")
    return parsed


def request_codes(note, model, api_key, provider="openrouter"):
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").rstrip("\n")
    if not system_prompt.strip():
        raise ValueError(f"{SYSTEM_PROMPT_PATH}: system prompt is empty")
    parsed = request_json(system_prompt, note, model, api_key, provider)
    codes = parsed.get("codes")
    if not isinstance(codes, list):
        raise RuntimeError(f'{provider} response must contain a "codes" list')
    return codes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="input JSONL with id and text fields")
    parser.add_argument("output", type=Path, help="output predictions JSONL")
    parser.add_argument(
        "--provider",
        choices=tuple(API_URLS),
        default="openai",
        help="model provider (default: openai)",
    )
    parser.add_argument(
        "--mms",
        type=Path,
        default=Path(__file__).parent / "data" / "mms-2026-01.json",
        help="ICD-11 MMS category JSON",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="model ID (default: provider's MODEL environment variable or built-in default)",
    )
    args = parser.parse_args()

    load_dotenv(Path(__file__).parent / ".env")
    key_name = f"{args.provider.upper()}_API_KEY"
    api_key = os.environ.get(key_name)
    if not api_key:
        parser.error(f"{key_name} is not set in the environment or project .env")
    model = args.model or os.environ.get(
        f"{args.provider.upper()}_MODEL", DEFAULT_MODELS[args.provider]
    )

    try:
        valid_codes = load_valid_codes(args.mms)
        with args.input.open(encoding="utf-8") as source, args.output.open(
            "w", encoding="utf-8"
        ) as destination:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    note = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{args.input}:{line_number}: invalid JSON") from error
                if not isinstance(note, dict) or "id" not in note or not isinstance(
                    note.get("text"), str
                ):
                    raise ValueError(
                        f"{args.input}:{line_number}: expected an object with id and text"
                    )

                suggested = request_codes(note["text"], model, api_key, args.provider)
                codes = []
                for code in suggested:
                    normalized = normalize_code(code)
                    if normalized in valid_codes and normalized not in codes:
                        codes.append(normalized)
                    if len(codes) == 3:
                        break
                destination.write(
                    json.dumps({"id": note["id"], "codes": codes}, ensure_ascii=False) + "\n"
                )
                destination.flush()
    except (OSError, ValueError, RuntimeError) as error:
        print(f"llm.py: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
