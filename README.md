# ICD-11 Consult Note Coder (`sugen_test`)

Proposes up to 3 ranked ICD-11 MMS (`2026-01`) diagnostic codes per clinical consult note using a two-agent Doctor + Judge workflow with local TF-IDF retrieval tools (`run/coder.py`), falling back automatically to offline retrieval when no API key is set or an API call fails.

## Quick Start (Docker)

Build and run from a clean clone:

```bash
docker compose build
docker compose run coder /in/notes.jsonl /out/predictions.jsonl
```

- **Input (`/in/notes.jsonl`):** Mapped to `./in/notes.jsonl` by default (or overridden with `-v /path/to/in:/in`). JSONL with one `{"id": "...", "text": "..."}` object per line.
- **Output (`/out/predictions.jsonl`):** Mapped to `./out/predictions.jsonl` by default (or overridden with `-v /path/to/out:/out`). Flushes predictions line-by-line as `{"id": "...", "codes": ["...", ...]}`.
- When gold labels exist for the input notes (such as dev notes), `score.py` runs automatically at the end and prints the score summary.

#