# ICD-11 Consult Note Coder (`sugen_test`)

Proposes up to three ranked ICD-11 MMS (`2026-01`) diagnostic codes per consult
note using a two-agent **Doctor + Judge** workflow (`run/coder.py`) grounded in
local retrieval over the real MMS code list and ICD descriptions. It falls back
to offline retrieval when no API key is set or an API call fails, so every note
always gets a valid prediction.

## Run

```bash
docker compose build
docker compose run coder /in/notes.jsonl /out/predictions.jsonl
```

- **Input** `/in/notes.jsonl` (mapped to `./in`): one `{"id", "text"}` per line.
- **Output** `/out/predictions.jsonl` (mapped to `./out`): one
  `{"id", "codes": [up to 3, best first]}` per line, flushed as it runs.
- **Key:** `OPENAI_API_KEY` is read from the command line / environment first,
  then `.env`; with no key it runs the offline retrieval fallback.
  When gold labels are available, `score.py` runs automatically and prints the
  summary at the end.

## Scores

From `score.py` (slice A, out of 75):

- **10 held-out dev notes (unseen during table/prompt building): 62.5 / 75**
  (top1 0.80, top3 0.90) — the honest held-out number.
- **All 60 dev notes: ~70 / 75** (top1 ~0.93, top3 ~0.95).

**Estimated hidden score: ~90 / 100.** Per-note retrieval grounds the model in
the actual MMS code space, which should transfer to the hidden core slice; the
main risk is the generalisation slice (distribution shift, non-English notes),
which the offline and cross-lingual character n-gram fallbacks are meant to
cushion.

## Data added

- `data/final-icd-codes.csv` — a Kaggle ICD-11 dataset of code titles,
  definitions and descriptive text (~20k rows). It enriches the title-only MMS
  list so both agents reason over real ICD descriptions and whole code families,
  not just code names.

## Evaluating beyond the 60 dev notes

- Split the 60 dev notes into **50 train / 10 held-out** (`dev_notes_train.jsonl`
  / `dev_notes_test.jsonl`) and scored only on the held-out 10 to get an
  unseen-data estimate instead of fitting to all 60.
- Retrieval tools (`training_examples`) never return the current note's own
  label, so held-out scores aren't inflated by self-matching.
- The system degrades gracefully on inputs unlike the dev set: offline TF-IDF
  plus an accent-stripped character n-gram fallback handles non-English and
  out-of-distribution notes so no note scores 0 for lack of a prediction.

## Where the AI tooling was wrong (and I caught it)

An early iteration (test02) had the AI build a closed-set condition→code table
from the 50 training notes and rank by cosine similarity. Scored against the
notes it was built from, it looked perfect — **75 / 75** — which is exactly the
tell. On the 10 held-out notes it scored **0 / 75**, because no held-out gold
code appears in the training golds, so a closed-set model can't predict them by
construction. I caught the leakage from that 75-vs-0 gap and dropped the approach
in favour of retrieval over the full MMS code space (test03 → test04).

See [DECISIONS.md](./DECISIONS.md) for the full iteration log and
[README_test04.md](./README_test04.md) for the Doctor + Judge architecture.
