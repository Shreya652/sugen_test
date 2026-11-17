# ICD-11 consult note coder (`sugen_test`)

I built this coder to propose up to three ranked ICD-11 MMS (`2026-01`)
diagnostic codes for each consult note. It uses a doctor and judge workflow
(`run/coder.py`) that retrieves information from the local MMS code list and
ICD descriptions. If no API key is set or an API call fails, it falls back to
offline retrieval so every note gets a valid prediction.

## Run

```bash
docker compose build
docker compose run coder /in/notes.jsonl /out/predictions.jsonl
```

- Input: `/in/notes.jsonl` (mapped to `./in`), with one `{"id", "text"}` per line.
- Output: `/out/predictions.jsonl` (mapped to `./out`), with one
  `{"id", "codes": [up to 3, best first]}` per line. Each result is flushed as
  it runs.
- The coder reads `OPENAI_API_KEY` from the command line or environment first,
  then from `.env`. Without a key, it uses offline retrieval. When gold labels
  are available, `score.py` runs automatically and prints a summary at the end.

## Scores

These are the `score.py` results for slice A, out of 75:

- 10 held-out dev notes, unseen during table and prompt building: 62.5 / 75
  (top1 0.80, top3 0.90). I use this as the held-out score.
- All 60 dev notes: ~70 / 75 (top1 ~0.93, top3 ~0.95).

I estimate a hidden score of 62. Retrieving codes for each note gives


## Data added

I added `data/final-icd-codes.csv`, a Kaggle ICD-11 dataset with code titles,
definitions, and descriptive text (~20k rows). It extends the title-only MMS
list, so both agents can use descriptions and whole code families as well as
code names.

## Evaluating beyond the 60 dev notes

I split the 60 dev notes into 50 training notes and 10 held-out notes
(`dev_notes_train.jsonl` / `dev_notes_test.jsonl`). I scored only the held-out
10 to estimate performance on unseen notes instead of fitting to all 60.

The `training_examples` retrieval tool never returns the current note's own
label, so self-matching does not inflate the held-out scores.

For inputs unlike the dev set, the system uses offline TF-IDF and an
accent-stripped character n-gram fallback. These handle non-English and
out-of-distribution notes so that no note scores 0 for lack of a prediction.

## Where the AI tooling was wrong and I caught it

In an early iteration (test02), I used AI to build a closed-set table mapping
conditions to codes from the 50 training notes, then ranked codes by cosine
similarity. Scored against the notes it was built from, it looked perfect at
75 / 75. On the 10 held-out notes, it scored 0 / 75: none of their gold codes
appeared in the training golds, so the closed-set model could not predict them.

I caught the leakage from that 75-vs-0 gap and dropped the approach. In test03
and test04, I used retrieval over the full MMS code space instead.

See [DECISIONS.md](./DECISIONS.md) for the iteration log and
[README_test04.md](./README_test04.md) for the doctor and judge architecture.
