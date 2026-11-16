# sugen_test
This is a take home assesment by Sugen

## Generate predictions

Set `OPENAI_API_KEY` in the environment or project `.env`, then run:

```sh
python3 llm.py data/dev_notes.jsonl predictions.jsonl
```

The default model is `gpt-6-luna`; override it with `--model` or
`OPENAI_MODEL`. To use OpenRouter instead, set `OPENROUTER_API_KEY` and pass
`--provider openrouter` (optionally set `OPENROUTER_MODEL`).

Edit `prompts/icd11_system.txt` to tune the ICD coding instructions. Keep the
requested JSON response shape `{"codes": ["CODE"]}`.

Open `evaluate_dev.ipynb` and run its cells to evaluate all dev notes with
`score.py`. Generating predictions sends notes to the model provider and may
incur API charges.

Open `build_condition_database.ipynb` to extract conditions from the 50 training
notes in `data/dev_notes_train.jsonl` and create `out/icd_conditions_train.csv`.
Edit `prompts/extract_conditions.txt` and
`prompts/match_conditions.txt` to tune extraction and column matching.

Open `evaluate_dev_test02.ipynb` to evaluate the 10 held-out notes in
`data/dev_notes_test.jsonl` against the training condition CSV. The original
`data/dev_notes.jsonl` remains unchanged. Since no test gold code appears in
training, this closed-set system cannot predict the test labels and must score
zero.
