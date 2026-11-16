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
