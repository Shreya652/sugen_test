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

Open `compare_icd_dev_test03.ipynb` to compare dev notes and gold codes with the
title, definition, and text in `data/final-icd-codes.csv`. This local
inspection does not call a model or establish clinical correctness.

Open `evaluate_dev_test03.ipynb` for the separate doctor-and-judge experiment
on the 10 held-out notes. Its prompts are `prompts/test03_doctor.txt` and
`prompts/test03_judge.txt`. It makes up to eight model calls per note and is
for research, not clinical use. The judge receives the candidates and all CSV
code-family entries sharing its first four stem characters; larger contexts can
increase API cost and latency. It reviews three ranked recommendations together;
partial results after the retry limit are marked incomplete.

Open `evaluate_dev_test04.ipynb` for the LangChain doctor-and-judge experiment.
Install `requirements_test04.txt` in the notebook's Python environment. It
defaults to a local TF-IDF baseline; selecting agent mode sends the held-out
notes and retrieved labeled training examples to OpenAI, incurs API charges,
and requires permission to share those notes. The agents can search MMS codes,
compare similarity, inspect complete ICD code families, and retrieve analogous
training examples. Outputs are saved separately in `out/` by mode. The agent
experiment is not for clinical use and does not enforce the assessment's
dollar or time budget.
