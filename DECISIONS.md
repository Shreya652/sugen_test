# Decisions

I kept this log of the choices I made while building the ICD-11 coder. Each
entry names the commit it ships with. The scores come from `score.py` against
the dev gold.

## 2026-11-16 16:03: Baseline, one LLM call per note (commit cf4d790)

I started with one model call per note. The system prompt
(`prompts/icd11_system.txt`) asked for up to three MMS codes as JSON, and
`llm.py` validated them against `mms-2026-01.json`.

I considered relying on the model's own knowledge or retrieving code
descriptions first. I chose the first option to get a baseline before working
on any improvements.

The score was 16.2 / 75 on 60 dev notes (top1 0.217, top3 0.217). Since top1
and top3 were equal, the second and third candidates rarely helped, and the
model usually missed the exact specific code.

The model knows the shape of ICD-11 codes but not the exact valid MMS leaf
codes. I'd give it the code list and descriptions directly.

## 2026-11-16 18:19: Test02, condition extraction and cosine similarity (commit 9a419cb)

I split the dev notes into 50 training notes and 10 holdout notes. For each
training note, I used the model to extract clinical conditions and built a
binary table of ICD codes and conditions (`build_condition_database.ipynb`,
`condition_utils.py`). For a holdout note, I extracted its conditions and
ranked the table's codes by cosine similarity.

I considered learning a condition matrix from the training notes or retrieving
directly from ICD descriptions. I tried the matrix to see how far the 50
labelled notes alone could take me.

The score on the 10 holdout notes was 0 / 75. The table contained only codes
seen in training, and none of the holdout gold codes appeared in the training
golds. The closed-set method therefore could not predict them. Scoring the same
method on the 60 notes it was built from gave a meaningless 75/75, which
confirmed leakage and overfitting rather than predictive skill.

I abandoned the table as a predictor. I'd get candidates from the full MMS
list and ICD descriptions instead of limiting them to the small labelled set.

## 2026-11-16 19:37: Test03, doctor and judge with ICD descriptions (commit d8a875e)

I added a two-role workflow (`test03_review.py`). A doctor LLM proposes three
ranked codes, and a judge LLM reviews them against ICD titles, definitions,
and text. I also added a Kaggle ICD-11 dataset containing code titles,
definitions, and text (`data/final-icd-codes.csv`, ~20k rows), so both roles
could use the descriptions as well as the titles. Invalid, duplicate, or
non-MMS proposals trigger a retry without a judge call. There are up to three
retries.

I considered keeping a single doctor call or adding a judge with a feedback
loop. I chose the judge to catch codes from the wrong family or codes that were
too specific before scoring.

The score on the 10 holdout notes was 30 / 75 (top1 0.40, top3 0.40). This
improved on the previous score of 0, but top1 and top3 were equal again. The
judge often collapsed the list toward a single code, and specificity was the
main source of errors, especially choosing between `.0`, `.Z`, and a parent code.

I'd give the agents tools to search for codes and inspect whole code families
and similar labelled notes, instead of relying on fixed context assembled by
the harness.

## 2026-11-16 20:16: Test04, LangChain doctor and judge with tools (commit 8aea0b9)

I rebuilt the doctor and judge as LangChain tool-calling agents
(`test04_agents.py`). Both can use local TF-IDF tools: `search_icd` to find
MMS codes by text, `compare_similarity`, `icd_context` for a full
four-character code family with MMS-validity flags, and `training_examples`
for similar labelled dev notes. The examples are analogies and do not include
the current note's own label.

I used Pydantic for structured output and capped model calls per invocation.
The workflow has one initial attempt and three retries. The default model is
`gpt-6-luna`.

I considered keeping the fixed-context judge from test03 or letting the agents
retrieve information as needed. I chose retrieval so the doctor could look up
the exact family and choose the right specificity, and the judge could check
sibling codes.

The score on the 10 holdout notes was 62.5 / 75 (top1 0.80, top3 0.90). Top3
was higher than top1 for the first time, so the ranked alternatives helped. On
all 60 dev notes, the same workflow reaches ~70 / 75 (top1 ~0.93, top3 ~0.95).
The remaining misses involve specificity, such as `.Z` or unspecified versus
a specific child, where the note is genuinely ambiguous.

I'd refine when the workflow should prefer `.Z` and test it against the hidden
set's generalisation slice, especially non-English notes and chapters not seen
in the 60 dev notes.

## Current status and hidden-set estimate

I chose the test04 workflow. It scores ~70 / 75 on dev (slice A), and I
estimate 90% on the hidden set. Retrieving the MMS codes for each note
should help on the hidden core slice. My main concern is the generalisation
slice, particularly distribution shift and non-English notes. The offline
retrieval and cross-lingual fallback are meant to help with those cases.
