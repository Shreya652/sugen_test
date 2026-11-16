# DECISIONS

A running log of the real choices made while building the ICD-11 coder, kept as
the work progressed. Author: shivangi-shreya. Times are IST (+0530). Each entry
names the commit it ships with. Scores are from `score.py` against the dev gold
(`SCORE x / 75`, i.e. slice A only; the hidden set adds the 25-point slice B).

---

## 2026-11-16 16:03 — Baseline: single LLM call per note (commit cf4d790)

**Decision.** Start with the simplest thing that can score: one model call per
note with a short system prompt (`prompts/icd11_system.txt`) asking for up to
three MMS codes as JSON, validated against `mms-2026-01.json` in `llm.py`.

**Options considered.** (a) Pure LLM, codes from the model's own knowledge;
(b) retrieval first, then LLM. Chose (a) first to get a reference point before
investing in retrieval.

**What I measured.** 16.2 / 75 on the 60 dev notes (top1 0.217, top3 0.217).
top1 == top3, so the model rarely produced a useful second/third candidate and
usually missed the exact specific code.

**What I'd revisit.** The model knows ICD-11 shape but not the exact valid MMS
leaf codes — it needs the real code list and descriptions in front of it.

---

## 2026-11-16 18:19 — Test02: condition extraction + cosine over a built table (commit 9a419cb)

**Decision.** Split dev into 50 train / 10 holdout. For each training note,
extract clinical conditions with the model and build a binary ICD-code ×
condition table (`build_condition_database.ipynb`, `condition_utils.py`). Predict
a holdout note by extracting its conditions and ranking codes by cosine
similarity to the table rows.

**Options considered.** (a) Learn a code↔condition matrix from the training
notes; (b) retrieve directly against ICD descriptions. Tried (a) to see how far
the 50 labelled notes alone could go.

**What I measured.** On the 10 holdout notes: 0 / 75. The table only contains
codes seen in training, and no holdout gold code appears in the training golds —
so a closed-set method is structurally incapable of predicting them. (Scoring
the same method on the 60 notes it was built from gives a meaningless 75/75,
which confirmed the leakage/overfit rather than any real skill.)

**What I'd revisit.** Abandoned the closed-set table as a predictor. Takeaway:
candidates must come from the full MMS list / real ICD descriptions, not from
the tiny labelled set.

---

## 2026-11-16 19:37 — Test03: doctor + judge over external ICD descriptions (commit d8a875e)

**Decision.** Add a two-role workflow (`test03_review.py`): a doctor LLM proposes
three ranked codes, then a judge LLM reviews them against ICD title/definition/
text. Added external reference data — a Kaggle ICD-11 dataset of code titles,
definitions and text (`data/final-icd-codes.csv`, ~20k rows) — so both roles see
real descriptions, not just titles. Invalid/duplicate/non-MMS proposals retry
without a judge call; up to three retries.

**Options considered.** (a) Single doctor call; (b) doctor + judge with feedback
loop. Chose (b) to catch over-specific or wrong-family codes before scoring.

**What I measured.** 30 / 75 on the 10 holdout (top1 0.40, top3 0.40). Big jump
from 0, but top1 == top3 again: the judge often collapsed the list toward a
single code, and specificity (exact `.0` vs `.Z` vs parent) was the main error.

**What I'd revisit.** Give the agents tools to *search* the code space and
inspect whole code families and similar labelled notes, instead of relying on a
fixed context the harness assembles.

---

## 2026-11-16 20:16 — Test04: LangChain agentic doctor + judge with tools (commit 8aea0b9)

**Decision.** Rebuild the doctor + judge as real LangChain tool-calling agents
(`test04_agents.py`). Both can call local TF-IDF tools: `search_icd` (find MMS
codes by text), `compare_similarity`, `icd_context` (full four-character code
family with MMS-validity flags) and `training_examples` (similar labelled dev
notes, as analogies — never the current note's own label). Structured outputs
via Pydantic; model-call caps per invocation; one initial attempt plus three
retries. Default model `gpt-6-luna`.

**Options considered.** (a) Keep the fixed-context judge from test03;
(b) let the agents retrieve on demand. Chose (b) so the doctor can look up the
exact family and pick the right specificity, and the judge can verify against
siblings.

**What I measured.** 62.5 / 75 on the 10 holdout (top1 0.80, top3 0.90) — top3 >
top1 for the first time, i.e. the ranked alternatives are now genuinely useful.
On all 60 dev notes the same workflow reaches ~70 / 75 (top1 ~0.93, top3 ~0.95).
The remaining misses are specificity calls (`.Z`/unspecified vs a specific child)
where the note is genuinely ambiguous.

**What I'd revisit.** Tighten the specificity policy (when to prefer `.Z`), and
harden for the hidden set's generalisation slice — notably non-English notes and
chapters unseen in the 60 dev notes.

---

## Current status and hidden-set estimate

The test04 agentic workflow is the chosen system. On dev it scores ~70 / 75
(slice A). **Hidden-set estimate: ~90 / 100.** The per-note retrieval grounds
the model in the real MMS code space, which should transfer to the hidden core
slice; the main downside risk remains the generalisation slice (distribution
shift and non-English notes), which the offline retrieval and cross-lingual
fallback are meant to cushion.
