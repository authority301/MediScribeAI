# MediScribeAI NLI Evaluation Benchmark (Step 14E)

A synthetic, human-labeled gold benchmark for evaluating whether the
existing Step 14B evidence-verification pipeline (local mDeBERTa-v3-base
XNLI) correctly distinguishes `SUPPORTED`, `CONTRADICTED`, and `UNGROUNDED`
for clinical claims, across English, Tanglish (Romanized Tamil-English
code-mixed), and mixed/code-switched clinical speech.

This dataset is **evaluation-only**. It does not modify, tune, or retrain
any part of the NLI model, retrieval algorithm, or aggregation logic.

## Files

- `benchmark.json` — 101 gold-labeled cases (see schema below)
- `schema.json` — JSON Schema describing `benchmark.json`
- `README.md` — this file

## Ground-truth methodology

Every `ground_truth` label was reasoned by hand directly from the synthetic
premise/claim pair, independent of any model. **No label in this dataset was
ever produced by running the NLI model and copying its prediction.** The NLI
model is the system being evaluated here (in a later step); the benchmark
itself must remain model-independent so that model behavior can be measured
against it honestly, including failures.

Where a case's correct label could plausibly be argued either way, it was
either rewritten so the ground truth became unambiguous, or excluded. Every
case includes an optional `rationale` field documenting the reasoning.

## Dataset size and distribution (actual, as generated)

Total cases: **101** (target was "approximately 100"; exact count follows
from balancing categories, languages, and labels rather than forcing a round
number).

### Language

| Language | Count |
|---|---|
| English (`en`) | 39 |
| Tanglish (`tanglish`) | 37 |
| Mixed/code-switched (`mixed`) | 25 |

(Target was ~40/40/20; actual came out close but not exact once every
category and label was covered in every language bucket without forcing
artificial cases.)

### Ground truth

| Label | Count |
|---|---|
| SUPPORTED | 37 |
| CONTRADICTED | 33 |
| UNGROUNDED | 31 |

Roughly even thirds (36.6% / 32.7% / 30.7%), no severe class imbalance.

### Category

| Category | Count |
|---|---|
| symptoms | 10 |
| medications | 8 |
| allergies | 5 |
| measurements | 8 |
| duration | 7 |
| frequency | 5 |
| medical_procedures | 5 |
| diagnoses | 5 |
| follow_up_instructions | 5 |
| referral_instructions | 5 |
| patient_history | 6 |
| doctor_assessment | 6 |
| negation | 11 |
| historical_statements | 6 |
| diagnosis_overreach | 9 |

`negation` and `diagnosis_overreach` are intentionally over-represented
relative to the others: negation handling and unsupported-diagnosis
generation are the two failure modes the Step 14B manual verification
already surfaced as weak points for the current model, and are explicitly
called out as major evaluation categories.

### Difficulty

| Difficulty | Count |
|---|---|
| easy | 16 |
| medium | 43 |
| hard | 42 |

`easy` = direct lexical match/restatement in English. `medium` = a single
complicating factor (paraphrase, negation, duration reasoning, or bare
code-switching). `hard` = combinations of the above (e.g. negation +
code-switching, historical context + code-switching) or diagnosis overreach
in any language, which is always `hard` regardless of language since it
requires recognizing that no diagnosis was actually stated.

## Schema

Each case in `benchmark.json`:

```json
{
  "id": "TL-002",
  "language": "tanglish",
  "premise": "Enakku chest pain illa.",
  "claim": "Patient reports chest pain.",
  "ground_truth": "CONTRADICTED",
  "category": "symptoms",
  "difficulty": "hard",
  "speaker_role": "PATIENT",
  "clinical_domain": "cardiology",
  "contains_negation": true,
  "contains_measurement": false,
  "contains_duration": false,
  "contains_medication": false,
  "contains_diagnosis": false,
  "contains_code_switching": true,
  "rationale": "'illa' negates chest pain in the premise; the claim asserts its presence."
}
```

`language` is one of `en | tanglish | mixed`. `ground_truth` is one of
`SUPPORTED | CONTRADICTED | UNGROUNDED`. `category` is one of the 15 fixed
clinical categories listed above. See `schema.json` for the full JSON Schema,
including field types and the explicit prohibition on model-derived fields.

## Model independence

`benchmark.json` contains **no** model predictions, confidence scores,
retrieval scores, NLI labels, or hallucination metrics of any kind. It
contains only synthetic premises, claims, and human-reasoned gold labels.
This is enforced structurally (`schema.json`'s `not`/`anyOf` clause rejects
any of those keys) and by the dataset validation tests in
`backend/tests/test_nli_dataset.py`.

This independence is what allows a later step to run the existing,
unmodified Step 14B pipeline against this benchmark and compare its output
to gold labels, without the benchmark having been contaminated by the
system it evaluates.

## Tanglish notes

The Tanglish and mixed examples use Romanized Tamil mixed with English
clinical terms, written in one plausible, readable spelling convention.
**Tamil-English code-switching spelling and phrasing vary considerably
across speakers, regions, and registers** — this benchmark does not claim to
represent all Tanglish dialects, spelling conventions, or usage patterns. It
is a controlled, synthetic sample for evaluating one specific pipeline's
behavior on one style of code-mixed clinical speech.

## Research integrity

This benchmark:

- is **synthetic** — no real patient data of any kind was used or referenced
- is for **research/evaluation only**
- does **not** establish clinical safety
- does **not** establish real-world medical accuracy
- does **not** establish HIPAA compliance
- measures model **behavior** on controlled synthetic examples, nothing more

Diagnosis-overreach cases in particular are designed to test whether the
verification pipeline correctly refuses to treat symptom-only evidence as
support for a diagnosis claim — they are about evidence grounding, not
medical plausibility. A medically "reasonable" inference (e.g. fever + cough
→ pneumonia) is still labeled `UNGROUNDED` here because the premise does not
itself state the diagnosis.

## What this dataset is NOT for

- Not for computing hallucination rate/percentage (Step 14C territory)
- Not for accuracy/precision/recall/F1 or confusion-matrix scoring (later
  step, if any)
- Not for threshold tuning — thresholds must never be adjusted to make the
  model perform better against this benchmark
- Not for model fine-tuning or training
