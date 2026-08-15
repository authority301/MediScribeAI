# Step 14I: Controlled Three-Way Evaluation (14B vs 14F vs 14H)

Controlled comparison of THREE systems against the exact same Step 14E 101-case gold benchmark, using the exact same, frozen Step 14B NLI model, thresholds, and aggregation rule throughout. Only the premise preprocessing differs between systems.

- **SYSTEM A (14B)**: raw premise -> mDeBERTa NLI -> aggregation. No Tanglish preprocessing.
- **SYSTEM B (14F)**: premise -> FROZEN Step 14F normalizer (byte-identical-logic snapshot of commit 5563418, before Step 14H) -> the SAME NLI -> the SAME aggregation.
- **SYSTEM C (14H)**: premise -> CURRENT Step 14H normalizer (numeric/measurement preservation + family/third-person attribution safety) -> the SAME NLI -> the SAME aggregation.

## Reproduction check (required before this comparison was produced)

Per the Step 14I spec, the current run's 14B and 14F reproduction was compared against the historical Step 14G results (`datasets/nli_evaluation/results/tanglish_comparison/`) BEFORE producing this comparison. If any value had differed beyond the documented tolerance, evaluation would have stopped without producing a final comparison.

- Tolerance: 1e-06
- All matched: **True**

| Check | Current run | Historical (Step 14G) | Diff | Matched |
|---|---|---|---|---|
| 14b_accuracy | 0.594059 | 0.594059 | 0.00e+00 | True |
| 14b_macro_f1 | 0.582938 | 0.582938 | 0.00e+00 | True |
| 14f_accuracy | 0.643564 | 0.643564 | 0.00e+00 | True |
| 14f_macro_f1 | 0.650216 | 0.650216 | 0.00e+00 | True |

## Research integrity

1. Synthetic benchmark (101 hand-authored cases), not a clinical dataset.
2. Model, thresholds (entailment=0.70, contradiction=0.70), and aggregation rule are IDENTICAL and UNCHANGED across all three systems; only the premise text fed to the model differs.
3. No fine-tuning, no new/alternate model, no external API.
4. Do NOT assume 14H is an improvement -- the conclusion below is drawn from the measured results.
5. Evidence Unsupported-Claim Rate (UCR) is an evidence-verification metric, not a clinical hallucination rate. Contradiction Rate (CR) is reported separately.
6. These results do NOT establish clinical safety, clinical validity, or real-world diagnostic accuracy.

## Reproducibility metadata

- Model provider: `local`
- Model name: `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`
- Device: `cpu`
- Entailment threshold: `0.7`
- Contradiction threshold: `0.7`
- System 14B configuration: raw premise, no Tanglish preprocessing (app.evaluation.runner, Step 14C, unmodified)
- System 14F configuration: premise normalized by FROZEN Step 14F normalizer (app.evaluation_tanglish_refinement.frozen_14f, byte-identical-logic snapshot of commit 5563418)
- System 14H configuration: premise normalized by CURRENT Step 14H normalizer (app.tanglish.normalizer, via app.evaluation_tanglish.runner.evaluate_case_mode(mode='tanglish_aware'), unmodified)
- Benchmark file: `D:\MediScribeAI\MediScribeAI\datasets\nli_evaluation\benchmark.json`
- Benchmark SHA-256: `db4d78a94879c2869d1abed423f46cbfd5a4a54e5d24d016c3c2f34faa4b0543`
- Number of cases evaluated: `101`
- Evaluation timestamp (UTC): `2026-08-15T08:41:39.697001+00:00`
- Python version: `3.11.15`
- Platform: `Windows-10-10.0.26200-SP0`

## Overall results

| System | Accuracy | Macro-F1 | UCR | CR |
|---|---|---|---|---|
| 14B Baseline | 59.41% | 0.5829 | 47.52% | 13.86% |
| 14F Tanglish-aware | 64.36% | 0.6502 | 48.51% | 22.77% |
| 14H Refined | 68.32% | 0.6888 | 46.53% | 24.75% |

## Main comparison (percentage-point changes)

| Transition | Accuracy Δ (pp) | Macro-F1 Δ (pp) |
|---|---|---|
| 14F − 14B | +4.95 | +6.73 |
| 14H − 14F | +3.96 | +3.86 |
| 14H − 14B | +8.91 | +10.58 |

## Language comparison (accuracy)

| Language | 14B | 14F | 14H |
|---|---|---|---|
| en         | 76.92% | 76.92% | 76.92% |
| tanglish   | 43.24% | 62.16% | 62.16% |
| mixed      | 56.00% | 48.00% | 64.00% |

See `language_comparison.json` for full precision/recall/F1/UCR/CR per language per system.

## Refinement change (14F -> 14H)

| Metric | 14F → 14H |
|---|---|
| Overall accuracy | +3.96 pp |
| Overall Macro-F1 | +3.86 pp |
| Tanglish accuracy | +0.00 pp |
| Tanglish Macro-F1 | -0.24 pp |
| Mixed accuracy | +16.00 pp |
| Mixed Macro-F1 | +15.38 pp |
| UCR | -1.98 pp |
| CR | +1.98 pp |

## Fixed / regressed (14F -> 14H)

- Fixed by 14H: **5**
- Regressed by 14H: **1**
- Unchanged correct: **64**
- Unchanged incorrect: **31**

## 14B -> 14F -> 14H transition highlights

- Preserved improvement (14B wrong, 14F correct, 14H correct): **10**
- Regression (14B wrong, 14F correct, 14H wrong): **0**
- New gain from 14H (14B wrong, 14F wrong, 14H correct): **1**

Full 8-way bucket counts (correct/wrong x 14B x 14F x 14H): see `comparison.json` -> `transition_analysis.counts`.

## Statistical tests (exact McNemar, binary correct/incorrect, paired)

| Pair | n discordant | p-value |
|---|---|---|
| 14B vs 14F | 15 | 0.3018 |
| 14F vs 14H | 6 | 0.2188 |
| 14B vs 14H | 13 | 0.0225 |

No claim of statistical significance beyond these reported p-values is made. The benchmark is 101 synthetic cases; these are descriptive, paired-by-case comparisons only.

## Targeted Step 14H analyses

See `targeted_analysis.json` for full numeric/measurement, attribution, negation, and historical-context breakdowns across all three systems, including 14F->14H fixed/regressed cases restricted to each category subset.

## Research interpretation

Research question: do the Step 14H numeric-preservation and attribution-safety refinements improve the Tanglish-aware NLI system relative to Step 14F, while preserving Step 14F's gains over the Step 14B baseline?

On this 101-case synthetic benchmark, Step 14H changed overall accuracy by +3.96 percentage points and overall Macro-F1 by +3.86 percentage points relative to Step 14F. The 14F-vs-14H paired McNemar comparison produced p = 0.2188 (n=6 discordant pairs). See the language/category breakdowns above and in the JSON result files for the full, unabridged picture (including any language- or category-specific regressions) before drawing a conclusion beyond this aggregate figure -- an aggregate accuracy change can mask offsetting gains and losses across languages, and the correct scientific statement should always cite the specific breakdown it is based on.

No claim of clinical validity, HIPAA compliance, or production accuracy is made.

## Discovered issues (Step 14I finding -- reported, not fixed)

Step 14I is evaluation-only and must not modify Step 14H while measuring it. The 1 case(s) below regressed from 14F to 14H on this run and were inspected for a GENERIC root cause (never a benchmark-specific one) before being reported here, unfixed:

- **TL-008** (tanglish/allergies): premise `"Enakku penicillin allergy irukku."`, claim `"Patient is allergic to penicillin."`, gold `SUPPORTED`. 14F -> `SUPPORTED` (normalized premise: `"Enakku penicillin allergy irukku."` -- i.e. UNRECOGNIZED, passed through unchanged under the frozen Step 14F lexicon); 14H -> `UNGROUNDED` (normalized premise: `"Patient has allergy."`).

**Likely generic root cause** for the case(s) flagged "UNRECOGNIZED" above: Step 14H added new bare clinical-entity vocabulary to `lexicon.py::CLINICAL_TERMS` (`hypertension`, `allergy`) to support its attribution-safety test coverage. Recognizing a previously-unrecognized entity is a double-edged change: it can also cause the composer to now rewrite a clause that used to pass through completely UNCHANGED under Step 14F -- and the composer only ever preserves the *entity itself* plus duration/frequency/measurement qualifiers, never an adjacent, non-entity qualifying noun (e.g. a specific allergen or drug name that is not itself in `CLINICAL_TERMS`). The result can be a *less* specific normalized sentence than the original raw text, for cases the frozen 14F lexicon simply never touched. This is a generic limitation of the compose-from-recognized-entity architecture (present since Step 14F, not introduced by Step 14H's own logic changes) that newly surfaced here because Step 14H's vocabulary additions widened lexicon coverage. Per the Step 14I instructions, this is reported as a finding for a future step, not patched during this evaluation-only step, and no benchmark-specific exception was added to work around it.

## Files

- `baseline_14b_results.json` / `tanglish_14f_results.json` / `tanglish_14h_results.json`: full single-system report (overall + language/category/difficulty breakdowns + bootstrap CIs + every per-case prediction) for each system.
- `comparison.json`: full three-way comparison (overall, language/category/difficulty, per-class, confusion matrices, 14F->14H fixed/regressed, 14B->14F->14H transitions, McNemar tests, targeted analyses).
- `language_comparison.json` / `category_comparison.json` / `difficulty_comparison.json` / `targeted_analysis.json` / `confusion_matrices.json` / `statistical_tests.json`: standalone copies of the corresponding `comparison.json` sections.
- `failure_analysis.json`: 14F->14H fixed/regressed/unchanged case-level detail plus the 14B->14F->14H transition highlight buckets.
