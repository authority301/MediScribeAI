# Step 14G: Controlled Baseline vs. Tanglish-Aware NLI Evaluation

Controlled, ablation-style comparison of two systems against the exact same Step 14E 101-case gold benchmark, using the exact same, frozen Step 14B NLI model, thresholds, and aggregation rule. The ONLY difference between the two systems is whether the Step 14F Tanglish-aware preprocessing layer runs before the premise reaches the NLI model.

- **SYSTEM A (baseline)**: raw premise -> mDeBERTa multilingual NLI -> aggregation.
- **SYSTEM B (tanglish_aware)**: premise -> Step 14F normalization -> the SAME mDeBERTa NLI -> the SAME aggregation.

## Research integrity

1. Synthetic benchmark (101 hand-authored cases), not a clinical dataset.
2. Model, thresholds (entailment=0.70, contradiction=0.70), and aggregation rule are IDENTICAL and UNCHANGED between systems; only the premise text fed to the model differs.
3. No fine-tuning, no new/alternate model, no external API.
4. Evidence Unsupported-Claim Rate (UCR) is an evidence-verification metric, not a clinical hallucination rate.
5. Contradiction Rate (CR) is reported separately from UCR.
6. These results do NOT establish clinical safety, clinical validity, or real-world diagnostic accuracy.
7. Results are descriptive and paired-by-case; see 'Statistical test' below for what was and was not tested.

## Reproducibility metadata

- Model provider: `local`
- Model name: `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`
- Device: `cpu`
- Entailment threshold: `0.7`
- Contradiction threshold: `0.7`
- Baseline mode: `baseline`
- Tanglish-aware mode: `tanglish_aware`
- Benchmark file: `D:\MediScribeAI\MediScribeAI\datasets\nli_evaluation\benchmark.json`
- Benchmark SHA-256: `db4d78a94879c2869d1abed423f46cbfd5a4a54e5d24d016c3c2f34faa4b0543`
- Number of cases evaluated: `101`
- Evaluation timestamp (UTC): `2026-08-15T07:17:11.735402+00:00`
- Python version: `3.11.15`
- Platform: `Windows-10-10.0.26200-SP0`

## Overall results

| System | Accuracy | Macro F1 | Weighted F1 | UCR (pred) | CR (pred) |
|---|---|---|---|---|---|
| Baseline 14B | 59.41% | 0.5829 | 0.5828 | 47.52% | 13.86% |
| Tanglish-aware 14F | 64.36% | 0.6502 | 0.6507 | 48.51% | 22.77% |

## By-language results (Accuracy, Macro F1)

| Language | Baseline | Tanglish-aware |
|---|---|---|
| en         | 76.92% (F1=0.7726) | 76.92% (F1=0.7726) |
| tanglish   | 43.24% (F1=0.4099) | 62.16% (F1=0.6239) |
| mixed      | 56.00% (F1=0.5343) | 48.00% (F1=0.4987) |

## Improvement metrics (overall, absolute change = tanglish_aware - baseline)

| Metric | Baseline | Tanglish-aware | Change (pp) |
|---|---|---|---|
| Accuracy | 0.5941 | 0.6436 | +4.95 |
| Macro F1 | 0.5829 | 0.6502 | +6.73 |
| UCR (pred) | 0.4752 | 0.4851 | +0.99 |
| CR (pred) | 0.1386 | 0.2277 | +8.91 |

Tanglish-specific improvement is the primary comparison of interest -- see `language_comparison.json` for the Tanglish row's own accuracy/macro-F1/UCR/CR baseline-vs-tanglish_aware figures.

## Per-class comparison

See `comparison.json` -> `per_class_comparison` for SUPPORTED/CONTRADICTED/UNGROUNDED precision/recall/F1 for both systems and their differences.

## Confusion matrices

Baseline (rows = ground truth, columns = prediction):
```
GOLD \ PRED   SUPPORTED     CONTRADICTED  UNGROUNDED    
SUPPORTED     23            0             14            
CONTRADICTED  12            12            9             
UNGROUNDED    4             2             25            
```

Tanglish-aware (rows = ground truth, columns = prediction):
```
GOLD \ PRED   SUPPORTED     CONTRADICTED  UNGROUNDED    
SUPPORTED     21            0             16            
CONTRADICTED  4             20            9             
UNGROUNDED    4             3             24            
```

See `confusion_matrices.json` for the difference matrix (tanglish_aware - baseline).

## Category / difficulty comparison

See `category_comparison.json` and `difficulty_comparison.json`.

## Negation category

See `comparison.json` -> `negation_analysis`. Fixed: 4, Regressed: 0.

## Historical statements / patient history

See `comparison.json` -> `historical_analysis`. Fixed: 1, Regressed: 1.

## Failure analysis (case-level)

- Fixed by Tanglish-aware preprocessing (baseline wrong, tanglish_aware right): **10**
- Regressed (baseline right, tanglish_aware wrong): **5**
- Unchanged and correct in both: **55**
- Unchanged and incorrect in both: **31**

Full per-case detail (premise, claim, gold label, both predictions, both NLI score sets, normalized premise, transformation trace) is in `failure_analysis.json`.

## Statistical test

- Method: McNemar's exact test (binomial form, two-sided, on paired binary correctness)
- Applies to: binary correct/incorrect per case, NOT the 3-class SUPPORTED/CONTRADICTED/UNGROUNDED label
- n (discordant pairs): 15 out of 101 total cases
- b (baseline correct, tanglish-aware incorrect): 5
- c (baseline incorrect, tanglish-aware correct): 10
- p-value: 0.3018
- Computed over a 101-case synthetic benchmark. This is a descriptive, paired-by-case statistical test, not a claim of clinical significance or real-world generalizability.

No claim of statistical significance beyond this reported p-value is made. The benchmark is 101 synthetic cases; this is a descriptive, paired-by-case comparison, not evidence of real-world generalization.

## Research interpretation

Research question: does deterministic Tanglish-aware clinical semantic normalization improve multilingual NLI evidence verification for Tanglish clinical text compared with the unmodified multilingual NLI baseline, on this 101-case synthetic benchmark?

Step 14F produced a +4.95 percentage-point change in overall accuracy and a +6.73 percentage-point change in overall macro F1 relative to the frozen Step 14B baseline, on this benchmark. See `language_comparison.json` for the Tanglish-only and Mixed-only figures, and English-only figures confirming (or not) that English performance was preserved. This experiment reports the measured result; it does not assert the hypothesis is confirmed beyond what these numbers show.

## Files

- `baseline_results.json` / `tanglish_aware_results.json`: full single-system report (overall + language/category/difficulty breakdowns + bootstrap CIs + every per-case prediction) for each system, built with the same `app.evaluation.reports.overall_report` function Step 14C uses.
- `comparison.json`: improvement metrics, grouped comparisons, per-class comparison, confusion matrices (+ diff), case-level fixed/regressed/unchanged lists, McNemar test, negation/historical focus.
- `language_comparison.json` / `category_comparison.json` / `difficulty_comparison.json`: standalone copies of the corresponding `comparison.json` sections.
- `confusion_matrices.json`: standalone copy of the confusion-matrix section.
- `failure_analysis.json`: fixed/regressed/unchanged case-level detail.

## Future work (explicitly out of scope for this step)

- No threshold recalibration was performed for either system.
- No model fine-tuning or training was performed.
- No new benchmark cases were added or removed.
