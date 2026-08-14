# Step 14C Evaluation Results

Results of running the EXISTING, UNMODIFIED Step 14B NLI evidence verifier against the Step 14E synthetic gold benchmark (`datasets/nli_evaluation/benchmark.json`).

## Research integrity

1. This is a synthetic benchmark (101 hand-authored cases), not a clinical dataset.
2. The NLI model (`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`) was evaluated as-is, without fine-tuning.
3. Thresholds (entailment=0.70, contradiction=0.70) were NOT optimized against this benchmark.
4. Tanglish is evaluated separately from English (see `language_results.json`).
5. Evidence Unsupported-Claim Rate (UCR) is an evidence-verification metric, not a clinical hallucination rate. UNGROUNDED claims represent evidence-unsupported claims and are used here as a measurable proxy for unsupported documentation behavior; they are not treated as a direct clinical hallucination rate.
6. Contradiction Rate (CR) is reported separately from UCR -- they represent different failure modes and are never merged.
7. These results do NOT establish clinical safety, clinical validity, real-world diagnostic accuracy, or HIPAA compliance.

## Reproducibility metadata

- Model provider: `local`
- Model name: `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`
- Device: `cpu`
- Entailment threshold: `0.7`
- Contradiction threshold: `0.7`
- Benchmark file: `D:\MediScribeAI\MediScribeAI\datasets\nli_evaluation\benchmark.json`
- Benchmark SHA-256: `db4d78a94879c2869d1abed423f46cbfd5a4a54e5d24d016c3c2f34faa4b0543`
- Number of cases evaluated: `101`
- Evaluation timestamp (UTC): `2026-08-14T18:08:15.511296+00:00`

## Overall results

- Cases: 101
- Accuracy: 0.5941 (59.41%)
- Macro precision: 0.6559
- Macro recall: 0.5972
- Macro F1: 0.5829
- Weighted F1: 0.5828
- Evidence Unsupported-Claim Rate (prediction-based): 0.4752 (47.52%)
- Evidence Unsupported-Claim Rate (gold-based, benchmark composition reference): 0.3069 (30.69%)
- Contradiction Rate (prediction-based): 0.1386 (13.86%)
- Contradiction Rate (gold-based, benchmark composition reference): 0.3267 (32.67%)

### Confusion matrix (rows = ground truth, columns = prediction)

```
GOLD \ PRED   SUPPORTED     CONTRADICTED  UNGROUNDED    
SUPPORTED     23            0             14            
CONTRADICTED  12            12            9             
UNGROUNDED    4             2             25            
```

## Language breakdown

See `language_results.json` for full per-language precision/recall/F1/UCR/CR. Summary:

| Language | Count | Accuracy | Macro F1 | UCR (pred) | CR (pred) |
|---|---|---|---|---|---|
| en | 39 | 76.92% | 0.7726 | 46.15% | 23.08% |
| tanglish | 37 | 43.24% | 0.4099 | 45.95% | 8.11% |
| mixed | 25 | 56.00% | 0.5343 | 52.00% | 8.00% |

## Category breakdown

See `category_results.json`. Categories with very small sample counts should not be over-interpreted.

## Difficulty breakdown

See `difficulty_results.json`.

## Failure analysis

41 of 101 cases had ground_truth != prediction. Full detail (including premise/claim text, since all data is synthetic) is in `failure_analysis.json`. 'possible_failure_categories' are advisory groupings based on case metadata, not asserted causes.

## Statistical caution

This benchmark contains ~100 synthetic cases. No claims of clinical safety, clinical accuracy, patient safety, HIPAA compliance, or real-world diagnostic accuracy are made or implied by these results. Any bootstrap confidence intervals reported reflect sampling uncertainty over this specific 101-case benchmark only, not real-world generalization.

## Future work (explicitly out of scope for this step)

- Threshold calibration/optimization was NOT performed and is left as future work.
- No model fine-tuning or training was performed.
