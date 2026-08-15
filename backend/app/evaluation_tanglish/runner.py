"""Step 14G evaluation runner.

Runs every Step 14E benchmark case through BOTH systems being compared:
  - SYSTEM A (baseline):       app.tanglish.service.verify(mode="baseline")
  - SYSTEM B (tanglish_aware): app.tanglish.service.verify(mode="tanglish_aware")

Both modes call the exact same, unmodified app.nli.model.run_nli and
app.nli.aggregation.aggregate_claim_verification functions with the exact
same thresholds (app.nli.config) -- see app/tanglish/service.py. This
module does not reimplement NLI inference, aggregation, or benchmark
loading/validation; benchmark loading/hashing is reused directly from
app.evaluation.runner (Step 14C), unmodified.
"""
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from app.evaluation.runner import (  # noqa: F401  (re-exported for callers)
    REQUIRED_FIELDS,
    BenchmarkValidationError,
    benchmark_hash,
    load_benchmark,
    validate_benchmark,
)
from app.nli.config import (
    NLI_CONTRADICTION_THRESHOLD,
    NLI_DEVICE,
    NLI_ENTAILMENT_THRESHOLD,
    NLI_MODEL_NAME,
    NLI_MODEL_PROVIDER,
)
from app.tanglish import service
from app.tanglish.config import MODE_BASELINE, MODE_TANGLISH_AWARE


def _transformation_dicts(normalization) -> list[dict]:
    if normalization is None:
        return []
    return [
        {"rule": t.rule, "input_span": t.input_span, "normalized_span": t.normalized_span}
        for t in normalization.transformations
    ]


def evaluate_case_mode(case: dict, mode: str) -> dict:
    """Run ONE benchmark case through ONE mode of the real Step 14B model
    (via app.tanglish.service.verify). Never mutates the input case dict.
    """
    result = service.verify(premise=case["premise"], hypothesis=case["claim"], mode=mode)
    normalization = result["normalization"]

    return {
        "case_id": case["id"],
        "language": case["language"],
        "category": case["category"],
        "difficulty": case["difficulty"],
        "ground_truth": case["ground_truth"],
        "mode": mode,
        "prediction": result["prediction"],
        "nli_label": result["nli_label"],
        "entailment_score": result["entailment_score"],
        "contradiction_score": result["contradiction_score"],
        "neutral_score": result["neutral_score"],
        "premise": case["premise"],
        # For baseline, normalized_premise always equals the original premise
        # (explicitly -- baseline never calls the normalizer). For
        # tanglish_aware, it is the text actually sent to run_nli.
        "normalized_premise": result["nli_premise"],
        "normalization_applied": mode == MODE_TANGLISH_AWARE,
        "detected_language": normalization.detected_language if normalization else None,
        "is_code_switched": normalization.is_code_switched if normalization else False,
        "transformations": _transformation_dicts(normalization),
    }


def run_mode(cases: list[dict], mode: str) -> list[dict]:
    return [evaluate_case_mode(case, mode) for case in cases]


def run_both_modes(cases: list[dict]) -> tuple[list[dict], list[dict]]:
    """Returns (baseline_records, tanglish_aware_records), same case order."""
    baseline_records = run_mode(cases, MODE_BASELINE)
    tanglish_records = run_mode(cases, MODE_TANGLISH_AWARE)
    return baseline_records, tanglish_records


def evaluation_metadata(benchmark_path: Path, num_cases: int) -> dict:
    return {
        "model_provider": NLI_MODEL_PROVIDER,
        "model_name": NLI_MODEL_NAME,
        "device": NLI_DEVICE,
        "entailment_threshold": NLI_ENTAILMENT_THRESHOLD,
        "contradiction_threshold": NLI_CONTRADICTION_THRESHOLD,
        "baseline_mode": MODE_BASELINE,
        "tanglish_aware_mode": MODE_TANGLISH_AWARE,
        "benchmark_path": str(benchmark_path),
        "benchmark_sha256": benchmark_hash(benchmark_path),
        "num_cases": num_cases,
        "evaluation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "thresholds_tuned_against_benchmark": False,
        "thresholds_tuned_separately_per_system": False,
        "model_fine_tuned": False,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }
