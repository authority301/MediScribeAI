"""Step 14I evaluation runner: runs every Step 14E benchmark case through
THREE systems for a controlled three-way comparison:

  SYSTEM A (14B):  raw premise -> real run_nli -> real aggregation.
                   Reuses app.evaluation.runner.evaluate_case (Step 14C,
                   unmodified) directly -- no NLI/aggregation logic is
                   reimplemented.
  SYSTEM B (14F):  premise -> FROZEN Step 14F normalizer (see
                   frozen_14f/) -> the SAME real run_nli -> the SAME real
                   aggregation.
  SYSTEM C (14H):  premise -> CURRENT (Step 14H) normalizer -> the SAME
                   real run_nli -> the SAME real aggregation. Reuses
                   app.evaluation_tanglish.runner.evaluate_case_mode(case,
                   "tanglish_aware") directly (Step 14G's own runner,
                   unmodified) -- which itself calls
                   app.tanglish.service.verify(mode="tanglish_aware"),
                   i.e. the real, current app.tanglish.normalizer.

All three paths converge on the exact same
app.nli.model.run_nli / app.nli.aggregation.aggregate_claim_verification /
app.nli.config thresholds. Only the text handed to run_nli as the premise
differs between systems.
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
from app.evaluation.runner import evaluate_case as evaluate_case_14b_raw
from app.evaluation_tanglish.runner import evaluate_case_mode as evaluate_case_14h_raw
from app.evaluation_tanglish_refinement.frozen_14f.normalizer import normalize as normalize_14f
from app.nli.aggregation import EvidenceNliResult, aggregate_claim_verification
from app.nli.config import (
    NLI_CONTRADICTION_THRESHOLD,
    NLI_DEVICE,
    NLI_ENTAILMENT_THRESHOLD,
    NLI_MODEL_NAME,
    NLI_MODEL_PROVIDER,
)
from app.nli.model import run_nli

SYSTEM_14B = "14B"
SYSTEM_14F = "14F"
SYSTEM_14H = "14H"


def _transformation_dicts(normalization) -> list[dict]:
    if normalization is None:
        return []
    return [
        {"rule": t.rule, "input_span": t.input_span, "normalized_span": t.normalized_span}
        for t in normalization.transformations
    ]


def evaluate_case_14b(case: dict) -> dict:
    """SYSTEM A. Delegates entirely to the frozen Step 14C runner; only adds
    the normalized_premise/transformations keys (both trivial for the
    baseline: premise unchanged, no rules fire) so all three systems' output
    records share one schema for the comparison code.
    """
    record = evaluate_case_14b_raw(case)
    record["system"] = SYSTEM_14B
    record["premise"] = case["premise"]
    record["normalized_premise"] = case["premise"]
    record["transformations"] = []
    return record


def evaluate_case_14f(case: dict) -> dict:
    """SYSTEM B. Mirrors app.tanglish.service.verify(mode="tanglish_aware")'s
    glue logic exactly, but with the FROZEN Step 14F normalizer substituted
    for the current (Step 14H) one. Never reimplements run_nli or
    aggregate_claim_verification -- both are the same real functions Step
    14B/14C/14G use.
    """
    normalization = normalize_14f(case["premise"])
    nli_premise = normalization.normalized_text

    nli_result = run_nli(premise=nli_premise, hypothesis=case["claim"])
    evidence_result = EvidenceNliResult(
        speaker_segment_id=case["id"],
        nli_label=nli_result.label,
        entailment_score=nli_result.entailment,
        contradiction_score=nli_result.contradiction,
        neutral_score=nli_result.neutral,
    )
    prediction = aggregate_claim_verification(
        [evidence_result], NLI_ENTAILMENT_THRESHOLD, NLI_CONTRADICTION_THRESHOLD
    )

    return {
        "case_id": case["id"],
        "language": case["language"],
        "category": case["category"],
        "difficulty": case["difficulty"],
        "ground_truth": case["ground_truth"],
        "system": SYSTEM_14F,
        "prediction": prediction,
        "nli_label": nli_result.label,
        "entailment_score": nli_result.entailment,
        "contradiction_score": nli_result.contradiction,
        "neutral_score": nli_result.neutral,
        "premise": case["premise"],
        "normalized_premise": nli_premise,
        "transformations": _transformation_dicts(normalization),
    }


def evaluate_case_14h(case: dict) -> dict:
    """SYSTEM C. Thin adapter over the frozen Step 14G runner (which itself
    calls the CURRENT, Step-14H-refined app.tanglish.normalizer via
    app.tanglish.service) -- only relabels/reshapes the record to match the
    shared three-system schema. Nothing about Step 14G's own module is
    modified or reimplemented here.
    """
    record = evaluate_case_14h_raw(case, "tanglish_aware")
    return {
        "case_id": record["case_id"],
        "language": record["language"],
        "category": record["category"],
        "difficulty": record["difficulty"],
        "ground_truth": record["ground_truth"],
        "system": SYSTEM_14H,
        "prediction": record["prediction"],
        "nli_label": record["nli_label"],
        "entailment_score": record["entailment_score"],
        "contradiction_score": record["contradiction_score"],
        "neutral_score": record["neutral_score"],
        "premise": record["premise"],
        "normalized_premise": record["normalized_premise"],
        "transformations": record["transformations"],
    }


def run_all_systems(cases: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Returns (records_14b, records_14f, records_14h), same case order."""
    records_14b = [evaluate_case_14b(case) for case in cases]
    records_14f = [evaluate_case_14f(case) for case in cases]
    records_14h = [evaluate_case_14h(case) for case in cases]
    return records_14b, records_14f, records_14h


def evaluation_metadata(benchmark_path: Path, num_cases: int) -> dict:
    return {
        "model_provider": NLI_MODEL_PROVIDER,
        "model_name": NLI_MODEL_NAME,
        "device": NLI_DEVICE,
        "entailment_threshold": NLI_ENTAILMENT_THRESHOLD,
        "contradiction_threshold": NLI_CONTRADICTION_THRESHOLD,
        "system_14b_configuration": "raw premise, no Tanglish preprocessing (app.evaluation.runner, Step 14C, unmodified)",
        "system_14f_configuration": (
            "premise normalized by FROZEN Step 14F normalizer "
            "(app.evaluation_tanglish_refinement.frozen_14f, byte-identical-logic snapshot of commit 5563418)"
        ),
        "system_14h_configuration": (
            "premise normalized by CURRENT Step 14H normalizer (app.tanglish.normalizer, "
            "via app.evaluation_tanglish.runner.evaluate_case_mode(mode='tanglish_aware'), unmodified)"
        ),
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
