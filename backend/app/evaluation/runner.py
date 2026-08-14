"""Step 14C evaluation runner: loads the Step 14E gold benchmark and scores
the EXISTING, UNMODIFIED Step 14B NLI verifier against it.

This module deliberately does NOT reimplement any NLI inference or
aggregation logic. Each benchmark case is a single premise/claim pair (no
consultation/SOAP/evidence-link records exist for it), so rather than
constructing a full fake production database per case, this runner adapts
the benchmark's premise/claim directly into the same functions production
uses:
  - app.nli.model.run_nli            (the real model wrapper)
  - app.nli.aggregation.EvidenceNliResult / aggregate_claim_verification
    (the real, unmodified claim-level decision rule)
  - app.nli.config thresholds        (the real, unmodified thresholds)

A benchmark case is treated as a claim with exactly one piece of evidence
(its own premise) -- aggregate_claim_verification is called with a
single-element list, which is exactly how it behaves for a Step 14A claim
that retrieved only one evidence segment. No second aggregation algorithm is
introduced.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from app.nli.aggregation import EvidenceNliResult, aggregate_claim_verification
from app.nli.config import (
    NLI_CONTRADICTION_THRESHOLD,
    NLI_DEVICE,
    NLI_ENTAILMENT_THRESHOLD,
    NLI_MODEL_NAME,
    NLI_MODEL_PROVIDER,
)
from app.nli.model import run_nli

REQUIRED_FIELDS = {"id", "language", "premise", "claim", "ground_truth", "category", "difficulty"}
VALID_LANGUAGES = {"en", "tanglish", "mixed"}
VALID_GROUND_TRUTH = {"SUPPORTED", "CONTRADICTED", "UNGROUNDED"}


class BenchmarkValidationError(Exception):
    pass


def validate_benchmark(cases: list[dict]) -> None:
    seen_ids = set()
    for case in cases:
        missing = REQUIRED_FIELDS - set(case.keys())
        if missing:
            raise BenchmarkValidationError(f"case {case.get('id')} missing required fields: {missing}")
        if case["id"] in seen_ids:
            raise BenchmarkValidationError(f"duplicate case id: {case['id']}")
        seen_ids.add(case["id"])
        if case["language"] not in VALID_LANGUAGES:
            raise BenchmarkValidationError(f"case {case['id']} has invalid language: {case['language']!r}")
        if case["ground_truth"] not in VALID_GROUND_TRUTH:
            raise BenchmarkValidationError(f"case {case['id']} has invalid ground_truth: {case['ground_truth']!r}")
        if not case["premise"].strip() or not case["claim"].strip():
            raise BenchmarkValidationError(f"case {case['id']} has an empty premise or claim")


def load_benchmark(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        cases = json.load(f)
    validate_benchmark(cases)
    return cases


def benchmark_hash(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def evaluate_case(case: dict) -> dict:
    """Run ONE benchmark case through the real Step 14B model + aggregation.
    Never mutates the input case dict -- the gold benchmark stays untouched
    and predictions are returned separately.
    """
    nli_result = run_nli(premise=case["premise"], hypothesis=case["claim"])

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
        "prediction": prediction,
        "nli_label": nli_result.label,
        "entailment_score": nli_result.entailment,
        "contradiction_score": nli_result.contradiction,
        "neutral_score": nli_result.neutral,
    }


def run_evaluation(cases: list[dict]) -> list[dict]:
    return [evaluate_case(case) for case in cases]


def evaluation_metadata(benchmark_path: Path, num_cases: int) -> dict:
    return {
        "model_provider": NLI_MODEL_PROVIDER,
        "model_name": NLI_MODEL_NAME,
        "device": NLI_DEVICE,
        "entailment_threshold": NLI_ENTAILMENT_THRESHOLD,
        "contradiction_threshold": NLI_CONTRADICTION_THRESHOLD,
        "benchmark_path": str(benchmark_path),
        "benchmark_sha256": benchmark_hash(benchmark_path),
        "num_cases": num_cases,
        "evaluation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "thresholds_tuned_against_benchmark": False,
        "model_fine_tuned": False,
    }
