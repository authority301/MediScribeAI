"""Validation for the Step 14E gold NLI evaluation benchmark
(datasets/nli_evaluation/benchmark.json).

This module NEVER imports the NLI model, transformers, or torch, and never
runs inference -- it only checks the structural and statistical integrity
of the gold dataset itself. Ground truth in that dataset was hand-reasoned,
independent of any model prediction; these tests exist to catch dataset
authoring mistakes (bad IDs, missing fields, duplicate cases, severe class
imbalance, accidental leakage of model-derived fields), not to measure model
accuracy.
"""
import json
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BENCHMARK_PATH = REPO_ROOT / "datasets" / "nli_evaluation" / "benchmark.json"

VALID_LANGUAGES = {"en", "tanglish", "mixed"}
VALID_GROUND_TRUTH = {"SUPPORTED", "CONTRADICTED", "UNGROUNDED"}
VALID_CATEGORIES = {
    "symptoms",
    "medications",
    "allergies",
    "measurements",
    "duration",
    "frequency",
    "medical_procedures",
    "diagnoses",
    "follow_up_instructions",
    "referral_instructions",
    "patient_history",
    "doctor_assessment",
    "negation",
    "historical_statements",
    "diagnosis_overreach",
}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}
REQUIRED_FIELDS = {"id", "language", "premise", "claim", "ground_truth", "category", "difficulty"}
FORBIDDEN_MODEL_DERIVED_FIELDS = {
    "nli_label",
    "entailment_score",
    "contradiction_score",
    "neutral_score",
    "model_prediction",
    "retrieval_score",
    "alignment_score",
    "hallucination_status",
    "hallucination_rate",
    "confidence",
    "verification_status",
}


@pytest.fixture(scope="module")
def benchmark():
    assert BENCHMARK_PATH.exists(), f"benchmark.json not found at {BENCHMARK_PATH}"
    with open(BENCHMARK_PATH, encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list)
    return data


def test_benchmark_file_exists_and_is_a_list(benchmark):
    assert len(benchmark) > 0


def test_approximately_100_cases(benchmark):
    assert 90 <= len(benchmark) <= 110


def test_all_ids_unique(benchmark):
    ids = [case.get("id") for case in benchmark]
    assert len(ids) == len(set(ids))


def test_required_fields_present(benchmark):
    for case in benchmark:
        missing = REQUIRED_FIELDS - set(case.keys())
        assert not missing, f"case {case.get('id')} missing fields: {missing}"


def test_language_values_are_valid(benchmark):
    for case in benchmark:
        assert case["language"] in VALID_LANGUAGES, f"case {case['id']} has invalid language {case['language']!r}"


def test_ground_truth_values_are_valid(benchmark):
    for case in benchmark:
        assert case["ground_truth"] in VALID_GROUND_TRUTH, (
            f"case {case['id']} has invalid ground_truth {case['ground_truth']!r}"
        )


def test_category_values_are_valid(benchmark):
    for case in benchmark:
        assert case["category"] in VALID_CATEGORIES, f"case {case['id']} has invalid category {case['category']!r}"


def test_difficulty_values_are_valid(benchmark):
    for case in benchmark:
        assert case["difficulty"] in VALID_DIFFICULTIES, (
            f"case {case['id']} has invalid difficulty {case['difficulty']!r}"
        )


def test_no_empty_premise(benchmark):
    for case in benchmark:
        assert isinstance(case["premise"], str) and case["premise"].strip(), (
            f"case {case['id']} has an empty premise"
        )


def test_no_empty_claim(benchmark):
    for case in benchmark:
        assert isinstance(case["claim"], str) and case["claim"].strip(), f"case {case['id']} has an empty claim"


def test_no_duplicate_premise_claim_combinations(benchmark):
    pairs = [(case["premise"], case["claim"]) for case in benchmark]
    duplicates = [pair for pair, count in Counter(pairs).items() if count > 1]
    assert not duplicates, f"duplicate premise+claim pairs found: {duplicates}"


def test_approximate_ground_truth_class_balance(benchmark):
    total = len(benchmark)
    counts = Counter(case["ground_truth"] for case in benchmark)
    assert set(counts.keys()) == VALID_GROUND_TRUTH, "every ground_truth label must appear at least once"
    for label, count in counts.items():
        proportion = count / total
        assert 0.20 <= proportion <= 0.50, (
            f"ground_truth {label!r} is {proportion:.0%} of the benchmark -- severe class imbalance"
        )


def test_approximate_language_balance(benchmark):
    total = len(benchmark)
    counts = Counter(case["language"] for case in benchmark)
    assert set(counts.keys()) == VALID_LANGUAGES, "every language must appear at least once"
    for language, count in counts.items():
        proportion = count / total
        assert 0.15 <= proportion <= 0.60, (
            f"language {language!r} is {proportion:.0%} of the benchmark -- severe imbalance"
        )


def test_every_category_is_represented(benchmark):
    counts = Counter(case["category"] for case in benchmark)
    missing = VALID_CATEGORIES - set(counts.keys())
    assert not missing, f"categories with zero cases: {missing}"


def test_no_model_derived_fields_present(benchmark):
    # The gold benchmark must remain strictly model-independent: no NLI
    # predictions, confidence, retrieval scores, or hallucination fields.
    for case in benchmark:
        leaked = FORBIDDEN_MODEL_DERIVED_FIELDS & set(case.keys())
        assert not leaked, f"case {case['id']} contains model-derived field(s): {leaked}"


def test_schema_json_is_valid_json():
    schema_path = REPO_ROOT / "datasets" / "nli_evaluation" / "schema.json"
    assert schema_path.exists()
    with open(schema_path, encoding="utf-8") as f:
        schema = json.load(f)
    assert schema["type"] == "array"
