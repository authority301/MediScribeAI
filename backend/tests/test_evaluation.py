"""Step 14C evaluation-harness tests.

NEVER loads the real NLI model: every test that exercises runner.evaluate_case
monkeypatches app.evaluation.runner.run_nli. Metrics tests use hand-computed
expected values on a tiny synthetic record set so the arithmetic itself is
verified independent of any model behavior.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from app.evaluation import metrics, reports, runner
from app.nli.model import NliScores

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BENCHMARK_PATH = REPO_ROOT / "datasets" / "nli_evaluation" / "benchmark.json"


# ---------------------------------------------------------------------------
# 1-3: benchmark loading / structure
# ---------------------------------------------------------------------------


def test_benchmark_loads_successfully():
    cases = runner.load_benchmark(BENCHMARK_PATH)
    assert isinstance(cases, list)
    assert len(cases) > 0


def test_exactly_101_cases_present():
    cases = runner.load_benchmark(BENCHMARK_PATH)
    assert len(cases) == 101


def test_required_fields_exist_on_every_case():
    cases = runner.load_benchmark(BENCHMARK_PATH)
    for case in cases:
        missing = runner.REQUIRED_FIELDS - set(case.keys())
        assert not missing, f"case {case.get('id')} missing {missing}"


def test_validate_benchmark_rejects_missing_field():
    bad_cases = [{"id": "X-1", "language": "en", "premise": "p", "claim": "c", "ground_truth": "SUPPORTED"}]
    with pytest.raises(runner.BenchmarkValidationError):
        runner.validate_benchmark(bad_cases)


def test_validate_benchmark_rejects_duplicate_id():
    case = {
        "id": "X-1", "language": "en", "premise": "p", "claim": "c",
        "ground_truth": "SUPPORTED", "category": "symptoms", "difficulty": "easy",
    }
    with pytest.raises(runner.BenchmarkValidationError):
        runner.validate_benchmark([case, dict(case)])


# ---------------------------------------------------------------------------
# 4: predictions stay separate from gold data
# ---------------------------------------------------------------------------


def test_evaluate_case_does_not_mutate_input_case(monkeypatch):
    case = {
        "id": "T-1", "language": "en", "premise": "I have fever.", "claim": "Patient has fever.",
        "ground_truth": "SUPPORTED", "category": "symptoms", "difficulty": "easy",
    }
    original_keys = set(case.keys())

    monkeypatch.setattr(
        "app.evaluation.runner.run_nli",
        lambda premise, hypothesis: NliScores(entailment=0.9, neutral=0.05, contradiction=0.05, label="ENTAILMENT"),
    )
    record = runner.evaluate_case(case)

    assert set(case.keys()) == original_keys  # gold case dict untouched
    assert "prediction" not in case
    assert record["prediction"] == "SUPPORTED"
    assert record is not case


# ---------------------------------------------------------------------------
# 5-11: metrics correctness on a tiny hand-computed example
#
# r1 gold=SUPPORTED    pred=SUPPORTED
# r2 gold=SUPPORTED    pred=UNGROUNDED
# r3 gold=CONTRADICTED pred=CONTRADICTED
# r4 gold=CONTRADICTED pred=SUPPORTED
# r5 gold=UNGROUNDED   pred=UNGROUNDED
# r6 gold=UNGROUNDED   pred=UNGROUNDED
# ---------------------------------------------------------------------------


def _tiny_records():
    return [
        {"ground_truth": "SUPPORTED", "prediction": "SUPPORTED"},
        {"ground_truth": "SUPPORTED", "prediction": "UNGROUNDED"},
        {"ground_truth": "CONTRADICTED", "prediction": "CONTRADICTED"},
        {"ground_truth": "CONTRADICTED", "prediction": "SUPPORTED"},
        {"ground_truth": "UNGROUNDED", "prediction": "UNGROUNDED"},
        {"ground_truth": "UNGROUNDED", "prediction": "UNGROUNDED"},
    ]


def test_confusion_matrix_orientation():
    matrix = metrics.confusion_matrix(_tiny_records())
    # rows = ground truth, columns = prediction
    assert matrix["SUPPORTED"]["SUPPORTED"] == 1
    assert matrix["SUPPORTED"]["UNGROUNDED"] == 1
    assert matrix["SUPPORTED"]["CONTRADICTED"] == 0
    assert matrix["CONTRADICTED"]["CONTRADICTED"] == 1
    assert matrix["CONTRADICTED"]["SUPPORTED"] == 1
    assert matrix["UNGROUNDED"]["UNGROUNDED"] == 2


def test_supported_precision_recall_f1():
    per_class = metrics.per_class_prf(metrics.confusion_matrix(_tiny_records()))
    supported = per_class["SUPPORTED"]
    assert supported["precision"] == pytest.approx(0.5)
    assert supported["recall"] == pytest.approx(0.5)
    assert supported["f1"] == pytest.approx(0.5)
    assert supported["support"] == 2


def test_contradicted_precision_recall_f1():
    per_class = metrics.per_class_prf(metrics.confusion_matrix(_tiny_records()))
    contradicted = per_class["CONTRADICTED"]
    assert contradicted["precision"] == pytest.approx(1.0)
    assert contradicted["recall"] == pytest.approx(0.5)
    assert contradicted["f1"] == pytest.approx(2 / 3)
    assert contradicted["support"] == 2


def test_ungrounded_precision_recall_f1():
    per_class = metrics.per_class_prf(metrics.confusion_matrix(_tiny_records()))
    ungrounded = per_class["UNGROUNDED"]
    assert ungrounded["precision"] == pytest.approx(2 / 3)
    assert ungrounded["recall"] == pytest.approx(1.0)
    assert ungrounded["f1"] == pytest.approx(0.8)
    assert ungrounded["support"] == 2


def test_macro_metrics():
    per_class = metrics.per_class_prf(metrics.confusion_matrix(_tiny_records()))
    macro = metrics.macro_metrics(per_class)
    assert macro["precision"] == pytest.approx((0.5 + 1.0 + 2 / 3) / 3)
    assert macro["recall"] == pytest.approx((0.5 + 0.5 + 1.0) / 3)
    assert macro["f1"] == pytest.approx((0.5 + 2 / 3 + 0.8) / 3)


def test_accuracy():
    assert metrics.accuracy(_tiny_records()) == pytest.approx(4 / 6)


def test_ucr_prediction_and_gold_based():
    records = _tiny_records()
    # prediction-based: r2, r5, r6 predicted UNGROUNDED -> 3/6
    assert metrics.unsupported_claim_rate(records, basis="prediction") == pytest.approx(3 / 6)
    # gold-based: r5, r6 are gold UNGROUNDED -> 2/6
    assert metrics.unsupported_claim_rate(records, basis="gold") == pytest.approx(2 / 6)


def test_cr_prediction_and_gold_based():
    records = _tiny_records()
    # prediction-based: only r3 predicted CONTRADICTED -> 1/6
    assert metrics.contradiction_rate(records, basis="prediction") == pytest.approx(1 / 6)
    # gold-based: r3, r4 are gold CONTRADICTED -> 2/6
    assert metrics.contradiction_rate(records, basis="gold") == pytest.approx(2 / 6)


def test_ucr_and_cr_remain_analytically_separate():
    records = _tiny_records()
    ucr = metrics.unsupported_claim_rate(records, basis="prediction")
    cr = metrics.contradiction_rate(records, basis="prediction")
    assert ucr != cr
    # Sanity: they must be computed independently, not derived from one another.
    assert ucr + cr <= 1.0


# ---------------------------------------------------------------------------
# 12: confusion matrix orientation is also verified above (test_confusion_matrix_orientation)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 13-15: language / category / difficulty breakdown
# ---------------------------------------------------------------------------


def _tagged_records():
    return [
        {"ground_truth": "SUPPORTED", "prediction": "SUPPORTED", "language": "en",
         "category": "symptoms", "difficulty": "easy"},
        {"ground_truth": "CONTRADICTED", "prediction": "SUPPORTED", "language": "en",
         "category": "negation", "difficulty": "medium"},
        {"ground_truth": "SUPPORTED", "prediction": "SUPPORTED", "language": "tanglish",
         "category": "symptoms", "difficulty": "medium"},
        {"ground_truth": "UNGROUNDED", "prediction": "UNGROUNDED", "language": "tanglish",
         "category": "diagnosis_overreach", "difficulty": "hard"},
        {"ground_truth": "CONTRADICTED", "prediction": "CONTRADICTED", "language": "mixed",
         "category": "negation", "difficulty": "hard"},
    ]


def test_language_breakdown():
    breakdown = reports.breakdown_by(_tagged_records(), "language", reports.LANGUAGES)
    assert breakdown["en"]["count"] == 2
    assert breakdown["en"]["accuracy"] == pytest.approx(0.5)
    assert breakdown["tanglish"]["count"] == 2
    assert breakdown["tanglish"]["accuracy"] == pytest.approx(1.0)
    assert breakdown["mixed"]["count"] == 1
    assert breakdown["mixed"]["accuracy"] == pytest.approx(1.0)


def test_category_breakdown():
    breakdown = reports.breakdown_by(_tagged_records(), "category")
    assert breakdown["symptoms"]["count"] == 2
    assert breakdown["negation"]["count"] == 2
    assert breakdown["negation"]["accuracy"] == pytest.approx(0.5)
    assert breakdown["diagnosis_overreach"]["count"] == 1


def test_difficulty_breakdown():
    breakdown = reports.breakdown_by(_tagged_records(), "difficulty", reports.DIFFICULTIES)
    assert breakdown["easy"]["count"] == 1
    assert breakdown["medium"]["count"] == 2
    assert breakdown["hard"]["count"] == 2
    assert breakdown["hard"]["accuracy"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 16: reproducibility
# ---------------------------------------------------------------------------


def test_evaluation_is_reproducible_with_mocked_inference(monkeypatch):
    case = {
        "id": "T-2", "language": "tanglish", "premise": "Enakku fever illa.",
        "claim": "Patient has fever.", "ground_truth": "CONTRADICTED",
        "category": "negation", "difficulty": "hard",
    }

    monkeypatch.setattr(
        "app.evaluation.runner.run_nli",
        lambda premise, hypothesis: NliScores(entailment=0.02, neutral=0.05, contradiction=0.93, label="CONTRADICTION"),
    )
    record_1 = runner.evaluate_case(case)
    record_2 = runner.evaluate_case(case)
    assert record_1 == record_2


def test_bootstrap_ci_is_reproducible_with_fixed_seed():
    records = _tiny_records()
    ci_1 = metrics.bootstrap_ci(records, metrics.accuracy, n_resamples=200, seed=7)
    ci_2 = metrics.bootstrap_ci(records, metrics.accuracy, n_resamples=200, seed=7)
    assert ci_1 == ci_2


def test_failure_analysis_marks_possible_categories_not_causes():
    records = [
        {"case_id": "TL-002", "language": "tanglish", "category": "symptoms", "difficulty": "hard",
         "ground_truth": "CONTRADICTED", "prediction": "SUPPORTED", "nli_label": "ENTAILMENT",
         "entailment_score": 0.8, "contradiction_score": 0.05, "neutral_score": 0.15},
    ]
    cases_by_id = {
        "TL-002": {
            "id": "TL-002", "premise": "Enakku chest pain illa.", "claim": "Patient reports chest pain.",
            "category": "symptoms", "contains_negation": True, "contains_code_switching": True,
        }
    }
    failures = reports.failure_analysis(records, cases_by_id)
    assert len(failures) == 1
    assert failures[0]["case_id"] == "TL-002"
    assert "negation + code-switching" in failures[0]["possible_failure_categories"]


def test_failure_analysis_skips_correct_predictions():
    records = [
        {"case_id": "EN-001", "language": "en", "category": "symptoms", "difficulty": "easy",
         "ground_truth": "SUPPORTED", "prediction": "SUPPORTED", "nli_label": "ENTAILMENT",
         "entailment_score": 0.9, "contradiction_score": 0.02, "neutral_score": 0.08},
    ]
    cases_by_id = {"EN-001": {"id": "EN-001", "premise": "p", "claim": "c", "category": "symptoms"}}
    assert reports.failure_analysis(records, cases_by_id) == []


def test_benchmark_json_is_not_modified_by_evaluation(monkeypatch):
    cases = runner.load_benchmark(BENCHMARK_PATH)
    before = BENCHMARK_PATH.read_text(encoding="utf-8")

    monkeypatch.setattr(
        "app.evaluation.runner.run_nli",
        lambda premise, hypothesis: NliScores(entailment=0.4, neutral=0.4, contradiction=0.2, label="NEUTRAL"),
    )
    runner.run_evaluation(cases[:5])

    after = BENCHMARK_PATH.read_text(encoding="utf-8")
    assert before == after
