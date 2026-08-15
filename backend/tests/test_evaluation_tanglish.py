"""Step 14G comparison-harness tests.

NEVER loads the real NLI model: every test that exercises
app.evaluation_tanglish.runner.evaluate_case_mode / run_mode / run_both_modes
monkeypatches app.tanglish.service.run_nli (the single point where the real
Step 14B model would be invoked). Comparison-math tests use hand-computed
expected values on small synthetic record sets, independent of any model
behavior.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from app.evaluation import runner as base_runner
from app.evaluation_tanglish import comparison as cmp
from app.evaluation_tanglish import runner as tg_runner
from app.nli.model import NliScores
from app.tanglish.config import MODE_BASELINE, MODE_TANGLISH_AWARE

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BENCHMARK_PATH = REPO_ROOT / "datasets" / "nli_evaluation" / "benchmark.json"


def _fixed_nli(premise, hypothesis):
    """Deterministic fake NLI: SUPPORTED-ish output regardless of input."""
    return NliScores(entailment=0.9, neutral=0.05, contradiction=0.05, label="ENTAILMENT")


def _echo_case(case_id="X-1", language="en", premise="I have fever.", claim="Patient has fever.",
               ground_truth="SUPPORTED", category="symptoms", difficulty="easy"):
    return {
        "id": case_id, "language": language, "premise": premise, "claim": claim,
        "ground_truth": ground_truth, "category": category, "difficulty": difficulty,
    }


# ---------------------------------------------------------------------------
# 1-3: both modes evaluate the same case set / case ids / ground truth
# ---------------------------------------------------------------------------
def test_both_modes_evaluate_same_case_set(monkeypatch):
    monkeypatch.setattr("app.tanglish.service.run_nli", _fixed_nli)
    cases = [
        _echo_case("A-1"),
        _echo_case("A-2", language="tanglish", premise="Enakku fever irukku.", ground_truth="SUPPORTED"),
    ]
    baseline_records, tanglish_records = tg_runner.run_both_modes(cases)
    assert len(baseline_records) == len(cases)
    assert len(tanglish_records) == len(cases)


def test_case_ids_match_exactly(monkeypatch):
    monkeypatch.setattr("app.tanglish.service.run_nli", _fixed_nli)
    cases = [_echo_case("B-1"), _echo_case("B-2")]
    baseline_records, tanglish_records = tg_runner.run_both_modes(cases)
    input_ids = {c["id"] for c in cases}
    assert {r["case_id"] for r in baseline_records} == input_ids
    assert {r["case_id"] for r in tanglish_records} == input_ids


def test_ground_truth_labels_match_exactly(monkeypatch):
    monkeypatch.setattr("app.tanglish.service.run_nli", _fixed_nli)
    cases = [_echo_case("C-1", ground_truth="CONTRADICTED"), _echo_case("C-2", ground_truth="UNGROUNDED")]
    baseline_records, tanglish_records = tg_runner.run_both_modes(cases)
    b_by_id = tg_runner.evaluate_case_mode  # noqa: F841 (not used directly)
    baseline_gt = {r["case_id"]: r["ground_truth"] for r in baseline_records}
    tanglish_gt = {r["case_id"]: r["ground_truth"] for r in tanglish_records}
    case_gt = {c["id"]: c["ground_truth"] for c in cases}
    assert baseline_gt == case_gt
    assert tanglish_gt == case_gt


# ---------------------------------------------------------------------------
# 4-5: reproducibility
# ---------------------------------------------------------------------------
def test_baseline_predictions_are_reproducible(monkeypatch):
    monkeypatch.setattr("app.tanglish.service.run_nli", _fixed_nli)
    case = _echo_case("D-1")
    record_1 = tg_runner.evaluate_case_mode(case, MODE_BASELINE)
    record_2 = tg_runner.evaluate_case_mode(case, MODE_BASELINE)
    assert record_1 == record_2


def test_tanglish_aware_predictions_are_reproducible(monkeypatch):
    monkeypatch.setattr("app.tanglish.service.run_nli", _fixed_nli)
    case = _echo_case("D-2", language="tanglish", premise="Enakku fever illa.", ground_truth="CONTRADICTED",
                       category="negation", difficulty="hard")
    record_1 = tg_runner.evaluate_case_mode(case, MODE_TANGLISH_AWARE)
    record_2 = tg_runner.evaluate_case_mode(case, MODE_TANGLISH_AWARE)
    assert record_1 == record_2


# ---------------------------------------------------------------------------
# 6, 9-10: improvement / UCR / CR calculation correctness (hand-computed)
# ---------------------------------------------------------------------------
def _baseline_records():
    return [
        {"case_id": "1", "ground_truth": "SUPPORTED", "prediction": "SUPPORTED", "language": "en", "category": "symptoms", "difficulty": "easy"},
        {"case_id": "2", "ground_truth": "SUPPORTED", "prediction": "UNGROUNDED", "language": "tanglish", "category": "symptoms", "difficulty": "medium"},
        {"case_id": "3", "ground_truth": "CONTRADICTED", "prediction": "SUPPORTED", "language": "tanglish", "category": "negation", "difficulty": "hard"},
        {"case_id": "4", "ground_truth": "UNGROUNDED", "prediction": "UNGROUNDED", "language": "mixed", "category": "negation", "difficulty": "hard"},
    ]


def _tanglish_records_all_correct():
    return [
        {"case_id": "1", "ground_truth": "SUPPORTED", "prediction": "SUPPORTED", "language": "en", "category": "symptoms", "difficulty": "easy"},
        {"case_id": "2", "ground_truth": "SUPPORTED", "prediction": "SUPPORTED", "language": "tanglish", "category": "symptoms", "difficulty": "medium"},
        {"case_id": "3", "ground_truth": "CONTRADICTED", "prediction": "CONTRADICTED", "language": "tanglish", "category": "negation", "difficulty": "hard"},
        {"case_id": "4", "ground_truth": "UNGROUNDED", "prediction": "UNGROUNDED", "language": "mixed", "category": "negation", "difficulty": "hard"},
    ]


def test_improvement_calculation_is_correct():
    baseline = _baseline_records()
    tanglish = _tanglish_records_all_correct()
    imp = cmp.improvement_metrics(baseline, tanglish)
    # baseline accuracy = 2/4 = 0.5, tanglish accuracy = 4/4 = 1.0
    assert imp["accuracy"]["baseline"] == pytest.approx(0.5)
    assert imp["accuracy"]["tanglish_aware"] == pytest.approx(1.0)
    assert imp["accuracy"]["absolute_change"] == pytest.approx(0.5)
    assert imp["accuracy"]["percentage_point_change"] == pytest.approx(50.0)


def test_ucr_comparison_is_correct():
    baseline = _baseline_records()
    tanglish = _tanglish_records_all_correct()
    imp = cmp.improvement_metrics(baseline, tanglish)
    # baseline UCR (pred): cases 2,4 predicted UNGROUNDED -> 2/4 = 0.5
    # tanglish UCR (pred): case 4 predicted UNGROUNDED -> 1/4 = 0.25
    assert imp["ucr_prediction_based"]["baseline"] == pytest.approx(0.5)
    assert imp["ucr_prediction_based"]["tanglish_aware"] == pytest.approx(0.25)
    assert imp["ucr_prediction_based"]["absolute_change"] == pytest.approx(-0.25)


def test_cr_comparison_is_correct():
    baseline = _baseline_records()
    tanglish = _tanglish_records_all_correct()
    imp = cmp.improvement_metrics(baseline, tanglish)
    # baseline CR (pred): case 3 predicted SUPPORTED -> 0/4; tanglish CR: case 3 predicted CONTRADICTED -> 1/4
    assert imp["cr_prediction_based"]["baseline"] == pytest.approx(0.0)
    assert imp["cr_prediction_based"]["tanglish_aware"] == pytest.approx(0.25)
    assert imp["cr_prediction_based"]["absolute_change"] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# 7: regression calculation correctness
# ---------------------------------------------------------------------------
def test_regression_calculation_is_correct():
    # Case where tanglish_aware breaks a case the baseline got right.
    baseline = [{"case_id": "R-1", "ground_truth": "SUPPORTED", "prediction": "SUPPORTED",
                 "language": "en", "category": "symptoms", "difficulty": "easy"}]
    tanglish = [{"case_id": "R-1", "ground_truth": "SUPPORTED", "prediction": "UNGROUNDED",
                 "language": "en", "category": "symptoms", "difficulty": "easy",
                 "nli_label": "NEUTRAL", "entailment_score": 0.1, "contradiction_score": 0.1,
                 "neutral_score": 0.8, "normalized_premise": "p", "transformations": []}]
    baseline[0].update({"nli_label": "ENTAILMENT", "entailment_score": 0.9,
                         "contradiction_score": 0.05, "neutral_score": 0.05})
    cases_by_id = {"R-1": {"id": "R-1", "premise": "p", "claim": "c"}}

    result = cmp.case_level_comparison(baseline, tanglish, cases_by_id)
    assert result["counts"]["regressed"] == 1
    assert result["counts"]["fixed"] == 0
    assert result["regressed_cases"][0]["case_id"] == "R-1"


def test_fixed_calculation_is_correct():
    baseline = [{"case_id": "F-1", "ground_truth": "CONTRADICTED", "prediction": "SUPPORTED",
                 "language": "tanglish", "category": "negation", "difficulty": "hard",
                 "nli_label": "ENTAILMENT", "entailment_score": 0.8, "contradiction_score": 0.1,
                 "neutral_score": 0.1}]
    tanglish = [{"case_id": "F-1", "ground_truth": "CONTRADICTED", "prediction": "CONTRADICTED",
                 "language": "tanglish", "category": "negation", "difficulty": "hard",
                 "nli_label": "CONTRADICTION", "entailment_score": 0.05, "contradiction_score": 0.9,
                 "neutral_score": 0.05, "normalized_premise": "Patient does not have fever.",
                 "transformations": [{"rule": "TANGLISH_NEGATION_STATE", "input_span": "fever illa",
                                       "normalized_span": "does not have fever"}]}]
    cases_by_id = {"F-1": {"id": "F-1", "premise": "Enakku fever illa.", "claim": "Patient has fever."}}

    result = cmp.case_level_comparison(baseline, tanglish, cases_by_id)
    assert result["counts"]["fixed"] == 1
    assert result["counts"]["regressed"] == 0
    assert result["fixed_cases"][0]["case_id"] == "F-1"
    assert result["fixed_cases"][0]["transformation_trace"][0]["rule"] == "TANGLISH_NEGATION_STATE"


# ---------------------------------------------------------------------------
# 8: percentage-point differences
# ---------------------------------------------------------------------------
def test_percentage_point_change_is_correct():
    assert cmp.pp_change(0.60, 0.50) == pytest.approx(10.0)
    assert cmp.pp_change(0.4324, 0.7692) == pytest.approx(-33.68, abs=0.01)
    assert cmp.pp_change(0.5, 0.5) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 11: confusion matrix orientation (difference matrix)
# ---------------------------------------------------------------------------
def test_confusion_matrix_difference_orientation():
    baseline = _baseline_records()
    tanglish = _tanglish_records_all_correct()
    matrices = cmp.confusion_matrices(baseline, tanglish)
    assert matrices["row_axis"] == "ground_truth"
    assert matrices["column_axis"] == "prediction"
    # baseline: gold SUPPORTED -> pred SUPPORTED once, pred UNGROUNDED once
    assert matrices["baseline"]["SUPPORTED"]["SUPPORTED"] == 1
    assert matrices["baseline"]["SUPPORTED"]["UNGROUNDED"] == 1
    # tanglish: gold SUPPORTED -> pred SUPPORTED twice
    assert matrices["tanglish_aware"]["SUPPORTED"]["SUPPORTED"] == 2
    # difference = tanglish - baseline
    assert matrices["difference_tanglish_minus_baseline"]["SUPPORTED"]["SUPPORTED"] == 1
    assert matrices["difference_tanglish_minus_baseline"]["SUPPORTED"]["UNGROUNDED"] == -1


# ---------------------------------------------------------------------------
# 12-14: language / category / difficulty grouping correctness
# ---------------------------------------------------------------------------
def test_language_grouping_is_correct():
    baseline = _baseline_records()
    tanglish = _tanglish_records_all_correct()
    result = cmp.language_comparison(baseline, tanglish)
    assert result["tanglish"]["count"] == 2
    assert result["tanglish"]["baseline"]["accuracy"] == pytest.approx(0.0)  # case 2 wrong, case 3 wrong -> 0/2
    assert result["tanglish"]["tanglish_aware"]["accuracy"] == pytest.approx(1.0)
    assert result["tanglish"]["accuracy_change"] == pytest.approx(1.0)


def test_category_grouping_is_correct():
    baseline = _baseline_records()
    tanglish = _tanglish_records_all_correct()
    result = cmp.category_comparison(baseline, tanglish)
    assert result["negation"]["count"] == 2
    assert result["symptoms"]["count"] == 2


def test_difficulty_grouping_is_correct():
    baseline = _baseline_records()
    tanglish = _tanglish_records_all_correct()
    result = cmp.difficulty_comparison(baseline, tanglish)
    assert result["hard"]["count"] == 2
    assert result["easy"]["count"] == 1
    assert result["medium"]["count"] == 1


# ---------------------------------------------------------------------------
# 15: no benchmark mutation
# ---------------------------------------------------------------------------
def test_no_benchmark_mutation_occurs(monkeypatch):
    cases = base_runner.load_benchmark(BENCHMARK_PATH)
    before = BENCHMARK_PATH.read_text(encoding="utf-8")

    monkeypatch.setattr("app.tanglish.service.run_nli", _fixed_nli)
    tg_runner.run_both_modes(cases[:5])

    after = BENCHMARK_PATH.read_text(encoding="utf-8")
    assert before == after


# ---------------------------------------------------------------------------
# 16-17: baseline never invokes the normalizer; tanglish_aware always does
# ---------------------------------------------------------------------------
def test_baseline_mode_never_invokes_normalizer(monkeypatch):
    calls = []
    monkeypatch.setattr("app.tanglish.service.run_nli", _fixed_nli)
    real_normalize = tg_runner.service.normalize

    def spy_normalize(text):
        calls.append(text)
        return real_normalize(text)

    monkeypatch.setattr("app.tanglish.service.normalize", spy_normalize)
    case = _echo_case("N-1", language="tanglish", premise="Enakku fever irukku.")
    tg_runner.evaluate_case_mode(case, MODE_BASELINE)
    assert calls == []


def test_tanglish_aware_mode_invokes_normalizer(monkeypatch):
    calls = []
    monkeypatch.setattr("app.tanglish.service.run_nli", _fixed_nli)
    real_normalize = tg_runner.service.normalize

    def spy_normalize(text):
        calls.append(text)
        return real_normalize(text)

    monkeypatch.setattr("app.tanglish.service.normalize", spy_normalize)
    case = _echo_case("N-2", language="tanglish", premise="Enakku fever irukku.")
    record = tg_runner.evaluate_case_mode(case, MODE_TANGLISH_AWARE)
    assert calls == ["Enakku fever irukku."]
    assert record["normalized_premise"] == "Patient has fever."
    assert record["normalization_applied"] is True


# ---------------------------------------------------------------------------
# 18: English inputs are semantically identical between the two modes when
# no Tanglish transformation is required (English bypass).
# ---------------------------------------------------------------------------
def test_english_inputs_identical_across_modes(monkeypatch):
    monkeypatch.setattr("app.tanglish.service.run_nli", _fixed_nli)
    case = _echo_case("E-1", language="en", premise="I have had fever for two days.")
    baseline_record = tg_runner.evaluate_case_mode(case, MODE_BASELINE)
    tanglish_record = tg_runner.evaluate_case_mode(case, MODE_TANGLISH_AWARE)

    assert baseline_record["normalized_premise"] == case["premise"]
    assert tanglish_record["normalized_premise"] == case["premise"]
    assert baseline_record["normalized_premise"] == tanglish_record["normalized_premise"]
    assert baseline_record["normalization_applied"] is False
    assert tanglish_record["normalization_applied"] is True  # mode flag, even though text is unchanged


# ---------------------------------------------------------------------------
# McNemar's exact test
# ---------------------------------------------------------------------------
def test_mcnemar_exact_test_counts_and_symmetry():
    baseline = _baseline_records()
    tanglish = _tanglish_records_all_correct()
    result = cmp.mcnemar_exact_test(baseline, tanglish)
    # baseline correct & tanglish incorrect: 0 cases (b)
    # baseline incorrect & tanglish correct: cases 2, 3 (c=2)
    assert result["b_baseline_correct_tanglish_aware_incorrect"] == 0
    assert result["c_baseline_incorrect_tanglish_aware_correct"] == 2
    assert result["n_discordant_pairs"] == 2
    assert 0.0 <= result["p_value"] <= 1.0


def test_mcnemar_exact_test_zero_discordant_pairs_gives_p_one():
    same = _baseline_records()
    result = cmp.mcnemar_exact_test(same, same)
    assert result["n_discordant_pairs"] == 0
    assert result["p_value"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Metadata / benchmark reuse sanity
# ---------------------------------------------------------------------------
def test_evaluation_metadata_reuses_frozen_thresholds():
    metadata = tg_runner.evaluation_metadata(BENCHMARK_PATH, 101)
    assert metadata["entailment_threshold"] == 0.70
    assert metadata["contradiction_threshold"] == 0.70
    assert metadata["baseline_mode"] == MODE_BASELINE
    assert metadata["tanglish_aware_mode"] == MODE_TANGLISH_AWARE
    assert metadata["thresholds_tuned_against_benchmark"] is False
    assert metadata["model_fine_tuned"] is False


def test_full_benchmark_runs_through_both_modes_with_mocked_inference(monkeypatch):
    monkeypatch.setattr("app.tanglish.service.run_nli", _fixed_nli)
    cases = base_runner.load_benchmark(BENCHMARK_PATH)
    baseline_records, tanglish_records = tg_runner.run_both_modes(cases)
    assert len(baseline_records) == 101
    assert len(tanglish_records) == 101
    assert {r["case_id"] for r in baseline_records} == {c["id"] for c in cases}
