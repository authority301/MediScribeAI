"""Step 14I comparison-harness tests.

NEVER loads the real NLI model: every test that exercises
app.evaluation_tanglish_refinement.runner.evaluate_case_14b/14f/14h or
run_all_systems monkeypatches every real call site (app.evaluation.runner.
run_nli, app.evaluation_tanglish_refinement.runner.run_nli, and
app.tanglish.service.run_nli -- the three places the real Step 14B model
would be invoked across the 14B/14F/14H paths respectively). Comparison-math
tests use hand-computed expected values on small synthetic record sets,
independent of any model behavior.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from app.evaluation.runner import load_benchmark
from app.evaluation_tanglish_refinement import comparison as cmp
from app.evaluation_tanglish_refinement import reports
from app.evaluation_tanglish_refinement import runner as tg_runner
from app.nli.model import NliScores

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BENCHMARK_PATH = REPO_ROOT / "datasets" / "nli_evaluation" / "benchmark.json"
HISTORICAL_DIR = REPO_ROOT / "datasets" / "nli_evaluation" / "results" / "tanglish_comparison"

_ALL_RUN_NLI_TARGETS = (
    "app.evaluation.runner.run_nli",
    "app.evaluation_tanglish_refinement.runner.run_nli",
    "app.tanglish.service.run_nli",
)


def _fixed_nli(premise, hypothesis):
    return NliScores(entailment=0.9, neutral=0.05, contradiction=0.05, label="ENTAILMENT")


def _patch_all_run_nli(monkeypatch, fn=_fixed_nli):
    for target in _ALL_RUN_NLI_TARGETS:
        monkeypatch.setattr(target, fn)


def _echo_case(case_id="X-1", language="en", premise="I have fever.", claim="Patient has fever.",
               ground_truth="SUPPORTED", category="symptoms", difficulty="easy"):
    return {
        "id": case_id, "language": language, "premise": premise, "claim": claim,
        "ground_truth": ground_truth, "category": category, "difficulty": difficulty,
    }


def _record(case_id, ground_truth, prediction, language="en", category="symptoms", difficulty="easy",
            nli_label="ENTAILMENT", entailment=0.9, contradiction=0.05, neutral=0.05,
            normalized_premise="p", transformations=None):
    return {
        "case_id": case_id, "language": language, "category": category, "difficulty": difficulty,
        "ground_truth": ground_truth, "prediction": prediction, "nli_label": nli_label,
        "entailment_score": entailment, "contradiction_score": contradiction, "neutral_score": neutral,
        "normalized_premise": normalized_premise, "transformations": transformations or [],
    }


# ---------------------------------------------------------------------------
# 1-3: all three systems evaluate the same 101 cases / ids / ground truth
# ---------------------------------------------------------------------------
def test_all_three_systems_evaluate_101_cases(monkeypatch):
    _patch_all_run_nli(monkeypatch)
    cases = load_benchmark(BENCHMARK_PATH)
    b, f, h = tg_runner.run_all_systems(cases)
    assert len(b) == len(f) == len(h) == 101


def test_case_ids_identical_across_systems(monkeypatch):
    _patch_all_run_nli(monkeypatch)
    cases = load_benchmark(BENCHMARK_PATH)[:8]
    b, f, h = tg_runner.run_all_systems(cases)
    input_ids = {c["id"] for c in cases}
    assert {r["case_id"] for r in b} == input_ids
    assert {r["case_id"] for r in f} == input_ids
    assert {r["case_id"] for r in h} == input_ids


def test_ground_truth_labels_identical_across_systems(monkeypatch):
    _patch_all_run_nli(monkeypatch)
    cases = load_benchmark(BENCHMARK_PATH)[:8]
    b, f, h = tg_runner.run_all_systems(cases)
    case_gt = {c["id"]: c["ground_truth"] for c in cases}
    for records in (b, f, h):
        assert {r["case_id"]: r["ground_truth"] for r in records} == case_gt


# ---------------------------------------------------------------------------
# 4: benchmark hash is recorded
# ---------------------------------------------------------------------------
def test_benchmark_hash_is_recorded():
    metadata = tg_runner.evaluation_metadata(BENCHMARK_PATH, 101)
    assert metadata["benchmark_sha256"] == tg_runner.benchmark_hash(BENCHMARK_PATH)
    assert len(metadata["benchmark_sha256"]) == 64  # sha256 hex digest
    assert metadata["entailment_threshold"] == 0.70
    assert metadata["contradiction_threshold"] == 0.70


# ---------------------------------------------------------------------------
# 5-7: reproducibility of each system
# ---------------------------------------------------------------------------
def test_14b_results_reproducible(monkeypatch):
    _patch_all_run_nli(monkeypatch)
    case = _echo_case("R-1")
    assert tg_runner.evaluate_case_14b(case) == tg_runner.evaluate_case_14b(case)


def test_14f_results_reproducible(monkeypatch):
    _patch_all_run_nli(monkeypatch)
    case = _echo_case("R-2", language="tanglish", premise="Enakku fever illa.", ground_truth="CONTRADICTED",
                       category="negation", difficulty="hard")
    assert tg_runner.evaluate_case_14f(case) == tg_runner.evaluate_case_14f(case)


def test_14h_results_reproducible(monkeypatch):
    _patch_all_run_nli(monkeypatch)
    case = _echo_case("R-3", language="tanglish", premise="Enakku cough irukku for three days.",
                       claim="Patient has cough for three days.", ground_truth="SUPPORTED",
                       category="duration", difficulty="medium")
    assert tg_runner.evaluate_case_14h(case) == tg_runner.evaluate_case_14h(case)


def test_14f_and_14h_diverge_on_numeric_preservation(monkeypatch):
    # Confirms the frozen 14F snapshot and the current 14H normalizer are
    # genuinely different code paths, not accidentally the same function.
    _patch_all_run_nli(monkeypatch)
    case = _echo_case("R-4", language="tanglish", premise="Enakku cough irukku for three days.",
                       claim="Patient has cough for three days.", ground_truth="SUPPORTED",
                       category="duration", difficulty="medium")
    record_14f = tg_runner.evaluate_case_14f(case)
    record_14h = tg_runner.evaluate_case_14h(case)
    assert record_14f["normalized_premise"] == "Patient has cough."
    assert record_14h["normalized_premise"] == "Patient has cough for three days."


# ---------------------------------------------------------------------------
# 8-9: 14F -> 14H fixed / regressed calculation correctness
# ---------------------------------------------------------------------------
def _cases_by_id_for(*case_ids):
    return {cid: {"id": cid, "premise": "p", "claim": "c"} for cid in case_ids}


def test_14f_to_14h_fixed_calculation_is_correct():
    records_14f = [_record("A", "CONTRADICTED", "SUPPORTED")]  # wrong
    records_14h = [_record("A", "CONTRADICTED", "CONTRADICTED")]  # now right
    result = cmp.fixed_regressed_14f_to_14h(records_14f, records_14h, _cases_by_id_for("A"))
    assert result["counts"]["fixed"] == 1
    assert result["counts"]["regressed"] == 0
    assert result["fixed_cases"][0]["case_id"] == "A"
    assert result["fixed_cases"][0]["prediction_14f"] == "SUPPORTED"
    assert result["fixed_cases"][0]["prediction_14h"] == "CONTRADICTED"


def test_14f_to_14h_regression_calculation_is_correct():
    records_14f = [_record("B", "SUPPORTED", "SUPPORTED")]  # right
    records_14h = [_record("B", "SUPPORTED", "UNGROUNDED")]  # now wrong
    result = cmp.fixed_regressed_14f_to_14h(records_14f, records_14h, _cases_by_id_for("B"))
    assert result["counts"]["regressed"] == 1
    assert result["counts"]["fixed"] == 0
    assert result["regressed_cases"][0]["case_id"] == "B"


def test_14f_to_14h_unchanged_buckets_correct():
    records_14f = [
        _record("C1", "SUPPORTED", "SUPPORTED"),      # unchanged correct
        _record("C2", "SUPPORTED", "CONTRADICTED"),    # unchanged incorrect
    ]
    records_14h = [
        _record("C1", "SUPPORTED", "SUPPORTED"),
        _record("C2", "SUPPORTED", "UNGROUNDED"),
    ]
    result = cmp.fixed_regressed_14f_to_14h(records_14f, records_14h, _cases_by_id_for("C1", "C2"))
    assert result["counts"]["unchanged_correct"] == 1
    assert result["counts"]["unchanged_incorrect"] == 1


# ---------------------------------------------------------------------------
# 10: 14B -> 14F -> 14H transition analysis correctness
# ---------------------------------------------------------------------------
def test_transition_analysis_buckets_correctly():
    # One case per interesting combination.
    records_14b = [
        _record("T1", "SUPPORTED", "UNGROUNDED"),   # wrong
        _record("T2", "SUPPORTED", "UNGROUNDED"),   # wrong
        _record("T3", "SUPPORTED", "SUPPORTED"),    # correct
    ]
    records_14f = [
        _record("T1", "SUPPORTED", "SUPPORTED"),    # correct (preserved improvement candidate)
        _record("T2", "SUPPORTED", "UNGROUNDED"),   # wrong (new-gain candidate)
        _record("T3", "SUPPORTED", "UNGROUNDED"),   # wrong (correct->wrong->? )
    ]
    records_14h = [
        _record("T1", "SUPPORTED", "SUPPORTED"),    # correct -> preserved improvement
        _record("T2", "SUPPORTED", "SUPPORTED"),    # correct -> new gain from 14H
        _record("T3", "SUPPORTED", "SUPPORTED"),    # correct
    ]
    cases_by_id = _cases_by_id_for("T1", "T2", "T3")
    result = cmp.transition_analysis(records_14b, records_14f, records_14h, cases_by_id)

    preserved = result["highlight"]["preserved_improvement__14b_wrong_14f_correct_14h_correct"]
    new_gain = result["highlight"]["new_gain_from_14h__14b_wrong_14f_wrong_14h_correct"]
    assert [c["case_id"] for c in preserved] == ["T1"]
    assert [c["case_id"] for c in new_gain] == ["T2"]
    assert result["counts"]["wrong_correct_correct"] == 1
    assert result["counts"]["wrong_wrong_correct"] == 1
    assert result["counts"]["correct_wrong_correct"] == 1
    assert sum(result["counts"].values()) == 3


# ---------------------------------------------------------------------------
# 11: percentage-point calculations
# ---------------------------------------------------------------------------
def test_percentage_point_calculation_is_correct():
    records_old = [_record("P1", "SUPPORTED", "UNGROUNDED"), _record("P2", "SUPPORTED", "SUPPORTED")]
    records_new = [_record("P1", "SUPPORTED", "SUPPORTED"), _record("P2", "SUPPORTED", "SUPPORTED")]
    result = cmp.pairwise_improvement(records_old, records_new, "old", "new")
    assert result["accuracy"]["old"] == pytest.approx(0.5)
    assert result["accuracy"]["new"] == pytest.approx(1.0)
    assert result["accuracy"]["percentage_point_change"] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# 12-13: UCR / CR calculations
# ---------------------------------------------------------------------------
def _three_system_ucr_cr_records():
    records_14b = [
        _record("U1", "SUPPORTED", "UNGROUNDED"),
        _record("U2", "CONTRADICTED", "CONTRADICTED"),
        _record("U3", "SUPPORTED", "SUPPORTED"),
        _record("U4", "UNGROUNDED", "UNGROUNDED"),
    ]
    records_14f = [
        _record("U1", "SUPPORTED", "SUPPORTED"),
        _record("U2", "CONTRADICTED", "CONTRADICTED"),
        _record("U3", "SUPPORTED", "UNGROUNDED"),
        _record("U4", "UNGROUNDED", "UNGROUNDED"),
    ]
    records_14h = [
        _record("U1", "SUPPORTED", "SUPPORTED"),
        _record("U2", "CONTRADICTED", "SUPPORTED"),
        _record("U3", "SUPPORTED", "SUPPORTED"),
        _record("U4", "UNGROUNDED", "UNGROUNDED"),
    ]
    return records_14b, records_14f, records_14h


def test_ucr_calculation_is_correct():
    b, f, h = _three_system_ucr_cr_records()
    overall = cmp.three_way_overall(b, f, h)
    assert overall["14b"]["ucr_prediction_based"] == pytest.approx(2 / 4)  # U1, U4
    assert overall["14f"]["ucr_prediction_based"] == pytest.approx(2 / 4)  # U3, U4
    assert overall["14h"]["ucr_prediction_based"] == pytest.approx(1 / 4)  # U4


def test_cr_calculation_is_correct():
    b, f, h = _three_system_ucr_cr_records()
    overall = cmp.three_way_overall(b, f, h)
    assert overall["14b"]["cr_prediction_based"] == pytest.approx(1 / 4)  # U2
    assert overall["14f"]["cr_prediction_based"] == pytest.approx(1 / 4)  # U2
    assert overall["14h"]["cr_prediction_based"] == pytest.approx(0 / 4)  # none


# ---------------------------------------------------------------------------
# 14: confusion matrix orientation
# ---------------------------------------------------------------------------
def test_confusion_matrix_orientation_and_diff():
    b, f, h = _three_system_ucr_cr_records()
    matrices = cmp.confusion_matrices(b, f, h)
    assert matrices["row_axis"] == "ground_truth"
    assert matrices["column_axis"] == "prediction"
    # 14b: gold SUPPORTED -> pred UNGROUNDED once (U1), pred SUPPORTED once (U3)
    assert matrices["14b"]["SUPPORTED"]["UNGROUNDED"] == 1
    assert matrices["14b"]["SUPPORTED"]["SUPPORTED"] == 1
    # 14h: gold CONTRADICTED -> pred SUPPORTED once (U2)
    assert matrices["14h"]["CONTRADICTED"]["SUPPORTED"] == 1
    # difference_14h_minus_14f: 14f gold CONTRADICTED->CONTRADICTED=1, 14h gold CONTRADICTED->CONTRADICTED=0
    assert matrices["difference_14h_minus_14f"]["CONTRADICTED"]["CONTRADICTED"] == -1
    assert matrices["difference_14h_minus_14f"]["CONTRADICTED"]["SUPPORTED"] == 1


# ---------------------------------------------------------------------------
# 15-16: language / category grouping
# ---------------------------------------------------------------------------
def test_language_grouping_is_correct():
    records_14b = [
        _record("L1", "SUPPORTED", "SUPPORTED", language="en"),
        _record("L2", "SUPPORTED", "UNGROUNDED", language="tanglish"),
    ]
    records_14f = [
        _record("L1", "SUPPORTED", "SUPPORTED", language="en"),
        _record("L2", "SUPPORTED", "SUPPORTED", language="tanglish"),
    ]
    records_14h = [
        _record("L1", "SUPPORTED", "SUPPORTED", language="en"),
        _record("L2", "SUPPORTED", "SUPPORTED", language="tanglish"),
    ]
    result = cmp.language_comparison(records_14b, records_14f, records_14h)
    assert result["tanglish"]["count"] == 1
    assert result["tanglish"]["14b"]["accuracy"] == pytest.approx(0.0)
    assert result["tanglish"]["14f"]["accuracy"] == pytest.approx(1.0)
    assert result["tanglish"]["14h"]["accuracy"] == pytest.approx(1.0)
    assert result["tanglish"]["14f_minus_14b"]["accuracy"] == pytest.approx(1.0)
    assert result["tanglish"]["14h_minus_14f"]["accuracy"] == pytest.approx(0.0)


def test_category_grouping_is_correct():
    records = [
        _record("G1", "SUPPORTED", "SUPPORTED", category="negation"),
        _record("G2", "SUPPORTED", "SUPPORTED", category="symptoms"),
    ]
    result = cmp.category_comparison(records, records, records)
    assert result["negation"]["count"] == 1
    assert result["symptoms"]["count"] == 1


# ---------------------------------------------------------------------------
# 17: numeric/measurement targeted analysis
# ---------------------------------------------------------------------------
def test_numeric_measurement_targeted_analysis_filters_correctly():
    records_14b = [
        _record("N1", "SUPPORTED", "UNGROUNDED", category="duration"),
        _record("N2", "SUPPORTED", "SUPPORTED", category="measurements"),
        _record("N3", "SUPPORTED", "SUPPORTED", category="negation"),  # must be excluded
    ]
    records_14f = [
        _record("N1", "SUPPORTED", "UNGROUNDED", category="duration"),
        _record("N2", "SUPPORTED", "SUPPORTED", category="measurements"),
        _record("N3", "SUPPORTED", "SUPPORTED", category="negation"),
    ]
    records_14h = [
        _record("N1", "SUPPORTED", "SUPPORTED", category="duration"),  # fixed
        _record("N2", "SUPPORTED", "SUPPORTED", category="measurements"),
        _record("N3", "SUPPORTED", "SUPPORTED", category="negation"),
    ]
    cases_by_id = _cases_by_id_for("N1", "N2", "N3")
    result = cmp.numeric_measurement_analysis(records_14b, records_14f, records_14h, cases_by_id)
    assert result["14b"]["count"] == 2  # N1, N2 only -- N3 (negation) excluded
    assert result["counts_14f_to_14h"]["fixed"] == 1
    assert [c["case_id"] for c in result["fixed_cases_14f_to_14h"]] == ["N1"]


# ---------------------------------------------------------------------------
# 18: attribution targeted analysis
# ---------------------------------------------------------------------------
def test_attribution_targeted_analysis_filters_correctly():
    records_14f = [
        _record("F1", "UNGROUNDED", "SUPPORTED", category="patient_history"),  # wrong (misattributed)
        _record("F2", "SUPPORTED", "SUPPORTED", category="symptoms"),  # must be excluded
    ]
    records_14h = [
        _record("F1", "UNGROUNDED", "UNGROUNDED", category="patient_history"),  # fixed
        _record("F2", "SUPPORTED", "SUPPORTED", category="symptoms"),
    ]
    records_14b = records_14f
    cases_by_id = _cases_by_id_for("F1", "F2")
    result = cmp.attribution_analysis(records_14b, records_14f, records_14h, cases_by_id)
    assert result["14f"]["count"] == 1  # F1 only
    assert result["counts_14f_to_14h"]["fixed"] == 1
    assert result["fixed_cases_14f_to_14h"][0]["case_id"] == "F1"


# ---------------------------------------------------------------------------
# 19: McNemar calculation
# ---------------------------------------------------------------------------
def test_mcnemar_calculation_is_correct():
    # b=1 (A right in first, wrong in second), c=1 (A wrong in first, right in second) -> n=2, symmetric -> p=1.0
    records_a = [_record("M1", "SUPPORTED", "SUPPORTED"), _record("M2", "SUPPORTED", "UNGROUNDED")]
    records_b = [_record("M1", "SUPPORTED", "UNGROUNDED"), _record("M2", "SUPPORTED", "SUPPORTED")]
    result = cmp.mcnemar_exact_test(records_a, records_b)  # reused from Step 14G, sanity-checked here
    assert result["n_discordant_pairs"] == 2
    assert result["p_value"] == pytest.approx(1.0)


def test_three_way_mcnemar_produces_all_three_pairs():
    b, f, h = _three_system_ucr_cr_records()
    result = cmp.three_way_mcnemar(b, f, h)
    assert set(result.keys()) == {"14b_vs_14f", "14f_vs_14h", "14b_vs_14h"}
    for pair_result in result.values():
        assert 0.0 <= pair_result["p_value"] <= 1.0


# ---------------------------------------------------------------------------
# 20: benchmark not modified by running the framework
# ---------------------------------------------------------------------------
def test_benchmark_not_modified_by_evaluation(monkeypatch):
    before = BENCHMARK_PATH.read_text(encoding="utf-8")
    _patch_all_run_nli(monkeypatch)
    cases = load_benchmark(BENCHMARK_PATH)[:5]
    tg_runner.run_all_systems(cases)
    after = BENCHMARK_PATH.read_text(encoding="utf-8")
    assert before == after


# ---------------------------------------------------------------------------
# 21: Step 14G result files not modified
# ---------------------------------------------------------------------------
def test_step_14g_results_not_modified_by_reading_them():
    baseline_path = HISTORICAL_DIR / "baseline_results.json"
    tanglish_path = HISTORICAL_DIR / "tanglish_aware_results.json"
    before_baseline = baseline_path.read_text(encoding="utf-8")
    before_tanglish = tanglish_path.read_text(encoding="utf-8")

    historical = reports.load_historical_reference(HISTORICAL_DIR)
    assert set(historical.keys()) == {"14b_accuracy", "14b_macro_f1", "14f_accuracy", "14f_macro_f1"}

    after_baseline = baseline_path.read_text(encoding="utf-8")
    after_tanglish = tanglish_path.read_text(encoding="utf-8")
    assert before_baseline == after_baseline
    assert before_tanglish == after_tanglish


# ---------------------------------------------------------------------------
# Reproduction-check mechanics (mismatch must raise, not silently continue)
# ---------------------------------------------------------------------------
def test_reproduction_check_passes_within_tolerance():
    historical = {"14b_accuracy": 0.5, "14b_macro_f1": 0.5, "14f_accuracy": 0.6, "14f_macro_f1": 0.6}
    records_b = [_record("C1", "SUPPORTED", "SUPPORTED"), _record("C2", "SUPPORTED", "UNGROUNDED")]
    # accuracy = 0.5, macro_f1 computed by metrics.summarize -- just verify it does not raise when
    # historical values are set to match the actual computed summary exactly.
    from app.evaluation import metrics
    summary = metrics.summarize(records_b)
    historical = {
        "14b_accuracy": summary["accuracy"], "14b_macro_f1": summary["macro_f1"],
        "14f_accuracy": summary["accuracy"], "14f_macro_f1": summary["macro_f1"],
    }
    result = reports.check_reproduction(records_b, records_b, historical)
    assert result["all_matched"] is True


def test_reproduction_check_raises_on_mismatch():
    records_b = [_record("C1", "SUPPORTED", "SUPPORTED"), _record("C2", "SUPPORTED", "UNGROUNDED")]
    historical = {"14b_accuracy": 0.999, "14b_macro_f1": 0.999, "14f_accuracy": 0.999, "14f_macro_f1": 0.999}
    with pytest.raises(reports.ReproductionMismatchError):
        reports.check_reproduction(records_b, records_b, historical)
