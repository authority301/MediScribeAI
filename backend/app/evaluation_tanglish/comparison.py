"""Step 14G comparison logic: pairs SYSTEM A (baseline) and SYSTEM B
(tanglish_aware) records by case_id and computes every comparison the
research spec asks for -- improvement metrics, per-language/category/
difficulty comparisons, per-class precision/recall/F1 comparison, a
confusion-matrix difference, case-level fixed/regressed/unchanged analysis,
and an exact (McNemar) paired-correctness significance test.

This module NEVER reimplements a classification metric -- every count
comes from app.evaluation.metrics / app.evaluation.reports (Step 14C,
unmodified). It only combines those already-computed numbers and does the
case-by-case pairing that Step 14C's single-system runner has no need for.
"""
import math

from app.evaluation import metrics
from app.evaluation import reports as base_reports

LANGUAGES = base_reports.LANGUAGES
DIFFICULTIES = base_reports.DIFFICULTIES

_SUMMARY_FIELDS = (
    "count",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "ucr_prediction_based",
    "cr_prediction_based",
)


def index_by_case_id(records: list[dict]) -> dict[str, dict]:
    return {r["case_id"]: r for r in records}


def _slim_summary(summary: dict) -> dict:
    return {field: summary[field] for field in _SUMMARY_FIELDS}


def pp_change(new: float, old: float) -> float:
    """Percentage-point change (never a relative-percentage change)."""
    return (new - old) * 100.0


# ---------------------------------------------------------------------------
# Overall improvement metrics
# ---------------------------------------------------------------------------
def improvement_metrics(baseline_records: list[dict], tanglish_records: list[dict]) -> dict:
    b = metrics.summarize(baseline_records)
    t = metrics.summarize(tanglish_records)

    def entry(key: str) -> dict:
        return {
            "baseline": b[key],
            "tanglish_aware": t[key],
            "absolute_change": t[key] - b[key],
            "percentage_point_change": pp_change(t[key], b[key]),
        }

    return {
        "accuracy": entry("accuracy"),
        "macro_precision": entry("macro_precision"),
        "macro_recall": entry("macro_recall"),
        "macro_f1": entry("macro_f1"),
        "weighted_f1": entry("weighted_f1"),
        "ucr_prediction_based": entry("ucr_prediction_based"),
        "cr_prediction_based": entry("cr_prediction_based"),
    }


# ---------------------------------------------------------------------------
# Grouped (language / category / difficulty) comparisons
# ---------------------------------------------------------------------------
def group_comparison(baseline_records: list[dict], tanglish_records: list[dict], key: str,
                      fixed_order: list[str] | None = None) -> dict:
    baseline_groups = base_reports.breakdown_by(baseline_records, key, fixed_order)
    tanglish_groups = base_reports.breakdown_by(tanglish_records, key, fixed_order)

    if fixed_order is not None:
        ordered_keys = fixed_order
    else:
        ordered_keys = sorted(set(baseline_groups) | set(tanglish_groups))

    comparison = {}
    for group_key in ordered_keys:
        b = baseline_groups.get(group_key)
        t = tanglish_groups.get(group_key)
        if b is None and t is None:
            continue
        entry = {"count": (b or t)["count"]}
        if b is not None:
            entry["baseline"] = _slim_summary(b)
        if t is not None:
            entry["tanglish_aware"] = _slim_summary(t)
        if b is not None and t is not None:
            entry["accuracy_change"] = t["accuracy"] - b["accuracy"]
            entry["macro_f1_change"] = t["macro_f1"] - b["macro_f1"]
            entry["ucr_change"] = t["ucr_prediction_based"] - b["ucr_prediction_based"]
            entry["cr_change"] = t["cr_prediction_based"] - b["cr_prediction_based"]
        comparison[group_key] = entry
    return comparison


def language_comparison(baseline_records: list[dict], tanglish_records: list[dict]) -> dict:
    return group_comparison(baseline_records, tanglish_records, "language", LANGUAGES)


def category_comparison(baseline_records: list[dict], tanglish_records: list[dict]) -> dict:
    return group_comparison(baseline_records, tanglish_records, "category")


def difficulty_comparison(baseline_records: list[dict], tanglish_records: list[dict]) -> dict:
    return group_comparison(baseline_records, tanglish_records, "difficulty", DIFFICULTIES)


# ---------------------------------------------------------------------------
# Per-class (SUPPORTED / CONTRADICTED / UNGROUNDED) comparison
# ---------------------------------------------------------------------------
def per_class_comparison(baseline_records: list[dict], tanglish_records: list[dict]) -> dict:
    b_matrix = metrics.confusion_matrix(baseline_records)
    t_matrix = metrics.confusion_matrix(tanglish_records)
    b_pc = metrics.per_class_prf(b_matrix)
    t_pc = metrics.per_class_prf(t_matrix)

    result = {}
    for label in metrics.LABELS:
        result[label] = {
            "baseline": b_pc[label],
            "tanglish_aware": t_pc[label],
            "precision_change": t_pc[label]["precision"] - b_pc[label]["precision"],
            "recall_change": t_pc[label]["recall"] - b_pc[label]["recall"],
            "f1_change": t_pc[label]["f1"] - b_pc[label]["f1"],
        }
    return result


# ---------------------------------------------------------------------------
# Confusion matrices
# ---------------------------------------------------------------------------
def confusion_matrices(baseline_records: list[dict], tanglish_records: list[dict]) -> dict:
    b_matrix = metrics.confusion_matrix(baseline_records)
    t_matrix = metrics.confusion_matrix(tanglish_records)
    diff = {
        gold: {pred: t_matrix[gold][pred] - b_matrix[gold][pred] for pred in metrics.LABELS}
        for gold in metrics.LABELS
    }
    return {
        "row_axis": "ground_truth",
        "column_axis": "prediction",
        "label_order": metrics.LABELS,
        "baseline": b_matrix,
        "tanglish_aware": t_matrix,
        "difference_tanglish_minus_baseline": diff,
    }


# ---------------------------------------------------------------------------
# Case-level fixed / regressed / unchanged analysis
# ---------------------------------------------------------------------------
def _case_detail(baseline_record: dict, tanglish_record: dict, case: dict) -> dict:
    gold = baseline_record["ground_truth"]
    return {
        "case_id": baseline_record["case_id"],
        "language": baseline_record["language"],
        "category": baseline_record["category"],
        "difficulty": baseline_record["difficulty"],
        "premise": case["premise"],
        "claim": case["claim"],
        "gold_label": gold,
        "baseline_prediction": baseline_record["prediction"],
        "tanglish_aware_prediction": tanglish_record["prediction"],
        "baseline_nli_scores": {
            "nli_label": baseline_record["nli_label"],
            "entailment_score": baseline_record["entailment_score"],
            "contradiction_score": baseline_record["contradiction_score"],
            "neutral_score": baseline_record["neutral_score"],
        },
        "tanglish_aware_nli_scores": {
            "nli_label": tanglish_record["nli_label"],
            "entailment_score": tanglish_record["entailment_score"],
            "contradiction_score": tanglish_record["contradiction_score"],
            "neutral_score": tanglish_record["neutral_score"],
        },
        "normalized_premise": tanglish_record["normalized_premise"],
        "transformation_trace": tanglish_record["transformations"],
    }


def case_level_comparison(baseline_records: list[dict], tanglish_records: list[dict],
                           cases_by_id: dict[str, dict]) -> dict:
    """fixed:   baseline prediction != gold AND tanglish_aware prediction == gold
    regressed: baseline prediction == gold AND tanglish_aware prediction != gold
    unchanged_correct / unchanged_incorrect: both systems agree with each other's correctness.
    """
    b_by_id = index_by_case_id(baseline_records)
    t_by_id = index_by_case_id(tanglish_records)

    fixed, regressed, unchanged_correct, unchanged_incorrect = [], [], [], []
    for case_id, b in b_by_id.items():
        t = t_by_id[case_id]
        gold = b["ground_truth"]
        b_correct = b["prediction"] == gold
        t_correct = t["prediction"] == gold
        detail = _case_detail(b, t, cases_by_id[case_id])

        if not b_correct and t_correct:
            fixed.append(detail)
        elif b_correct and not t_correct:
            regressed.append(detail)
        elif b_correct and t_correct:
            unchanged_correct.append(detail)
        else:
            unchanged_incorrect.append(detail)

    return {
        "fixed_cases": fixed,
        "regressed_cases": regressed,
        "unchanged_correct_cases": unchanged_correct,
        "unchanged_incorrect_cases": unchanged_incorrect,
        "counts": {
            "fixed": len(fixed),
            "regressed": len(regressed),
            "unchanged_correct": len(unchanged_correct),
            "unchanged_incorrect": len(unchanged_incorrect),
        },
    }


# ---------------------------------------------------------------------------
# McNemar's exact test on paired binary correctness (never on 3-class labels)
# ---------------------------------------------------------------------------
def mcnemar_exact_test(baseline_records: list[dict], tanglish_records: list[dict]) -> dict:
    b_by_id = index_by_case_id(baseline_records)
    t_by_id = index_by_case_id(tanglish_records)

    baseline_correct_tanglish_incorrect = 0  # "b" in the usual 2x2 notation
    baseline_incorrect_tanglish_correct = 0  # "c"
    for case_id, b in b_by_id.items():
        t = t_by_id[case_id]
        gold = b["ground_truth"]
        b_correct = b["prediction"] == gold
        t_correct = t["prediction"] == gold
        if b_correct and not t_correct:
            baseline_correct_tanglish_incorrect += 1
        elif t_correct and not b_correct:
            baseline_incorrect_tanglish_correct += 1

    n = baseline_correct_tanglish_incorrect + baseline_incorrect_tanglish_correct
    if n == 0:
        p_value = 1.0
    else:
        k = min(baseline_correct_tanglish_incorrect, baseline_incorrect_tanglish_correct)
        one_tail = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
        p_value = min(1.0, 2 * one_tail)

    return {
        "test": "McNemar's exact test (binomial form, two-sided, on paired binary correctness)",
        "applies_to": "binary correct/incorrect per case, NOT the 3-class SUPPORTED/CONTRADICTED/UNGROUNDED label",
        "n_cases_total": len(b_by_id),
        "n_discordant_pairs": n,
        "b_baseline_correct_tanglish_aware_incorrect": baseline_correct_tanglish_incorrect,
        "c_baseline_incorrect_tanglish_aware_correct": baseline_incorrect_tanglish_correct,
        "p_value": p_value,
        "note": (
            "Computed over a 101-case synthetic benchmark. This is a descriptive, paired-by-case "
            "statistical test, not a claim of clinical significance or real-world generalizability."
        ),
    }


# ---------------------------------------------------------------------------
# Focused negation / historical-context analyses
# ---------------------------------------------------------------------------
def _filtered(records: list[dict], categories: set[str]) -> list[dict]:
    return [r for r in records if r["category"] in categories]


def category_focus_analysis(baseline_records: list[dict], tanglish_records: list[dict],
                             cases_by_id: dict[str, dict], categories: set[str]) -> dict:
    b_subset = _filtered(baseline_records, categories)
    t_subset = _filtered(tanglish_records, categories)
    b_summary = metrics.summarize(b_subset) if b_subset else None
    t_summary = metrics.summarize(t_subset) if t_subset else None
    case_level = case_level_comparison(b_subset, t_subset, cases_by_id) if b_subset else {
        "fixed_cases": [], "regressed_cases": [], "unchanged_correct_cases": [], "unchanged_incorrect_cases": [],
        "counts": {"fixed": 0, "regressed": 0, "unchanged_correct": 0, "unchanged_incorrect": 0},
    }
    return {
        "categories": sorted(categories),
        "baseline": _slim_summary(b_summary) if b_summary else None,
        "tanglish_aware": _slim_summary(t_summary) if t_summary else None,
        "fixed_cases": case_level["fixed_cases"],
        "regressed_cases": case_level["regressed_cases"],
        "counts": case_level["counts"],
    }


def negation_analysis(baseline_records: list[dict], tanglish_records: list[dict],
                       cases_by_id: dict[str, dict]) -> dict:
    return category_focus_analysis(baseline_records, tanglish_records, cases_by_id, {"negation"})


def historical_analysis(baseline_records: list[dict], tanglish_records: list[dict],
                         cases_by_id: dict[str, dict]) -> dict:
    return category_focus_analysis(
        baseline_records, tanglish_records, cases_by_id, {"historical_statements", "patient_history"}
    )


def build_comparison(baseline_records: list[dict], tanglish_records: list[dict],
                      cases_by_id: dict[str, dict]) -> dict:
    return {
        "improvement": improvement_metrics(baseline_records, tanglish_records),
        "language_comparison": language_comparison(baseline_records, tanglish_records),
        "category_comparison": category_comparison(baseline_records, tanglish_records),
        "difficulty_comparison": difficulty_comparison(baseline_records, tanglish_records),
        "per_class_comparison": per_class_comparison(baseline_records, tanglish_records),
        "confusion_matrices": confusion_matrices(baseline_records, tanglish_records),
        "case_level": case_level_comparison(baseline_records, tanglish_records, cases_by_id),
        "mcnemar_exact_test": mcnemar_exact_test(baseline_records, tanglish_records),
        "negation_analysis": negation_analysis(baseline_records, tanglish_records, cases_by_id),
        "historical_analysis": historical_analysis(baseline_records, tanglish_records, cases_by_id),
    }
