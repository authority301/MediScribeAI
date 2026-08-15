"""Step 14I comparison logic: three-way (14B baseline / 14F / 14H) analysis
over the same 101-case benchmark.

Reuses, without modification:
  - app.evaluation.metrics       (Step 14C -- summarize/confusion_matrix/
                                   per_class_prf/accuracy/UCR/CR, all
                                   pairwise arithmetic below is built from
                                   these, never reimplemented)
  - app.evaluation.reports       (Step 14C -- breakdown_by grouping)
  - app.evaluation_tanglish.comparison.pp_change / index_by_case_id /
    mcnemar_exact_test           (Step 14G -- generic, record-schema-only
                                   helpers; mcnemar_exact_test in particular
                                   is reused verbatim for all three pairwise
                                   tests below, since it only assumes
                                   case_id/ground_truth/prediction keys)

This module only adds the THREE-WAY assembly (14B/14F/14H side by side,
14B->14F->14H transition buckets) that Step 14G's two-system comparison
had no need for.
"""
from app.evaluation import metrics
from app.evaluation import reports as base_reports
from app.evaluation_tanglish.comparison import index_by_case_id, mcnemar_exact_test, pp_change

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


def _slim_summary(summary: dict) -> dict:
    return {field: summary[field] for field in _SUMMARY_FIELDS}


def _change(new_summary: dict, old_summary: dict) -> dict:
    return {
        "accuracy": new_summary["accuracy"] - old_summary["accuracy"],
        "macro_f1": new_summary["macro_f1"] - old_summary["macro_f1"],
        "ucr": new_summary["ucr_prediction_based"] - old_summary["ucr_prediction_based"],
        "cr": new_summary["cr_prediction_based"] - old_summary["cr_prediction_based"],
    }


# ---------------------------------------------------------------------------
# Overall / pairwise improvement metrics
# ---------------------------------------------------------------------------
def pairwise_improvement(records_old: list[dict], records_new: list[dict], label_old: str, label_new: str) -> dict:
    old = metrics.summarize(records_old)
    new = metrics.summarize(records_new)

    def entry(key: str) -> dict:
        return {
            label_old: old[key],
            label_new: new[key],
            "absolute_change": new[key] - old[key],
            "percentage_point_change": pp_change(new[key], old[key]),
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


def three_way_overall(records_14b: list[dict], records_14f: list[dict], records_14h: list[dict]) -> dict:
    return {
        "14b": _slim_summary(metrics.summarize(records_14b)),
        "14f": _slim_summary(metrics.summarize(records_14f)),
        "14h": _slim_summary(metrics.summarize(records_14h)),
        "14f_minus_14b": pairwise_improvement(records_14b, records_14f, "14b", "14f"),
        "14h_minus_14f": pairwise_improvement(records_14f, records_14h, "14f", "14h"),
        "14h_minus_14b": pairwise_improvement(records_14b, records_14h, "14b", "14h"),
    }


# ---------------------------------------------------------------------------
# Three-way grouped (language / category / difficulty) comparisons
# ---------------------------------------------------------------------------
def three_way_group_comparison(
    records_14b: list[dict], records_14f: list[dict], records_14h: list[dict], key: str,
    fixed_order: list[str] | None = None,
) -> dict:
    groups_14b = base_reports.breakdown_by(records_14b, key, fixed_order)
    groups_14f = base_reports.breakdown_by(records_14f, key, fixed_order)
    groups_14h = base_reports.breakdown_by(records_14h, key, fixed_order)

    ordered_keys = fixed_order if fixed_order is not None else sorted(
        set(groups_14b) | set(groups_14f) | set(groups_14h)
    )

    comparison = {}
    for group_key in ordered_keys:
        b, f, h = groups_14b.get(group_key), groups_14f.get(group_key), groups_14h.get(group_key)
        if b is None and f is None and h is None:
            continue
        entry = {"count": (b or f or h)["count"]}
        if b is not None:
            entry["14b"] = _slim_summary(b)
        if f is not None:
            entry["14f"] = _slim_summary(f)
        if h is not None:
            entry["14h"] = _slim_summary(h)
        if b is not None and f is not None:
            entry["14f_minus_14b"] = _change(f, b)
        if f is not None and h is not None:
            entry["14h_minus_14f"] = _change(h, f)
        if b is not None and h is not None:
            entry["14h_minus_14b"] = _change(h, b)
        comparison[group_key] = entry
    return comparison


def language_comparison(records_14b, records_14f, records_14h) -> dict:
    return three_way_group_comparison(records_14b, records_14f, records_14h, "language", LANGUAGES)


def category_comparison(records_14b, records_14f, records_14h) -> dict:
    return three_way_group_comparison(records_14b, records_14f, records_14h, "category")


def difficulty_comparison(records_14b, records_14f, records_14h) -> dict:
    return three_way_group_comparison(records_14b, records_14f, records_14h, "difficulty", DIFFICULTIES)


# ---------------------------------------------------------------------------
# Per-class (SUPPORTED / CONTRADICTED / UNGROUNDED) three-way comparison
# ---------------------------------------------------------------------------
def per_class_comparison(records_14b: list[dict], records_14f: list[dict], records_14h: list[dict]) -> dict:
    per_class_by_system = {
        "14b": metrics.per_class_prf(metrics.confusion_matrix(records_14b)),
        "14f": metrics.per_class_prf(metrics.confusion_matrix(records_14f)),
        "14h": metrics.per_class_prf(metrics.confusion_matrix(records_14h)),
    }
    result = {}
    for label in metrics.LABELS:
        result[label] = {system: per_class_by_system[system][label] for system in ("14b", "14f", "14h")}
        result[label]["f1_change_14f_minus_14b"] = (
            per_class_by_system["14f"][label]["f1"] - per_class_by_system["14b"][label]["f1"]
        )
        result[label]["f1_change_14h_minus_14f"] = (
            per_class_by_system["14h"][label]["f1"] - per_class_by_system["14f"][label]["f1"]
        )
        result[label]["f1_change_14h_minus_14b"] = (
            per_class_by_system["14h"][label]["f1"] - per_class_by_system["14b"][label]["f1"]
        )
    return result


# ---------------------------------------------------------------------------
# Confusion matrices
# ---------------------------------------------------------------------------
def confusion_matrices(records_14b: list[dict], records_14f: list[dict], records_14h: list[dict]) -> dict:
    m_b = metrics.confusion_matrix(records_14b)
    m_f = metrics.confusion_matrix(records_14f)
    m_h = metrics.confusion_matrix(records_14h)
    diff_h_minus_f = {
        gold: {pred: m_h[gold][pred] - m_f[gold][pred] for pred in metrics.LABELS} for gold in metrics.LABELS
    }
    return {
        "row_axis": "ground_truth",
        "column_axis": "prediction",
        "label_order": metrics.LABELS,
        "14b": m_b,
        "14f": m_f,
        "14h": m_h,
        "difference_14h_minus_14f": diff_h_minus_f,
    }


# ---------------------------------------------------------------------------
# 14F -> 14H fixed / regressed case-level analysis (the primary Step 14H
# refinement effect, isolated from the earlier 14B -> 14F change).
# ---------------------------------------------------------------------------
def _case_detail_14f_14h(record_14f: dict, record_14h: dict, case: dict) -> dict:
    gold = record_14f["ground_truth"]
    return {
        "case_id": record_14f["case_id"],
        "language": record_14f["language"],
        "category": record_14f["category"],
        "difficulty": record_14f["difficulty"],
        "premise": case["premise"],
        "claim": case["claim"],
        "ground_truth": gold,
        "prediction_14f": record_14f["prediction"],
        "prediction_14h": record_14h["prediction"],
        "nli_scores_14f": {
            "nli_label": record_14f["nli_label"],
            "entailment_score": record_14f["entailment_score"],
            "contradiction_score": record_14f["contradiction_score"],
            "neutral_score": record_14f["neutral_score"],
        },
        "nli_scores_14h": {
            "nli_label": record_14h["nli_label"],
            "entailment_score": record_14h["entailment_score"],
            "contradiction_score": record_14h["contradiction_score"],
            "neutral_score": record_14h["neutral_score"],
        },
        "normalized_premise_14f": record_14f["normalized_premise"],
        "normalized_premise_14h": record_14h["normalized_premise"],
        "transformation_trace_14h": record_14h["transformations"],
    }


def fixed_regressed_14f_to_14h(records_14f: list[dict], records_14h: list[dict], cases_by_id: dict[str, dict]) -> dict:
    """FIXED BY 14H:   14f prediction != gold AND 14h prediction == gold.
    REGRESSED BY 14H: 14f prediction == gold AND 14h prediction != gold.
    """
    f_by_id = index_by_case_id(records_14f)
    h_by_id = index_by_case_id(records_14h)

    fixed, regressed, unchanged_correct, unchanged_incorrect = [], [], [], []
    for case_id, f in f_by_id.items():
        h = h_by_id[case_id]
        gold = f["ground_truth"]
        f_correct = f["prediction"] == gold
        h_correct = h["prediction"] == gold
        detail = _case_detail_14f_14h(f, h, cases_by_id[case_id])

        if not f_correct and h_correct:
            fixed.append(detail)
        elif f_correct and not h_correct:
            regressed.append(detail)
        elif f_correct and h_correct:
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
# 14B -> 14F -> 14H transition analysis (all 8 correct/wrong combinations;
# the three combinations the Step 14I spec calls out by name are surfaced
# separately under "highlight").
# ---------------------------------------------------------------------------
def transition_analysis(
    records_14b: list[dict], records_14f: list[dict], records_14h: list[dict], cases_by_id: dict[str, dict]
) -> dict:
    b_by_id = index_by_case_id(records_14b)
    f_by_id = index_by_case_id(records_14f)
    h_by_id = index_by_case_id(records_14h)

    buckets: dict[str, list[dict]] = {
        f"{bc}_{fc}_{hc}": []
        for bc in ("correct", "wrong")
        for fc in ("correct", "wrong")
        for hc in ("correct", "wrong")
    }

    for case_id, b in b_by_id.items():
        f = f_by_id[case_id]
        h = h_by_id[case_id]
        gold = b["ground_truth"]
        bc = "correct" if b["prediction"] == gold else "wrong"
        fc = "correct" if f["prediction"] == gold else "wrong"
        hc = "correct" if h["prediction"] == gold else "wrong"
        case = cases_by_id[case_id]
        buckets[f"{bc}_{fc}_{hc}"].append({
            "case_id": case_id,
            "language": b["language"],
            "category": b["category"],
            "difficulty": b["difficulty"],
            "premise": case["premise"],
            "claim": case["claim"],
            "ground_truth": gold,
            "prediction_14b": b["prediction"],
            "prediction_14f": f["prediction"],
            "prediction_14h": h["prediction"],
        })

    highlight = {
        "preserved_improvement__14b_wrong_14f_correct_14h_correct": buckets["wrong_correct_correct"],
        "regression__14b_wrong_14f_correct_14h_wrong": buckets["wrong_correct_wrong"],
        "new_gain_from_14h__14b_wrong_14f_wrong_14h_correct": buckets["wrong_wrong_correct"],
    }
    counts = {key: len(records) for key, records in buckets.items()}

    return {"buckets": buckets, "highlight": highlight, "counts": counts}


# ---------------------------------------------------------------------------
# McNemar exact tests -- reuses app.evaluation_tanglish.comparison's
# generic, record-schema-only implementation for all three pairs. Field
# names inside each result (b_baseline_correct_tanglish_aware_incorrect /
# c_baseline_incorrect_tanglish_aware_correct) are the reused function's
# own generic terminology: "baseline" = first argument, "tanglish_aware" =
# second argument, for whichever pair is being compared.
# ---------------------------------------------------------------------------
def three_way_mcnemar(records_14b: list[dict], records_14f: list[dict], records_14h: list[dict]) -> dict:
    return {
        "14b_vs_14f": mcnemar_exact_test(records_14b, records_14f),
        "14f_vs_14h": mcnemar_exact_test(records_14f, records_14h),
        "14b_vs_14h": mcnemar_exact_test(records_14b, records_14h),
    }


# ---------------------------------------------------------------------------
# Targeted Step 14H analyses: numeric/measurement, attribution, negation,
# historical context. All built from the same three_way_group_comparison /
# fixed_regressed_14f_to_14h primitives above, restricted to a category
# subset -- no new metric logic.
# ---------------------------------------------------------------------------
def _filtered(records: list[dict], categories: set[str]) -> list[dict]:
    return [r for r in records if r["category"] in categories]


def category_focus_analysis(
    records_14b: list[dict], records_14f: list[dict], records_14h: list[dict],
    cases_by_id: dict[str, dict], categories: set[str],
) -> dict:
    b_subset = _filtered(records_14b, categories)
    f_subset = _filtered(records_14f, categories)
    h_subset = _filtered(records_14h, categories)

    result = {
        "categories": sorted(categories),
        "14b": _slim_summary(metrics.summarize(b_subset)) if b_subset else None,
        "14f": _slim_summary(metrics.summarize(f_subset)) if f_subset else None,
        "14h": _slim_summary(metrics.summarize(h_subset)) if h_subset else None,
    }
    if f_subset and h_subset:
        case_level = fixed_regressed_14f_to_14h(f_subset, h_subset, cases_by_id)
        result["fixed_cases_14f_to_14h"] = case_level["fixed_cases"]
        result["regressed_cases_14f_to_14h"] = case_level["regressed_cases"]
        result["counts_14f_to_14h"] = case_level["counts"]
    return result


def numeric_measurement_analysis(records_14b, records_14f, records_14h, cases_by_id) -> dict:
    return category_focus_analysis(
        records_14b, records_14f, records_14h, cases_by_id, {"measurements", "duration", "frequency"}
    )


def attribution_analysis(records_14b, records_14f, records_14h, cases_by_id) -> dict:
    return category_focus_analysis(
        records_14b, records_14f, records_14h, cases_by_id, {"patient_history", "historical_statements", "diagnoses"}
    )


def negation_analysis(records_14b, records_14f, records_14h, cases_by_id) -> dict:
    return category_focus_analysis(records_14b, records_14f, records_14h, cases_by_id, {"negation"})


def historical_context_analysis(records_14b, records_14f, records_14h, cases_by_id) -> dict:
    return category_focus_analysis(
        records_14b, records_14f, records_14h, cases_by_id, {"historical_statements", "patient_history"}
    )


def build_comparison(
    records_14b: list[dict], records_14f: list[dict], records_14h: list[dict], cases_by_id: dict[str, dict]
) -> dict:
    return {
        "overall": three_way_overall(records_14b, records_14f, records_14h),
        "language_comparison": language_comparison(records_14b, records_14f, records_14h),
        "category_comparison": category_comparison(records_14b, records_14f, records_14h),
        "difficulty_comparison": difficulty_comparison(records_14b, records_14f, records_14h),
        "per_class_comparison": per_class_comparison(records_14b, records_14f, records_14h),
        "confusion_matrices": confusion_matrices(records_14b, records_14f, records_14h),
        "fixed_regressed_14f_to_14h": fixed_regressed_14f_to_14h(records_14f, records_14h, cases_by_id),
        "transition_analysis": transition_analysis(records_14b, records_14f, records_14h, cases_by_id),
        "mcnemar_tests": three_way_mcnemar(records_14b, records_14f, records_14h),
        "numeric_measurement_analysis": numeric_measurement_analysis(records_14b, records_14f, records_14h, cases_by_id),
        "attribution_analysis": attribution_analysis(records_14b, records_14f, records_14h, cases_by_id),
        "negation_analysis": negation_analysis(records_14b, records_14f, records_14h, cases_by_id),
        "historical_context_analysis": historical_context_analysis(records_14b, records_14f, records_14h, cases_by_id),
    }
