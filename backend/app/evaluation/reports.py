"""Step 14C report assembly: groups evaluation records by language / category
/ difficulty, builds a failure-analysis listing, and writes the result files
under datasets/nli_evaluation/results/. Never touches benchmark.json.
"""
import json
from pathlib import Path

from app.evaluation import metrics

LANGUAGES = ["en", "tanglish", "mixed"]
DIFFICULTIES = ["easy", "medium", "hard"]

# Possible-cause tags are advisory only ("possible failure category"), never
# an asserted causal diagnosis of why a given prediction was wrong.
NEGATION_HISTORICAL_CATEGORY = "historical_statements"
OVERREACH_CATEGORY = "diagnosis_overreach"


def group_by(records: list[dict], key: str) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(record[key], []).append(record)
    return groups


def breakdown_by(records: list[dict], key: str, fixed_order: list[str] | None = None) -> dict:
    groups = group_by(records, key)
    order = fixed_order if fixed_order is not None else sorted(groups.keys())
    result = {}
    for group_key in order:
        group_records = groups.get(group_key, [])
        if not group_records:
            continue
        result[group_key] = metrics.summarize(group_records)
    return result


def overall_report(records: list[dict], metadata: dict) -> dict:
    summary = metrics.summarize(records)
    accuracy_ci = metrics.bootstrap_ci(records, metrics.accuracy)
    macro_f1_ci = metrics.bootstrap_ci(
        records, lambda recs: metrics.macro_metrics(metrics.per_class_prf(metrics.confusion_matrix(recs)))["f1"]
    )
    ucr_ci = metrics.bootstrap_ci(records, lambda recs: metrics.unsupported_claim_rate(recs, basis="prediction"))

    return {
        "metadata": metadata,
        "overall": summary,
        "bootstrap_confidence_intervals": {
            "note": (
                "95% bootstrap CIs computed with a fixed seed (see 'seed') over "
                "n_resamples resamples of the evaluated cases. Optional per the "
                "Step 14C spec; included here for transparency, not as a "
                "substitute for a larger benchmark."
            ),
            "accuracy": accuracy_ci,
            "macro_f1": macro_f1_ci,
            "ucr_prediction_based": ucr_ci,
        },
        "language_results": breakdown_by(records, "language", LANGUAGES),
        "category_results": breakdown_by(records, "category"),
        "difficulty_results": breakdown_by(records, "difficulty", DIFFICULTIES),
    }


def failure_analysis(records: list[dict], cases_by_id: dict[str, dict]) -> list[dict]:
    failures = []
    for record in records:
        if record["ground_truth"] == record["prediction"]:
            continue
        case = cases_by_id[record["case_id"]]

        possible_causes = []
        has_negation = bool(case.get("contains_negation"))
        has_code_switching = bool(case.get("contains_code_switching"))
        if has_negation and has_code_switching:
            possible_causes.append("negation + code-switching")
        elif has_negation:
            possible_causes.append("negation")
        elif has_code_switching:
            possible_causes.append("code-switching / Tanglish")
        if case["category"] == OVERREACH_CATEGORY:
            possible_causes.append("diagnosis overreach")
        if case["category"] == NEGATION_HISTORICAL_CATEGORY:
            possible_causes.append("historical context")
        if case.get("contains_measurement"):
            possible_causes.append("measurement")
        if case.get("contains_medication"):
            possible_causes.append("medication")
        if not possible_causes:
            possible_causes.append("unspecified")

        failures.append({
            "case_id": record["case_id"],
            "language": record["language"],
            "category": record["category"],
            "difficulty": record["difficulty"],
            "premise": case["premise"],
            "claim": case["claim"],
            "ground_truth": record["ground_truth"],
            "prediction": record["prediction"],
            "nli_label": record["nli_label"],
            "entailment_score": record["entailment_score"],
            "contradiction_score": record["contradiction_score"],
            "neutral_score": record["neutral_score"],
            "possible_failure_categories": possible_causes,
        })
    return failures


def confusion_matrix_json(records: list[dict]) -> dict:
    matrix = metrics.confusion_matrix(records)
    return {
        "row_axis": "ground_truth",
        "column_axis": "prediction",
        "label_order": metrics.LABELS,
        "matrix": matrix,
    }


def render_readme(report: dict, failures: list[dict]) -> str:
    md = report["metadata"]
    overall = report["overall"]
    lines = [
        "# Step 14C Evaluation Results",
        "",
        "Results of running the EXISTING, UNMODIFIED Step 14B NLI evidence "
        "verifier against the Step 14E synthetic gold benchmark "
        "(`datasets/nli_evaluation/benchmark.json`).",
        "",
        "## Research integrity",
        "",
        "1. This is a synthetic benchmark (101 hand-authored cases), not a clinical dataset.",
        "2. The NLI model (`{}`) was evaluated as-is, without fine-tuning.".format(md["model_name"]),
        "3. Thresholds (entailment={:.2f}, contradiction={:.2f}) were NOT optimized against this benchmark.".format(
            md["entailment_threshold"], md["contradiction_threshold"]
        ),
        "4. Tanglish is evaluated separately from English (see `language_results.json`).",
        "5. Evidence Unsupported-Claim Rate (UCR) is an evidence-verification metric, not a clinical hallucination rate. "
        "UNGROUNDED claims represent evidence-unsupported claims and are used here as a measurable proxy for "
        "unsupported documentation behavior; they are not treated as a direct clinical hallucination rate.",
        "6. Contradiction Rate (CR) is reported separately from UCR -- they represent different failure modes and are never merged.",
        "7. These results do NOT establish clinical safety, clinical validity, real-world diagnostic accuracy, or HIPAA compliance.",
        "",
        "## Reproducibility metadata",
        "",
        f"- Model provider: `{md['model_provider']}`",
        f"- Model name: `{md['model_name']}`",
        f"- Device: `{md['device']}`",
        f"- Entailment threshold: `{md['entailment_threshold']}`",
        f"- Contradiction threshold: `{md['contradiction_threshold']}`",
        f"- Benchmark file: `{md['benchmark_path']}`",
        f"- Benchmark SHA-256: `{md['benchmark_sha256']}`",
        f"- Number of cases evaluated: `{md['num_cases']}`",
        f"- Evaluation timestamp (UTC): `{md['evaluation_timestamp_utc']}`",
        "",
        "## Overall results",
        "",
        f"- Cases: {overall['count']}",
        f"- Accuracy: {overall['accuracy']:.4f} ({overall['accuracy']:.2%})",
        f"- Macro precision: {overall['macro_precision']:.4f}",
        f"- Macro recall: {overall['macro_recall']:.4f}",
        f"- Macro F1: {overall['macro_f1']:.4f}",
        f"- Weighted F1: {overall['weighted_f1']:.4f}",
        f"- Evidence Unsupported-Claim Rate (prediction-based): {overall['ucr_prediction_based']:.4f} "
        f"({overall['ucr_prediction_based']:.2%})",
        f"- Evidence Unsupported-Claim Rate (gold-based, benchmark composition reference): "
        f"{overall['ucr_gold_based']:.4f} ({overall['ucr_gold_based']:.2%})",
        f"- Contradiction Rate (prediction-based): {overall['cr_prediction_based']:.4f} "
        f"({overall['cr_prediction_based']:.2%})",
        f"- Contradiction Rate (gold-based, benchmark composition reference): "
        f"{overall['cr_gold_based']:.4f} ({overall['cr_gold_based']:.2%})",
        "",
        "### Confusion matrix (rows = ground truth, columns = prediction)",
        "",
        "```",
        metrics.confusion_matrix_text(overall["confusion_matrix"]),
        "```",
        "",
        "## Language breakdown",
        "",
        "See `language_results.json` for full per-language precision/recall/F1/UCR/CR. Summary:",
        "",
        "| Language | Count | Accuracy | Macro F1 | UCR (pred) | CR (pred) |",
        "|---|---|---|---|---|---|",
    ]
    for language in LANGUAGES:
        stats = report["language_results"].get(language)
        if stats:
            lines.append(
                f"| {language} | {stats['count']} | {stats['accuracy']:.2%} | {stats['macro_f1']:.4f} | "
                f"{stats['ucr_prediction_based']:.2%} | {stats['cr_prediction_based']:.2%} |"
            )
    lines += [
        "",
        "## Category breakdown",
        "",
        "See `category_results.json`. Categories with very small sample counts should not be over-interpreted.",
        "",
        "## Difficulty breakdown",
        "",
        "See `difficulty_results.json`.",
        "",
        "## Failure analysis",
        "",
        f"{len(failures)} of {overall['count']} cases had ground_truth != prediction. "
        "Full detail (including premise/claim text, since all data is synthetic) is in `failure_analysis.json`. "
        "'possible_failure_categories' are advisory groupings based on case metadata, not asserted causes.",
        "",
        "## Statistical caution",
        "",
        "This benchmark contains ~100 synthetic cases. No claims of clinical safety, clinical accuracy, patient "
        "safety, HIPAA compliance, or real-world diagnostic accuracy are made or implied by these results. Any "
        "bootstrap confidence intervals reported reflect sampling uncertainty over this specific 101-case benchmark "
        "only, not real-world generalization.",
        "",
        "## Future work (explicitly out of scope for this step)",
        "",
        "- Threshold calibration/optimization was NOT performed and is left as future work.",
        "- No model fine-tuning or training was performed.",
        "",
    ]
    return "\n".join(lines)


def save_results(records: list[dict], cases_by_id: dict[str, dict], metadata: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    report = overall_report(records, metadata)
    failures = failure_analysis(records, cases_by_id)
    confusion = confusion_matrix_json(records)

    def dump(name: str, payload) -> None:
        with open(out_dir / name, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")

    dump("results.json", report)
    dump("classification_report.json", {"overall": report["overall"], "metadata": metadata})
    dump("confusion_matrix.json", confusion)
    dump("language_results.json", report["language_results"])
    dump("category_results.json", report["category_results"])
    dump("difficulty_results.json", report["difficulty_results"])
    dump("predictions.json", records)
    dump("failure_analysis.json", failures)

    with open(out_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(render_readme(report, failures))
