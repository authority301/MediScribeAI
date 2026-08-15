"""Step 14G report assembly: writes every result file under
datasets/nli_evaluation/results/tanglish_comparison/. Never touches
benchmark.json or any existing datasets/nli_evaluation/results/*.json file
(Step 14C's own output directory).
"""
import json
from pathlib import Path

from app.evaluation import metrics
from app.evaluation import reports as base_reports
from app.evaluation_tanglish import comparison as cmp
from app.evaluation_tanglish.runner import evaluation_metadata


def _dump(out_dir: Path, name: str, payload) -> None:
    with open(out_dir / name, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _system_report(records: list[dict], metadata: dict) -> dict:
    """Full single-system report (overall + language/category/difficulty
    breakdowns + bootstrap CIs), built with the exact same, unmodified
    app.evaluation.reports.overall_report function Step 14C uses -- plus the
    raw per-case predictions for this system.
    """
    report = base_reports.overall_report(records, metadata)
    report["predictions"] = records
    return report


def _table_row(language: str, lang_cmp: dict, side: str) -> str:
    stats = lang_cmp.get(language, {}).get(side)
    if not stats:
        return "n/a"
    return f"{stats['accuracy']:.2%} (F1={stats['macro_f1']:.4f})"


def render_readme(baseline_report: dict, tanglish_report: dict, comparison: dict, metadata: dict) -> str:
    b_overall = baseline_report["overall"]
    t_overall = tanglish_report["overall"]
    lang_cmp = comparison["language_comparison"]
    imp = comparison["improvement"]
    counts = comparison["case_level"]["counts"]
    mcnemar = comparison["mcnemar_exact_test"]

    def lang_line(lang: str) -> str:
        return f"| {lang:<10} | {_table_row(lang, lang_cmp, 'baseline')} | {_table_row(lang, lang_cmp, 'tanglish_aware')} |"

    lines = [
        "# Step 14G: Controlled Baseline vs. Tanglish-Aware NLI Evaluation",
        "",
        "Controlled, ablation-style comparison of two systems against the exact "
        "same Step 14E 101-case gold benchmark, using the exact same, frozen "
        "Step 14B NLI model, thresholds, and aggregation rule. The ONLY "
        "difference between the two systems is whether the Step 14F "
        "Tanglish-aware preprocessing layer runs before the premise reaches "
        "the NLI model.",
        "",
        "- **SYSTEM A (baseline)**: raw premise -> mDeBERTa multilingual NLI -> aggregation.",
        "- **SYSTEM B (tanglish_aware)**: premise -> Step 14F normalization -> the SAME mDeBERTa NLI -> the SAME aggregation.",
        "",
        "## Research integrity",
        "",
        "1. Synthetic benchmark (101 hand-authored cases), not a clinical dataset.",
        "2. Model, thresholds (entailment=0.70, contradiction=0.70), and aggregation rule are IDENTICAL and UNCHANGED "
        "between systems; only the premise text fed to the model differs.",
        "3. No fine-tuning, no new/alternate model, no external API.",
        "4. Evidence Unsupported-Claim Rate (UCR) is an evidence-verification metric, not a clinical hallucination rate.",
        "5. Contradiction Rate (CR) is reported separately from UCR.",
        "6. These results do NOT establish clinical safety, clinical validity, or real-world diagnostic accuracy.",
        "7. Results are descriptive and paired-by-case; see 'Statistical test' below for what was and was not tested.",
        "",
        "## Reproducibility metadata",
        "",
        f"- Model provider: `{metadata['model_provider']}`",
        f"- Model name: `{metadata['model_name']}`",
        f"- Device: `{metadata['device']}`",
        f"- Entailment threshold: `{metadata['entailment_threshold']}`",
        f"- Contradiction threshold: `{metadata['contradiction_threshold']}`",
        f"- Baseline mode: `{metadata['baseline_mode']}`",
        f"- Tanglish-aware mode: `{metadata['tanglish_aware_mode']}`",
        f"- Benchmark file: `{metadata['benchmark_path']}`",
        f"- Benchmark SHA-256: `{metadata['benchmark_sha256']}`",
        f"- Number of cases evaluated: `{metadata['num_cases']}`",
        f"- Evaluation timestamp (UTC): `{metadata['evaluation_timestamp_utc']}`",
        f"- Python version: `{metadata['python_version']}`",
        f"- Platform: `{metadata['platform']}`",
        "",
        "## Overall results",
        "",
        "| System | Accuracy | Macro F1 | Weighted F1 | UCR (pred) | CR (pred) |",
        "|---|---|---|---|---|---|",
        f"| Baseline 14B | {b_overall['accuracy']:.2%} | {b_overall['macro_f1']:.4f} | {b_overall['weighted_f1']:.4f} | "
        f"{b_overall['ucr_prediction_based']:.2%} | {b_overall['cr_prediction_based']:.2%} |",
        f"| Tanglish-aware 14F | {t_overall['accuracy']:.2%} | {t_overall['macro_f1']:.4f} | {t_overall['weighted_f1']:.4f} | "
        f"{t_overall['ucr_prediction_based']:.2%} | {t_overall['cr_prediction_based']:.2%} |",
        "",
        "## By-language results (Accuracy, Macro F1)",
        "",
        "| Language | Baseline | Tanglish-aware |",
        "|---|---|---|",
        lang_line("en"),
        lang_line("tanglish"),
        lang_line("mixed"),
        "",
        "## Improvement metrics (overall, absolute change = tanglish_aware - baseline)",
        "",
        "| Metric | Baseline | Tanglish-aware | Change (pp) |",
        "|---|---|---|---|",
        f"| Accuracy | {imp['accuracy']['baseline']:.4f} | {imp['accuracy']['tanglish_aware']:.4f} | "
        f"{imp['accuracy']['percentage_point_change']:+.2f} |",
        f"| Macro F1 | {imp['macro_f1']['baseline']:.4f} | {imp['macro_f1']['tanglish_aware']:.4f} | "
        f"{imp['macro_f1']['percentage_point_change']:+.2f} |",
        f"| UCR (pred) | {imp['ucr_prediction_based']['baseline']:.4f} | {imp['ucr_prediction_based']['tanglish_aware']:.4f} | "
        f"{imp['ucr_prediction_based']['percentage_point_change']:+.2f} |",
        f"| CR (pred) | {imp['cr_prediction_based']['baseline']:.4f} | {imp['cr_prediction_based']['tanglish_aware']:.4f} | "
        f"{imp['cr_prediction_based']['percentage_point_change']:+.2f} |",
        "",
        "Tanglish-specific improvement is the primary comparison of interest -- see `language_comparison.json` "
        "for the Tanglish row's own accuracy/macro-F1/UCR/CR baseline-vs-tanglish_aware figures.",
        "",
        "## Per-class comparison",
        "",
        "See `comparison.json` -> `per_class_comparison` for SUPPORTED/CONTRADICTED/UNGROUNDED "
        "precision/recall/F1 for both systems and their differences.",
        "",
        "## Confusion matrices",
        "",
        "Baseline (rows = ground truth, columns = prediction):",
        "```",
        metrics.confusion_matrix_text(comparison["confusion_matrices"]["baseline"]),
        "```",
        "",
        "Tanglish-aware (rows = ground truth, columns = prediction):",
        "```",
        metrics.confusion_matrix_text(comparison["confusion_matrices"]["tanglish_aware"]),
        "```",
        "",
        "See `confusion_matrices.json` for the difference matrix (tanglish_aware - baseline).",
        "",
        "## Category / difficulty comparison",
        "",
        "See `category_comparison.json` and `difficulty_comparison.json`.",
        "",
        "## Negation category",
        "",
        f"See `comparison.json` -> `negation_analysis`. Fixed: {len(comparison['negation_analysis']['fixed_cases'])}, "
        f"Regressed: {len(comparison['negation_analysis']['regressed_cases'])}.",
        "",
        "## Historical statements / patient history",
        "",
        f"See `comparison.json` -> `historical_analysis`. Fixed: {len(comparison['historical_analysis']['fixed_cases'])}, "
        f"Regressed: {len(comparison['historical_analysis']['regressed_cases'])}.",
        "",
        "## Failure analysis (case-level)",
        "",
        f"- Fixed by Tanglish-aware preprocessing (baseline wrong, tanglish_aware right): **{counts['fixed']}**",
        f"- Regressed (baseline right, tanglish_aware wrong): **{counts['regressed']}**",
        f"- Unchanged and correct in both: **{counts['unchanged_correct']}**",
        f"- Unchanged and incorrect in both: **{counts['unchanged_incorrect']}**",
        "",
        "Full per-case detail (premise, claim, gold label, both predictions, both NLI score sets, normalized "
        "premise, transformation trace) is in `failure_analysis.json`.",
        "",
        "## Statistical test",
        "",
        f"- Method: {mcnemar['test']}",
        f"- Applies to: {mcnemar['applies_to']}",
        f"- n (discordant pairs): {mcnemar['n_discordant_pairs']} out of {mcnemar['n_cases_total']} total cases",
        f"- b (baseline correct, tanglish-aware incorrect): {mcnemar['b_baseline_correct_tanglish_aware_incorrect']}",
        f"- c (baseline incorrect, tanglish-aware correct): {mcnemar['c_baseline_incorrect_tanglish_aware_correct']}",
        f"- p-value: {mcnemar['p_value']:.4f}",
        f"- {mcnemar['note']}",
        "",
        "No claim of statistical significance beyond this reported p-value is made. The benchmark is 101 "
        "synthetic cases; this is a descriptive, paired-by-case comparison, not evidence of real-world "
        "generalization.",
        "",
        "## Research interpretation",
        "",
        "Research question: does deterministic Tanglish-aware clinical semantic normalization improve "
        "multilingual NLI evidence verification for Tanglish clinical text compared with the unmodified "
        "multilingual NLI baseline, on this 101-case synthetic benchmark?",
        "",
        f"Step 14F produced a {imp['accuracy']['percentage_point_change']:+.2f} percentage-point change in overall "
        f"accuracy and a {imp['macro_f1']['percentage_point_change']:+.2f} percentage-point change in overall "
        "macro F1 relative to the frozen Step 14B baseline, on this benchmark. See `language_comparison.json` "
        "for the Tanglish-only and Mixed-only figures, and English-only figures confirming (or not) that "
        "English performance was preserved. This experiment reports the measured result; it does not assert "
        "the hypothesis is confirmed beyond what these numbers show.",
        "",
        "## Files",
        "",
        "- `baseline_results.json` / `tanglish_aware_results.json`: full single-system report (overall + "
        "language/category/difficulty breakdowns + bootstrap CIs + every per-case prediction) for each system, "
        "built with the same `app.evaluation.reports.overall_report` function Step 14C uses.",
        "- `comparison.json`: improvement metrics, grouped comparisons, per-class comparison, confusion "
        "matrices (+ diff), case-level fixed/regressed/unchanged lists, McNemar test, negation/historical focus.",
        "- `language_comparison.json` / `category_comparison.json` / `difficulty_comparison.json`: standalone "
        "copies of the corresponding `comparison.json` sections.",
        "- `confusion_matrices.json`: standalone copy of the confusion-matrix section.",
        "- `failure_analysis.json`: fixed/regressed/unchanged case-level detail.",
        "",
        "## Future work (explicitly out of scope for this step)",
        "",
        "- No threshold recalibration was performed for either system.",
        "- No model fine-tuning or training was performed.",
        "- No new benchmark cases were added or removed.",
        "",
    ]
    return "\n".join(lines)


def save_comparison_results(baseline_records: list[dict], tanglish_records: list[dict],
                             cases_by_id: dict[str, dict], benchmark_path: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = evaluation_metadata(benchmark_path, len(baseline_records))

    baseline_report = _system_report(baseline_records, metadata)
    tanglish_report = _system_report(tanglish_records, metadata)
    comparison = cmp.build_comparison(baseline_records, tanglish_records, cases_by_id)

    _dump(out_dir, "baseline_results.json", baseline_report)
    _dump(out_dir, "tanglish_aware_results.json", tanglish_report)
    _dump(out_dir, "comparison.json", comparison)
    _dump(out_dir, "language_comparison.json", comparison["language_comparison"])
    _dump(out_dir, "category_comparison.json", comparison["category_comparison"])
    _dump(out_dir, "difficulty_comparison.json", comparison["difficulty_comparison"])
    _dump(out_dir, "confusion_matrices.json", comparison["confusion_matrices"])
    _dump(out_dir, "failure_analysis.json", {
        "fixed_cases": comparison["case_level"]["fixed_cases"],
        "regressed_cases": comparison["case_level"]["regressed_cases"],
        "unchanged_correct_cases": comparison["case_level"]["unchanged_correct_cases"],
        "unchanged_incorrect_cases": comparison["case_level"]["unchanged_incorrect_cases"],
        "counts": comparison["case_level"]["counts"],
    })

    with open(out_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(render_readme(baseline_report, tanglish_report, comparison, metadata))

    return {
        "metadata": metadata,
        "baseline_report": baseline_report,
        "tanglish_report": tanglish_report,
        "comparison": comparison,
    }
