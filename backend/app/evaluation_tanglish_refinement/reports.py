"""Step 14I report assembly: writes every result file under
datasets/nli_evaluation/results/tanglish_refinement/. Never touches
benchmark.json or any existing datasets/nli_evaluation/results/*.json file,
including the Step 14G tanglish_comparison/ directory (read-only reference
for the reproduction check below).
"""
import json
from pathlib import Path

from app.evaluation import metrics
from app.evaluation import reports as base_reports
from app.evaluation_tanglish_refinement import comparison as cmp
from app.evaluation_tanglish_refinement.runner import evaluation_metadata

REPRODUCTION_TOLERANCE = 1e-6


def _dump(out_dir: Path, name: str, payload) -> None:
    with open(out_dir / name, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _system_report(records: list[dict], metadata: dict) -> dict:
    report = base_reports.overall_report(records, metadata)
    report["predictions"] = records
    return report


class ReproductionMismatchError(Exception):
    """Raised when the current-run 14B/14F reproduction diverges from the
    historical Step 14G results beyond REPRODUCTION_TOLERANCE. Per the Step
    14I spec, evaluation MUST stop here rather than silently continuing to
    the final three-way comparison.
    """


def load_historical_reference(tanglish_comparison_dir: Path) -> dict:
    """Read-only load of Step 14G's own results, for reproduction
    comparison purposes only. Never writes to this directory."""
    with open(tanglish_comparison_dir / "baseline_results.json", encoding="utf-8") as f:
        historical_14b = json.load(f)
    with open(tanglish_comparison_dir / "tanglish_aware_results.json", encoding="utf-8") as f:
        historical_14f = json.load(f)
    return {
        "14b_accuracy": historical_14b["overall"]["accuracy"],
        "14b_macro_f1": historical_14b["overall"]["macro_f1"],
        "14f_accuracy": historical_14f["overall"]["accuracy"],
        "14f_macro_f1": historical_14f["overall"]["macro_f1"],
    }


def check_reproduction(records_14b: list[dict], records_14f: list[dict], historical: dict) -> dict:
    """Compares the CURRENT run's 14B/14F overall accuracy/macro-F1 against
    the historical Step 14G reference values. Returns a report dict;
    raises ReproductionMismatchError if any value differs beyond
    REPRODUCTION_TOLERANCE (both systems are fully deterministic given the
    same model/thresholds/benchmark, so an exact or near-exact match is
    expected -- a mismatch would indicate an unintended change somewhere
    in the pipeline and must be investigated, not silently accepted).
    """
    current_14b = metrics.summarize(records_14b)
    current_14f = metrics.summarize(records_14f)

    checks = {
        "14b_accuracy": (current_14b["accuracy"], historical["14b_accuracy"]),
        "14b_macro_f1": (current_14b["macro_f1"], historical["14b_macro_f1"]),
        "14f_accuracy": (current_14f["accuracy"], historical["14f_accuracy"]),
        "14f_macro_f1": (current_14f["macro_f1"], historical["14f_macro_f1"]),
    }

    result = {"tolerance": REPRODUCTION_TOLERANCE, "checks": {}, "all_matched": True}
    for name, (current, expected) in checks.items():
        diff = abs(current - expected)
        matched = diff <= REPRODUCTION_TOLERANCE
        result["checks"][name] = {"current": current, "historical": expected, "diff": diff, "matched": matched}
        if not matched:
            result["all_matched"] = False

    if not result["all_matched"]:
        raise ReproductionMismatchError(
            "Current 14B/14F reproduction diverges from historical Step 14G results: "
            f"{json.dumps(result['checks'], indent=2)}"
        )

    return result


def _render_discovered_issues(comparison: dict) -> list[str]:
    """Reports any 14F -> 14H regressions found during Step 14I's real
    evaluation run, entirely data-driven from the actual comparison result
    (never a hardcoded case list). Step 14I is evaluation-only: per its own
    spec, a genuine generic bug discovered here must be REPORTED, not
    silently patched -- this section exists so that reporting happens in
    the result artifact itself, not only in chat.
    """
    regressed = comparison["fixed_regressed_14f_to_14h"]["regressed_cases"]
    if not regressed:
        return ["## Discovered issues", "", "No 14F -> 14H regressions were observed on this benchmark run.", ""]

    lines = [
        "## Discovered issues (Step 14I finding -- reported, not fixed)",
        "",
        f"Step 14I is evaluation-only and must not modify Step 14H while measuring it. The "
        f"{len(regressed)} case(s) below regressed from 14F to 14H on this run and were inspected for a "
        "GENERIC root cause (never a benchmark-specific one) before being reported here, unfixed:",
        "",
    ]
    unrecognized_by_14f_count = 0
    for case in regressed:
        was_unrecognized_by_14f = case["normalized_premise_14f"] == case["premise"]
        if was_unrecognized_by_14f:
            unrecognized_by_14f_count += 1
        lines.append(
            f"- **{case['case_id']}** ({case['language']}/{case['category']}): premise "
            f"`\"{case['premise']}\"`, claim `\"{case['claim']}\"`, gold `{case['ground_truth']}`. "
            f"14F -> `{case['prediction_14f']}` (normalized premise: `\"{case['normalized_premise_14f']}\"`"
            f"{' -- i.e. UNRECOGNIZED, passed through unchanged under the frozen Step 14F lexicon' if was_unrecognized_by_14f else ''}); "
            f"14H -> `{case['prediction_14h']}` (normalized premise: `\"{case['normalized_premise_14h']}\"`)."
        )

    if unrecognized_by_14f_count:
        lines += [
            "",
            "**Likely generic root cause** for the case(s) flagged \"UNRECOGNIZED\" above: Step 14H added "
            "new bare clinical-entity vocabulary to `lexicon.py::CLINICAL_TERMS` (`hypertension`, "
            "`allergy`) to support its attribution-safety test coverage. Recognizing a previously-"
            "unrecognized entity is a double-edged change: it can also cause the composer to now rewrite a "
            "clause that used to pass through completely UNCHANGED under Step 14F -- and the composer only "
            "ever preserves the *entity itself* plus duration/frequency/measurement qualifiers, never an "
            "adjacent, non-entity qualifying noun (e.g. a specific allergen or drug name that is not itself "
            "in `CLINICAL_TERMS`). The result can be a *less* specific normalized sentence than the "
            "original raw text, for cases the frozen 14F lexicon simply never touched. This is a generic "
            "limitation of the compose-from-recognized-entity architecture (present since Step 14F, not "
            "introduced by Step 14H's own logic changes) that newly surfaced here because Step 14H's "
            "vocabulary additions widened lexicon coverage. Per the Step 14I instructions, this is reported "
            "as a finding for a future step, not patched during this evaluation-only step, and no "
            "benchmark-specific exception was added to work around it.",
            "",
        ]
    return lines


def render_readme(
    baseline_report: dict, tanglish_14f_report: dict, tanglish_14h_report: dict,
    comparison: dict, metadata: dict, reproduction_check: dict,
) -> str:
    overall = comparison["overall"]
    lang = comparison["language_comparison"]
    transition_counts = comparison["transition_analysis"]["counts"]
    highlight = comparison["transition_analysis"]["highlight"]
    fixed_regressed = comparison["fixed_regressed_14f_to_14h"]["counts"]
    mcnemar = comparison["mcnemar_tests"]

    def lang_row(language: str) -> str:
        entry = lang.get(language, {})
        b = entry.get("14b", {})
        f = entry.get("14f", {})
        h = entry.get("14h", {})
        return (
            f"| {language:<10} | {b.get('accuracy', float('nan')):.2%} | "
            f"{f.get('accuracy', float('nan')):.2%} | {h.get('accuracy', float('nan')):.2%} |"
        )

    lines = [
        "# Step 14I: Controlled Three-Way Evaluation (14B vs 14F vs 14H)",
        "",
        "Controlled comparison of THREE systems against the exact same Step 14E "
        "101-case gold benchmark, using the exact same, frozen Step 14B NLI model, "
        "thresholds, and aggregation rule throughout. Only the premise "
        "preprocessing differs between systems.",
        "",
        "- **SYSTEM A (14B)**: raw premise -> mDeBERTa NLI -> aggregation. No Tanglish preprocessing.",
        "- **SYSTEM B (14F)**: premise -> FROZEN Step 14F normalizer (byte-identical-logic snapshot of "
        "commit 5563418, before Step 14H) -> the SAME NLI -> the SAME aggregation.",
        "- **SYSTEM C (14H)**: premise -> CURRENT Step 14H normalizer (numeric/measurement preservation + "
        "family/third-person attribution safety) -> the SAME NLI -> the SAME aggregation.",
        "",
        "## Reproduction check (required before this comparison was produced)",
        "",
        "Per the Step 14I spec, the current run's 14B and 14F reproduction was compared against the "
        "historical Step 14G results (`datasets/nli_evaluation/results/tanglish_comparison/`) BEFORE "
        "producing this comparison. If any value had differed beyond the documented tolerance, evaluation "
        "would have stopped without producing a final comparison.",
        "",
        f"- Tolerance: {reproduction_check['tolerance']}",
        f"- All matched: **{reproduction_check['all_matched']}**",
        "",
        "| Check | Current run | Historical (Step 14G) | Diff | Matched |",
        "|---|---|---|---|---|",
    ]
    for name, check in reproduction_check["checks"].items():
        lines.append(
            f"| {name} | {check['current']:.6f} | {check['historical']:.6f} | {check['diff']:.2e} | {check['matched']} |"
        )
    lines += [
        "",
        "## Research integrity",
        "",
        "1. Synthetic benchmark (101 hand-authored cases), not a clinical dataset.",
        "2. Model, thresholds (entailment=0.70, contradiction=0.70), and aggregation rule are IDENTICAL "
        "and UNCHANGED across all three systems; only the premise text fed to the model differs.",
        "3. No fine-tuning, no new/alternate model, no external API.",
        "4. Do NOT assume 14H is an improvement -- the conclusion below is drawn from the measured results.",
        "5. Evidence Unsupported-Claim Rate (UCR) is an evidence-verification metric, not a clinical "
        "hallucination rate. Contradiction Rate (CR) is reported separately.",
        "6. These results do NOT establish clinical safety, clinical validity, or real-world diagnostic accuracy.",
        "",
        "## Reproducibility metadata",
        "",
        f"- Model provider: `{metadata['model_provider']}`",
        f"- Model name: `{metadata['model_name']}`",
        f"- Device: `{metadata['device']}`",
        f"- Entailment threshold: `{metadata['entailment_threshold']}`",
        f"- Contradiction threshold: `{metadata['contradiction_threshold']}`",
        f"- System 14B configuration: {metadata['system_14b_configuration']}",
        f"- System 14F configuration: {metadata['system_14f_configuration']}",
        f"- System 14H configuration: {metadata['system_14h_configuration']}",
        f"- Benchmark file: `{metadata['benchmark_path']}`",
        f"- Benchmark SHA-256: `{metadata['benchmark_sha256']}`",
        f"- Number of cases evaluated: `{metadata['num_cases']}`",
        f"- Evaluation timestamp (UTC): `{metadata['evaluation_timestamp_utc']}`",
        f"- Python version: `{metadata['python_version']}`",
        f"- Platform: `{metadata['platform']}`",
        "",
        "## Overall results",
        "",
        "| System | Accuracy | Macro-F1 | UCR | CR |",
        "|---|---|---|---|---|",
        f"| 14B Baseline | {overall['14b']['accuracy']:.2%} | {overall['14b']['macro_f1']:.4f} | "
        f"{overall['14b']['ucr_prediction_based']:.2%} | {overall['14b']['cr_prediction_based']:.2%} |",
        f"| 14F Tanglish-aware | {overall['14f']['accuracy']:.2%} | {overall['14f']['macro_f1']:.4f} | "
        f"{overall['14f']['ucr_prediction_based']:.2%} | {overall['14f']['cr_prediction_based']:.2%} |",
        f"| 14H Refined | {overall['14h']['accuracy']:.2%} | {overall['14h']['macro_f1']:.4f} | "
        f"{overall['14h']['ucr_prediction_based']:.2%} | {overall['14h']['cr_prediction_based']:.2%} |",
        "",
        "## Main comparison (percentage-point changes)",
        "",
        "| Transition | Accuracy Δ (pp) | Macro-F1 Δ (pp) |",
        "|---|---|---|",
        f"| 14F − 14B | {overall['14f_minus_14b']['accuracy']['percentage_point_change']:+.2f} | "
        f"{overall['14f_minus_14b']['macro_f1']['percentage_point_change']:+.2f} |",
        f"| 14H − 14F | {overall['14h_minus_14f']['accuracy']['percentage_point_change']:+.2f} | "
        f"{overall['14h_minus_14f']['macro_f1']['percentage_point_change']:+.2f} |",
        f"| 14H − 14B | {overall['14h_minus_14b']['accuracy']['percentage_point_change']:+.2f} | "
        f"{overall['14h_minus_14b']['macro_f1']['percentage_point_change']:+.2f} |",
        "",
        "## Language comparison (accuracy)",
        "",
        "| Language | 14B | 14F | 14H |",
        "|---|---|---|---|",
        lang_row("en"),
        lang_row("tanglish"),
        lang_row("mixed"),
        "",
        "See `language_comparison.json` for full precision/recall/F1/UCR/CR per language per system.",
        "",
        "## Refinement change (14F -> 14H)",
        "",
        "| Metric | 14F → 14H |",
        "|---|---|",
        f"| Overall accuracy | {overall['14h_minus_14f']['accuracy']['percentage_point_change']:+.2f} pp |",
        f"| Overall Macro-F1 | {overall['14h_minus_14f']['macro_f1']['percentage_point_change']:+.2f} pp |",
        f"| Tanglish accuracy | {lang.get('tanglish', {}).get('14h_minus_14f', {}).get('accuracy', float('nan')) * 100:+.2f} pp |",
        f"| Tanglish Macro-F1 | {lang.get('tanglish', {}).get('14h_minus_14f', {}).get('macro_f1', float('nan')) * 100:+.2f} pp |",
        f"| Mixed accuracy | {lang.get('mixed', {}).get('14h_minus_14f', {}).get('accuracy', float('nan')) * 100:+.2f} pp |",
        f"| Mixed Macro-F1 | {lang.get('mixed', {}).get('14h_minus_14f', {}).get('macro_f1', float('nan')) * 100:+.2f} pp |",
        f"| UCR | {overall['14h_minus_14f']['ucr_prediction_based']['percentage_point_change']:+.2f} pp |",
        f"| CR | {overall['14h_minus_14f']['cr_prediction_based']['percentage_point_change']:+.2f} pp |",
        "",
        "## Fixed / regressed (14F -> 14H)",
        "",
        f"- Fixed by 14H: **{fixed_regressed['fixed']}**",
        f"- Regressed by 14H: **{fixed_regressed['regressed']}**",
        f"- Unchanged correct: **{fixed_regressed['unchanged_correct']}**",
        f"- Unchanged incorrect: **{fixed_regressed['unchanged_incorrect']}**",
        "",
        "## 14B -> 14F -> 14H transition highlights",
        "",
        f"- Preserved improvement (14B wrong, 14F correct, 14H correct): "
        f"**{len(highlight['preserved_improvement__14b_wrong_14f_correct_14h_correct'])}**",
        f"- Regression (14B wrong, 14F correct, 14H wrong): "
        f"**{len(highlight['regression__14b_wrong_14f_correct_14h_wrong'])}**",
        f"- New gain from 14H (14B wrong, 14F wrong, 14H correct): "
        f"**{len(highlight['new_gain_from_14h__14b_wrong_14f_wrong_14h_correct'])}**",
        "",
        "Full 8-way bucket counts (correct/wrong x 14B x 14F x 14H): see `comparison.json` -> "
        "`transition_analysis.counts`.",
        "",
        "## Statistical tests (exact McNemar, binary correct/incorrect, paired)",
        "",
        "| Pair | n discordant | p-value |",
        "|---|---|---|",
        f"| 14B vs 14F | {mcnemar['14b_vs_14f']['n_discordant_pairs']} | {mcnemar['14b_vs_14f']['p_value']:.4f} |",
        f"| 14F vs 14H | {mcnemar['14f_vs_14h']['n_discordant_pairs']} | {mcnemar['14f_vs_14h']['p_value']:.4f} |",
        f"| 14B vs 14H | {mcnemar['14b_vs_14h']['n_discordant_pairs']} | {mcnemar['14b_vs_14h']['p_value']:.4f} |",
        "",
        "No claim of statistical significance beyond these reported p-values is made. The benchmark is "
        "101 synthetic cases; these are descriptive, paired-by-case comparisons only.",
        "",
        "## Targeted Step 14H analyses",
        "",
        "See `targeted_analysis.json` for full numeric/measurement, attribution, negation, and "
        "historical-context breakdowns across all three systems, including 14F->14H fixed/regressed "
        "cases restricted to each category subset.",
        "",
        "## Research interpretation",
        "",
        "Research question: do the Step 14H numeric-preservation and attribution-safety refinements "
        "improve the Tanglish-aware NLI system relative to Step 14F, while preserving Step 14F's gains "
        "over the Step 14B baseline?",
        "",
        f"On this 101-case synthetic benchmark, Step 14H changed overall accuracy by "
        f"{overall['14h_minus_14f']['accuracy']['percentage_point_change']:+.2f} percentage points and "
        f"overall Macro-F1 by {overall['14h_minus_14f']['macro_f1']['percentage_point_change']:+.2f} "
        f"percentage points relative to Step 14F. The 14F-vs-14H paired McNemar comparison produced "
        f"p = {mcnemar['14f_vs_14h']['p_value']:.4f} "
        f"(n={mcnemar['14f_vs_14h']['n_discordant_pairs']} discordant pairs). See the language/category "
        "breakdowns above and in the JSON result files for the full, unabridged picture (including any "
        "language- or category-specific regressions) before drawing a conclusion beyond this aggregate "
        "figure -- an aggregate accuracy change can mask offsetting gains and losses across languages, "
        "and the correct scientific statement should always cite the specific breakdown it is based on.",
        "",
        "No claim of clinical validity, HIPAA compliance, or production accuracy is made.",
        "",
    ]
    lines += _render_discovered_issues(comparison)
    lines += [
        "## Files",
        "",
        "- `baseline_14b_results.json` / `tanglish_14f_results.json` / `tanglish_14h_results.json`: full "
        "single-system report (overall + language/category/difficulty breakdowns + bootstrap CIs + every "
        "per-case prediction) for each system.",
        "- `comparison.json`: full three-way comparison (overall, language/category/difficulty, per-class, "
        "confusion matrices, 14F->14H fixed/regressed, 14B->14F->14H transitions, McNemar tests, targeted analyses).",
        "- `language_comparison.json` / `category_comparison.json` / `difficulty_comparison.json` / "
        "`targeted_analysis.json` / `confusion_matrices.json` / `statistical_tests.json`: standalone copies "
        "of the corresponding `comparison.json` sections.",
        "- `failure_analysis.json`: 14F->14H fixed/regressed/unchanged case-level detail plus the "
        "14B->14F->14H transition highlight buckets.",
        "",
    ]
    return "\n".join(lines)


def save_refinement_results(
    records_14b: list[dict], records_14f: list[dict], records_14h: list[dict],
    cases_by_id: dict[str, dict], benchmark_path: Path, out_dir: Path, reproduction_check: dict,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = evaluation_metadata(benchmark_path, len(records_14b))

    report_14b = _system_report(records_14b, metadata)
    report_14f = _system_report(records_14f, metadata)
    report_14h = _system_report(records_14h, metadata)
    comparison = cmp.build_comparison(records_14b, records_14f, records_14h, cases_by_id)

    _dump(out_dir, "baseline_14b_results.json", report_14b)
    _dump(out_dir, "tanglish_14f_results.json", report_14f)
    _dump(out_dir, "tanglish_14h_results.json", report_14h)
    _dump(out_dir, "comparison.json", comparison)
    _dump(out_dir, "language_comparison.json", comparison["language_comparison"])
    _dump(out_dir, "category_comparison.json", comparison["category_comparison"])
    _dump(out_dir, "difficulty_comparison.json", comparison["difficulty_comparison"])
    _dump(out_dir, "targeted_analysis.json", {
        "numeric_measurement_analysis": comparison["numeric_measurement_analysis"],
        "attribution_analysis": comparison["attribution_analysis"],
        "negation_analysis": comparison["negation_analysis"],
        "historical_context_analysis": comparison["historical_context_analysis"],
    })
    _dump(out_dir, "confusion_matrices.json", comparison["confusion_matrices"])
    _dump(out_dir, "statistical_tests.json", comparison["mcnemar_tests"])
    _dump(out_dir, "failure_analysis.json", {
        "fixed_regressed_14f_to_14h": comparison["fixed_regressed_14f_to_14h"],
        "transition_highlight": comparison["transition_analysis"]["highlight"],
        "transition_counts": comparison["transition_analysis"]["counts"],
    })
    _dump(out_dir, "reproduction_check.json", reproduction_check)

    with open(out_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(render_readme(report_14b, report_14f, report_14h, comparison, metadata, reproduction_check))

    return {
        "metadata": metadata,
        "report_14b": report_14b,
        "report_14f": report_14f,
        "report_14h": report_14h,
        "comparison": comparison,
    }
