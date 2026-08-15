"""Run the Step 14G controlled baseline-vs-Tanglish-aware comparison with
the real local NLI model.

Usage:
    backend\\.venv\\Scripts\\python.exe -m app.evaluation_tanglish

Loads datasets/nli_evaluation/benchmark.json (unchanged, same file Step 14C
used), runs every case through BOTH the real, unmodified Step 14B baseline
and the Step 14F Tanglish-aware preprocessing layer -- same model, same
thresholds, same aggregation, same benchmark, same hardware -- and writes
result files to datasets/nli_evaluation/results/tanglish_comparison/. Does
not modify benchmark.json or any existing Step 14C result file.
"""
from pathlib import Path

from app.evaluation_tanglish import runner
from app.evaluation_tanglish.reports import save_comparison_results

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
BENCHMARK_PATH = REPO_ROOT / "datasets" / "nli_evaluation" / "benchmark.json"
RESULTS_DIR = REPO_ROOT / "datasets" / "nli_evaluation" / "results" / "tanglish_comparison"


def main() -> None:
    print(f"Loading benchmark from {BENCHMARK_PATH} ...")
    cases = runner.load_benchmark(BENCHMARK_PATH)
    print(f"Loaded {len(cases)} cases.")
    cases_by_id = {case["id"]: case for case in cases}

    print("Running SYSTEM A (baseline) ... (this loads the real model)")
    baseline_records = runner.run_mode(cases, "baseline")

    print("Running SYSTEM B (tanglish_aware) ...")
    tanglish_records = runner.run_mode(cases, "tanglish_aware")

    print(f"Writing results to {RESULTS_DIR} ...")
    save_comparison_results(baseline_records, tanglish_records, cases_by_id, BENCHMARK_PATH, RESULTS_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
