"""Run the Step 14I controlled three-way comparison (14B vs 14F vs 14H)
with the real local NLI model.

Usage:
    backend\\.venv\\Scripts\\python.exe -m app.evaluation_tanglish_refinement

Run order (per the Step 14I spec):
  1. Load datasets/nli_evaluation/benchmark.json (unchanged).
  2. Run SYSTEM A (14B baseline) with the real model.
  3. Run SYSTEM B (14F, via the frozen commit-5563418 snapshot) with the
     real model.
  4. Compare both against the historical Step 14G results
     (datasets/nli_evaluation/results/tanglish_comparison/, read-only). If
     either diverges beyond the documented tolerance, STOP -- do not run
     14H or produce a final comparison.
  5. If reproduction succeeds: run SYSTEM C (14H, current normalizer).
  6. Write the full three-way comparison to
     datasets/nli_evaluation/results/tanglish_refinement/.

Never modifies benchmark.json, backend/app/nli/, or the Step 14G
tanglish_comparison/ directory.
"""
import sys
from pathlib import Path

from app.evaluation_tanglish_refinement import runner
from app.evaluation_tanglish_refinement.reports import (
    ReproductionMismatchError,
    check_reproduction,
    load_historical_reference,
    save_refinement_results,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
BENCHMARK_PATH = REPO_ROOT / "datasets" / "nli_evaluation" / "benchmark.json"
HISTORICAL_DIR = REPO_ROOT / "datasets" / "nli_evaluation" / "results" / "tanglish_comparison"
RESULTS_DIR = REPO_ROOT / "datasets" / "nli_evaluation" / "results" / "tanglish_refinement"


def main() -> None:
    print(f"Loading benchmark from {BENCHMARK_PATH} ...")
    cases = runner.load_benchmark(BENCHMARK_PATH)
    print(f"Loaded {len(cases)} cases.")
    cases_by_id = {case["id"]: case for case in cases}

    print("Running SYSTEM A (14B baseline) ... (this loads the real model)")
    records_14b = [runner.evaluate_case_14b(case) for case in cases]

    print("Running SYSTEM B (14F, frozen snapshot) ...")
    records_14f = [runner.evaluate_case_14f(case) for case in cases]

    print("Checking 14B/14F reproduction against historical Step 14G results ...")
    historical = load_historical_reference(HISTORICAL_DIR)
    try:
        reproduction_check = check_reproduction(records_14b, records_14f, historical)
    except ReproductionMismatchError as exc:
        print("REPRODUCTION MISMATCH -- stopping before producing the final Step 14I comparison.")
        print(str(exc))
        sys.exit(1)
    print(f"Reproduction check passed (tolerance={reproduction_check['tolerance']}).")

    print("Running SYSTEM C (14H, current refined normalizer) ...")
    records_14h = [runner.evaluate_case_14h(case) for case in cases]

    print(f"Writing results to {RESULTS_DIR} ...")
    save_refinement_results(records_14b, records_14f, records_14h, cases_by_id, BENCHMARK_PATH, RESULTS_DIR, reproduction_check)
    print("Done.")


if __name__ == "__main__":
    main()
