"""Run the Step 14C gold-benchmark evaluation with the real local NLI model.

Usage:
    backend\\.venv\\Scripts\\python.exe -m app.evaluation

Loads datasets/nli_evaluation/benchmark.json, runs every case through the
real, unmodified Step 14B model + aggregation (app.nli.model.run_nli +
app.nli.aggregation.aggregate_claim_verification), and writes result files
to datasets/nli_evaluation/results/. Does not modify benchmark.json.
"""
from pathlib import Path

from app.evaluation import runner
from app.evaluation.reports import save_results

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
BENCHMARK_PATH = REPO_ROOT / "datasets" / "nli_evaluation" / "benchmark.json"
RESULTS_DIR = REPO_ROOT / "datasets" / "nli_evaluation" / "results"


def main() -> None:
    print(f"Loading benchmark from {BENCHMARK_PATH} ...")
    cases = runner.load_benchmark(BENCHMARK_PATH)
    print(f"Loaded {len(cases)} cases. Running real NLI evaluation (this loads the real model)...")

    records = runner.run_evaluation(cases)
    cases_by_id = {case["id"]: case for case in cases}
    metadata = runner.evaluation_metadata(BENCHMARK_PATH, len(cases))

    print(f"Writing results to {RESULTS_DIR} ...")
    save_results(records, cases_by_id, metadata, RESULTS_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
