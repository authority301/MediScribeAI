"""Pure-Python classification metrics for the Step 14C gold-benchmark
evaluation. No new dependency is introduced (no numpy/sklearn/scipy) -- this
module only does counting, division, and (for the optional bootstrap
confidence intervals) fixed-seed resampling with the standard library
`random` module.

============================================================================
RESEARCH INTEGRITY
============================================================================
These functions measure agreement between the existing, UNMODIFIED Step 14B
NLI verifier's output and a synthetic gold benchmark. They are NOT clinical
accuracy/safety metrics. See datasets/nli_evaluation/results/README.md for
the full research-integrity statement.

Evidence Unsupported-Claim Rate (UCR) and Contradiction Rate (CR) are
reported on two different bases, both provided for transparency:
  - "prediction" basis: how often the pipeline's OWN OUTPUT was
    UNGROUNDED/CONTRADICTED. This is the operationally meaningful rate --
    what you would observe running this pipeline on new, unlabeled input.
  - "gold" basis: how often the BENCHMARK ITSELF was labeled
    UNGROUNDED/CONTRADICTED. This reflects benchmark composition, not model
    behavior, and is included only as reference context.
Unless stated otherwise, "UCR"/"CR" in the human-readable report refers to
the prediction-based figure, since that is what actually characterizes the
pipeline being evaluated.
"""
import random

LABELS = ["SUPPORTED", "CONTRADICTED", "UNGROUNDED"]


def confusion_matrix(records: list[dict]) -> dict:
    """Rows = ground truth, columns = prediction. records: list of dicts with
    'ground_truth' and 'prediction' keys, both in LABELS.
    """
    matrix = {gold: {pred: 0 for pred in LABELS} for gold in LABELS}
    for record in records:
        matrix[record["ground_truth"]][record["prediction"]] += 1
    return matrix


def confusion_matrix_text(matrix: dict) -> str:
    header = "GOLD \\ PRED".ljust(14) + "".join(label.ljust(14) for label in LABELS)
    lines = [header]
    for gold in LABELS:
        row = gold.ljust(14) + "".join(str(matrix[gold][pred]).ljust(14) for pred in LABELS)
        lines.append(row)
    return "\n".join(lines)


def per_class_prf(matrix: dict) -> dict:
    result = {}
    for label in LABELS:
        tp = matrix[label][label]
        fn = sum(matrix[label][pred] for pred in LABELS if pred != label)
        fp = sum(matrix[gold][label] for gold in LABELS if gold != label)
        support = tp + fn
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        result[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    return result


def macro_metrics(per_class: dict) -> dict:
    n = len(per_class)
    return {
        "precision": sum(v["precision"] for v in per_class.values()) / n,
        "recall": sum(v["recall"] for v in per_class.values()) / n,
        "f1": sum(v["f1"] for v in per_class.values()) / n,
    }


def weighted_metrics(per_class: dict) -> dict:
    total_support = sum(v["support"] for v in per_class.values())
    if total_support == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {
        "precision": sum(v["precision"] * v["support"] for v in per_class.values()) / total_support,
        "recall": sum(v["recall"] * v["support"] for v in per_class.values()) / total_support,
        "f1": sum(v["f1"] * v["support"] for v in per_class.values()) / total_support,
    }


def accuracy(records: list[dict]) -> float:
    if not records:
        return 0.0
    correct = sum(1 for r in records if r["ground_truth"] == r["prediction"])
    return correct / len(records)


def unsupported_claim_rate(records: list[dict], basis: str = "prediction") -> float:
    """basis: 'prediction' (pipeline output, operationally meaningful) or
    'gold' (benchmark composition, reference only)."""
    if not records:
        return 0.0
    key = "prediction" if basis == "prediction" else "ground_truth"
    count = sum(1 for r in records if r[key] == "UNGROUNDED")
    return count / len(records)


def contradiction_rate(records: list[dict], basis: str = "prediction") -> float:
    if not records:
        return 0.0
    key = "prediction" if basis == "prediction" else "ground_truth"
    count = sum(1 for r in records if r[key] == "CONTRADICTED")
    return count / len(records)


def summarize(records: list[dict]) -> dict:
    """Full classification summary for one group of records."""
    matrix = confusion_matrix(records)
    per_class = per_class_prf(matrix)
    macro = macro_metrics(per_class)
    weighted = weighted_metrics(per_class)
    return {
        "count": len(records),
        "accuracy": accuracy(records),
        "macro_precision": macro["precision"],
        "macro_recall": macro["recall"],
        "macro_f1": macro["f1"],
        "weighted_precision": weighted["precision"],
        "weighted_recall": weighted["recall"],
        "weighted_f1": weighted["f1"],
        "per_class": per_class,
        "confusion_matrix": matrix,
        "ucr_prediction_based": unsupported_claim_rate(records, basis="prediction"),
        "ucr_gold_based": unsupported_claim_rate(records, basis="gold"),
        "cr_prediction_based": contradiction_rate(records, basis="prediction"),
        "cr_gold_based": contradiction_rate(records, basis="gold"),
    }


def bootstrap_ci(
    records: list[dict],
    metric_fn,
    n_resamples: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """95% (by default) bootstrap confidence interval for a scalar metric
    computed by metric_fn(records) -> float. Fixed seed for reproducibility.
    Pure standard-library implementation (no numpy/scipy).
    """
    n = len(records)
    if n == 0:
        return {"point_estimate": 0.0, "lower": 0.0, "upper": 0.0, "n_resamples": n_resamples, "seed": seed}

    rng = random.Random(seed)
    resampled_values = []
    for _ in range(n_resamples):
        sample = [records[rng.randrange(n)] for _ in range(n)]
        resampled_values.append(metric_fn(sample))
    resampled_values.sort()

    lower_index = int((alpha / 2) * n_resamples)
    upper_index = min(int((1 - alpha / 2) * n_resamples), n_resamples - 1)

    return {
        "point_estimate": metric_fn(records),
        "lower": resampled_values[lower_index],
        "upper": resampled_values[upper_index],
        "n_resamples": n_resamples,
        "seed": seed,
        "confidence_level": 1 - alpha,
    }
