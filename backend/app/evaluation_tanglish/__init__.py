"""Step 14G: controlled baseline-vs-Tanglish-aware NLI evaluation.

Evaluation-only package. Never imported by app.main or any HTTP route.
Reuses the frozen Step 14B model/aggregation/thresholds (via
app.tanglish.service, itself a thin wrapper around app.nli) and the frozen
Step 14C metrics/report-grouping helpers (app.evaluation.metrics /
app.evaluation.reports) without modifying either.
"""
