"""Step 14I: controlled three-way evaluation (14B baseline vs. 14F vs. 14H
Tanglish-aware preprocessing). Evaluation-only package, never imported by
app.main or any HTTP route. Reuses the frozen Step 14B model/aggregation/
thresholds and the frozen Step 14C metrics/report-grouping helpers without
modifying either. See runner.py for how each of the three systems is
evaluated, and frozen_14f/ for why a separate snapshot package was needed
to reproduce Step 14F exactly as it behaved before Step 14H's in-place
refinement.
"""
