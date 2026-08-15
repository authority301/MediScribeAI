"""Frozen, read-only snapshot of `backend/app/tanglish/` as it existed at
commit 5563418 ("Add Tanglish-aware NLI evaluation" -- the Step 14G
checkpoint, i.e. the Step 14F implementation BEFORE the Step 14H numeric/
attribution refinement changed `app/tanglish/normalizer.py`,
`lexicon.py`, and `schemas.py` in place).

WHY THIS EXISTS: Step 14H refined the Tanglish preprocessing layer
in-place -- there is no separately-maintained "Step 14F" package in the
current codebase to import. Step 14I needs to run the EXACT Step 14F
behavior (System B) as one arm of a three-way comparison against Step 14B
(System A, unaffected -- baseline mode never calls the normalizer at all)
and current Step 14H (System C, `app.tanglish.normalizer`). This package
is the only way to do that without checking out a different git commit
(which would disturb the working tree) or modifying `app/tanglish/` itself
(which Step 14H's instructions and this step's instructions both forbid).

`config.py`, `lexicon.py`, and `schemas.py` here are BYTE-IDENTICAL to
their commit-5563418 versions (verified via `git show 5563418:...` --
Step 14H's own diff never touched `config.py`, and this snapshot was taken
before comparing against the current `lexicon.py`/`schemas.py`).
`normalizer.py` is identical in logic; only its `from app.tanglish...`
imports were rewritten to `from app.evaluation_tanglish_refinement.
frozen_14f...` so it resolves against this snapshot instead of the current
(Step 14H) package.

NEVER imported by production code, by `app/tanglish/` itself, or by any
other evaluation step. Evaluation-only, exactly like the rest of
`app/evaluation_tanglish_refinement/`.
"""
