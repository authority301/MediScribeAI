"""Step 14F service layer: exposes BASELINE and TANGLISH_AWARE verification
modes for controlled comparison (the comparison itself is Step 14G's job).

This module reuses the EXISTING, UNMODIFIED Step 14B components directly:
  - app.nli.model.run_nli                              (the real model)
  - app.nli.aggregation.EvidenceNliResult / aggregate_claim_verification
  - app.nli.config thresholds

No NLI model loading, label mapping, or aggregation logic is duplicated
here. The only new responsibility is choosing what text reaches run_nli's
`premise` argument:
  - BASELINE mode:       the original premise, unchanged.
  - TANGLISH_AWARE mode: the premise after app.tanglish.normalizer.normalize().

Nothing in this module is wired into the production
/consultations/{id}/soap/evidence/verify endpoint or into main.py --
Tanglish normalization is never silently enabled for real consultations.
This is an evaluation-only service, callable programmatically (as Step 14G
requires), not exposed over HTTP -- see backend/app/tanglish/README.md for
the rationale.
"""
from app.nli.aggregation import EvidenceNliResult, aggregate_claim_verification
from app.nli.config import NLI_CONTRADICTION_THRESHOLD, NLI_ENTAILMENT_THRESHOLD
from app.nli.model import run_nli
from app.tanglish.config import MODE_BASELINE, MODE_TANGLISH_AWARE, VALID_MODES
from app.tanglish.normalizer import normalize


def verify(premise: str, hypothesis: str, mode: str = MODE_BASELINE) -> dict:
    """Run one premise/hypothesis pair through the real Step 14B NLI model
    and aggregation, optionally preceded by Tanglish normalization.

    Returns a dict (not a DB record -- no evidence_links row is created;
    this never touches the database) with the normalization detail (if any),
    the raw NLI scores, and the aggregated SUPPORTED/CONTRADICTED/UNGROUNDED
    prediction -- using the exact same decision rule as production.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"unknown mode: {mode!r}; expected one of {VALID_MODES}")

    normalization = None
    nli_premise = premise
    if mode == MODE_TANGLISH_AWARE:
        normalization = normalize(premise)
        nli_premise = normalization.normalized_text

    nli_result = run_nli(premise=nli_premise, hypothesis=hypothesis)

    evidence_result = EvidenceNliResult(
        speaker_segment_id=None,
        nli_label=nli_result.label,
        entailment_score=nli_result.entailment,
        contradiction_score=nli_result.contradiction,
        neutral_score=nli_result.neutral,
    )
    prediction = aggregate_claim_verification(
        [evidence_result], NLI_ENTAILMENT_THRESHOLD, NLI_CONTRADICTION_THRESHOLD
    )

    return {
        "mode": mode,
        "premise": premise,
        "nli_premise": nli_premise,
        "hypothesis": hypothesis,
        "normalization": normalization,
        "nli_label": nli_result.label,
        "entailment_score": nli_result.entailment,
        "contradiction_score": nli_result.contradiction,
        "neutral_score": nli_result.neutral,
        "prediction": prediction,
    }


def verify_baseline(premise: str, hypothesis: str) -> dict:
    return verify(premise, hypothesis, mode=MODE_BASELINE)


def verify_tanglish_aware(premise: str, hypothesis: str) -> dict:
    return verify(premise, hypothesis, mode=MODE_TANGLISH_AWARE)
