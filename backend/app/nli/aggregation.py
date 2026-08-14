"""Deterministic claim-level aggregation of per-evidence NLI results.

============================================================================
RESEARCH INTEGRITY
============================================================================
NLI prediction != medical truth.

SUPPORTED means the retrieved evidence semantically entails the SOAP claim
according to the configured NLI model and thresholds -- NOT that the claim
is clinically correct.

CONTRADICTED means the evidence semantically conflicts with the claim
according to the model -- NOT a clinical determination.

UNGROUNDED means the available evidence did not provide sufficient
entailment or contradiction under this decision rule. It does NOT mean the
claim is false -- only that this retrieval+NLI pipeline could not confirm or
refute it from the available transcript evidence.

These labels are model-derived verification outcomes, not clinical
diagnoses, and not absolute proof of truth or falsehood.

============================================================================
CONFLICTING EVIDENCE
============================================================================
One claim can have multiple evidence links (Step 14A may retrieve several
segments). If some evidence entails the claim while other evidence
contradicts it, this is resolved to UNGROUNDED rather than a naive majority
vote, and rather than inventing a new public label -- this keeps the public
claim-level vocabulary strictly to SUPPORTED / CONTRADICTED / UNGROUNDED, as
required. Individual evidence-level results (each link's own nli_label and
scores) are preserved regardless, so the conflict remains inspectable.

Decision rule (applied per claim, across ALL its evidence-level results):
  qualifying_entailment    = any evidence with label ENTAILMENT
                              and entailment_score    >= NLI_ENTAILMENT_THRESHOLD
  qualifying_contradiction = any evidence with label CONTRADICTION
                              and contradiction_score >= NLI_CONTRADICTION_THRESHOLD

  qualifying_entailment and not qualifying_contradiction  -> SUPPORTED
  qualifying_contradiction and not qualifying_entailment  -> CONTRADICTED
  both, or neither                                        -> UNGROUNDED

NLI_ENTAILMENT_THRESHOLD / NLI_CONTRADICTION_THRESHOLD (default 0.70) are
initial experimental baseline thresholds, not calibrated probabilities.
"""
from dataclasses import dataclass


@dataclass
class EvidenceNliResult:
    speaker_segment_id: object
    nli_label: str  # ENTAILMENT | NEUTRAL | CONTRADICTION
    entailment_score: float
    contradiction_score: float
    neutral_score: float


def aggregate_claim_verification(
    evidence_results: list[EvidenceNliResult],
    entailment_threshold: float,
    contradiction_threshold: float,
) -> str:
    """Return SUPPORTED | CONTRADICTED | UNGROUNDED for one claim, given all
    of its evidence-level NLI results. Deterministic -- see module docstring
    for the exact, conservative decision rule.
    """
    qualifying_entailment = any(
        r.nli_label == "ENTAILMENT" and r.entailment_score >= entailment_threshold
        for r in evidence_results
    )
    qualifying_contradiction = any(
        r.nli_label == "CONTRADICTION" and r.contradiction_score >= contradiction_threshold
        for r in evidence_results
    )

    if qualifying_entailment and not qualifying_contradiction:
        return "SUPPORTED"
    if qualifying_contradiction and not qualifying_entailment:
        return "CONTRADICTED"
    return "UNGROUNDED"
