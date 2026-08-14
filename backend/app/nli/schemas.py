import uuid

from pydantic import BaseModel


class VerifyEvidenceRequest(BaseModel):
    soap_note_id: uuid.UUID


class EvidenceVerificationOut(BaseModel):
    speaker_segment_id: uuid.UUID
    nli_label: str
    # Raw NLI probabilities -- NOT clinical certainty. See app/nli/model.py.
    entailment_score: float
    contradiction_score: float
    neutral_score: float


class ClaimVerificationOut(BaseModel):
    claim_id: uuid.UUID
    claim_text: str
    section: str
    verification_status: str  # SUPPORTED | CONTRADICTED | UNGROUNDED
    evidence: list[EvidenceVerificationOut]


class VerifyEvidenceResponse(BaseModel):
    consultation_id: uuid.UUID
    soap_note_id: uuid.UUID
    claims_processed: int
    evidence_links_verified: int
    supported: int
    contradicted: int
    ungrounded: int
    status: str
    per_claim: list[ClaimVerificationOut]
