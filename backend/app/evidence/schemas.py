import uuid

from pydantic import BaseModel


class RetrieveEvidenceRequest(BaseModel):
    soap_note_id: uuid.UUID


class EvidenceCandidateOut(BaseModel):
    speaker_segment_id: uuid.UUID
    # Lexical relevance only -- NOT a truth/confidence judgement. See
    # app/evidence/retrieval.py's module docstring.
    retrieval_score: float
    rank: int


class ClaimEvidenceOut(BaseModel):
    claim_id: uuid.UUID
    evidence: list[EvidenceCandidateOut]


class RetrieveEvidenceResponse(BaseModel):
    consultation_id: uuid.UUID
    soap_note_id: uuid.UUID
    claims_processed: int
    evidence_links_created: int
    status: str
    per_claim: list[ClaimEvidenceOut]
