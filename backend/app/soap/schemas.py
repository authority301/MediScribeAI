import uuid

from pydantic import BaseModel


class GenerateSoapRequest(BaseModel):
    transcript_id: uuid.UUID


class SoapClaimOut(BaseModel):
    section: str
    claim_text: str
    sequence_index: int
    # Generation/evidence-availability confidence, NOT clinical correctness.
    generation_confidence: float | None


class GenerateSoapResponse(BaseModel):
    soap_note_id: uuid.UUID
    consultation_id: uuid.UUID
    transcript_id: uuid.UUID
    status: str
    subjective_claim_count: int
    objective_claim_count: int
    assessment_claim_count: int
    plan_claim_count: int
    claims: list[SoapClaimOut]
