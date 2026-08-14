import uuid

from pydantic import BaseModel


class ExtractEntitiesRequest(BaseModel):
    transcript_id: uuid.UUID


class EntitySummary(BaseModel):
    entity_type: str
    # For PHI entity types this is always a redaction placeholder (e.g.
    # "[PERSON_NAME]"), never the raw value -- see app/medical_entities/phi.py.
    entity_text: str
    normalized_text: str | None
    negated: bool
    historical: bool


class ExtractEntitiesResponse(BaseModel):
    consultation_id: uuid.UUID
    transcript_id: uuid.UUID
    medical_entity_count: int
    phi_detected: bool
    deidentified_text_available: bool
    processing_status: str
    entities: list[EntitySummary]
