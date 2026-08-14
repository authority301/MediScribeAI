import uuid

from pydantic import BaseModel


class DiarizeRequest(BaseModel):
    audio_id: uuid.UUID


class DiarizeResponse(BaseModel):
    consultation_id: uuid.UUID
    audio_id: uuid.UUID
    segment_count: int
    speaker_count: int
    speakers: list[str]
    processing_status: str
