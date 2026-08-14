import uuid
from datetime import datetime

from pydantic import BaseModel


class TranscribeRequest(BaseModel):
    audio_id: uuid.UUID


class TranscribeResponse(BaseModel):
    transcript_id: uuid.UUID
    consultation_id: uuid.UUID
    audio_id: uuid.UUID
    language: str | None
    text: str
    segment_count: int
    processing_status: str
    created_at: datetime | None
