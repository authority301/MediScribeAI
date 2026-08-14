import uuid
from datetime import datetime

from pydantic import BaseModel


class ProcessAudioRequest(BaseModel):
    audio_id: uuid.UUID


class ProcessAudioResponse(BaseModel):
    audio_id: uuid.UUID
    consultation_id: uuid.UUID
    processing_status: str
    processed_storage_path: str | None
    sample_rate: int | None
    channels: int | None
    format: str
    created_at: datetime | None
