import uuid

from pydantic import BaseModel


class IdentifySpeakersRequest(BaseModel):
    audio_id: uuid.UUID


class SpeakerRoleOut(BaseModel):
    speaker_label: str
    role: str
    confidence: float


class IdentifySpeakersResponse(BaseModel):
    consultation_id: uuid.UUID
    audio_id: uuid.UUID
    speakers: list[SpeakerRoleOut]
    status: str
