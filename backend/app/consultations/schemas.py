import uuid
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict


class ConsultationStatus(str, Enum):
    draft = "draft"
    active = "active"
    completed = "completed"
    cancelled = "cancelled"


class ConsultationCreate(BaseModel):
    patient_reference: str | None = None


class ConsultationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    patient_reference: str | None
    status: str
    created_at: datetime


class ConsultationListOut(BaseModel):
    items: list[ConsultationOut]


class ConsultationStatusUpdate(BaseModel):
    status: ConsultationStatus
