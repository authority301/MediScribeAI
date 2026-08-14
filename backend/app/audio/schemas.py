import uuid
from datetime import datetime

from pydantic import BaseModel


class AudioRecordOut(BaseModel):
    id: uuid.UUID
    consultation_id: uuid.UUID
    original_filename: str
    content_type: str
    file_size: int
    storage_path: str
    created_at: datetime
