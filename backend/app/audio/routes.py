import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.audio.schemas import AudioRecordOut
from app.audio.storage import ALLOWED_CONTENT_TYPES, MAX_UPLOAD_BYTES, UPLOAD_ROOT, save_upload_streaming
from app.auth.dependencies import get_current_doctor
from app.consultations.routes import _get_owned_consultation_or_404
from app.database import get_db
from app.models import AudioRecord, Doctor

router = APIRouter(prefix="/consultations", tags=["audio"])


@router.post(
    "/{consultation_id}/audio", response_model=AudioRecordOut, status_code=status.HTTP_201_CREATED
)
async def upload_audio(
    consultation_id: uuid.UUID,
    file: UploadFile = File(...),
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    consultation = _get_owned_consultation_or_404(consultation_id, current_doctor, db)

    if consultation.status != "active":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Audio can only be uploaded to an active consultation",
        )

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported content type '{file.content_type}'",
        )

    extension = ALLOWED_CONTENT_TYPES[file.content_type]
    generated_name = f"{uuid.uuid4().hex}{extension}"
    destination = UPLOAD_ROOT / generated_name
    relative_storage_path = f"uploads/audio/{generated_name}"

    file_size = await save_upload_streaming(file, destination, MAX_UPLOAD_BYTES)

    # The client-supplied filename is never used as a path; only its basename
    # is kept as metadata, preventing any directory-traversal via the name.
    original_filename = Path(file.filename or "recording").name

    audio_record = AudioRecord(
        consultation_id=consultation.id,
        storage_path=relative_storage_path,
        original_filename=original_filename,
        content_type=file.content_type,
        file_size_bytes=file_size,
    )
    db.add(audio_record)
    db.commit()
    db.refresh(audio_record)

    return AudioRecordOut(
        id=audio_record.id,
        consultation_id=audio_record.consultation_id,
        original_filename=audio_record.original_filename,
        content_type=audio_record.content_type,
        file_size=audio_record.file_size_bytes,
        storage_path=audio_record.storage_path,
        created_at=audio_record.created_at,
    )
