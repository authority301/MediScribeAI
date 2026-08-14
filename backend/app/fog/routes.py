import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_doctor
from app.consultations.routes import _get_owned_consultation_or_404
from app.database import get_db
from app.fog.schemas import ProcessAudioRequest, ProcessAudioResponse
from app.fog.service import FogProcessingError, process_audio_record
from app.models import AudioRecord, Doctor

router = APIRouter(prefix="/consultations", tags=["fog"])


@router.post("/{consultation_id}/audio/process", response_model=ProcessAudioResponse)
def process_audio(
    consultation_id: uuid.UUID,
    payload: ProcessAudioRequest,
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    consultation = _get_owned_consultation_or_404(consultation_id, current_doctor, db)

    if consultation.status != "active":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Audio can only be processed for an active consultation",
        )

    audio_record = (
        db.query(AudioRecord)
        .filter(AudioRecord.id == payload.audio_id, AudioRecord.consultation_id == consultation.id)
        .first()
    )
    if audio_record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Audio record not found")

    try:
        processed = process_audio_record(audio_record, db)
    except FogProcessingError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    return ProcessAudioResponse(
        audio_id=processed.id,
        consultation_id=processed.consultation_id,
        processing_status=processed.processing_status,
        processed_storage_path=processed.processed_storage_path,
        sample_rate=processed.processed_sample_rate_hz,
        channels=processed.processed_channels,
        format="wav",
        created_at=processed.processed_at,
    )
