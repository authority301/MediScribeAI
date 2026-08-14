import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_doctor
from app.consultations.routes import _get_owned_consultation_or_404
from app.database import get_db
from app.diarization.schemas import DiarizeRequest, DiarizeResponse
from app.diarization.service import DiarizationError, run_diarization
from app.models import AudioRecord, Doctor

router = APIRouter(prefix="/consultations", tags=["diarization"])


@router.post(
    "/{consultation_id}/audio/diarize",
    response_model=DiarizeResponse,
    status_code=status.HTTP_201_CREATED,
)
def diarize_audio(
    consultation_id: uuid.UUID,
    payload: DiarizeRequest,
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    consultation = _get_owned_consultation_or_404(consultation_id, current_doctor, db)

    if consultation.status != "active":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Audio can only be diarized for an active consultation",
        )

    audio_record = (
        db.query(AudioRecord)
        .filter(AudioRecord.id == payload.audio_id, AudioRecord.consultation_id == consultation.id)
        .first()
    )
    if audio_record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Audio record not found")

    try:
        result = run_diarization(audio_record, db)
    except DiarizationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    return DiarizeResponse(
        consultation_id=consultation.id,
        audio_id=audio_record.id,
        segment_count=result["segment_count"],
        speaker_count=result["speaker_count"],
        speakers=result["speakers"],
        processing_status=audio_record.diarization_status,
    )
