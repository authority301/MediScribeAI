import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.asr.schemas import TranscribeRequest, TranscribeResponse
from app.asr.service import AsrError, run_transcription
from app.auth.dependencies import get_current_doctor
from app.consultations.routes import _get_owned_consultation_or_404
from app.database import get_db
from app.models import AudioRecord, Doctor

router = APIRouter(prefix="/consultations", tags=["asr"])


@router.post(
    "/{consultation_id}/audio/transcribe",
    response_model=TranscribeResponse,
    status_code=status.HTTP_201_CREATED,
)
def transcribe_audio(
    consultation_id: uuid.UUID,
    payload: TranscribeRequest,
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    consultation = _get_owned_consultation_or_404(consultation_id, current_doctor, db)

    if consultation.status != "active":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Audio can only be transcribed for an active consultation",
        )

    audio_record = (
        db.query(AudioRecord)
        .filter(AudioRecord.id == payload.audio_id, AudioRecord.consultation_id == consultation.id)
        .first()
    )
    if audio_record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Audio record not found")

    try:
        transcript = run_transcription(audio_record, db)
    except AsrError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    return TranscribeResponse(
        transcript_id=transcript.id,
        consultation_id=transcript.consultation_id,
        audio_id=audio_record.id,
        language=transcript.language,
        text=transcript.full_text or "",
        segment_count=len(transcript.speaker_segments),
        processing_status=transcript.processing_status,
        created_at=transcript.created_at,
    )
