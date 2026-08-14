import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_doctor
from app.consultations.routes import _get_owned_consultation_or_404
from app.database import get_db
from app.models import AudioRecord, Doctor
from app.speaker_roles.schemas import (
    IdentifySpeakersRequest,
    IdentifySpeakersResponse,
    SpeakerRoleOut,
)
from app.speaker_roles.service import RoleIdentificationError, run_role_identification

router = APIRouter(prefix="/consultations", tags=["speaker-roles"])


@router.post(
    "/{consultation_id}/audio/identify-speakers",
    response_model=IdentifySpeakersResponse,
    status_code=status.HTTP_200_OK,
)
def identify_speakers(
    consultation_id: uuid.UUID,
    payload: IdentifySpeakersRequest,
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    consultation = _get_owned_consultation_or_404(consultation_id, current_doctor, db)

    if consultation.status != "active":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Speaker roles can only be identified for an active consultation",
        )

    audio_record = (
        db.query(AudioRecord)
        .filter(AudioRecord.id == payload.audio_id, AudioRecord.consultation_id == consultation.id)
        .first()
    )
    if audio_record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Audio record not found")

    try:
        classifications, overall_status = run_role_identification(audio_record, db)
    except RoleIdentificationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    return IdentifySpeakersResponse(
        consultation_id=consultation.id,
        audio_id=audio_record.id,
        speakers=[
            SpeakerRoleOut(speaker_label=c.speaker_label, role=c.role, confidence=c.confidence)
            for c in classifications
        ],
        status=overall_status,
    )
