import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_doctor
from app.consultations.routes import _get_owned_consultation_or_404
from app.database import get_db
from app.medical_entities.schemas import ExtractEntitiesRequest, ExtractEntitiesResponse
from app.medical_entities.service import EntityExtractionError, run_entity_extraction
from app.models import Doctor, Transcript

router = APIRouter(prefix="/consultations", tags=["medical-entities"])


@router.post(
    "/{consultation_id}/audio/extract-entities",
    response_model=ExtractEntitiesResponse,
    status_code=status.HTTP_200_OK,
)
def extract_entities_endpoint(
    consultation_id: uuid.UUID,
    payload: ExtractEntitiesRequest,
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    consultation = _get_owned_consultation_or_404(consultation_id, current_doctor, db)

    if consultation.status != "active":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Entities can only be extracted for an active consultation",
        )

    transcript = (
        db.query(Transcript)
        .filter(
            Transcript.id == payload.transcript_id, Transcript.consultation_id == consultation.id
        )
        .first()
    )
    if transcript is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transcript not found")

    try:
        result = run_entity_extraction(transcript, db)
    except EntityExtractionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    return ExtractEntitiesResponse(
        consultation_id=consultation.id,
        transcript_id=transcript.id,
        medical_entity_count=result["medical_entity_count"],
        phi_detected=result["phi_detected"],
        deidentified_text_available=result["deidentified_text_available"],
        processing_status=transcript.entity_extraction_status,
        entities=result["entities"],
    )
