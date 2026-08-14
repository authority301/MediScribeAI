import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_doctor
from app.consultations.routes import _get_owned_consultation_or_404
from app.database import get_db
from app.models import Doctor, Transcript
from app.soap.generator import ASSESSMENT, OBJECTIVE, PLAN, SUBJECTIVE
from app.soap.schemas import GenerateSoapRequest, GenerateSoapResponse, SoapClaimOut
from app.soap.service import SoapGenerationError, run_soap_generation

router = APIRouter(prefix="/consultations", tags=["soap"])


@router.post(
    "/{consultation_id}/soap/generate",
    response_model=GenerateSoapResponse,
    status_code=status.HTTP_201_CREATED,
)
def generate_soap_note(
    consultation_id: uuid.UUID,
    payload: GenerateSoapRequest,
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    consultation = _get_owned_consultation_or_404(consultation_id, current_doctor, db)

    if consultation.status != "active":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="SOAP notes can only be generated for an active consultation",
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
        result = run_soap_generation(consultation, transcript, db)
    except SoapGenerationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    soap_note = result["soap_note"]
    sections = result["sections"]

    claims_out = [
        SoapClaimOut(
            section=section,
            claim_text=claim.text,
            sequence_index=index,
            generation_confidence=claim.confidence,
        )
        for section in (SUBJECTIVE, OBJECTIVE, ASSESSMENT, PLAN)
        for index, claim in enumerate(sections[section])
    ]

    return GenerateSoapResponse(
        soap_note_id=soap_note.id,
        consultation_id=consultation.id,
        transcript_id=transcript.id,
        status=soap_note.generation_status,
        subjective_claim_count=len(sections[SUBJECTIVE]),
        objective_claim_count=len(sections[OBJECTIVE]),
        assessment_claim_count=len(sections[ASSESSMENT]),
        plan_claim_count=len(sections[PLAN]),
        claims=claims_out,
    )
