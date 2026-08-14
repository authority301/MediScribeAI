import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_doctor
from app.consultations.routes import _get_owned_consultation_or_404
from app.database import get_db
from app.evidence.schemas import (
    ClaimEvidenceOut,
    EvidenceCandidateOut,
    RetrieveEvidenceRequest,
    RetrieveEvidenceResponse,
)
from app.evidence.service import EvidenceRetrievalError, run_evidence_retrieval
from app.models import Doctor, SOAPNote

router = APIRouter(prefix="/consultations", tags=["evidence"])


@router.post(
    "/{consultation_id}/soap/evidence/retrieve",
    response_model=RetrieveEvidenceResponse,
    status_code=status.HTTP_200_OK,
)
def retrieve_evidence_endpoint(
    consultation_id: uuid.UUID,
    payload: RetrieveEvidenceRequest,
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    consultation = _get_owned_consultation_or_404(consultation_id, current_doctor, db)

    if consultation.status != "active":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Evidence retrieval requires an active consultation",
        )

    soap_note = (
        db.query(SOAPNote)
        .filter(SOAPNote.id == payload.soap_note_id, SOAPNote.consultation_id == consultation.id)
        .first()
    )
    if soap_note is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="SOAP note not found")

    try:
        result = run_evidence_retrieval(consultation, soap_note, db)
    except EvidenceRetrievalError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    return RetrieveEvidenceResponse(
        consultation_id=consultation.id,
        soap_note_id=soap_note.id,
        claims_processed=result["claims_processed"],
        evidence_links_created=result["evidence_links_created"],
        status="completed",
        per_claim=[
            ClaimEvidenceOut(
                claim_id=item["claim_id"],
                evidence=[
                    EvidenceCandidateOut(
                        speaker_segment_id=c.speaker_segment_id,
                        retrieval_score=c.retrieval_score,
                        rank=c.rank,
                    )
                    for c in item["evidence"]
                ],
            )
            for item in result["per_claim"]
        ],
    )
