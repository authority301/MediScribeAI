import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_doctor
from app.consultations.routes import _get_owned_consultation_or_404
from app.database import get_db
from app.models import Doctor, SOAPNote
from app.nli.schemas import (
    ClaimVerificationOut,
    EvidenceVerificationOut,
    VerifyEvidenceRequest,
    VerifyEvidenceResponse,
)
from app.nli.service import EvidenceVerificationError, run_evidence_verification

router = APIRouter(prefix="/consultations", tags=["nli"])


@router.post(
    "/{consultation_id}/soap/evidence/verify",
    response_model=VerifyEvidenceResponse,
    status_code=status.HTTP_200_OK,
)
def verify_evidence_endpoint(
    consultation_id: uuid.UUID,
    payload: VerifyEvidenceRequest,
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    consultation = _get_owned_consultation_or_404(consultation_id, current_doctor, db)

    if consultation.status != "active":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Evidence verification requires an active consultation",
        )

    soap_note = (
        db.query(SOAPNote)
        .filter(SOAPNote.id == payload.soap_note_id, SOAPNote.consultation_id == consultation.id)
        .first()
    )
    if soap_note is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="SOAP note not found")

    try:
        result = run_evidence_verification(consultation, soap_note, db)
    except EvidenceVerificationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    return VerifyEvidenceResponse(
        consultation_id=consultation.id,
        soap_note_id=soap_note.id,
        claims_processed=result["claims_processed"],
        evidence_links_verified=result["evidence_links_verified"],
        supported=result["supported"],
        contradicted=result["contradicted"],
        ungrounded=result["ungrounded"],
        status="completed",
        per_claim=[
            ClaimVerificationOut(
                claim_id=item["claim_id"],
                claim_text=item["claim_text"],
                section=item["section"],
                verification_status=item["verification_status"],
                evidence=[
                    EvidenceVerificationOut(
                        speaker_segment_id=link.speaker_segment_id,
                        nli_label=link.nli_label,
                        entailment_score=float(link.entailment_score),
                        contradiction_score=float(link.contradiction_score),
                        neutral_score=float(link.neutral_score),
                    )
                    for link in item["links"]
                ],
            )
            for item in result["per_claim"]
        ],
    )
