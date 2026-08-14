import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_doctor
from app.consultations.schemas import (
    ConsultationCreate,
    ConsultationListOut,
    ConsultationOut,
    ConsultationStatusUpdate,
)
from app.database import get_db
from app.models import Consultation, Doctor

router = APIRouter(prefix="/consultations", tags=["consultations"])

_ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "draft": {"active", "cancelled"},
    "active": {"completed", "cancelled"},
    "completed": set(),
    "cancelled": set(),
}


def _get_owned_consultation_or_404(
    consultation_id: uuid.UUID, doctor: Doctor, db: Session
) -> Consultation:
    consultation = (
        db.query(Consultation)
        .filter(Consultation.id == consultation_id, Consultation.doctor_id == doctor.id)
        .first()
    )
    if consultation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Consultation not found")
    return consultation


@router.post("", response_model=ConsultationOut, status_code=status.HTTP_201_CREATED)
def create_consultation(
    payload: ConsultationCreate,
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    consultation = Consultation(
        doctor_id=current_doctor.id,
        patient_reference=payload.patient_reference,
        consultation_date=datetime.now(timezone.utc),
        status="draft",
    )
    db.add(consultation)
    db.commit()
    db.refresh(consultation)
    return consultation


@router.get("", response_model=ConsultationListOut)
def list_consultations(
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    consultations = (
        db.query(Consultation)
        .filter(Consultation.doctor_id == current_doctor.id)
        .order_by(Consultation.created_at.desc())
        .all()
    )
    return ConsultationListOut(items=consultations)


@router.get("/{consultation_id}", response_model=ConsultationOut)
def get_consultation(
    consultation_id: uuid.UUID,
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    return _get_owned_consultation_or_404(consultation_id, current_doctor, db)


@router.patch("/{consultation_id}/status", response_model=ConsultationOut)
def update_consultation_status(
    consultation_id: uuid.UUID,
    payload: ConsultationStatusUpdate,
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    consultation = _get_owned_consultation_or_404(consultation_id, current_doctor, db)

    new_status = payload.status.value
    allowed_next = _ALLOWED_TRANSITIONS.get(consultation.status, set())
    if new_status not in allowed_next:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Cannot transition consultation from '{consultation.status}' to '{new_status}'",
        )

    consultation.status = new_status
    db.commit()
    db.refresh(consultation)
    return consultation
