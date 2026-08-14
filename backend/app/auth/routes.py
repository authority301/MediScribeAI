from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_doctor
from app.auth.jwt import create_access_token
from app.auth.schemas import (
    DoctorPublic,
    LoginRequest,
    RegisterRequest,
    RegisterResponse,
    TokenResponse,
)
from app.auth.security import hash_password, verify_password
from app.database import get_db
from app.models import Doctor

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=RegisterResponse, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    existing = db.query(Doctor).filter(Doctor.email == payload.email).first()
    if existing is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")

    doctor = Doctor(
        full_name=payload.name,
        email=payload.email,
        password_hash=hash_password(payload.password),
    )
    db.add(doctor)
    db.commit()
    db.refresh(doctor)

    return RegisterResponse(
        message="Doctor registered successfully",
        doctor=DoctorPublic(id=doctor.id, name=doctor.full_name, email=doctor.email),
    )


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    doctor = db.query(Doctor).filter(Doctor.email == payload.email).first()
    if doctor is None or not verify_password(payload.password, doctor.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")

    access_token = create_access_token(subject=str(doctor.id))
    return TokenResponse(access_token=access_token)


@router.get("/me", response_model=DoctorPublic)
def read_current_doctor(current_doctor: Doctor = Depends(get_current_doctor)):
    return DoctorPublic(id=current_doctor.id, name=current_doctor.full_name, email=current_doctor.email)
