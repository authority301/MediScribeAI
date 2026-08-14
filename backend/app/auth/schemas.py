import uuid

from pydantic import BaseModel, Field


class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str = Field(min_length=8)


class LoginRequest(BaseModel):
    email: str
    password: str


class DoctorPublic(BaseModel):
    id: uuid.UUID
    name: str
    email: str


class RegisterResponse(BaseModel):
    message: str
    doctor: DoctorPublic


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
