"""Doctor authentication tests: register, login, and the JWT-protected /auth/me."""
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

from app.database import SessionLocal
from app.database.init_db import init_db
from app.main import app
from app.models import Doctor

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def _ensure_schema():
    init_db()


@pytest.fixture()
def doctor_email():
    email = f"test-{uuid.uuid4()}@example.com"
    yield email
    db = SessionLocal()
    try:
        db.query(Doctor).filter(Doctor.email == email).delete()
        db.commit()
    finally:
        db.close()


def _register(email, password="example-password", name="Dr. Example"):
    return client.post(
        "/auth/register",
        json={"name": name, "email": email, "password": password},
    )


def test_register_success(doctor_email):
    response = _register(doctor_email)
    assert response.status_code == 201

    body = response.json()
    assert body["message"] == "Doctor registered successfully"
    assert body["doctor"]["email"] == doctor_email
    assert body["doctor"]["name"] == "Dr. Example"
    assert "password" not in body["doctor"]
    assert "password_hash" not in body["doctor"]
    assert "password" not in body
    assert "password_hash" not in body


def test_password_is_stored_as_secure_hash_not_plaintext(doctor_email):
    _register(doctor_email, password="example-password")

    db = SessionLocal()
    try:
        doctor = db.query(Doctor).filter(Doctor.email == doctor_email).first()
        assert doctor is not None
        assert doctor.password_hash != "example-password"
        assert doctor.password_hash.startswith("$argon2")
    finally:
        db.close()


def test_duplicate_email_registration_fails(doctor_email):
    first = _register(doctor_email)
    assert first.status_code == 201

    second = _register(doctor_email)
    assert second.status_code == 409


def test_login_with_correct_credentials_returns_jwt(doctor_email):
    _register(doctor_email, password="example-password")

    response = client.post(
        "/auth/login", json={"email": doctor_email, "password": "example-password"}
    )
    assert response.status_code == 200

    body = response.json()
    assert "access_token" in body
    assert body["token_type"] == "bearer"
    assert "password" not in body
    assert "password_hash" not in body


def test_login_with_incorrect_credentials_fails(doctor_email):
    _register(doctor_email, password="example-password")

    response = client.post(
        "/auth/login", json={"email": doctor_email, "password": "wrong-password"}
    )
    assert response.status_code == 401


def test_me_with_valid_jwt_succeeds(doctor_email):
    _register(doctor_email, password="example-password")
    login_response = client.post(
        "/auth/login", json={"email": doctor_email, "password": "example-password"}
    )
    token = login_response.json()["access_token"]

    response = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200

    body = response.json()
    assert body["email"] == doctor_email
    assert body["name"] == "Dr. Example"
    assert "password_hash" not in body


def test_me_without_jwt_returns_401():
    response = client.get("/auth/me")
    assert response.status_code == 401


def test_me_with_invalid_jwt_returns_401():
    response = client.get("/auth/me", headers={"Authorization": "Bearer not-a-real-token"})
    assert response.status_code == 401
