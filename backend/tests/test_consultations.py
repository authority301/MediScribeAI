"""Consultation management tests: create, list, retrieve, ownership, status lifecycle."""
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

from app.database import SessionLocal
from app.database.init_db import init_db
from app.main import app
from app.models import Consultation, Doctor

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def _ensure_schema():
    init_db()


def _register_and_login(name="Dr. Consult"):
    email = f"test-{uuid.uuid4()}@example.com"
    password = "example-password"
    client.post("/auth/register", json={"name": name, "email": email, "password": password})
    login_response = client.post("/auth/login", json={"email": email, "password": password})
    token = login_response.json()["access_token"]
    return email, token


def _cleanup_doctor(email):
    db = SessionLocal()
    try:
        doctor = db.query(Doctor).filter(Doctor.email == email).first()
        if doctor is not None:
            db.query(Consultation).filter(Consultation.doctor_id == doctor.id).delete()
            db.query(Doctor).filter(Doctor.id == doctor.id).delete()
            db.commit()
    finally:
        db.close()


def _headers(token):
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def auth_doctor():
    email, token = _register_and_login()
    yield {"email": email, "token": token}
    _cleanup_doctor(email)


def test_create_consultation(auth_doctor):
    response = client.post(
        "/consultations",
        json={"patient_reference": "PATIENT-001"},
        headers=_headers(auth_doctor["token"]),
    )
    assert response.status_code == 201

    body = response.json()
    assert body["patient_reference"] == "PATIENT-001"
    assert body["status"] == "draft"
    assert "id" in body
    assert "created_at" in body
    assert "doctor_id" not in body


def test_create_consultation_ignores_client_supplied_doctor_id(auth_doctor):
    spoofed_id = str(uuid.uuid4())
    response = client.post(
        "/consultations",
        json={"patient_reference": "PATIENT-002", "doctor_id": spoofed_id},
        headers=_headers(auth_doctor["token"]),
    )
    assert response.status_code == 201

    list_response = client.get("/consultations", headers=_headers(auth_doctor["token"]))
    ids = [item["id"] for item in list_response.json()["items"]]
    assert response.json()["id"] in ids


def test_list_consultations_returns_only_own(auth_doctor):
    client.post(
        "/consultations",
        json={"patient_reference": "PATIENT-003"},
        headers=_headers(auth_doctor["token"]),
    )
    response = client.get("/consultations", headers=_headers(auth_doctor["token"]))
    assert response.status_code == 200
    body = response.json()
    assert len(body["items"]) >= 1
    for item in body["items"]:
        assert set(item.keys()) == {"id", "patient_reference", "status", "created_at"}


def test_list_consultations_empty_for_new_doctor():
    email, token = _register_and_login(name="Dr. Empty")
    response = client.get("/consultations", headers=_headers(token))
    assert response.status_code == 200
    assert response.json() == {"items": []}
    _cleanup_doctor(email)


def test_get_consultation_by_id(auth_doctor):
    create_response = client.post(
        "/consultations",
        json={"patient_reference": "PATIENT-004"},
        headers=_headers(auth_doctor["token"]),
    )
    consultation_id = create_response.json()["id"]

    response = client.get(f"/consultations/{consultation_id}", headers=_headers(auth_doctor["token"]))
    assert response.status_code == 200
    assert response.json()["id"] == consultation_id


def test_get_nonexistent_consultation_returns_404(auth_doctor):
    response = client.get(f"/consultations/{uuid.uuid4()}", headers=_headers(auth_doctor["token"]))
    assert response.status_code == 404


def test_cannot_view_another_doctors_consultation(auth_doctor):
    create_response = client.post(
        "/consultations",
        json={"patient_reference": "PATIENT-005"},
        headers=_headers(auth_doctor["token"]),
    )
    consultation_id = create_response.json()["id"]

    other_email, other_token = _register_and_login(name="Dr. Other")
    response = client.get(f"/consultations/{consultation_id}", headers=_headers(other_token))
    assert response.status_code == 404
    _cleanup_doctor(other_email)


def test_status_transition_draft_to_active(auth_doctor):
    create_response = client.post(
        "/consultations",
        json={"patient_reference": "PATIENT-006"},
        headers=_headers(auth_doctor["token"]),
    )
    consultation_id = create_response.json()["id"]

    response = client.patch(
        f"/consultations/{consultation_id}/status",
        json={"status": "active"},
        headers=_headers(auth_doctor["token"]),
    )
    assert response.status_code == 200
    assert response.json()["status"] == "active"


def test_status_transition_active_to_completed(auth_doctor):
    create_response = client.post(
        "/consultations",
        json={"patient_reference": "PATIENT-007"},
        headers=_headers(auth_doctor["token"]),
    )
    consultation_id = create_response.json()["id"]

    client.patch(
        f"/consultations/{consultation_id}/status",
        json={"status": "active"},
        headers=_headers(auth_doctor["token"]),
    )
    response = client.patch(
        f"/consultations/{consultation_id}/status",
        json={"status": "completed"},
        headers=_headers(auth_doctor["token"]),
    )
    assert response.status_code == 200
    assert response.json()["status"] == "completed"


def test_invalid_status_transition_rejected(auth_doctor):
    create_response = client.post(
        "/consultations",
        json={"patient_reference": "PATIENT-008"},
        headers=_headers(auth_doctor["token"]),
    )
    consultation_id = create_response.json()["id"]

    response = client.patch(
        f"/consultations/{consultation_id}/status",
        json={"status": "completed"},
        headers=_headers(auth_doctor["token"]),
    )
    assert response.status_code == 409


def test_transition_from_terminal_status_rejected(auth_doctor):
    create_response = client.post(
        "/consultations",
        json={"patient_reference": "PATIENT-009"},
        headers=_headers(auth_doctor["token"]),
    )
    consultation_id = create_response.json()["id"]

    client.patch(
        f"/consultations/{consultation_id}/status",
        json={"status": "cancelled"},
        headers=_headers(auth_doctor["token"]),
    )
    response = client.patch(
        f"/consultations/{consultation_id}/status",
        json={"status": "active"},
        headers=_headers(auth_doctor["token"]),
    )
    assert response.status_code == 409


@pytest.mark.parametrize(
    "reached_status, attempted_status",
    [
        ("completed", "active"),
        ("completed", "cancelled"),
        ("cancelled", "active"),
        ("cancelled", "completed"),
    ],
)
def test_invalid_transitions_from_terminal_states_return_409(
    auth_doctor, reached_status, attempted_status
):
    create_response = client.post(
        "/consultations",
        json={"patient_reference": "PATIENT-TERMINAL"},
        headers=_headers(auth_doctor["token"]),
    )
    consultation_id = create_response.json()["id"]

    if reached_status == "completed":
        client.patch(
            f"/consultations/{consultation_id}/status",
            json={"status": "active"},
            headers=_headers(auth_doctor["token"]),
        )
    client.patch(
        f"/consultations/{consultation_id}/status",
        json={"status": reached_status},
        headers=_headers(auth_doctor["token"]),
    )

    response = client.patch(
        f"/consultations/{consultation_id}/status",
        json={"status": attempted_status},
        headers=_headers(auth_doctor["token"]),
    )
    assert response.status_code == 409


def test_invalid_status_value_returns_422(auth_doctor):
    create_response = client.post(
        "/consultations",
        json={"patient_reference": "PATIENT-012"},
        headers=_headers(auth_doctor["token"]),
    )
    consultation_id = create_response.json()["id"]

    response = client.patch(
        f"/consultations/{consultation_id}/status",
        json={"status": "unknown"},
        headers=_headers(auth_doctor["token"]),
    )
    assert response.status_code == 422


def test_cannot_update_another_doctors_consultation_status(auth_doctor):
    create_response = client.post(
        "/consultations",
        json={"patient_reference": "PATIENT-010"},
        headers=_headers(auth_doctor["token"]),
    )
    consultation_id = create_response.json()["id"]

    other_email, other_token = _register_and_login(name="Dr. Other2")
    response = client.patch(
        f"/consultations/{consultation_id}/status",
        json={"status": "active"},
        headers=_headers(other_token),
    )
    assert response.status_code == 404
    _cleanup_doctor(other_email)


def test_consultations_require_authentication():
    response = client.post("/consultations", json={"patient_reference": "PATIENT-011"})
    assert response.status_code == 401

    response = client.get("/consultations")
    assert response.status_code == 401
