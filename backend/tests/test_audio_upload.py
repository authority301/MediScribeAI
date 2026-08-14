"""Audio upload tests: ownership, status gating, MIME/size validation, storage safety."""
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

from app.audio.storage import UPLOAD_ROOT
from app.database import SessionLocal
from app.database.init_db import init_db
from app.main import app
from app.models import AudioRecord, Consultation, Doctor

client = TestClient(app)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture(scope="module", autouse=True)
def _ensure_schema():
    init_db()


def _register_and_login(name="Dr. Audio"):
    email = f"test-{uuid.uuid4()}@example.com"
    password = "example-password"
    client.post("/auth/register", json={"name": name, "email": email, "password": password})
    login_response = client.post("/auth/login", json={"email": email, "password": password})
    token = login_response.json()["access_token"]
    return email, token


def _headers(token):
    return {"Authorization": f"Bearer {token}"}


def _cleanup_doctor(email):
    db = SessionLocal()
    try:
        doctor = db.query(Doctor).filter(Doctor.email == email).first()
        if doctor is None:
            return
        consultation_ids = [
            c.id for c in db.query(Consultation).filter(Consultation.doctor_id == doctor.id)
        ]
        if consultation_ids:
            audio_records = (
                db.query(AudioRecord)
                .filter(AudioRecord.consultation_id.in_(consultation_ids))
                .all()
            )
            for audio_record in audio_records:
                file_path = REPO_ROOT / audio_record.storage_path
                if file_path.exists():
                    file_path.unlink()
            db.query(AudioRecord).filter(
                AudioRecord.consultation_id.in_(consultation_ids)
            ).delete(synchronize_session=False)
        db.query(Consultation).filter(Consultation.doctor_id == doctor.id).delete(
            synchronize_session=False
        )
        db.query(Doctor).filter(Doctor.id == doctor.id).delete(synchronize_session=False)
        db.commit()
    finally:
        db.close()


def _webm_bytes(size=1024):
    return b"\x1aE\xdf\xa3" + (b"0" * max(size - 4, 0))


@pytest.fixture()
def active_consultation():
    email, token = _register_and_login()
    create_response = client.post(
        "/consultations", json={"patient_reference": "PATIENT-AUDIO"}, headers=_headers(token)
    )
    consultation_id = create_response.json()["id"]
    client.patch(
        f"/consultations/{consultation_id}/status",
        json={"status": "active"},
        headers=_headers(token),
    )
    yield {"token": token, "consultation_id": consultation_id, "email": email}
    _cleanup_doctor(email)


def test_upload_requires_authentication(active_consultation):
    response = client.post(
        f"/consultations/{active_consultation['consultation_id']}/audio",
        files={"file": ("recording.webm", _webm_bytes(), "audio/webm")},
    )
    assert response.status_code == 401


def test_upload_to_unknown_consultation_returns_404(active_consultation):
    response = client.post(
        f"/consultations/{uuid.uuid4()}/audio",
        files={"file": ("recording.webm", _webm_bytes(), "audio/webm")},
        headers=_headers(active_consultation["token"]),
    )
    assert response.status_code == 404


def test_upload_to_another_doctors_consultation_returns_404(active_consultation):
    other_email, other_token = _register_and_login(name="Dr. Other")
    response = client.post(
        f"/consultations/{active_consultation['consultation_id']}/audio",
        files={"file": ("recording.webm", _webm_bytes(), "audio/webm")},
        headers=_headers(other_token),
    )
    assert response.status_code == 404
    _cleanup_doctor(other_email)


def test_upload_to_draft_consultation_returns_409():
    email, token = _register_and_login(name="Dr. Draft")
    create_response = client.post(
        "/consultations", json={"patient_reference": "PATIENT-DRAFT"}, headers=_headers(token)
    )
    consultation_id = create_response.json()["id"]

    response = client.post(
        f"/consultations/{consultation_id}/audio",
        files={"file": ("recording.webm", _webm_bytes(), "audio/webm")},
        headers=_headers(token),
    )
    assert response.status_code == 409
    _cleanup_doctor(email)


def test_upload_to_completed_consultation_returns_409():
    email, token = _register_and_login(name="Dr. Completed")
    create_response = client.post(
        "/consultations", json={"patient_reference": "PATIENT-COMPLETED"}, headers=_headers(token)
    )
    consultation_id = create_response.json()["id"]
    client.patch(
        f"/consultations/{consultation_id}/status", json={"status": "active"}, headers=_headers(token)
    )
    client.patch(
        f"/consultations/{consultation_id}/status",
        json={"status": "completed"},
        headers=_headers(token),
    )

    response = client.post(
        f"/consultations/{consultation_id}/audio",
        files={"file": ("recording.webm", _webm_bytes(), "audio/webm")},
        headers=_headers(token),
    )
    assert response.status_code == 409
    _cleanup_doctor(email)


def test_upload_to_cancelled_consultation_returns_409():
    email, token = _register_and_login(name="Dr. Cancelled")
    create_response = client.post(
        "/consultations", json={"patient_reference": "PATIENT-CANCELLED"}, headers=_headers(token)
    )
    consultation_id = create_response.json()["id"]
    client.patch(
        f"/consultations/{consultation_id}/status",
        json={"status": "cancelled"},
        headers=_headers(token),
    )

    response = client.post(
        f"/consultations/{consultation_id}/audio",
        files={"file": ("recording.webm", _webm_bytes(), "audio/webm")},
        headers=_headers(token),
    )
    assert response.status_code == 409
    _cleanup_doctor(email)


def test_upload_to_active_consultation_succeeds_and_creates_audio_record(active_consultation):
    audio_bytes = _webm_bytes(2048)
    response = client.post(
        f"/consultations/{active_consultation['consultation_id']}/audio",
        files={"file": ("recording.webm", audio_bytes, "audio/webm")},
        headers=_headers(active_consultation["token"]),
    )
    assert response.status_code == 201

    body = response.json()
    assert body["consultation_id"] == active_consultation["consultation_id"]
    assert body["original_filename"] == "recording.webm"
    assert body["content_type"] == "audio/webm"
    assert body["file_size"] == len(audio_bytes)
    assert body["storage_path"].startswith("uploads/audio/")
    assert body["storage_path"].endswith(".webm")
    assert "id" in body
    assert "created_at" in body

    db = SessionLocal()
    try:
        record = db.query(AudioRecord).filter(AudioRecord.id == uuid.UUID(body["id"])).first()
        assert record is not None
        assert record.storage_path == body["storage_path"]
        assert record.file_size_bytes == len(audio_bytes)
    finally:
        db.close()

    stored_file = REPO_ROOT / body["storage_path"]
    assert stored_file.exists()
    assert stored_file.parent == UPLOAD_ROOT
    assert stored_file.read_bytes() == audio_bytes


def test_upload_unsupported_mime_type_returns_415(active_consultation):
    response = client.post(
        f"/consultations/{active_consultation['consultation_id']}/audio",
        files={"file": ("recording.wav", b"RIFF....WAVEfmt ", "audio/wav")},
        headers=_headers(active_consultation["token"]),
    )
    assert response.status_code == 415


def test_upload_oversized_file_returns_413(active_consultation):
    oversized = b"0" * (26 * 1024 * 1024)
    files_before = set(UPLOAD_ROOT.glob("*"))

    response = client.post(
        f"/consultations/{active_consultation['consultation_id']}/audio",
        files={"file": ("large.webm", oversized, "audio/webm")},
        headers=_headers(active_consultation["token"]),
    )
    assert response.status_code == 413

    files_after = set(UPLOAD_ROOT.glob("*"))
    assert files_after == files_before


def test_upload_filename_cannot_cause_path_traversal(active_consultation):
    audio_bytes = _webm_bytes(512)
    response = client.post(
        f"/consultations/{active_consultation['consultation_id']}/audio",
        files={"file": ("../../../etc/passwd.webm", audio_bytes, "audio/webm")},
        headers=_headers(active_consultation["token"]),
    )
    assert response.status_code == 201

    body = response.json()
    assert "../" not in body["storage_path"]
    assert "/" not in body["original_filename"]
    assert body["original_filename"] == "passwd.webm"

    stored_file = REPO_ROOT / body["storage_path"]
    assert stored_file.resolve().parent == UPLOAD_ROOT.resolve()
    assert stored_file.exists()


def test_audio_records_table_stores_metadata_not_binary():
    column_types = {c.name: type(c.type).__name__ for c in AudioRecord.__table__.columns}
    assert "LargeBinary" not in column_types.values()
    assert column_types["storage_path"] == "Text"
