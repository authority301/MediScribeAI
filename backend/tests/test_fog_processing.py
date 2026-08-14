"""Fog local audio-processing tests: ownership, status gating, normalization, safety."""
import io
import struct
import sys
import uuid
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

from app.database import SessionLocal
from app.database.init_db import init_db
from app.main import app
from app.models import AudioRecord, Consultation, Doctor

client = TestClient(app)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture(scope="module", autouse=True)
def _ensure_schema():
    init_db()


def _register_and_login(name="Dr. Fog"):
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
                for path_str in (audio_record.storage_path, audio_record.processed_storage_path):
                    if path_str:
                        file_path = REPO_ROOT / path_str
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


def _generate_wav_bytes(duration_seconds=0.5, sample_rate=44100, channels=2, amplitude=8000):
    """Real, valid stereo/44.1kHz WAV bytes -- exercises actual resampling/downmixing."""
    n_frames = int(duration_seconds * sample_rate)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        frames = bytearray()
        for i in range(n_frames):
            value = int(amplitude * ((i % 100) / 100.0 - 0.5))
            for _ in range(channels):
                frames += struct.pack("<h", value)
        wav_file.writeframes(bytes(frames))
    return buffer.getvalue()


@pytest.fixture()
def active_consultation_with_audio():
    email, token = _register_and_login()
    create_response = client.post(
        "/consultations", json={"patient_reference": "PATIENT-FOG"}, headers=_headers(token)
    )
    consultation_id = create_response.json()["id"]
    client.patch(
        f"/consultations/{consultation_id}/status",
        json={"status": "active"},
        headers=_headers(token),
    )

    audio_bytes = _generate_wav_bytes()
    upload_response = client.post(
        f"/consultations/{consultation_id}/audio",
        files={"file": ("recording.webm", audio_bytes, "audio/webm")},
        headers=_headers(token),
    )
    audio_id = upload_response.json()["id"]
    storage_path = upload_response.json()["storage_path"]

    yield {
        "token": token,
        "consultation_id": consultation_id,
        "audio_id": audio_id,
        "storage_path": storage_path,
        "email": email,
    }
    _cleanup_doctor(email)


def test_process_requires_authentication(active_consultation_with_audio):
    response = client.post(
        f"/consultations/{active_consultation_with_audio['consultation_id']}/audio/process",
        json={"audio_id": active_consultation_with_audio["audio_id"]},
    )
    assert response.status_code == 401


def test_process_unknown_consultation_returns_404(active_consultation_with_audio):
    response = client.post(
        f"/consultations/{uuid.uuid4()}/audio/process",
        json={"audio_id": active_consultation_with_audio["audio_id"]},
        headers=_headers(active_consultation_with_audio["token"]),
    )
    assert response.status_code == 404


def test_process_another_doctors_consultation_returns_404(active_consultation_with_audio):
    other_email, other_token = _register_and_login(name="Dr. Other")
    response = client.post(
        f"/consultations/{active_consultation_with_audio['consultation_id']}/audio/process",
        json={"audio_id": active_consultation_with_audio["audio_id"]},
        headers=_headers(other_token),
    )
    assert response.status_code == 404
    _cleanup_doctor(other_email)


def test_process_draft_consultation_returns_409():
    email, token = _register_and_login(name="Dr. Draft")
    create_response = client.post(
        "/consultations", json={"patient_reference": "PATIENT-DRAFT"}, headers=_headers(token)
    )
    consultation_id = create_response.json()["id"]

    response = client.post(
        f"/consultations/{consultation_id}/audio/process",
        json={"audio_id": str(uuid.uuid4())},
        headers=_headers(token),
    )
    assert response.status_code == 409
    _cleanup_doctor(email)


def test_process_completed_consultation_returns_409():
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
        f"/consultations/{consultation_id}/audio/process",
        json={"audio_id": str(uuid.uuid4())},
        headers=_headers(token),
    )
    assert response.status_code == 409
    _cleanup_doctor(email)


def test_process_cancelled_consultation_returns_409():
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
        f"/consultations/{consultation_id}/audio/process",
        json={"audio_id": str(uuid.uuid4())},
        headers=_headers(token),
    )
    assert response.status_code == 409
    _cleanup_doctor(email)


def test_process_unknown_audio_id_returns_404(active_consultation_with_audio):
    response = client.post(
        f"/consultations/{active_consultation_with_audio['consultation_id']}/audio/process",
        json={"audio_id": str(uuid.uuid4())},
        headers=_headers(active_consultation_with_audio["token"]),
    )
    assert response.status_code == 404


def test_process_audio_belonging_to_another_consultation_returns_404(active_consultation_with_audio):
    token = active_consultation_with_audio["token"]
    create_response = client.post(
        "/consultations", json={"patient_reference": "PATIENT-SECOND"}, headers=_headers(token)
    )
    second_consultation_id = create_response.json()["id"]
    client.patch(
        f"/consultations/{second_consultation_id}/status",
        json={"status": "active"},
        headers=_headers(token),
    )

    # first consultation's audio_id, addressed through the second consultation's URL
    response = client.post(
        f"/consultations/{second_consultation_id}/audio/process",
        json={"audio_id": active_consultation_with_audio["audio_id"]},
        headers=_headers(token),
    )
    assert response.status_code == 404


def test_process_active_consultation_succeeds(active_consultation_with_audio):
    response = client.post(
        f"/consultations/{active_consultation_with_audio['consultation_id']}/audio/process",
        json={"audio_id": active_consultation_with_audio["audio_id"]},
        headers=_headers(active_consultation_with_audio["token"]),
    )
    assert response.status_code == 200

    body = response.json()
    assert body["audio_id"] == active_consultation_with_audio["audio_id"]
    assert body["consultation_id"] == active_consultation_with_audio["consultation_id"]
    assert body["processing_status"] == "completed"
    assert body["processed_storage_path"].startswith("uploads/processed_audio/")
    assert body["processed_storage_path"].endswith(".wav")
    assert body["sample_rate"] == 16000
    assert body["channels"] == 1
    assert body["format"] == "wav"
    assert "created_at" in body

    output_path = REPO_ROOT / body["processed_storage_path"]
    assert output_path.exists()

    with wave.open(str(output_path), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getframerate() == 16000
        assert wav_file.getsampwidth() == 2  # 16-bit PCM

    # original audio remains unchanged (still stereo/44.1kHz as uploaded)
    original_path = REPO_ROOT / active_consultation_with_audio["storage_path"]
    assert original_path.exists()
    with wave.open(str(original_path), "rb") as original_wav:
        assert original_wav.getnchannels() == 2
        assert original_wav.getframerate() == 44100

    db = SessionLocal()
    try:
        record = (
            db.query(AudioRecord)
            .filter(AudioRecord.id == uuid.UUID(active_consultation_with_audio["audio_id"]))
            .first()
        )
        assert record.processing_status == "completed"
        assert record.processed_storage_path == body["processed_storage_path"]
        assert record.processed_sample_rate_hz == 16000
        assert record.processed_channels == 1
        assert record.processed_at is not None
    finally:
        db.close()


def test_process_ignores_client_supplied_path_fields(active_consultation_with_audio):
    response = client.post(
        f"/consultations/{active_consultation_with_audio['consultation_id']}/audio/process",
        json={
            "audio_id": active_consultation_with_audio["audio_id"],
            "path": "../../../etc/passwd",
            "source_path": "/etc/passwd",
        },
        headers=_headers(active_consultation_with_audio["token"]),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["processed_storage_path"].startswith("uploads/processed_audio/")
    assert "../" not in body["processed_storage_path"]


def test_process_corrupt_audio_is_handled_safely():
    email, token = _register_and_login(name="Dr. Corrupt")
    create_response = client.post(
        "/consultations", json={"patient_reference": "PATIENT-CORRUPT"}, headers=_headers(token)
    )
    consultation_id = create_response.json()["id"]
    client.patch(
        f"/consultations/{consultation_id}/status", json={"status": "active"}, headers=_headers(token)
    )

    garbage_bytes = b"this is not audio data at all, just garbage bytes" * 10
    upload_response = client.post(
        f"/consultations/{consultation_id}/audio",
        files={"file": ("garbage.webm", garbage_bytes, "audio/webm")},
        headers=_headers(token),
    )
    audio_id = upload_response.json()["id"]

    response = client.post(
        f"/consultations/{consultation_id}/audio/process",
        json={"audio_id": audio_id},
        headers=_headers(token),
    )
    assert response.status_code in (422, 500)
    assert "traceback" not in response.text.lower()
    assert REPO_ROOT.drive not in response.text if REPO_ROOT.drive else True

    db = SessionLocal()
    try:
        record = db.query(AudioRecord).filter(AudioRecord.id == uuid.UUID(audio_id)).first()
        assert record.processing_status == "failed"
    finally:
        db.close()

    _cleanup_doctor(email)


def test_audio_records_processing_columns_have_no_binary_type():
    column_types = {c.name: type(c.type).__name__ for c in AudioRecord.__table__.columns}
    assert "LargeBinary" not in column_types.values()
    assert column_types["processed_storage_path"] == "Text"
