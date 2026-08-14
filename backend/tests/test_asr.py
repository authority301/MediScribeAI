"""ASR (Faster-Whisper baseline) tests: ownership, status gating, mocked transcription.

Automated tests never load the real model -- transcribe_file is monkeypatched
wherever a transcription actually needs to "succeed" or "fail".
"""
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

from app.asr.model import TranscriptionResult, TranscriptionSegment
from app.database import SessionLocal
from app.database.init_db import init_db
from app.fog.storage import PROCESSED_ROOT
from app.main import app
from app.models import AudioRecord, Consultation, Doctor, SpeakerSegment, Transcript

client = TestClient(app)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture(scope="module", autouse=True)
def _ensure_schema():
    init_db()


def _register_and_login(name="Dr. ASR"):
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

            transcripts = (
                db.query(Transcript).filter(Transcript.consultation_id.in_(consultation_ids)).all()
            )
            transcript_ids = [t.id for t in transcripts]
            if transcript_ids:
                db.query(SpeakerSegment).filter(
                    SpeakerSegment.transcript_id.in_(transcript_ids)
                ).delete(synchronize_session=False)
            db.query(Transcript).filter(Transcript.consultation_id.in_(consultation_ids)).delete(
                synchronize_session=False
            )
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


def _fake_transcribe_file(_path):
    return TranscriptionResult(
        language="en",
        segments=[
            TranscriptionSegment(sequence_index=0, text="Hello doctor.", start_ms=0, end_ms=1200),
            TranscriptionSegment(
                sequence_index=1, text="I have a headache.", start_ms=1200, end_ms=2600
            ),
        ],
    )


def _mark_audio_processed(audio_id: str) -> str:
    """Simulate a completed Fog-processing result without running real ffmpeg.

    Fog processing itself is exercised in test_fog_processing.py; here we only
    need audio_records.processing_status == "completed" pointing at a real
    file, since the ASR call itself is mocked in these tests.
    """
    processed_name = f"{uuid.uuid4().hex}.wav"
    processed_path = PROCESSED_ROOT / processed_name
    processed_path.write_bytes(b"placeholder-processed-audio")
    relative_path = f"uploads/processed_audio/{processed_name}"

    db = SessionLocal()
    try:
        audio_record = db.query(AudioRecord).filter(AudioRecord.id == uuid.UUID(audio_id)).first()
        audio_record.processing_status = "completed"
        audio_record.processed_storage_path = relative_path
        audio_record.processed_content_type = "audio/wav"
        audio_record.processed_sample_rate_hz = 16000
        audio_record.processed_channels = 1
        db.commit()
    finally:
        db.close()
    return relative_path


@pytest.fixture()
def processed_audio():
    email, token = _register_and_login()
    create_response = client.post(
        "/consultations", json={"patient_reference": "PATIENT-ASR"}, headers=_headers(token)
    )
    consultation_id = create_response.json()["id"]
    client.patch(
        f"/consultations/{consultation_id}/status",
        json={"status": "active"},
        headers=_headers(token),
    )

    upload_response = client.post(
        f"/consultations/{consultation_id}/audio",
        files={"file": ("recording.webm", b"\x1aE\xdf\xa3fake", "audio/webm")},
        headers=_headers(token),
    )
    audio_id = upload_response.json()["id"]
    processed_storage_path = _mark_audio_processed(audio_id)

    yield {
        "token": token,
        "consultation_id": consultation_id,
        "audio_id": audio_id,
        "processed_storage_path": processed_storage_path,
        "email": email,
    }
    _cleanup_doctor(email)


def test_transcribe_requires_authentication(processed_audio):
    response = client.post(
        f"/consultations/{processed_audio['consultation_id']}/audio/transcribe",
        json={"audio_id": processed_audio["audio_id"]},
    )
    assert response.status_code == 401


def test_transcribe_unknown_consultation_returns_404(processed_audio):
    response = client.post(
        f"/consultations/{uuid.uuid4()}/audio/transcribe",
        json={"audio_id": processed_audio["audio_id"]},
        headers=_headers(processed_audio["token"]),
    )
    assert response.status_code == 404


def test_transcribe_another_doctors_consultation_returns_404(processed_audio):
    other_email, other_token = _register_and_login(name="Dr. Other")
    response = client.post(
        f"/consultations/{processed_audio['consultation_id']}/audio/transcribe",
        json={"audio_id": processed_audio["audio_id"]},
        headers=_headers(other_token),
    )
    assert response.status_code == 404
    _cleanup_doctor(other_email)


def test_transcribe_draft_consultation_returns_409():
    email, token = _register_and_login(name="Dr. Draft")
    create_response = client.post(
        "/consultations", json={"patient_reference": "PATIENT-DRAFT"}, headers=_headers(token)
    )
    consultation_id = create_response.json()["id"]

    response = client.post(
        f"/consultations/{consultation_id}/audio/transcribe",
        json={"audio_id": str(uuid.uuid4())},
        headers=_headers(token),
    )
    assert response.status_code == 409
    _cleanup_doctor(email)


def test_transcribe_completed_consultation_returns_409():
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
        f"/consultations/{consultation_id}/audio/transcribe",
        json={"audio_id": str(uuid.uuid4())},
        headers=_headers(token),
    )
    assert response.status_code == 409
    _cleanup_doctor(email)


def test_transcribe_cancelled_consultation_returns_409():
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
        f"/consultations/{consultation_id}/audio/transcribe",
        json={"audio_id": str(uuid.uuid4())},
        headers=_headers(token),
    )
    assert response.status_code == 409
    _cleanup_doctor(email)


def test_transcribe_unknown_audio_id_returns_404(processed_audio):
    response = client.post(
        f"/consultations/{processed_audio['consultation_id']}/audio/transcribe",
        json={"audio_id": str(uuid.uuid4())},
        headers=_headers(processed_audio["token"]),
    )
    assert response.status_code == 404


def test_transcribe_audio_belonging_to_another_consultation_returns_404(processed_audio):
    token = processed_audio["token"]
    create_response = client.post(
        "/consultations", json={"patient_reference": "PATIENT-SECOND"}, headers=_headers(token)
    )
    second_consultation_id = create_response.json()["id"]
    client.patch(
        f"/consultations/{second_consultation_id}/status",
        json={"status": "active"},
        headers=_headers(token),
    )

    response = client.post(
        f"/consultations/{second_consultation_id}/audio/transcribe",
        json={"audio_id": processed_audio["audio_id"]},
        headers=_headers(token),
    )
    assert response.status_code == 404


def test_transcribe_requires_fog_processing_completed():
    email, token = _register_and_login(name="Dr. Unprocessed")
    create_response = client.post(
        "/consultations", json={"patient_reference": "PATIENT-UNPROCESSED"}, headers=_headers(token)
    )
    consultation_id = create_response.json()["id"]
    client.patch(
        f"/consultations/{consultation_id}/status", json={"status": "active"}, headers=_headers(token)
    )
    upload_response = client.post(
        f"/consultations/{consultation_id}/audio",
        files={"file": ("recording.webm", b"\x1aE\xdf\xa3fake", "audio/webm")},
        headers=_headers(token),
    )
    audio_id = upload_response.json()["id"]  # never Fog-processed; still "pending"

    response = client.post(
        f"/consultations/{consultation_id}/audio/transcribe",
        json={"audio_id": audio_id},
        headers=_headers(token),
    )
    assert response.status_code == 409
    _cleanup_doctor(email)


def test_transcribe_missing_processed_file_handled_safely(processed_audio):
    # simulate the DB saying "completed" while the file is actually gone
    missing_path = REPO_ROOT / processed_audio["processed_storage_path"]
    missing_path.unlink()

    response = client.post(
        f"/consultations/{processed_audio['consultation_id']}/audio/transcribe",
        json={"audio_id": processed_audio["audio_id"]},
        headers=_headers(processed_audio["token"]),
    )
    assert response.status_code == 500
    assert "traceback" not in response.text.lower()
    assert str(REPO_ROOT) not in response.text


def test_transcribe_succeeds_with_mocked_model(processed_audio, monkeypatch):
    monkeypatch.setattr("app.asr.service.transcribe_file", _fake_transcribe_file)

    response = client.post(
        f"/consultations/{processed_audio['consultation_id']}/audio/transcribe",
        json={"audio_id": processed_audio["audio_id"]},
        headers=_headers(processed_audio["token"]),
    )
    assert response.status_code == 201

    body = response.json()
    assert body["consultation_id"] == processed_audio["consultation_id"]
    assert body["audio_id"] == processed_audio["audio_id"]
    assert body["language"] == "en"
    assert body["text"] == "Hello doctor. I have a headache."
    assert body["segment_count"] == 2
    assert body["processing_status"] == "completed"
    assert "transcript_id" in body
    assert "created_at" in body

    db = SessionLocal()
    try:
        transcript = (
            db.query(Transcript).filter(Transcript.id == uuid.UUID(body["transcript_id"])).first()
        )
        assert transcript is not None
        assert transcript.processing_status == "completed"
        assert transcript.language == "en"
        assert transcript.full_text == "Hello doctor. I have a headache."
        assert transcript.asr_model is not None

        segments = (
            db.query(SpeakerSegment)
            .filter(SpeakerSegment.transcript_id == transcript.id)
            .order_by(SpeakerSegment.sequence_index)
            .all()
        )
        assert len(segments) == 2
        assert segments[0].segment_text == "Hello doctor."
        assert segments[0].start_time_ms == 0
        assert segments[0].end_time_ms == 1200
        assert segments[0].speaker_label is None  # no fabricated speaker identity
        assert segments[1].segment_text == "I have a headache."
    finally:
        db.close()


def test_transcribe_failure_sets_failed_status(processed_audio, monkeypatch):
    def _raise(_path):
        raise RuntimeError("simulated model crash")

    monkeypatch.setattr("app.asr.service.transcribe_file", _raise)

    response = client.post(
        f"/consultations/{processed_audio['consultation_id']}/audio/transcribe",
        json={"audio_id": processed_audio["audio_id"]},
        headers=_headers(processed_audio["token"]),
    )
    assert response.status_code == 500
    assert "traceback" not in response.text.lower()

    db = SessionLocal()
    try:
        transcript = (
            db.query(Transcript)
            .filter(Transcript.consultation_id == uuid.UUID(processed_audio["consultation_id"]))
            .first()
        )
        assert transcript is not None
        assert transcript.processing_status == "failed"
        assert transcript.processing_error is not None
    finally:
        db.close()
