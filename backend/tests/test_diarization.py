"""Diarization (PyAnnote baseline) tests: ownership, status gating, mocked pipeline,
anonymous speaker labels, and deterministic temporal-overlap alignment with ASR.

Automated tests never load the real model -- diarize_file is monkeypatched
wherever diarization actually needs to "succeed" or "fail".
"""
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import or_

from app.database import SessionLocal
from app.database.init_db import init_db
from app.diarization.model import DiarizationTurn
from app.fog.storage import PROCESSED_ROOT
from app.main import app
from app.models import AudioRecord, Consultation, Doctor, SpeakerSegment, Transcript

client = TestClient(app)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture(scope="module", autouse=True)
def _ensure_schema():
    init_db()


def _register_and_login(name="Dr. Diarize"):
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
            audio_record_ids = [a.id for a in audio_records]
            segment_conditions = [SpeakerSegment.audio_record_id.in_(audio_record_ids)]
            if transcript_ids:
                segment_conditions.append(SpeakerSegment.transcript_id.in_(transcript_ids))
            db.query(SpeakerSegment).filter(or_(*segment_conditions)).delete(
                synchronize_session=False
            )
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


def _mark_audio_processed(audio_id: str) -> str:
    """Simulate a completed Fog-processing result without running real ffmpeg."""
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


def _create_transcript_with_segments(audio_id: str, segment_specs):
    """segment_specs: list of (start_ms, end_ms, text). Simulates Step 9A ASR output."""
    db = SessionLocal()
    try:
        audio_record = db.query(AudioRecord).filter(AudioRecord.id == uuid.UUID(audio_id)).first()
        transcript = Transcript(
            consultation_id=audio_record.consultation_id,
            audio_record_id=audio_record.id,
            full_text=" ".join(spec[2] for spec in segment_specs),
            language="en",
            asr_model="faster-whisper-small",
            processing_status="completed",
        )
        db.add(transcript)
        db.commit()
        db.refresh(transcript)
        for index, (start_ms, end_ms, text) in enumerate(segment_specs):
            db.add(
                SpeakerSegment(
                    transcript_id=transcript.id,
                    sequence_index=index,
                    start_time_ms=start_ms,
                    end_time_ms=end_ms,
                    segment_text=text,
                    speaker_label=None,
                )
            )
        db.commit()
        return str(transcript.id)
    finally:
        db.close()


def _fake_diarize_two_speakers(_path):
    return [
        DiarizationTurn(speaker_label="SPEAKER_00", start_ms=0, end_ms=1500),
        DiarizationTurn(speaker_label="SPEAKER_01", start_ms=1500, end_ms=3000),
    ]


@pytest.fixture()
def processed_audio():
    email, token = _register_and_login()
    create_response = client.post(
        "/consultations", json={"patient_reference": "PATIENT-DIARIZE"}, headers=_headers(token)
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


def test_diarize_requires_authentication(processed_audio):
    response = client.post(
        f"/consultations/{processed_audio['consultation_id']}/audio/diarize",
        json={"audio_id": processed_audio["audio_id"]},
    )
    assert response.status_code == 401


def test_diarize_unknown_consultation_returns_404(processed_audio):
    response = client.post(
        f"/consultations/{uuid.uuid4()}/audio/diarize",
        json={"audio_id": processed_audio["audio_id"]},
        headers=_headers(processed_audio["token"]),
    )
    assert response.status_code == 404


def test_diarize_another_doctors_consultation_returns_404(processed_audio):
    other_email, other_token = _register_and_login(name="Dr. Other")
    response = client.post(
        f"/consultations/{processed_audio['consultation_id']}/audio/diarize",
        json={"audio_id": processed_audio["audio_id"]},
        headers=_headers(other_token),
    )
    assert response.status_code == 404
    _cleanup_doctor(other_email)


def test_diarize_draft_consultation_returns_409():
    email, token = _register_and_login(name="Dr. Draft")
    create_response = client.post(
        "/consultations", json={"patient_reference": "PATIENT-DRAFT"}, headers=_headers(token)
    )
    consultation_id = create_response.json()["id"]

    response = client.post(
        f"/consultations/{consultation_id}/audio/diarize",
        json={"audio_id": str(uuid.uuid4())},
        headers=_headers(token),
    )
    assert response.status_code == 409
    _cleanup_doctor(email)


def test_diarize_completed_consultation_returns_409():
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
        f"/consultations/{consultation_id}/audio/diarize",
        json={"audio_id": str(uuid.uuid4())},
        headers=_headers(token),
    )
    assert response.status_code == 409
    _cleanup_doctor(email)


def test_diarize_cancelled_consultation_returns_409():
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
        f"/consultations/{consultation_id}/audio/diarize",
        json={"audio_id": str(uuid.uuid4())},
        headers=_headers(token),
    )
    assert response.status_code == 409
    _cleanup_doctor(email)


def test_diarize_unknown_audio_id_returns_404(processed_audio):
    response = client.post(
        f"/consultations/{processed_audio['consultation_id']}/audio/diarize",
        json={"audio_id": str(uuid.uuid4())},
        headers=_headers(processed_audio["token"]),
    )
    assert response.status_code == 404


def test_diarize_audio_belonging_to_another_consultation_returns_404(processed_audio):
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
        f"/consultations/{second_consultation_id}/audio/diarize",
        json={"audio_id": processed_audio["audio_id"]},
        headers=_headers(token),
    )
    assert response.status_code == 404


def test_diarize_requires_fog_processing_completed():
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
        f"/consultations/{consultation_id}/audio/diarize",
        json={"audio_id": audio_id},
        headers=_headers(token),
    )
    assert response.status_code == 409
    _cleanup_doctor(email)


def test_diarize_missing_processed_file_handled_safely(processed_audio):
    missing_path = REPO_ROOT / processed_audio["processed_storage_path"]
    missing_path.unlink()

    response = client.post(
        f"/consultations/{processed_audio['consultation_id']}/audio/diarize",
        json={"audio_id": processed_audio["audio_id"]},
        headers=_headers(processed_audio["token"]),
    )
    assert response.status_code == 500
    assert "traceback" not in response.text.lower()
    assert str(REPO_ROOT) not in response.text


def test_diarize_succeeds_with_mocked_pipeline_no_transcript(processed_audio, monkeypatch):
    monkeypatch.setattr("app.diarization.service.diarize_file", _fake_diarize_two_speakers)

    response = client.post(
        f"/consultations/{processed_audio['consultation_id']}/audio/diarize",
        json={"audio_id": processed_audio["audio_id"]},
        headers=_headers(processed_audio["token"]),
    )
    assert response.status_code == 201

    body = response.json()
    assert body["consultation_id"] == processed_audio["consultation_id"]
    assert body["audio_id"] == processed_audio["audio_id"]
    assert body["processing_status"] == "completed"
    assert body["speaker_count"] == 2
    assert sorted(body["speakers"]) == ["SPEAKER_00", "SPEAKER_01"]
    assert body["segment_count"] == 2  # two standalone diarization turns, no ASR yet

    # anonymous labels only -- never Doctor/Patient/Nurse
    for label in body["speakers"]:
        assert label.startswith("SPEAKER_")
    assert "Doctor" not in body["speakers"]
    assert "Patient" not in body["speakers"]

    db = SessionLocal()
    try:
        segments = (
            db.query(SpeakerSegment)
            .filter(SpeakerSegment.audio_record_id == uuid.UUID(processed_audio["audio_id"]))
            .order_by(SpeakerSegment.start_time_ms)
            .all()
        )
        assert len(segments) == 2
        for segment in segments:
            assert segment.speaker_label in ("SPEAKER_00", "SPEAKER_01")
            assert segment.speaker_label not in ("Doctor", "Patient", "Nurse")
            assert segment.end_time_ms > segment.start_time_ms  # valid start/end
            assert segment.segment_text is None  # no ASR text to align to
            assert segment.transcript_id is None
    finally:
        db.close()


def test_diarize_aligns_asr_segments_using_temporal_overlap(processed_audio, monkeypatch):
    monkeypatch.setattr("app.diarization.service.diarize_file", _fake_diarize_two_speakers)

    # ASR segment 0 overlaps SPEAKER_00's turn (0-1500ms) almost entirely.
    # ASR segment 1 overlaps SPEAKER_01's turn (1500-3000ms) almost entirely.
    _create_transcript_with_segments(
        processed_audio["audio_id"],
        [
            (0, 1400, "Hello doctor."),
            (1600, 2900, "I have a headache."),
        ],
    )

    response = client.post(
        f"/consultations/{processed_audio['consultation_id']}/audio/diarize",
        json={"audio_id": processed_audio["audio_id"]},
        headers=_headers(processed_audio["token"]),
    )
    assert response.status_code == 201
    body = response.json()
    assert body["segment_count"] == 2  # both turns matched an ASR segment; no orphans
    assert sorted(body["speakers"]) == ["SPEAKER_00", "SPEAKER_01"]

    db = SessionLocal()
    try:
        segments = (
            db.query(SpeakerSegment)
            .join(Transcript, SpeakerSegment.transcript_id == Transcript.id)
            .filter(Transcript.audio_record_id == uuid.UUID(processed_audio["audio_id"]))
            .order_by(SpeakerSegment.sequence_index)
            .all()
        )
        assert len(segments) == 2
        assert segments[0].segment_text == "Hello doctor."
        assert segments[0].speaker_label == "SPEAKER_00"
        assert segments[0].diarization_confidence is not None
        assert segments[1].segment_text == "I have a headache."
        assert segments[1].speaker_label == "SPEAKER_01"
        assert segments[1].diarization_confidence is not None
    finally:
        db.close()


def test_diarize_weak_overlap_does_not_fabricate_speaker(processed_audio, monkeypatch):
    monkeypatch.setattr("app.diarization.service.diarize_file", _fake_diarize_two_speakers)

    # ASR segment entirely outside both diarization turns (turns span 0-3000ms).
    _create_transcript_with_segments(
        processed_audio["audio_id"],
        [(5000, 6000, "Unrelated later speech.")],
    )

    response = client.post(
        f"/consultations/{processed_audio['consultation_id']}/audio/diarize",
        json={"audio_id": processed_audio["audio_id"]},
        headers=_headers(processed_audio["token"]),
    )
    assert response.status_code == 201

    db = SessionLocal()
    try:
        segment = (
            db.query(SpeakerSegment)
            .join(Transcript, SpeakerSegment.transcript_id == Transcript.id)
            .filter(Transcript.audio_record_id == uuid.UUID(processed_audio["audio_id"]))
            .first()
        )
        assert segment.segment_text == "Unrelated later speech."
        assert segment.speaker_label is None  # no overlap -- never fabricated
        assert segment.diarization_confidence is None
    finally:
        db.close()


def test_diarize_failure_sets_failed_status(processed_audio, monkeypatch):
    def _raise(_path):
        raise RuntimeError("simulated model crash")

    monkeypatch.setattr("app.diarization.service.diarize_file", _raise)

    response = client.post(
        f"/consultations/{processed_audio['consultation_id']}/audio/diarize",
        json={"audio_id": processed_audio["audio_id"]},
        headers=_headers(processed_audio["token"]),
    )
    assert response.status_code == 500
    assert "traceback" not in response.text.lower()

    db = SessionLocal()
    try:
        audio_record = (
            db.query(AudioRecord)
            .filter(AudioRecord.id == uuid.UUID(processed_audio["audio_id"]))
            .first()
        )
        assert audio_record.diarization_status == "failed"
        assert audio_record.diarization_error is not None
    finally:
        db.close()
