"""Speaker role identification tests: ownership, status gating, deterministic
heuristic scoring, two-speaker consistency, and idempotency.

This module is pure Python heuristics (no model, no external service), so no
mocking of ASR/diarization/HF is needed -- tests set up preconditions
(processing_status/diarization_status, and speaker_segments rows) directly.
"""
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

from app.database import SessionLocal
from app.database.init_db import init_db
from app.main import app
from app.models import AudioRecord, Consultation, Doctor, SpeakerSegment

client = TestClient(app)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture(scope="module", autouse=True)
def _ensure_schema():
    init_db()


def _register_and_login(name="Dr. Roles"):
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
                if audio_record.storage_path:
                    file_path = REPO_ROOT / audio_record.storage_path
                    if file_path.exists():
                        file_path.unlink()
            audio_record_ids = [a.id for a in audio_records]
            if audio_record_ids:
                db.query(SpeakerSegment).filter(
                    SpeakerSegment.audio_record_id.in_(audio_record_ids)
                ).delete(synchronize_session=False)
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


def _mark_audio_ready_for_role_id(audio_id: str) -> None:
    """Simulate completed Fog processing + diarization without running either."""
    db = SessionLocal()
    try:
        audio_record = db.query(AudioRecord).filter(AudioRecord.id == uuid.UUID(audio_id)).first()
        audio_record.processing_status = "completed"
        audio_record.diarization_status = "completed"
        db.commit()
    finally:
        db.close()


def _create_speaker_segments(audio_id: str, segments_by_speaker: dict) -> None:
    """segments_by_speaker: {speaker_label: [(start_ms, end_ms, text_or_None), ...]}"""
    db = SessionLocal()
    try:
        sequence_index = 0
        for speaker_label, specs in segments_by_speaker.items():
            for start_ms, end_ms, text in specs:
                db.add(
                    SpeakerSegment(
                        audio_record_id=uuid.UUID(audio_id),
                        sequence_index=sequence_index,
                        speaker_label=speaker_label,
                        start_time_ms=start_ms,
                        end_time_ms=end_ms,
                        segment_text=text,
                    )
                )
                sequence_index += 1
        db.commit()
    finally:
        db.close()


@pytest.fixture()
def ready_audio():
    email, token = _register_and_login()
    create_response = client.post(
        "/consultations", json={"patient_reference": "PATIENT-ROLES"}, headers=_headers(token)
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

    yield {"token": token, "consultation_id": consultation_id, "audio_id": audio_id, "email": email}
    _cleanup_doctor(email)


def _identify(consultation_id, audio_id, token):
    return client.post(
        f"/consultations/{consultation_id}/audio/identify-speakers",
        json={"audio_id": audio_id},
        headers=_headers(token),
    )


def test_identify_requires_authentication(ready_audio):
    response = client.post(
        f"/consultations/{ready_audio['consultation_id']}/audio/identify-speakers",
        json={"audio_id": ready_audio["audio_id"]},
    )
    assert response.status_code == 401


def test_identify_unknown_consultation_returns_404(ready_audio):
    response = _identify(str(uuid.uuid4()), ready_audio["audio_id"], ready_audio["token"])
    assert response.status_code == 404


def test_identify_another_doctors_consultation_returns_404(ready_audio):
    other_email, other_token = _register_and_login(name="Dr. Other")
    response = _identify(ready_audio["consultation_id"], ready_audio["audio_id"], other_token)
    assert response.status_code == 404
    _cleanup_doctor(other_email)


def test_identify_draft_consultation_returns_409():
    email, token = _register_and_login(name="Dr. Draft")
    create_response = client.post(
        "/consultations", json={"patient_reference": "PATIENT-DRAFT"}, headers=_headers(token)
    )
    consultation_id = create_response.json()["id"]

    response = _identify(consultation_id, str(uuid.uuid4()), token)
    assert response.status_code == 409
    _cleanup_doctor(email)


def test_identify_completed_consultation_returns_409():
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

    response = _identify(consultation_id, str(uuid.uuid4()), token)
    assert response.status_code == 409
    _cleanup_doctor(email)


def test_identify_cancelled_consultation_returns_409():
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

    response = _identify(consultation_id, str(uuid.uuid4()), token)
    assert response.status_code == 409
    _cleanup_doctor(email)


def test_identify_unknown_audio_id_returns_404(ready_audio):
    response = _identify(ready_audio["consultation_id"], str(uuid.uuid4()), ready_audio["token"])
    assert response.status_code == 404


def test_identify_requires_diarization_completed(ready_audio):
    # processing_status/diarization_status both still "pending" (fixture default)
    response = _identify(
        ready_audio["consultation_id"], ready_audio["audio_id"], ready_audio["token"]
    )
    assert response.status_code == 409


def test_identify_no_speaker_segments_returns_safe_error(ready_audio):
    _mark_audio_ready_for_role_id(ready_audio["audio_id"])
    # no speaker_segments created at all
    response = _identify(
        ready_audio["consultation_id"], ready_audio["audio_id"], ready_audio["token"]
    )
    assert response.status_code == 409
    assert "traceback" not in response.text.lower()


def test_strong_doctor_cues_classified_as_doctor(ready_audio):
    _mark_audio_ready_for_role_id(ready_audio["audio_id"])
    _create_speaker_segments(
        ready_audio["audio_id"],
        {
            "SPEAKER_00": [
                (0, 1500, "What brings you in today?"),
                (1500, 3000, "How long have you had this pain?"),
                (3000, 4000, "Any allergies?"),
            ]
        },
    )

    response = _identify(
        ready_audio["consultation_id"], ready_audio["audio_id"], ready_audio["token"]
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert len(body["speakers"]) == 1
    speaker = body["speakers"][0]
    assert speaker["speaker_label"] == "SPEAKER_00"
    assert speaker["role"] == "DOCTOR"
    assert 0.0 <= speaker["confidence"] <= 1.0


def test_strong_patient_cues_classified_as_patient(ready_audio):
    _mark_audio_ready_for_role_id(ready_audio["audio_id"])
    _create_speaker_segments(
        ready_audio["audio_id"],
        {
            "SPEAKER_00": [
                (0, 1500, "I have been feeling very tired lately."),
                (1500, 3000, "It hurts when I move my arm."),
                (3000, 4000, "I take medication for my blood pressure."),
            ]
        },
    )

    response = _identify(
        ready_audio["consultation_id"], ready_audio["audio_id"], ready_audio["token"]
    )
    assert response.status_code == 200
    speaker = response.json()["speakers"][0]
    assert speaker["role"] == "PATIENT"
    assert 0.0 <= speaker["confidence"] <= 1.0


def test_ambiguous_speaker_remains_unknown(ready_audio):
    _mark_audio_ready_for_role_id(ready_audio["audio_id"])
    _create_speaker_segments(
        ready_audio["audio_id"],
        {"SPEAKER_00": [(0, 500, "Okay."), (500, 1000, "Yes."), (1000, 1500, "Alright then.")]},
    )

    response = _identify(
        ready_audio["consultation_id"], ready_audio["audio_id"], ready_audio["token"]
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "uncertain"  # UNKNOWN is a legitimate outcome, not a failure
    speaker = body["speakers"][0]
    assert speaker["role"] == "UNKNOWN"
    assert 0.0 <= speaker["confidence"] <= 1.0

    db = SessionLocal()
    try:
        segment = (
            db.query(SpeakerSegment)
            .filter(SpeakerSegment.audio_record_id == uuid.UUID(ready_audio["audio_id"]))
            .first()
        )
        assert segment.role_identification_status == "uncertain"
    finally:
        db.close()


def test_role_evidence_is_stored(ready_audio):
    _mark_audio_ready_for_role_id(ready_audio["audio_id"])
    _create_speaker_segments(
        ready_audio["audio_id"],
        {"SPEAKER_00": [(0, 1500, "What brings you in today?")]},
    )

    response = _identify(
        ready_audio["consultation_id"], ready_audio["audio_id"], ready_audio["token"]
    )
    assert response.status_code == 200

    db = SessionLocal()
    try:
        segment = (
            db.query(SpeakerSegment)
            .filter(SpeakerSegment.audio_record_id == uuid.UUID(ready_audio["audio_id"]))
            .first()
        )
        assert segment.inferred_role == "DOCTOR"
        assert segment.role_confidence is not None
        assert 0.0 <= float(segment.role_confidence) <= 1.0
        assert segment.role_identification_status == "completed"
        assert segment.role_evidence is not None
        assert "matched_doctor_cues" in segment.role_evidence
        assert "what brings you in" in segment.role_evidence["matched_doctor_cues"]
        assert "note" in segment.role_evidence
    finally:
        db.close()


def test_two_speaker_consistency_infers_opposite_role_with_zero_evidence(ready_audio):
    _mark_audio_ready_for_role_id(ready_audio["audio_id"])
    _create_speaker_segments(
        ready_audio["audio_id"],
        {
            "SPEAKER_00": [
                (0, 1500, "What brings you in today?"),
                (1500, 3000, "How long have you had this pain?"),
                (3000, 4000, "Any allergies?"),
                (4000, 5000, "Let's check your blood pressure."),
            ],
            # zero text at all -- diarization-only turn, no ASR alignment
            "SPEAKER_01": [(5000, 6000, None)],
        },
    )

    response = _identify(
        ready_audio["consultation_id"], ready_audio["audio_id"], ready_audio["token"]
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"  # both speakers resolved to a confident role
    speakers = {s["speaker_label"]: s for s in body["speakers"]}
    assert speakers["SPEAKER_00"]["role"] == "DOCTOR"
    assert speakers["SPEAKER_00"]["confidence"] >= 0.80
    assert speakers["SPEAKER_01"]["role"] == "PATIENT"  # inferred, not fabricated from text


def test_two_speaker_consistency_does_not_force_weak_evidence(ready_audio):
    _mark_audio_ready_for_role_id(ready_audio["audio_id"])
    _create_speaker_segments(
        ready_audio["audio_id"],
        {
            "SPEAKER_00": [
                (0, 1500, "What brings you in today?"),
                (1500, 3000, "How long have you had this pain?"),
                (3000, 4000, "Any allergies?"),
                (4000, 5000, "Let's check your blood pressure."),
            ],
            # balanced/ambiguous evidence of its own (doctor_score=1.5 from "any
            # pain", patient_score=1.0 from "i have" -> confidence 0.6, below the
            # 0.65 threshold, but nonzero -- must NOT be overridden by the
            # two-speaker fallback, which only applies to literally zero evidence
            "SPEAKER_01": [(5000, 5500, "I have any pain today.")],
        },
    )

    response = _identify(
        ready_audio["consultation_id"], ready_audio["audio_id"], ready_audio["token"]
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "uncertain"  # SPEAKER_01 stayed UNKNOWN
    speakers = {s["speaker_label"]: s for s in body["speakers"]}
    assert speakers["SPEAKER_00"]["role"] == "DOCTOR"
    assert speakers["SPEAKER_01"]["role"] == "UNKNOWN"  # weak evidence never forced


def test_speaker_segment_defaults_to_pending_before_role_identification(ready_audio):
    _mark_audio_ready_for_role_id(ready_audio["audio_id"])
    _create_speaker_segments(
        ready_audio["audio_id"],
        {"SPEAKER_00": [(0, 1500, "What brings you in today?")]},
    )

    db = SessionLocal()
    try:
        segment = (
            db.query(SpeakerSegment)
            .filter(SpeakerSegment.audio_record_id == uuid.UUID(ready_audio["audio_id"]))
            .first()
        )
        assert segment.role_identification_status == "pending"
        assert segment.inferred_role is None
    finally:
        db.close()


def test_unexpected_processing_error_sets_failed_status(ready_audio, monkeypatch):
    _mark_audio_ready_for_role_id(ready_audio["audio_id"])
    _create_speaker_segments(
        ready_audio["audio_id"],
        {"SPEAKER_00": [(0, 1500, "What brings you in today?")]},
    )

    def _raise(*_args, **_kwargs):
        raise RuntimeError("simulated unexpected processing error")

    monkeypatch.setattr("app.speaker_roles.service.classify_speaker", _raise)

    response = _identify(
        ready_audio["consultation_id"], ready_audio["audio_id"], ready_audio["token"]
    )
    assert response.status_code == 500
    assert "traceback" not in response.text.lower()
    assert str(REPO_ROOT) not in response.text

    db = SessionLocal()
    try:
        segment = (
            db.query(SpeakerSegment)
            .filter(SpeakerSegment.audio_record_id == uuid.UUID(ready_audio["audio_id"]))
            .first()
        )
        assert segment.role_identification_status == "failed"
    finally:
        db.close()


def test_repeated_request_does_not_duplicate_speaker_segments(ready_audio):
    _mark_audio_ready_for_role_id(ready_audio["audio_id"])
    _create_speaker_segments(
        ready_audio["audio_id"],
        {"SPEAKER_00": [(0, 1500, "What brings you in today?")]},
    )

    first = _identify(ready_audio["consultation_id"], ready_audio["audio_id"], ready_audio["token"])
    assert first.status_code == 200

    db = SessionLocal()
    try:
        count_after_first = (
            db.query(SpeakerSegment)
            .filter(SpeakerSegment.audio_record_id == uuid.UUID(ready_audio["audio_id"]))
            .count()
        )
    finally:
        db.close()

    second = _identify(
        ready_audio["consultation_id"], ready_audio["audio_id"], ready_audio["token"]
    )
    assert second.status_code == 200
    assert second.json()["speakers"] == first.json()["speakers"]

    db = SessionLocal()
    try:
        count_after_second = (
            db.query(SpeakerSegment)
            .filter(SpeakerSegment.audio_record_id == uuid.UUID(ready_audio["audio_id"]))
            .count()
        )
    finally:
        db.close()

    assert count_after_first == count_after_second == 1
