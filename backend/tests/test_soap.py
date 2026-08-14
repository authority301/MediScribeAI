"""SOAP note generation tests: ownership, status gating, deterministic
claim generation, evidence-only content, negation/historical handling,
idempotency, and safe failure.

Uses the REAL Step 12 extraction logic (extract_entities) to build medical
entities from synthetic segment text, so these tests exercise the actual
extraction -> generation pipeline rather than hand-crafted entity rows. All
transcript text is synthetic, never real patient data.
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
from app.medical_entities.extraction import extract_entities
from app.models import Consultation, Doctor, MedicalEntity, SOAPClaim, SOAPNote, SpeakerSegment, Transcript

client = TestClient(app)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Matches the manual-verification conversation from the task spec.
SAMPLE_CONVERSATION = [
    ("DOCTOR", "How are you feeling today?"),
    ("PATIENT", "I have had fever and cough for two days."),
    ("DOCTOR", "Your temperature is 101.2 degrees Fahrenheit."),
    ("DOCTOR", "This appears to be a viral infection."),
    ("DOCTOR", "Take paracetamol and return in three days."),
]


@pytest.fixture(scope="module", autouse=True)
def _ensure_schema():
    init_db()


def _register_and_login(name="Dr. Soap"):
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
            soap_note_ids = [
                n.id for n in db.query(SOAPNote).filter(SOAPNote.consultation_id.in_(consultation_ids))
            ]
            if soap_note_ids:
                db.query(SOAPClaim).filter(SOAPClaim.soap_note_id.in_(soap_note_ids)).delete(
                    synchronize_session=False
                )
            db.query(SOAPNote).filter(SOAPNote.consultation_id.in_(consultation_ids)).delete(
                synchronize_session=False
            )
            transcript_ids = [
                t.id for t in db.query(Transcript).filter(Transcript.consultation_id.in_(consultation_ids))
            ]
            if transcript_ids:
                db.query(MedicalEntity).filter(MedicalEntity.transcript_id.in_(transcript_ids)).delete(
                    synchronize_session=False
                )
                db.query(SpeakerSegment).filter(SpeakerSegment.transcript_id.in_(transcript_ids)).delete(
                    synchronize_session=False
                )
            db.query(Transcript).filter(Transcript.consultation_id.in_(consultation_ids)).delete(
                synchronize_session=False
            )
        db.query(Consultation).filter(Consultation.doctor_id == doctor.id).delete(
            synchronize_session=False
        )
        db.query(Doctor).filter(Doctor.id == doctor.id).delete(synchronize_session=False)
        db.commit()
    finally:
        db.close()


def _create_transcript_with_conversation(
    consultation_id: str, turns: list, entity_extraction_status: str = "completed"
) -> str:
    """turns: list of (role, text) with role in ('DOCTOR', 'PATIENT').
    Runs the REAL Step 12 extractor on each turn's text.
    """
    db = SessionLocal()
    try:
        full_text = " ".join(text for _role, text in turns)
        transcript = Transcript(
            consultation_id=uuid.UUID(consultation_id),
            full_text=full_text,
            language="en",
            asr_model="faster-whisper-small",
            processing_status="completed",
            entity_extraction_status=entity_extraction_status,
        )
        db.add(transcript)
        db.commit()
        db.refresh(transcript)

        cumulative_ms = 0
        for index, (role, text) in enumerate(turns):
            segment = SpeakerSegment(
                transcript_id=transcript.id,
                sequence_index=index,
                speaker_label="SPEAKER_00" if role == "DOCTOR" else "SPEAKER_01",
                inferred_role=role,
                start_time_ms=cumulative_ms,
                end_time_ms=cumulative_ms + 1000,
                segment_text=text,
            )
            db.add(segment)
            db.commit()
            db.refresh(segment)
            cumulative_ms += 1000

            for entity in extract_entities(text):
                db.add(
                    MedicalEntity(
                        consultation_id=uuid.UUID(consultation_id),
                        transcript_id=transcript.id,
                        speaker_segment_id=segment.id,
                        entity_type=entity.entity_type,
                        entity_text=entity.entity_text,
                        normalized_value=entity.normalized_text,
                        start_char=entity.start_offset,
                        end_char=entity.end_offset,
                        confidence_score=entity.confidence,
                        negated=entity.negated,
                        historical=entity.historical,
                    )
                )
            db.commit()

        return str(transcript.id)
    finally:
        db.close()


@pytest.fixture()
def active_consultation():
    email, token = _register_and_login()
    create_response = client.post(
        "/consultations", json={"patient_reference": "PATIENT-SOAP"}, headers=_headers(token)
    )
    consultation_id = create_response.json()["id"]
    client.patch(
        f"/consultations/{consultation_id}/status",
        json={"status": "active"},
        headers=_headers(token),
    )
    yield {"token": token, "consultation_id": consultation_id, "email": email}
    _cleanup_doctor(email)


def _generate(consultation_id, transcript_id, token):
    return client.post(
        f"/consultations/{consultation_id}/soap/generate",
        json={"transcript_id": transcript_id},
        headers=_headers(token),
    )


def test_generate_requires_authentication(active_consultation):
    transcript_id = _create_transcript_with_conversation(
        active_consultation["consultation_id"], SAMPLE_CONVERSATION
    )
    response = client.post(
        f"/consultations/{active_consultation['consultation_id']}/soap/generate",
        json={"transcript_id": transcript_id},
    )
    assert response.status_code == 401


def test_generate_unknown_consultation_returns_404(active_consultation):
    transcript_id = _create_transcript_with_conversation(
        active_consultation["consultation_id"], SAMPLE_CONVERSATION
    )
    response = _generate(str(uuid.uuid4()), transcript_id, active_consultation["token"])
    assert response.status_code == 404


def test_generate_another_doctors_consultation_returns_404(active_consultation):
    transcript_id = _create_transcript_with_conversation(
        active_consultation["consultation_id"], SAMPLE_CONVERSATION
    )
    other_email, other_token = _register_and_login(name="Dr. Other")
    response = _generate(active_consultation["consultation_id"], transcript_id, other_token)
    assert response.status_code == 404
    _cleanup_doctor(other_email)


def test_generate_draft_consultation_returns_409():
    email, token = _register_and_login(name="Dr. Draft")
    create_response = client.post(
        "/consultations", json={"patient_reference": "PATIENT-DRAFT"}, headers=_headers(token)
    )
    consultation_id = create_response.json()["id"]

    response = _generate(consultation_id, str(uuid.uuid4()), token)
    assert response.status_code == 409
    _cleanup_doctor(email)


def test_generate_completed_consultation_returns_409():
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

    response = _generate(consultation_id, str(uuid.uuid4()), token)
    assert response.status_code == 409
    _cleanup_doctor(email)


def test_generate_cancelled_consultation_returns_409():
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

    response = _generate(consultation_id, str(uuid.uuid4()), token)
    assert response.status_code == 409
    _cleanup_doctor(email)


def test_generate_unknown_transcript_returns_404(active_consultation):
    response = _generate(
        active_consultation["consultation_id"], str(uuid.uuid4()), active_consultation["token"]
    )
    assert response.status_code == 404


def test_generate_transcript_belonging_to_another_consultation_returns_404(active_consultation):
    token = active_consultation["token"]
    create_response = client.post(
        "/consultations", json={"patient_reference": "PATIENT-SECOND"}, headers=_headers(token)
    )
    second_consultation_id = create_response.json()["id"]
    client.patch(
        f"/consultations/{second_consultation_id}/status",
        json={"status": "active"},
        headers=_headers(token),
    )
    transcript_id = _create_transcript_with_conversation(
        active_consultation["consultation_id"], SAMPLE_CONVERSATION
    )

    response = _generate(second_consultation_id, transcript_id, token)
    assert response.status_code == 404


def test_generate_transcript_not_completed_returns_409(active_consultation):
    db = SessionLocal()
    try:
        transcript = Transcript(
            consultation_id=uuid.UUID(active_consultation["consultation_id"]),
            processing_status="pending",
            entity_extraction_status="pending",
        )
        db.add(transcript)
        db.commit()
        transcript_id = str(transcript.id)
    finally:
        db.close()

    response = _generate(
        active_consultation["consultation_id"], transcript_id, active_consultation["token"]
    )
    assert response.status_code == 409


def test_generate_entity_extraction_not_completed_returns_409(active_consultation):
    transcript_id = _create_transcript_with_conversation(
        active_consultation["consultation_id"], SAMPLE_CONVERSATION, entity_extraction_status="pending"
    )
    response = _generate(
        active_consultation["consultation_id"], transcript_id, active_consultation["token"]
    )
    assert response.status_code == 409


def test_successful_soap_generation(active_consultation):
    transcript_id = _create_transcript_with_conversation(
        active_consultation["consultation_id"], SAMPLE_CONVERSATION
    )
    response = _generate(
        active_consultation["consultation_id"], transcript_id, active_consultation["token"]
    )
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "completed"
    assert body["consultation_id"] == active_consultation["consultation_id"]
    assert body["transcript_id"] == transcript_id
    assert body["subjective_claim_count"] >= 1
    assert body["objective_claim_count"] >= 1
    assert body["assessment_claim_count"] >= 1
    assert body["plan_claim_count"] >= 1


def test_subjective_claims_from_patient_evidence(active_consultation):
    transcript_id = _create_transcript_with_conversation(
        active_consultation["consultation_id"], SAMPLE_CONVERSATION
    )
    response = _generate(
        active_consultation["consultation_id"], transcript_id, active_consultation["token"]
    )
    body = response.json()
    subjective_texts = [c["claim_text"] for c in body["claims"] if c["section"] == "SUBJECTIVE"]
    assert any("fever" in t and "two days" in t for t in subjective_texts)
    assert any("cough" in t and "two days" in t for t in subjective_texts)
    for t in subjective_texts:
        assert t.startswith("Patient")


def test_objective_measurements_preserved(active_consultation):
    transcript_id = _create_transcript_with_conversation(
        active_consultation["consultation_id"], SAMPLE_CONVERSATION
    )
    response = _generate(
        active_consultation["consultation_id"], transcript_id, active_consultation["token"]
    )
    body = response.json()
    objective_texts = [c["claim_text"] for c in body["claims"] if c["section"] == "OBJECTIVE"]
    assert any("101.2" in t for t in objective_texts)


def test_explicit_doctor_assessment_preserved(active_consultation):
    transcript_id = _create_transcript_with_conversation(
        active_consultation["consultation_id"], SAMPLE_CONVERSATION
    )
    response = _generate(
        active_consultation["consultation_id"], transcript_id, active_consultation["token"]
    )
    body = response.json()
    assessment_texts = [c["claim_text"] for c in body["claims"] if c["section"] == "ASSESSMENT"]
    assert len(assessment_texts) == 1
    assert "viral infection" in assessment_texts[0]
    assert assessment_texts[0].startswith("Doctor assessed")


def test_explicit_doctor_plan_preserved(active_consultation):
    transcript_id = _create_transcript_with_conversation(
        active_consultation["consultation_id"], SAMPLE_CONVERSATION
    )
    response = _generate(
        active_consultation["consultation_id"], transcript_id, active_consultation["token"]
    )
    body = response.json()
    plan_texts = [c["claim_text"] for c in body["claims"] if c["section"] == "PLAN"]
    assert any("paracetamol" in t for t in plan_texts)
    assert any("follow-up in three days" in t for t in plan_texts)


def test_unsupported_diagnosis_is_not_generated(active_consultation):
    # Symptoms mentioned, but the doctor never states an assessment --
    # must NOT invent "pneumonia" or any other diagnosis.
    conversation = [
        ("PATIENT", "I have cough and fever."),
        ("DOCTOR", "Okay, thank you for letting me know."),
    ]
    transcript_id = _create_transcript_with_conversation(
        active_consultation["consultation_id"], conversation
    )
    response = _generate(
        active_consultation["consultation_id"], transcript_id, active_consultation["token"]
    )
    assert response.status_code == 201
    body = response.json()
    assert body["assessment_claim_count"] == 0
    assert not any(c["section"] == "ASSESSMENT" for c in body["claims"])
    all_text = " ".join(c["claim_text"] for c in body["claims"])
    assert "pneumonia" not in all_text.lower()


def test_negated_symptoms_remain_negated(active_consultation):
    conversation = [("PATIENT", "I don't have chest pain.")]
    transcript_id = _create_transcript_with_conversation(
        active_consultation["consultation_id"], conversation
    )
    response = _generate(
        active_consultation["consultation_id"], transcript_id, active_consultation["token"]
    )
    body = response.json()
    subjective_texts = [c["claim_text"] for c in body["claims"] if c["section"] == "SUBJECTIVE"]
    assert len(subjective_texts) == 1
    assert subjective_texts[0] == "Patient denies chest pain."
    assert "Patient reports chest pain" not in subjective_texts[0]


def test_historical_conditions_not_treated_as_current(active_consultation):
    conversation = [("PATIENT", "I had asthma as a child.")]
    transcript_id = _create_transcript_with_conversation(
        active_consultation["consultation_id"], conversation
    )
    response = _generate(
        active_consultation["consultation_id"], transcript_id, active_consultation["token"]
    )
    body = response.json()
    subjective_texts = [c["claim_text"] for c in body["claims"] if c["section"] == "SUBJECTIVE"]
    assert subjective_texts == ["Patient reports history of asthma."]
    # never surfaced as a current assessment/diagnosis
    assert body["assessment_claim_count"] == 0


def test_individual_soap_claims_are_stored_with_correct_sections(active_consultation):
    transcript_id = _create_transcript_with_conversation(
        active_consultation["consultation_id"], SAMPLE_CONVERSATION
    )
    response = _generate(
        active_consultation["consultation_id"], transcript_id, active_consultation["token"]
    )
    assert response.status_code == 201
    soap_note_id = response.json()["soap_note_id"]

    db = SessionLocal()
    try:
        claims = (
            db.query(SOAPClaim)
            .filter(SOAPClaim.soap_note_id == uuid.UUID(soap_note_id))
            .order_by(SOAPClaim.section, SOAPClaim.sequence_index)
            .all()
        )
        assert len(claims) >= 4
        sections_seen = {c.section for c in claims}
        assert sections_seen == {"SUBJECTIVE", "OBJECTIVE", "ASSESSMENT", "PLAN"}
        for claim in claims:
            assert claim.generation_confidence is not None
            assert 0.0 <= float(claim.generation_confidence) <= 1.0

        note = db.query(SOAPNote).filter(SOAPNote.id == uuid.UUID(soap_note_id)).first()
        assert note.generation_status == "completed"
        assert note.subjective != "Not documented."
        assert note.objective != "Not documented."
        assert note.assessment != "Not documented."
        assert note.plan != "Not documented."
    finally:
        db.close()


def test_repeated_generation_does_not_create_uncontrolled_duplicates(active_consultation):
    transcript_id = _create_transcript_with_conversation(
        active_consultation["consultation_id"], SAMPLE_CONVERSATION
    )
    first = _generate(
        active_consultation["consultation_id"], transcript_id, active_consultation["token"]
    )
    assert first.status_code == 201
    first_note_id = first.json()["soap_note_id"]

    second = _generate(
        active_consultation["consultation_id"], transcript_id, active_consultation["token"]
    )
    assert second.status_code == 201
    second_note_id = second.json()["soap_note_id"]

    assert first_note_id == second_note_id  # same row updated, not a new version

    db = SessionLocal()
    try:
        note_count = (
            db.query(SOAPNote)
            .filter(SOAPNote.consultation_id == uuid.UUID(active_consultation["consultation_id"]))
            .count()
        )
        claim_count = (
            db.query(SOAPClaim).filter(SOAPClaim.soap_note_id == uuid.UUID(first_note_id)).count()
        )
    finally:
        db.close()

    assert note_count == 1
    assert claim_count == len(first.json()["claims"])  # not doubled


def test_empty_transcript_handled_safely(active_consultation):
    conversation = [("DOCTOR", "Hello."), ("PATIENT", "Hi.")]
    transcript_id = _create_transcript_with_conversation(
        active_consultation["consultation_id"], conversation
    )
    response = _generate(
        active_consultation["consultation_id"], transcript_id, active_consultation["token"]
    )
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "completed"  # sparse evidence is not a failure
    assert body["subjective_claim_count"] == 0
    assert body["objective_claim_count"] == 0
    assert body["assessment_claim_count"] == 0
    assert body["plan_claim_count"] == 0

    db = SessionLocal()
    try:
        note = (
            db.query(SOAPNote)
            .filter(SOAPNote.id == uuid.UUID(body["soap_note_id"]))
            .first()
        )
        assert note.subjective == "Not documented."
        assert note.objective == "Not documented."
        assert note.assessment == "Not documented."
        assert note.plan == "Not documented."
    finally:
        db.close()


def test_generation_failure_is_represented_safely(active_consultation, monkeypatch):
    transcript_id = _create_transcript_with_conversation(
        active_consultation["consultation_id"], SAMPLE_CONVERSATION
    )

    def _raise(*_args, **_kwargs):
        raise RuntimeError("simulated unexpected generation error")

    monkeypatch.setattr("app.soap.service.generate_claims", _raise)

    response = _generate(
        active_consultation["consultation_id"], transcript_id, active_consultation["token"]
    )
    assert response.status_code == 500
    assert "traceback" not in response.text.lower()
    assert str(REPO_ROOT) not in response.text

    db = SessionLocal()
    try:
        note = (
            db.query(SOAPNote)
            .filter(SOAPNote.consultation_id == uuid.UUID(active_consultation["consultation_id"]))
            .first()
        )
        assert note.generation_status == "failed"
        assert note.generation_error is not None
    finally:
        db.close()
