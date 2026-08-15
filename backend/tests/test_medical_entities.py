"""Medical entity extraction + PHI protection tests: ownership, status gating,
deterministic rule-based extraction, negation/historical detection, speaker
association, PHI detection/de-identification, idempotency, and safe failure.

Pure Python heuristics (no model, no external service) -- no mocking needed
for the extraction logic itself; only the "unexpected failure" test mocks.
All test text is synthetic, never real patient data.
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
from app.models import Consultation, Doctor, MedicalEntity, SpeakerSegment, Transcript

client = TestClient(app)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture(scope="module", autouse=True)
def _ensure_schema():
    init_db()


def _register_and_login(name="Dr. Entities"):
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
            transcript_ids = [
                t.id for t in db.query(Transcript).filter(Transcript.consultation_id.in_(consultation_ids))
            ]
            if transcript_ids:
                db.query(MedicalEntity).filter(
                    MedicalEntity.transcript_id.in_(transcript_ids)
                ).delete(synchronize_session=False)
                db.query(SpeakerSegment).filter(
                    SpeakerSegment.transcript_id.in_(transcript_ids)
                ).delete(synchronize_session=False)
            db.query(MedicalEntity).filter(
                MedicalEntity.consultation_id.in_(consultation_ids)
            ).delete(synchronize_session=False)
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


def _create_transcript(consultation_id: str, full_text: str, segments: list) -> str:
    """segments: list of (start_ms, end_ms, text)"""
    db = SessionLocal()
    try:
        transcript = Transcript(
            consultation_id=uuid.UUID(consultation_id),
            full_text=full_text,
            language="en",
            asr_model="faster-whisper-small",
            processing_status="completed",
        )
        db.add(transcript)
        db.commit()
        db.refresh(transcript)
        for index, (start_ms, end_ms, text) in enumerate(segments):
            db.add(
                SpeakerSegment(
                    transcript_id=transcript.id,
                    sequence_index=index,
                    start_time_ms=start_ms,
                    end_time_ms=end_ms,
                    segment_text=text,
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
        "/consultations", json={"patient_reference": "PATIENT-ENTITIES"}, headers=_headers(token)
    )
    consultation_id = create_response.json()["id"]
    client.patch(
        f"/consultations/{consultation_id}/status",
        json={"status": "active"},
        headers=_headers(token),
    )
    yield {"token": token, "consultation_id": consultation_id, "email": email}
    _cleanup_doctor(email)


def _extract(consultation_id, transcript_id, token):
    return client.post(
        f"/consultations/{consultation_id}/audio/extract-entities",
        json={"transcript_id": transcript_id},
        headers=_headers(token),
    )


def _entities_for(transcript_id: str):
    db = SessionLocal()
    try:
        return (
            db.query(MedicalEntity)
            .filter(MedicalEntity.transcript_id == uuid.UUID(transcript_id))
            .all()
        )
    finally:
        db.close()


def test_extract_requires_authentication(active_consultation):
    transcript_id = _create_transcript(
        active_consultation["consultation_id"], "I have a fever.", [(0, 1000, "I have a fever.")]
    )
    response = client.post(
        f"/consultations/{active_consultation['consultation_id']}/audio/extract-entities",
        json={"transcript_id": transcript_id},
    )
    assert response.status_code == 401


def test_extract_unknown_consultation_returns_404(active_consultation):
    transcript_id = _create_transcript(
        active_consultation["consultation_id"], "I have a fever.", [(0, 1000, "I have a fever.")]
    )
    response = _extract(str(uuid.uuid4()), transcript_id, active_consultation["token"])
    assert response.status_code == 404


def test_extract_another_doctors_consultation_returns_404(active_consultation):
    transcript_id = _create_transcript(
        active_consultation["consultation_id"], "I have a fever.", [(0, 1000, "I have a fever.")]
    )
    other_email, other_token = _register_and_login(name="Dr. Other")
    response = _extract(active_consultation["consultation_id"], transcript_id, other_token)
    assert response.status_code == 404
    _cleanup_doctor(other_email)


def test_extract_draft_consultation_returns_409():
    email, token = _register_and_login(name="Dr. Draft")
    create_response = client.post(
        "/consultations", json={"patient_reference": "PATIENT-DRAFT"}, headers=_headers(token)
    )
    consultation_id = create_response.json()["id"]

    response = _extract(consultation_id, str(uuid.uuid4()), token)
    assert response.status_code == 409
    _cleanup_doctor(email)


def test_extract_completed_consultation_returns_409():
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

    response = _extract(consultation_id, str(uuid.uuid4()), token)
    assert response.status_code == 409
    _cleanup_doctor(email)


def test_extract_cancelled_consultation_returns_409():
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

    response = _extract(consultation_id, str(uuid.uuid4()), token)
    assert response.status_code == 409
    _cleanup_doctor(email)


def test_extract_unknown_transcript_returns_404(active_consultation):
    response = _extract(
        active_consultation["consultation_id"], str(uuid.uuid4()), active_consultation["token"]
    )
    assert response.status_code == 404


def test_extract_transcript_belonging_to_another_consultation_returns_404(active_consultation):
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
    transcript_id = _create_transcript(
        active_consultation["consultation_id"], "I have a fever.", [(0, 1000, "I have a fever.")]
    )

    response = _extract(second_consultation_id, transcript_id, token)
    assert response.status_code == 404


def test_extract_transcript_not_completed_returns_409(active_consultation):
    db = SessionLocal()
    try:
        transcript = Transcript(
            consultation_id=uuid.UUID(active_consultation["consultation_id"]),
            processing_status="pending",
        )
        db.add(transcript)
        db.commit()
        transcript_id = str(transcript.id)
    finally:
        db.close()

    response = _extract(
        active_consultation["consultation_id"], transcript_id, active_consultation["token"]
    )
    assert response.status_code == 409


def test_symptom_extraction(active_consultation):
    transcript_id = _create_transcript(
        active_consultation["consultation_id"],
        "I have a fever and a cough.",
        [(0, 2000, "I have a fever and a cough.")],
    )
    response = _extract(
        active_consultation["consultation_id"], transcript_id, active_consultation["token"]
    )
    assert response.status_code == 200
    assert response.json()["medical_entity_count"] >= 2

    entities = _entities_for(transcript_id)
    symptom_types = {e.normalized_value for e in entities if e.entity_type == "SYMPTOM"}
    assert "fever" in symptom_types
    assert "cough" in symptom_types


def test_medication_extraction(active_consultation):
    transcript_id = _create_transcript(
        active_consultation["consultation_id"],
        "The doctor prescribed paracetamol.",
        [(0, 2000, "The doctor prescribed paracetamol.")],
    )
    response = _extract(
        active_consultation["consultation_id"], transcript_id, active_consultation["token"]
    )
    assert response.status_code == 200

    entities = _entities_for(transcript_id)
    medications = [e for e in entities if e.entity_type == "MEDICATION"]
    assert len(medications) == 1
    assert medications[0].normalized_value == "paracetamol"


def test_allergy_extraction(active_consultation):
    transcript_id = _create_transcript(
        active_consultation["consultation_id"],
        "I am allergic to penicillin.",
        [(0, 2000, "I am allergic to penicillin.")],
    )
    response = _extract(
        active_consultation["consultation_id"], transcript_id, active_consultation["token"]
    )
    assert response.status_code == 200

    entities = _entities_for(transcript_id)
    allergies = [e for e in entities if e.entity_type == "ALLERGY"]
    assert len(allergies) == 1
    assert allergies[0].normalized_value == "penicillin allergy"


def test_medical_procedure_extraction(active_consultation):
    transcript_id = _create_transcript(
        active_consultation["consultation_id"],
        "You will need a vaccination today.",
        [(0, 2000, "You will need a vaccination today.")],
    )
    response = _extract(
        active_consultation["consultation_id"], transcript_id, active_consultation["token"]
    )
    assert response.status_code == 200

    entities = _entities_for(transcript_id)
    procedures = [e for e in entities if e.entity_type == "MEDICAL_PROCEDURE"]
    assert len(procedures) == 1
    assert procedures[0].normalized_value == "vaccination"


def test_measurement_extraction(active_consultation):
    transcript_id = _create_transcript(
        active_consultation["consultation_id"],
        "Your blood pressure is 120/80.",
        [(0, 2000, "Your blood pressure is 120/80.")],
    )
    response = _extract(
        active_consultation["consultation_id"], transcript_id, active_consultation["token"]
    )
    assert response.status_code == 200

    entities = _entities_for(transcript_id)
    measurements = [e for e in entities if e.entity_type == "MEASUREMENT"]
    assert len(measurements) == 1
    assert "120/80" in measurements[0].entity_text


def test_negation_detection(active_consultation):
    transcript_id = _create_transcript(
        active_consultation["consultation_id"],
        "I don't have chest pain.",
        [(0, 2000, "I don't have chest pain.")],
    )
    response = _extract(
        active_consultation["consultation_id"], transcript_id, active_consultation["token"]
    )
    assert response.status_code == 200

    entities = _entities_for(transcript_id)
    symptoms = [e for e in entities if e.entity_type == "SYMPTOM"]
    assert len(symptoms) == 1
    assert symptoms[0].normalized_value == "chest pain"
    assert symptoms[0].negated is True


def test_historical_mention_detection(active_consultation):
    transcript_id = _create_transcript(
        active_consultation["consultation_id"],
        "I had asthma as a child.",
        [(0, 2000, "I had asthma as a child.")],
    )
    response = _extract(
        active_consultation["consultation_id"], transcript_id, active_consultation["token"]
    )
    assert response.status_code == 200

    entities = _entities_for(transcript_id)
    diagnoses = [e for e in entities if e.entity_type == "DIAGNOSIS_MENTION"]
    assert len(diagnoses) == 1
    assert diagnoses[0].normalized_value == "asthma"
    assert diagnoses[0].historical is True
    assert diagnoses[0].negated is False


def test_family_mention_is_not_attributed_to_patient(active_consultation):
    transcript_id = _create_transcript(
        active_consultation["consultation_id"],
        "My mother has diabetes.",
        [(0, 2000, "My mother has diabetes.")],
    )
    response = _extract(
        active_consultation["consultation_id"], transcript_id, active_consultation["token"]
    )
    assert response.status_code == 200
    assert response.json()["medical_entity_count"] == 0  # conservatively not extracted at all


def test_speaker_association_where_available(active_consultation):
    transcript_id = _create_transcript(
        active_consultation["consultation_id"],
        "What brings you in today? I have a fever.",
        [
            (0, 1500, "What brings you in today?"),
            (1500, 3000, "I have a fever."),
        ],
    )
    response = _extract(
        active_consultation["consultation_id"], transcript_id, active_consultation["token"]
    )
    assert response.status_code == 200

    db = SessionLocal()
    try:
        segments = (
            db.query(SpeakerSegment)
            .filter(SpeakerSegment.transcript_id == uuid.UUID(transcript_id))
            .order_by(SpeakerSegment.sequence_index)
            .all()
        )
        fever_segment_id = segments[1].id
    finally:
        db.close()

    entities = _entities_for(transcript_id)
    symptoms = [e for e in entities if e.entity_type == "SYMPTOM"]
    assert len(symptoms) == 1
    assert symptoms[0].speaker_segment_id == fever_segment_id


def test_phi_detection(active_consultation):
    transcript_id = _create_transcript(
        active_consultation["consultation_id"],
        "My name is John Miller. Call me at 9876543210.",
        [(0, 3000, "My name is John Miller. Call me at 9876543210.")],
    )
    response = _extract(
        active_consultation["consultation_id"], transcript_id, active_consultation["token"]
    )
    assert response.status_code == 200
    assert response.json()["phi_detected"] is True

    entities = _entities_for(transcript_id)
    phi_entities = [e for e in entities if e.entity_type.startswith("PHI_")]
    phi_types = {e.entity_type for e in phi_entities}
    assert "PHI_PERSON_NAME" in phi_types
    assert "PHI_PHONE_NUMBER" in phi_types
    # raw PHI value must never be stored -- only redaction placeholders
    for entity in phi_entities:
        assert "John" not in entity.entity_text
        assert "9876543210" not in entity.entity_text
        assert entity.entity_text.startswith("[") and entity.entity_text.endswith("]")


def test_phi_deidentification_and_original_transcript_unchanged(active_consultation):
    original_text = "My name is John Miller. Call me at 9876543210."
    transcript_id = _create_transcript(
        active_consultation["consultation_id"], original_text, [(0, 3000, original_text)]
    )
    response = _extract(
        active_consultation["consultation_id"], transcript_id, active_consultation["token"]
    )
    assert response.status_code == 200
    assert response.json()["deidentified_text_available"] is True

    db = SessionLocal()
    try:
        transcript = db.query(Transcript).filter(Transcript.id == uuid.UUID(transcript_id)).first()
        # original transcript (source evidence) is never modified
        assert transcript.full_text == original_text
        # de-identified representation is a separate field with PHI redacted
        assert transcript.deidentified_text != original_text
        assert "John Miller" not in transcript.deidentified_text
        assert "9876543210" not in transcript.deidentified_text
        assert "[PERSON_NAME]" in transcript.deidentified_text
        assert "[PHONE_NUMBER]" in transcript.deidentified_text
    finally:
        db.close()


def test_no_raw_phi_appears_in_application_logs(active_consultation, caplog):
    original_text = "My name is John Miller. Call me at 9876543210."
    transcript_id = _create_transcript(
        active_consultation["consultation_id"], original_text, [(0, 3000, original_text)]
    )
    with caplog.at_level("DEBUG"):
        response = _extract(
            active_consultation["consultation_id"], transcript_id, active_consultation["token"]
        )
    assert response.status_code == 200
    assert "John Miller" not in caplog.text
    assert "9876543210" not in caplog.text


def test_empty_extraction_succeeds_with_zero_count(active_consultation):
    transcript_id = _create_transcript(
        active_consultation["consultation_id"], "Hello, thank you.", [(0, 1000, "Hello, thank you.")]
    )
    response = _extract(
        active_consultation["consultation_id"], transcript_id, active_consultation["token"]
    )
    assert response.status_code == 200
    body = response.json()
    assert body["medical_entity_count"] == 0
    assert body["phi_detected"] is False
    assert body["processing_status"] == "completed"  # empty result is not a failure


def test_extraction_failure_is_represented_safely(active_consultation, monkeypatch):
    transcript_id = _create_transcript(
        active_consultation["consultation_id"], "I have a fever.", [(0, 1000, "I have a fever.")]
    )

    def _raise(_text):
        raise RuntimeError("simulated unexpected extraction error")

    monkeypatch.setattr("app.medical_entities.service.extract_entities", _raise)

    response = _extract(
        active_consultation["consultation_id"], transcript_id, active_consultation["token"]
    )
    assert response.status_code == 500
    assert "traceback" not in response.text.lower()
    assert str(REPO_ROOT) not in response.text

    db = SessionLocal()
    try:
        transcript = db.query(Transcript).filter(Transcript.id == uuid.UUID(transcript_id)).first()
        assert transcript.entity_extraction_status == "failed"
        assert transcript.entity_extraction_error is not None
    finally:
        db.close()


def test_repeated_extraction_does_not_duplicate_entities(active_consultation):
    transcript_id = _create_transcript(
        active_consultation["consultation_id"],
        "I have a fever and a cough.",
        [(0, 2000, "I have a fever and a cough.")],
    )

    first = _extract(
        active_consultation["consultation_id"], transcript_id, active_consultation["token"]
    )
    assert first.status_code == 200
    count_after_first = len(_entities_for(transcript_id))

    second = _extract(
        active_consultation["consultation_id"], transcript_id, active_consultation["token"]
    )
    assert second.status_code == 200
    assert second.json()["medical_entity_count"] == first.json()["medical_entity_count"]
    count_after_second = len(_entities_for(transcript_id))

    assert count_after_first == count_after_second
    assert count_after_first > 0


# ---------------------------------------------------------------------------
# Step 15A: vocabulary boundary-matching regression tests.
#
# _find_vocab_matches() used to be a plain substring search, so short
# vocabulary terms could match inside unrelated words (e.g. BODY_PART "ear"
# matching inside "appears"). These call extract_entities() directly (same
# real code path used by the API, see test_soap.py for precedent) since the
# fix is pure text-processing logic and doesn't need the DB/HTTP round trip.
# ---------------------------------------------------------------------------


def _normalized_values(entities, entity_type=None):
    return {
        e.normalized_text for e in entities if entity_type is None or e.entity_type == entity_type
    }


def test_partial_word_match_inside_unrelated_word_is_rejected():
    # Exact Step 15 false positive: "ear" must not be extracted from "appears".
    entities = extract_entities("This appears to be a viral infection.")
    assert "ear" not in _normalized_values(entities, "BODY_PART")
    assert entities == []  # no other vocabulary term in this sentence either


def test_standalone_short_vocab_term_still_matches():
    entities = extract_entities("The patient has ear pain.")
    assert "ear" in _normalized_values(entities, "BODY_PART")


def test_multiword_vocab_entity_still_matches_as_one_span():
    entities = extract_entities("I have a stomach ache.")
    symptoms = _normalized_values(entities, "SYMPTOM")
    assert "stomach pain" in symptoms
    # the multi-word phrase should claim the span; "stomach" must not also
    # fire separately as a BODY_PART match on the same text.
    assert "stomach" not in _normalized_values(entities, "BODY_PART")


def test_case_insensitive_matching_still_works():
    entities = extract_entities("I have a FEVER today.")
    fevers = [e for e in entities if e.normalized_text == "fever"]
    assert len(fevers) == 1
    assert fevers[0].entity_text == "FEVER"  # original casing preserved


def test_punctuation_immediately_after_entity_does_not_block_match():
    entities = extract_entities("He has a headache, and nausea.")
    symptoms = _normalized_values(entities, "SYMPTOM")
    assert "headache" in symptoms
    assert "nausea" in symptoms


def test_hyphenated_vocab_entity_still_matches():
    entities = extract_entities("The test confirmed covid-19.")
    diagnoses = [e for e in entities if e.entity_type == "DIAGNOSIS_MENTION"]
    assert len(diagnoses) == 1
    assert diagnoses[0].normalized_text == "covid-19"
    assert diagnoses[0].entity_text == "covid-19"


def test_tanglish_vocabulary_matching_still_works():
    # Tamil combining marks (vowel signs / virama) are not \w in Python's re
    # module, so a naive \b-based boundary fix would silently break Tamil
    # vocabulary matching. This proves the fix still matches it correctly.
    entities = extract_entities("எனக்கு காய்ச்சல் இருக்கு.")
    symptoms = [e for e in entities if e.entity_type == "SYMPTOM"]
    assert len(symptoms) == 1
    assert symptoms[0].normalized_text == "fever"


def test_medication_extraction_unaffected_by_boundary_fix():
    entities = extract_entities("The doctor prescribed paracetamol.")
    medications = [e for e in entities if e.entity_type == "MEDICATION"]
    assert len(medications) == 1
    assert medications[0].normalized_text == "paracetamol"


def test_measurement_extraction_unaffected_by_boundary_fix():
    entities = extract_entities("Your blood pressure is 120/80.")
    measurements = [e for e in entities if e.entity_type == "MEASUREMENT"]
    assert len(measurements) == 1
    assert "120/80" in measurements[0].entity_text
