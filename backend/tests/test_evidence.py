"""Evidence retrieval (Step 14A) tests: ownership, status gating, deterministic
lexical scoring, role-based ranking (not filtering), top-K/threshold behavior,
idempotency, and the negation-preservation research principle.

Pure Python heuristics (no model, no external service) -- no mocking needed
except for the "unexpected failure" test. All conversation text is synthetic.
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
from app.models import (
    Consultation,
    Doctor,
    EvidenceLink,
    SOAPClaim,
    SOAPNote,
    SpeakerSegment,
    Transcript,
)

client = TestClient(app)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture(scope="module", autouse=True)
def _ensure_schema():
    init_db()


def _register_and_login(name="Dr. Evidence"):
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
                claim_ids = [
                    c.id for c in db.query(SOAPClaim).filter(SOAPClaim.soap_note_id.in_(soap_note_ids))
                ]
                if claim_ids:
                    db.query(EvidenceLink).filter(EvidenceLink.soap_claim_id.in_(claim_ids)).delete(
                        synchronize_session=False
                    )
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


def _create_transcript_with_segments(consultation_id: str, turns: list) -> tuple:
    """turns: list of (role, text) with role in ('DOCTOR', 'PATIENT', 'UNKNOWN', None).
    Returns (transcript_id, [segment_id, ...]) in turn order.
    """
    db = SessionLocal()
    try:
        full_text = " ".join(text for _role, text in turns)
        transcript = Transcript(
            consultation_id=uuid.UUID(consultation_id),
            full_text=full_text,
            language="en",
            processing_status="completed",
            entity_extraction_status="completed",
        )
        db.add(transcript)
        db.commit()
        db.refresh(transcript)

        segment_ids = []
        cumulative_ms = 0
        for index, (role, text) in enumerate(turns):
            segment = SpeakerSegment(
                transcript_id=transcript.id,
                sequence_index=index,
                speaker_label=f"SPEAKER_{index:02d}",
                inferred_role=role,
                start_time_ms=cumulative_ms,
                end_time_ms=cumulative_ms + 1000,
                segment_text=text,
            )
            db.add(segment)
            db.commit()
            db.refresh(segment)
            segment_ids.append(str(segment.id))
            cumulative_ms += 1000

        return str(transcript.id), segment_ids
    finally:
        db.close()


def _create_soap_note_with_claims(consultation_id: str, transcript_id: str, claims: list) -> str:
    """claims: list of (section, claim_text). Returns soap_note_id."""
    db = SessionLocal()
    try:
        soap_note = SOAPNote(
            consultation_id=uuid.UUID(consultation_id),
            transcript_id=uuid.UUID(transcript_id) if transcript_id else None,
            version=1,
            status="generated",
            generation_status="completed",
        )
        db.add(soap_note)
        db.commit()
        db.refresh(soap_note)
        for index, (section, text) in enumerate(claims):
            db.add(
                SOAPClaim(
                    soap_note_id=soap_note.id, section=section, claim_text=text, sequence_index=index
                )
            )
        db.commit()
        return str(soap_note.id)
    finally:
        db.close()


@pytest.fixture()
def active_consultation():
    email, token = _register_and_login()
    create_response = client.post(
        "/consultations", json={"patient_reference": "PATIENT-EVIDENCE"}, headers=_headers(token)
    )
    consultation_id = create_response.json()["id"]
    client.patch(
        f"/consultations/{consultation_id}/status",
        json={"status": "active"},
        headers=_headers(token),
    )
    yield {"token": token, "consultation_id": consultation_id, "email": email}
    _cleanup_doctor(email)


def _retrieve(consultation_id, soap_note_id, token):
    return client.post(
        f"/consultations/{consultation_id}/soap/evidence/retrieve",
        json={"soap_note_id": soap_note_id},
        headers=_headers(token),
    )


def test_retrieve_requires_authentication(active_consultation):
    transcript_id, _ = _create_transcript_with_segments(
        active_consultation["consultation_id"], [("PATIENT", "I have fever.")]
    )
    soap_note_id = _create_soap_note_with_claims(
        active_consultation["consultation_id"], transcript_id, [("SUBJECTIVE", "Patient reports fever.")]
    )
    response = client.post(
        f"/consultations/{active_consultation['consultation_id']}/soap/evidence/retrieve",
        json={"soap_note_id": soap_note_id},
    )
    assert response.status_code == 401


def test_retrieve_unknown_consultation_returns_404(active_consultation):
    transcript_id, _ = _create_transcript_with_segments(
        active_consultation["consultation_id"], [("PATIENT", "I have fever.")]
    )
    soap_note_id = _create_soap_note_with_claims(
        active_consultation["consultation_id"], transcript_id, [("SUBJECTIVE", "Patient reports fever.")]
    )
    response = _retrieve(str(uuid.uuid4()), soap_note_id, active_consultation["token"])
    assert response.status_code == 404


def test_retrieve_another_doctors_consultation_returns_404(active_consultation):
    transcript_id, _ = _create_transcript_with_segments(
        active_consultation["consultation_id"], [("PATIENT", "I have fever.")]
    )
    soap_note_id = _create_soap_note_with_claims(
        active_consultation["consultation_id"], transcript_id, [("SUBJECTIVE", "Patient reports fever.")]
    )
    other_email, other_token = _register_and_login(name="Dr. Other")
    response = _retrieve(active_consultation["consultation_id"], soap_note_id, other_token)
    assert response.status_code == 404
    _cleanup_doctor(other_email)


def test_retrieve_draft_consultation_returns_409():
    email, token = _register_and_login(name="Dr. Draft")
    create_response = client.post(
        "/consultations", json={"patient_reference": "PATIENT-DRAFT"}, headers=_headers(token)
    )
    consultation_id = create_response.json()["id"]

    response = _retrieve(consultation_id, str(uuid.uuid4()), token)
    assert response.status_code == 409
    _cleanup_doctor(email)


def test_retrieve_completed_consultation_returns_409():
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

    response = _retrieve(consultation_id, str(uuid.uuid4()), token)
    assert response.status_code == 409
    _cleanup_doctor(email)


def test_retrieve_cancelled_consultation_returns_409():
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

    response = _retrieve(consultation_id, str(uuid.uuid4()), token)
    assert response.status_code == 409
    _cleanup_doctor(email)


def test_retrieve_unknown_soap_note_returns_404(active_consultation):
    response = _retrieve(
        active_consultation["consultation_id"], str(uuid.uuid4()), active_consultation["token"]
    )
    assert response.status_code == 404


def test_retrieve_soap_note_belonging_to_another_consultation_returns_404(active_consultation):
    token = active_consultation["token"]
    transcript_id, _ = _create_transcript_with_segments(
        active_consultation["consultation_id"], [("PATIENT", "I have fever.")]
    )
    soap_note_id = _create_soap_note_with_claims(
        active_consultation["consultation_id"], transcript_id, [("SUBJECTIVE", "Patient reports fever.")]
    )

    create_response = client.post(
        "/consultations", json={"patient_reference": "PATIENT-SECOND"}, headers=_headers(token)
    )
    second_consultation_id = create_response.json()["id"]
    client.patch(
        f"/consultations/{second_consultation_id}/status",
        json={"status": "active"},
        headers=_headers(token),
    )

    response = _retrieve(second_consultation_id, soap_note_id, token)
    assert response.status_code == 404


def test_retrieve_soap_note_not_completed_returns_409(active_consultation):
    db = SessionLocal()
    try:
        soap_note = SOAPNote(
            consultation_id=uuid.UUID(active_consultation["consultation_id"]),
            version=1,
            generation_status="pending",
        )
        db.add(soap_note)
        db.commit()
        soap_note_id = str(soap_note.id)
    finally:
        db.close()

    response = _retrieve(
        active_consultation["consultation_id"], soap_note_id, active_consultation["token"]
    )
    assert response.status_code == 409


def test_retrieve_soap_note_with_no_claims_returns_safe_zero_result(active_consultation):
    transcript_id, _ = _create_transcript_with_segments(
        active_consultation["consultation_id"], [("PATIENT", "I have fever.")]
    )
    soap_note_id = _create_soap_note_with_claims(
        active_consultation["consultation_id"], transcript_id, []
    )

    response = _retrieve(
        active_consultation["consultation_id"], soap_note_id, active_consultation["token"]
    )
    assert response.status_code == 200
    body = response.json()
    assert body["claims_processed"] == 0
    assert body["evidence_links_created"] == 0
    assert body["status"] == "completed"


def test_basic_lexical_retrieval(active_consultation):
    transcript_id, segment_ids = _create_transcript_with_segments(
        active_consultation["consultation_id"],
        [("PATIENT", "I have had fever and cough for two days.")],
    )
    soap_note_id = _create_soap_note_with_claims(
        active_consultation["consultation_id"],
        transcript_id,
        [("SUBJECTIVE", "Patient reports fever for two days.")],
    )

    response = _retrieve(
        active_consultation["consultation_id"], soap_note_id, active_consultation["token"]
    )
    assert response.status_code == 200
    body = response.json()
    assert body["claims_processed"] == 1
    assert body["evidence_links_created"] == 1
    evidence = body["per_claim"][0]["evidence"]
    assert len(evidence) == 1
    assert evidence[0]["speaker_segment_id"] == segment_ids[0]
    assert evidence[0]["rank"] == 1


def test_token_overlap_retrieval(active_consultation):
    transcript_id, segment_ids = _create_transcript_with_segments(
        active_consultation["consultation_id"],
        [("DOCTOR", "Take paracetamol and return in three days.")],
    )
    soap_note_id = _create_soap_note_with_claims(
        active_consultation["consultation_id"], transcript_id, [("PLAN", "Doctor advised paracetamol.")]
    )

    response = _retrieve(
        active_consultation["consultation_id"], soap_note_id, active_consultation["token"]
    )
    body = response.json()
    evidence = body["per_claim"][0]["evidence"]
    assert len(evidence) == 1
    assert evidence[0]["speaker_segment_id"] == segment_ids[0]


def test_multiple_candidates_are_correctly_ranked(active_consultation):
    transcript_id, segment_ids = _create_transcript_with_segments(
        active_consultation["consultation_id"],
        [
            ("PATIENT", "I have fever and cough."),  # weaker overlap with the claim below
            ("PATIENT", "I have had fever and cough for two days."),  # strongest overlap
            ("DOCTOR", "How are you feeling today?"),  # irrelevant
        ],
    )
    soap_note_id = _create_soap_note_with_claims(
        active_consultation["consultation_id"],
        transcript_id,
        [("SUBJECTIVE", "Patient reports fever and cough for two days.")],
    )

    response = _retrieve(
        active_consultation["consultation_id"], soap_note_id, active_consultation["token"]
    )
    evidence = response.json()["per_claim"][0]["evidence"]
    assert len(evidence) >= 2
    assert evidence[0]["speaker_segment_id"] == segment_ids[1]  # strongest match ranked first
    assert evidence[0]["rank"] == 1
    scores = [e["retrieval_score"] for e in evidence]
    assert scores == sorted(scores, reverse=True)  # descending order


def test_top_k_limit_works(active_consultation):
    # 5 segments all strongly overlapping the claim; default EVIDENCE_TOP_K=3.
    turns = [("PATIENT", "I have fever and cough.") for _ in range(5)]
    transcript_id, _ = _create_transcript_with_segments(
        active_consultation["consultation_id"], turns
    )
    soap_note_id = _create_soap_note_with_claims(
        active_consultation["consultation_id"], transcript_id, [("SUBJECTIVE", "Patient reports fever and cough.")]
    )

    response = _retrieve(
        active_consultation["consultation_id"], soap_note_id, active_consultation["token"]
    )
    body = response.json()
    assert body["evidence_links_created"] == 3  # capped at default top-K
    assert len(body["per_claim"][0]["evidence"]) == 3


def test_minimum_score_threshold_works(active_consultation):
    transcript_id, segment_ids = _create_transcript_with_segments(
        active_consultation["consultation_id"],
        [
            ("PATIENT", "I have had fever and cough for two days."),  # strong match
            ("DOCTOR", "Okay understood."),  # weak/no match
        ],
    )
    soap_note_id = _create_soap_note_with_claims(
        active_consultation["consultation_id"],
        transcript_id,
        [("SUBJECTIVE", "Patient reports fever and cough for two days.")],
    )

    response = _retrieve(
        active_consultation["consultation_id"], soap_note_id, active_consultation["token"]
    )
    evidence = response.json()["per_claim"][0]["evidence"]
    retrieved_ids = {e["speaker_segment_id"] for e in evidence}
    assert segment_ids[0] in retrieved_ids
    assert segment_ids[1] not in retrieved_ids  # below EVIDENCE_MIN_SCORE


def test_no_candidate_above_threshold_returns_zero_evidence_links(active_consultation):
    transcript_id, _ = _create_transcript_with_segments(
        active_consultation["consultation_id"], [("DOCTOR", "How are you feeling today?")]
    )
    soap_note_id = _create_soap_note_with_claims(
        active_consultation["consultation_id"], transcript_id, [("PLAN", "Doctor advised paracetamol.")]
    )

    response = _retrieve(
        active_consultation["consultation_id"], soap_note_id, active_consultation["token"]
    )
    body = response.json()
    assert body["evidence_links_created"] == 0
    assert body["per_claim"][0]["evidence"] == []


def test_patient_role_preference_for_subjective_claims(active_consultation):
    # Identical text, different roles -- PATIENT segment should score higher
    # for a SUBJECTIVE claim (role is a ranking signal, not a filter).
    transcript_id, segment_ids = _create_transcript_with_segments(
        active_consultation["consultation_id"],
        [
            ("DOCTOR", "I have had fever and cough for two days."),
            ("PATIENT", "I have had fever and cough for two days."),
        ],
    )
    soap_note_id = _create_soap_note_with_claims(
        active_consultation["consultation_id"],
        transcript_id,
        [("SUBJECTIVE", "Patient reports fever and cough for two days.")],
    )

    response = _retrieve(
        active_consultation["consultation_id"], soap_note_id, active_consultation["token"]
    )
    evidence = response.json()["per_claim"][0]["evidence"]
    scores_by_segment = {e["speaker_segment_id"]: e["retrieval_score"] for e in evidence}
    assert scores_by_segment[segment_ids[1]] > scores_by_segment[segment_ids[0]]  # PATIENT > DOCTOR
    assert evidence[0]["speaker_segment_id"] == segment_ids[1]


def test_doctor_role_preference_for_assessment_claims(active_consultation):
    transcript_id, segment_ids = _create_transcript_with_segments(
        active_consultation["consultation_id"],
        [
            ("PATIENT", "This appears to be a viral infection."),
            ("DOCTOR", "This appears to be a viral infection."),
        ],
    )
    soap_note_id = _create_soap_note_with_claims(
        active_consultation["consultation_id"],
        transcript_id,
        [("ASSESSMENT", "Doctor assessed the condition as a viral infection.")],
    )

    response = _retrieve(
        active_consultation["consultation_id"], soap_note_id, active_consultation["token"]
    )
    evidence = response.json()["per_claim"][0]["evidence"]
    scores_by_segment = {e["speaker_segment_id"]: e["retrieval_score"] for e in evidence}
    assert scores_by_segment[segment_ids[1]] > scores_by_segment[segment_ids[0]]  # DOCTOR > PATIENT


def test_unknown_speakers_remain_eligible(active_consultation):
    transcript_id, segment_ids = _create_transcript_with_segments(
        active_consultation["consultation_id"],
        [("UNKNOWN", "I have had fever and cough for two days.")],
    )
    soap_note_id = _create_soap_note_with_claims(
        active_consultation["consultation_id"],
        transcript_id,
        [("SUBJECTIVE", "Patient reports fever and cough for two days.")],
    )

    response = _retrieve(
        active_consultation["consultation_id"], soap_note_id, active_consultation["token"]
    )
    evidence = response.json()["per_claim"][0]["evidence"]
    assert len(evidence) == 1  # not excluded for being UNKNOWN
    assert evidence[0]["speaker_segment_id"] == segment_ids[0]


def test_negated_segments_are_not_incorrectly_filtered_out(active_consultation):
    # CORE research principle: retrieval is lexical only. A negated segment
    # must still be retrieved for a claim that shares its vocabulary --
    # Step 14A does not decide whether it supports or contradicts the claim.
    transcript_id, segment_ids = _create_transcript_with_segments(
        active_consultation["consultation_id"], [("PATIENT", "I don't have fever.")]
    )
    soap_note_id = _create_soap_note_with_claims(
        active_consultation["consultation_id"], transcript_id, [("SUBJECTIVE", "Patient reports fever.")]
    )

    response = _retrieve(
        active_consultation["consultation_id"], soap_note_id, active_consultation["token"]
    )
    evidence = response.json()["per_claim"][0]["evidence"]
    assert len(evidence) == 1
    assert evidence[0]["speaker_segment_id"] == segment_ids[0]


def test_retrieval_score_is_between_zero_and_one(active_consultation):
    transcript_id, _ = _create_transcript_with_segments(
        active_consultation["consultation_id"],
        [("PATIENT", "I have had fever and cough for two days.")],
    )
    soap_note_id = _create_soap_note_with_claims(
        active_consultation["consultation_id"],
        transcript_id,
        [("SUBJECTIVE", "Patient reports fever for two days.")],
    )

    response = _retrieve(
        active_consultation["consultation_id"], soap_note_id, active_consultation["token"]
    )
    for claim in response.json()["per_claim"]:
        for evidence in claim["evidence"]:
            assert 0.0 <= evidence["retrieval_score"] <= 1.0


def test_evidence_links_are_created_correctly(active_consultation):
    transcript_id, segment_ids = _create_transcript_with_segments(
        active_consultation["consultation_id"],
        [("PATIENT", "I have had fever and cough for two days.")],
    )
    soap_note_id = _create_soap_note_with_claims(
        active_consultation["consultation_id"],
        transcript_id,
        [("SUBJECTIVE", "Patient reports fever for two days.")],
    )

    response = _retrieve(
        active_consultation["consultation_id"], soap_note_id, active_consultation["token"]
    )
    claim_id = response.json()["per_claim"][0]["claim_id"]

    db = SessionLocal()
    try:
        link = (
            db.query(EvidenceLink)
            .filter(EvidenceLink.soap_claim_id == uuid.UUID(claim_id))
            .first()
        )
        assert link is not None
        assert str(link.speaker_segment_id) == segment_ids[0]
        assert link.relationship_type == "candidate"
        assert link.alignment_score is not None
        assert 0.0 <= float(link.alignment_score) <= 1.0
    finally:
        db.close()


def test_repeated_retrieval_does_not_create_duplicates(active_consultation):
    transcript_id, _ = _create_transcript_with_segments(
        active_consultation["consultation_id"],
        [("PATIENT", "I have had fever and cough for two days.")],
    )
    soap_note_id = _create_soap_note_with_claims(
        active_consultation["consultation_id"],
        transcript_id,
        [("SUBJECTIVE", "Patient reports fever for two days.")],
    )

    first = _retrieve(
        active_consultation["consultation_id"], soap_note_id, active_consultation["token"]
    )
    assert first.status_code == 200
    second = _retrieve(
        active_consultation["consultation_id"], soap_note_id, active_consultation["token"]
    )
    assert second.status_code == 200
    assert first.json()["evidence_links_created"] == second.json()["evidence_links_created"]

    db = SessionLocal()
    try:
        claim_id = uuid.UUID(first.json()["per_claim"][0]["claim_id"])
        count = db.query(EvidenceLink).filter(EvidenceLink.soap_claim_id == claim_id).count()
    finally:
        db.close()

    assert count == first.json()["evidence_links_created"]  # not doubled


def test_no_nli_fields_populated(active_consultation):
    # Step 14A (evidence retrieval) must never itself decide or store any
    # NLI verification outcome. The entailment/contradiction/nli_label
    # columns exist on evidence_links (added in Step 14B) but retrieval
    # alone must leave them unset; only running verification populates them.
    transcript_id, _ = _create_transcript_with_segments(
        active_consultation["consultation_id"],
        [("PATIENT", "I have had fever and cough for two days.")],
    )
    soap_note_id = _create_soap_note_with_claims(
        active_consultation["consultation_id"],
        transcript_id,
        [("SUBJECTIVE", "Patient reports fever for two days.")],
    )

    response = _retrieve(
        active_consultation["consultation_id"], soap_note_id, active_consultation["token"]
    )
    assert response.json()["evidence_links_created"] >= 1

    db = SessionLocal()
    try:
        claim_id = uuid.UUID(response.json()["per_claim"][0]["claim_id"])
        links = db.query(EvidenceLink).filter(EvidenceLink.soap_claim_id == claim_id).all()
        assert links  # this test's own claim produced at least one evidence link
        for link in links:
            assert link.relationship_type == "candidate"  # never supports/contradicts/insufficient
            assert link.entailment_score is None
            assert link.contradiction_score is None
            assert link.nli_label is None
            assert link.verification_status is None
        column_names = {c.name for c in EvidenceLink.__table__.columns}
        assert "hallucination_status" not in column_names
    finally:
        db.close()


def test_retrieval_failure_is_represented_safely(active_consultation, monkeypatch):
    transcript_id, _ = _create_transcript_with_segments(
        active_consultation["consultation_id"],
        [("PATIENT", "I have had fever and cough for two days.")],
    )
    soap_note_id = _create_soap_note_with_claims(
        active_consultation["consultation_id"],
        transcript_id,
        [("SUBJECTIVE", "Patient reports fever for two days.")],
    )

    def _raise(*_args, **_kwargs):
        raise RuntimeError("simulated unexpected retrieval error")

    monkeypatch.setattr("app.evidence.service.retrieve_evidence", _raise)

    response = _retrieve(
        active_consultation["consultation_id"], soap_note_id, active_consultation["token"]
    )
    assert response.status_code == 500
    assert "traceback" not in response.text.lower()
    assert str(REPO_ROOT) not in response.text
