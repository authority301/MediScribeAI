"""NLI evidence verification (Step 14B) tests: ownership, status gating,
deterministic claim-level aggregation, conflicting-evidence handling,
idempotency (update-in-place, never duplicate), and the research-integrity
requirement that negated evidence text reaches NLI unmodified.

Automated tests NEVER load the real mDeBERTa model -- run_nli is
monkeypatched everywhere a verification actually needs to "succeed". All
conversation text is synthetic, never real patient data.
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
from app.nli.model import NliScores

client = TestClient(app)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture(scope="module", autouse=True)
def _ensure_schema():
    init_db()


def _register_and_login(name="Dr. NLI"):
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
    db = SessionLocal()
    try:
        full_text = " ".join(text for _role, text in turns)
        transcript = Transcript(
            consultation_id=uuid.UUID(consultation_id),
            full_text=full_text,
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


def _create_soap_note_with_claims(
    consultation_id: str, transcript_id: str, claims: list, version: int = 1
) -> tuple:
    """claims: list of (section, claim_text). Returns (soap_note_id, [claim_id, ...])."""
    db = SessionLocal()
    try:
        soap_note = SOAPNote(
            consultation_id=uuid.UUID(consultation_id),
            transcript_id=uuid.UUID(transcript_id) if transcript_id else None,
            version=version,
            status="generated",
            generation_status="completed",
        )
        db.add(soap_note)
        db.commit()
        db.refresh(soap_note)
        claim_ids = []
        for index, (section, text) in enumerate(claims):
            claim = SOAPClaim(
                soap_note_id=soap_note.id, section=section, claim_text=text, sequence_index=index
            )
            db.add(claim)
            db.commit()
            db.refresh(claim)
            claim_ids.append(str(claim.id))
        return str(soap_note.id), claim_ids
    finally:
        db.close()


def _create_evidence_link(claim_id: str, segment_id: str, alignment_score: float = 0.6) -> str:
    db = SessionLocal()
    try:
        link = EvidenceLink(
            soap_claim_id=uuid.UUID(claim_id),
            speaker_segment_id=uuid.UUID(segment_id),
            relationship_type="candidate",
            alignment_score=alignment_score,
        )
        db.add(link)
        db.commit()
        db.refresh(link)
        return str(link.id)
    finally:
        db.close()


def _mock_run_nli_factory(responses_by_premise: dict):
    """responses_by_premise: dict[premise_text] -> NliScores."""

    def _mock(premise, hypothesis):
        if premise in responses_by_premise:
            return responses_by_premise[premise]
        return NliScores(entailment=0.1, neutral=0.8, contradiction=0.1, label="NEUTRAL")

    return _mock


@pytest.fixture()
def active_consultation():
    email, token = _register_and_login()
    create_response = client.post(
        "/consultations", json={"patient_reference": "PATIENT-NLI"}, headers=_headers(token)
    )
    consultation_id = create_response.json()["id"]
    client.patch(
        f"/consultations/{consultation_id}/status",
        json={"status": "active"},
        headers=_headers(token),
    )
    yield {"token": token, "consultation_id": consultation_id, "email": email}
    _cleanup_doctor(email)


def _verify(consultation_id, soap_note_id, token):
    return client.post(
        f"/consultations/{consultation_id}/soap/evidence/verify",
        json={"soap_note_id": soap_note_id},
        headers=_headers(token),
    )


# ---------------------------------------------------------------------------
# Ownership / precondition tests
# ---------------------------------------------------------------------------


def test_verify_requires_authentication(active_consultation):
    transcript_id, segment_ids = _create_transcript_with_segments(
        active_consultation["consultation_id"], [("PATIENT", "I have fever.")]
    )
    soap_note_id, claim_ids = _create_soap_note_with_claims(
        active_consultation["consultation_id"], transcript_id, [("SUBJECTIVE", "Patient reports fever.")]
    )
    _create_evidence_link(claim_ids[0], segment_ids[0])

    response = client.post(
        f"/consultations/{active_consultation['consultation_id']}/soap/evidence/verify",
        json={"soap_note_id": soap_note_id},
    )
    assert response.status_code == 401


def test_verify_unknown_consultation_returns_404(active_consultation):
    transcript_id, segment_ids = _create_transcript_with_segments(
        active_consultation["consultation_id"], [("PATIENT", "I have fever.")]
    )
    soap_note_id, claim_ids = _create_soap_note_with_claims(
        active_consultation["consultation_id"], transcript_id, [("SUBJECTIVE", "Patient reports fever.")]
    )
    _create_evidence_link(claim_ids[0], segment_ids[0])

    response = _verify(str(uuid.uuid4()), soap_note_id, active_consultation["token"])
    assert response.status_code == 404


def test_verify_another_doctors_consultation_returns_404(active_consultation):
    transcript_id, segment_ids = _create_transcript_with_segments(
        active_consultation["consultation_id"], [("PATIENT", "I have fever.")]
    )
    soap_note_id, claim_ids = _create_soap_note_with_claims(
        active_consultation["consultation_id"], transcript_id, [("SUBJECTIVE", "Patient reports fever.")]
    )
    _create_evidence_link(claim_ids[0], segment_ids[0])

    other_email, other_token = _register_and_login(name="Dr. Other")
    response = _verify(active_consultation["consultation_id"], soap_note_id, other_token)
    assert response.status_code == 404
    _cleanup_doctor(other_email)


def test_verify_draft_consultation_returns_409():
    email, token = _register_and_login(name="Dr. Draft")
    create_response = client.post(
        "/consultations", json={"patient_reference": "PATIENT-DRAFT"}, headers=_headers(token)
    )
    consultation_id = create_response.json()["id"]

    response = _verify(consultation_id, str(uuid.uuid4()), token)
    assert response.status_code == 409
    _cleanup_doctor(email)


def test_verify_completed_consultation_returns_409():
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

    response = _verify(consultation_id, str(uuid.uuid4()), token)
    assert response.status_code == 409
    _cleanup_doctor(email)


def test_verify_cancelled_consultation_returns_409():
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

    response = _verify(consultation_id, str(uuid.uuid4()), token)
    assert response.status_code == 409
    _cleanup_doctor(email)


def test_verify_unknown_soap_note_returns_404(active_consultation):
    response = _verify(
        active_consultation["consultation_id"], str(uuid.uuid4()), active_consultation["token"]
    )
    assert response.status_code == 404


def test_verify_soap_note_belonging_to_another_consultation_returns_404(active_consultation):
    token = active_consultation["token"]
    transcript_id, segment_ids = _create_transcript_with_segments(
        active_consultation["consultation_id"], [("PATIENT", "I have fever.")]
    )
    soap_note_id, claim_ids = _create_soap_note_with_claims(
        active_consultation["consultation_id"], transcript_id, [("SUBJECTIVE", "Patient reports fever.")]
    )
    _create_evidence_link(claim_ids[0], segment_ids[0])

    create_response = client.post(
        "/consultations", json={"patient_reference": "PATIENT-SECOND"}, headers=_headers(token)
    )
    second_consultation_id = create_response.json()["id"]
    client.patch(
        f"/consultations/{second_consultation_id}/status",
        json={"status": "active"},
        headers=_headers(token),
    )

    response = _verify(second_consultation_id, soap_note_id, token)
    assert response.status_code == 404


def test_verify_soap_generation_not_completed_returns_409(active_consultation):
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

    response = _verify(
        active_consultation["consultation_id"], soap_note_id, active_consultation["token"]
    )
    assert response.status_code == 409


def test_verify_no_evidence_links_returns_409(active_consultation):
    transcript_id, _ = _create_transcript_with_segments(
        active_consultation["consultation_id"], [("PATIENT", "I have fever.")]
    )
    soap_note_id, _ = _create_soap_note_with_claims(
        active_consultation["consultation_id"], transcript_id, [("SUBJECTIVE", "Patient reports fever.")]
    )
    # no evidence_links created -- Step 14A was never run

    response = _verify(
        active_consultation["consultation_id"], soap_note_id, active_consultation["token"]
    )
    assert response.status_code == 409


# ---------------------------------------------------------------------------
# Core NLI aggregation behavior (mocked inference)
# ---------------------------------------------------------------------------


def test_basic_entailment_yields_supported(active_consultation, monkeypatch):
    premise = "I have had fever for two days."
    transcript_id, segment_ids = _create_transcript_with_segments(
        active_consultation["consultation_id"], [("PATIENT", premise)]
    )
    soap_note_id, claim_ids = _create_soap_note_with_claims(
        active_consultation["consultation_id"],
        transcript_id,
        [("SUBJECTIVE", "Patient reports fever for two days.")],
    )
    _create_evidence_link(claim_ids[0], segment_ids[0])

    monkeypatch.setattr(
        "app.nli.service.run_nli",
        _mock_run_nli_factory(
            {premise: NliScores(entailment=0.94, neutral=0.04, contradiction=0.02, label="ENTAILMENT")}
        ),
    )

    response = _verify(
        active_consultation["consultation_id"], soap_note_id, active_consultation["token"]
    )
    assert response.status_code == 200
    body = response.json()
    assert body["supported"] == 1
    assert body["contradicted"] == 0
    assert body["ungrounded"] == 0
    assert body["per_claim"][0]["verification_status"] == "SUPPORTED"
    assert body["per_claim"][0]["evidence"][0]["nli_label"] == "ENTAILMENT"


def test_basic_contradiction_yields_contradicted(active_consultation, monkeypatch):
    premise = "I don't have fever."
    transcript_id, segment_ids = _create_transcript_with_segments(
        active_consultation["consultation_id"], [("PATIENT", premise)]
    )
    soap_note_id, claim_ids = _create_soap_note_with_claims(
        active_consultation["consultation_id"], transcript_id, [("SUBJECTIVE", "Patient reports fever.")]
    )
    _create_evidence_link(claim_ids[0], segment_ids[0])

    monkeypatch.setattr(
        "app.nli.service.run_nli",
        _mock_run_nli_factory(
            {premise: NliScores(entailment=0.03, neutral=0.09, contradiction=0.88, label="CONTRADICTION")}
        ),
    )

    response = _verify(
        active_consultation["consultation_id"], soap_note_id, active_consultation["token"]
    )
    body = response.json()
    assert body["contradicted"] == 1
    assert body["per_claim"][0]["verification_status"] == "CONTRADICTED"


def test_basic_neutral_yields_ungrounded(active_consultation, monkeypatch):
    premise = "I have fever and cough."
    transcript_id, segment_ids = _create_transcript_with_segments(
        active_consultation["consultation_id"], [("PATIENT", premise)]
    )
    soap_note_id, claim_ids = _create_soap_note_with_claims(
        active_consultation["consultation_id"], transcript_id, [("ASSESSMENT", "Patient has pneumonia.")]
    )
    _create_evidence_link(claim_ids[0], segment_ids[0])

    monkeypatch.setattr(
        "app.nli.service.run_nli",
        _mock_run_nli_factory(
            {premise: NliScores(entailment=0.15, neutral=0.75, contradiction=0.10, label="NEUTRAL")}
        ),
    )

    response = _verify(
        active_consultation["consultation_id"], soap_note_id, active_consultation["token"]
    )
    body = response.json()
    assert body["ungrounded"] == 1
    assert body["per_claim"][0]["verification_status"] == "UNGROUNDED"


def test_entailment_below_threshold_is_not_supported(active_consultation, monkeypatch):
    premise = "I have had fever for two days."
    transcript_id, segment_ids = _create_transcript_with_segments(
        active_consultation["consultation_id"], [("PATIENT", premise)]
    )
    soap_note_id, claim_ids = _create_soap_note_with_claims(
        active_consultation["consultation_id"],
        transcript_id,
        [("SUBJECTIVE", "Patient reports fever for two days.")],
    )
    _create_evidence_link(claim_ids[0], segment_ids[0])

    # ENTAILMENT wins but below NLI_ENTAILMENT_THRESHOLD (default 0.70)
    monkeypatch.setattr(
        "app.nli.service.run_nli",
        _mock_run_nli_factory(
            {premise: NliScores(entailment=0.55, neutral=0.35, contradiction=0.10, label="ENTAILMENT")}
        ),
    )

    response = _verify(
        active_consultation["consultation_id"], soap_note_id, active_consultation["token"]
    )
    body = response.json()
    assert body["per_claim"][0]["verification_status"] == "UNGROUNDED"


def test_contradiction_below_threshold_is_not_contradicted(active_consultation, monkeypatch):
    premise = "I don't have fever."
    transcript_id, segment_ids = _create_transcript_with_segments(
        active_consultation["consultation_id"], [("PATIENT", premise)]
    )
    soap_note_id, claim_ids = _create_soap_note_with_claims(
        active_consultation["consultation_id"], transcript_id, [("SUBJECTIVE", "Patient reports fever.")]
    )
    _create_evidence_link(claim_ids[0], segment_ids[0])

    monkeypatch.setattr(
        "app.nli.service.run_nli",
        _mock_run_nli_factory(
            {premise: NliScores(entailment=0.10, neutral=0.40, contradiction=0.50, label="CONTRADICTION")}
        ),
    )

    response = _verify(
        active_consultation["consultation_id"], soap_note_id, active_consultation["token"]
    )
    body = response.json()
    assert body["per_claim"][0]["verification_status"] == "UNGROUNDED"


def test_conflicting_evidence_yields_ungrounded(active_consultation, monkeypatch):
    premise_support = "I have fever."
    premise_contradict = "I don't have fever."
    transcript_id, segment_ids = _create_transcript_with_segments(
        active_consultation["consultation_id"],
        [("PATIENT", premise_support), ("PATIENT", premise_contradict)],
    )
    soap_note_id, claim_ids = _create_soap_note_with_claims(
        active_consultation["consultation_id"], transcript_id, [("SUBJECTIVE", "Patient reports fever.")]
    )
    _create_evidence_link(claim_ids[0], segment_ids[0])
    _create_evidence_link(claim_ids[0], segment_ids[1])

    monkeypatch.setattr(
        "app.nli.service.run_nli",
        _mock_run_nli_factory(
            {
                premise_support: NliScores(
                    entailment=0.91, neutral=0.06, contradiction=0.03, label="ENTAILMENT"
                ),
                premise_contradict: NliScores(
                    entailment=0.05, neutral=0.07, contradiction=0.88, label="CONTRADICTION"
                ),
            }
        ),
    )

    response = _verify(
        active_consultation["consultation_id"], soap_note_id, active_consultation["token"]
    )
    body = response.json()
    # conservative rule: conflicting qualifying evidence -> UNGROUNDED, never
    # a silent majority vote and never a new public label
    assert body["per_claim"][0]["verification_status"] == "UNGROUNDED"
    assert len(body["per_claim"][0]["evidence"]) == 2
    labels = {e["nli_label"] for e in body["per_claim"][0]["evidence"]}
    assert labels == {"ENTAILMENT", "CONTRADICTION"}  # individual results preserved


def test_multiple_evidence_segments_all_individually_verified(active_consultation, monkeypatch):
    premise1 = "I have had fever for two days."
    premise2 = "The fever started two days ago."
    transcript_id, segment_ids = _create_transcript_with_segments(
        active_consultation["consultation_id"], [("PATIENT", premise1), ("PATIENT", premise2)]
    )
    soap_note_id, claim_ids = _create_soap_note_with_claims(
        active_consultation["consultation_id"],
        transcript_id,
        [("SUBJECTIVE", "Patient reports fever for two days.")],
    )
    _create_evidence_link(claim_ids[0], segment_ids[0])
    _create_evidence_link(claim_ids[0], segment_ids[1])

    monkeypatch.setattr(
        "app.nli.service.run_nli",
        _mock_run_nli_factory(
            {
                premise1: NliScores(entailment=0.90, neutral=0.07, contradiction=0.03, label="ENTAILMENT"),
                premise2: NliScores(entailment=0.85, neutral=0.10, contradiction=0.05, label="ENTAILMENT"),
            }
        ),
    )

    response = _verify(
        active_consultation["consultation_id"], soap_note_id, active_consultation["token"]
    )
    body = response.json()
    assert body["evidence_links_verified"] == 2
    assert len(body["per_claim"][0]["evidence"]) == 2
    assert body["per_claim"][0]["verification_status"] == "SUPPORTED"


def test_raw_probabilities_stored_and_valid(active_consultation, monkeypatch):
    premise = "I have had fever for two days."
    transcript_id, segment_ids = _create_transcript_with_segments(
        active_consultation["consultation_id"], [("PATIENT", premise)]
    )
    soap_note_id, claim_ids = _create_soap_note_with_claims(
        active_consultation["consultation_id"],
        transcript_id,
        [("SUBJECTIVE", "Patient reports fever for two days.")],
    )
    link_id = _create_evidence_link(claim_ids[0], segment_ids[0])

    monkeypatch.setattr(
        "app.nli.service.run_nli",
        _mock_run_nli_factory(
            {premise: NliScores(entailment=0.94, neutral=0.04, contradiction=0.02, label="ENTAILMENT")}
        ),
    )

    _verify(active_consultation["consultation_id"], soap_note_id, active_consultation["token"])

    db = SessionLocal()
    try:
        link = db.query(EvidenceLink).filter(EvidenceLink.id == uuid.UUID(link_id)).first()
        assert float(link.entailment_score) == pytest.approx(0.94)
        assert float(link.neutral_score) == pytest.approx(0.04)
        assert float(link.contradiction_score) == pytest.approx(0.02)
        assert link.nli_label == "ENTAILMENT"
        for score in (link.entailment_score, link.neutral_score, link.contradiction_score):
            assert 0.0 <= float(score) <= 1.0
        total = float(link.entailment_score) + float(link.neutral_score) + float(link.contradiction_score)
        assert total == pytest.approx(1.0, abs=0.01)
    finally:
        db.close()


def test_relationship_type_updated_and_alignment_score_preserved(active_consultation, monkeypatch):
    premise = "I don't have fever."
    transcript_id, segment_ids = _create_transcript_with_segments(
        active_consultation["consultation_id"], [("PATIENT", premise)]
    )
    soap_note_id, claim_ids = _create_soap_note_with_claims(
        active_consultation["consultation_id"], transcript_id, [("SUBJECTIVE", "Patient reports fever.")]
    )
    link_id = _create_evidence_link(claim_ids[0], segment_ids[0], alignment_score=0.633)

    monkeypatch.setattr(
        "app.nli.service.run_nli",
        _mock_run_nli_factory(
            {premise: NliScores(entailment=0.03, neutral=0.09, contradiction=0.88, label="CONTRADICTION")}
        ),
    )

    _verify(active_consultation["consultation_id"], soap_note_id, active_consultation["token"])

    db = SessionLocal()
    try:
        link = db.query(EvidenceLink).filter(EvidenceLink.id == uuid.UUID(link_id)).first()
        assert link.relationship_type == "contradicts"  # updated from "candidate"
        assert float(link.alignment_score) == pytest.approx(0.633)  # Step 14A score untouched
        assert link.verification_status == "CONTRADICTED"
    finally:
        db.close()


def test_repeated_verification_does_not_duplicate_links(active_consultation, monkeypatch):
    premise = "I have had fever for two days."
    transcript_id, segment_ids = _create_transcript_with_segments(
        active_consultation["consultation_id"], [("PATIENT", premise)]
    )
    soap_note_id, claim_ids = _create_soap_note_with_claims(
        active_consultation["consultation_id"],
        transcript_id,
        [("SUBJECTIVE", "Patient reports fever for two days.")],
    )
    link_id = _create_evidence_link(claim_ids[0], segment_ids[0])

    monkeypatch.setattr(
        "app.nli.service.run_nli",
        _mock_run_nli_factory(
            {premise: NliScores(entailment=0.94, neutral=0.04, contradiction=0.02, label="ENTAILMENT")}
        ),
    )

    first = _verify(active_consultation["consultation_id"], soap_note_id, active_consultation["token"])
    assert first.status_code == 200
    second = _verify(active_consultation["consultation_id"], soap_note_id, active_consultation["token"])
    assert second.status_code == 200
    assert first.json()["evidence_links_verified"] == second.json()["evidence_links_verified"] == 1

    db = SessionLocal()
    try:
        count = db.query(EvidenceLink).filter(EvidenceLink.soap_claim_id == uuid.UUID(claim_ids[0])).count()
        still_same_row = db.query(EvidenceLink).filter(EvidenceLink.id == uuid.UUID(link_id)).first()
    finally:
        db.close()
    assert count == 1  # not duplicated
    assert still_same_row is not None  # same row updated in place, never deleted/recreated


def test_negated_evidence_text_reaches_nli_unmodified(active_consultation, monkeypatch):
    premise = "I don't have fever."
    transcript_id, segment_ids = _create_transcript_with_segments(
        active_consultation["consultation_id"], [("PATIENT", premise)]
    )
    soap_note_id, claim_ids = _create_soap_note_with_claims(
        active_consultation["consultation_id"], transcript_id, [("SUBJECTIVE", "Patient reports fever.")]
    )
    _create_evidence_link(claim_ids[0], segment_ids[0])

    captured_premises = []

    def _capture(premise, hypothesis):
        captured_premises.append(premise)
        return NliScores(entailment=0.03, neutral=0.09, contradiction=0.88, label="CONTRADICTION")

    monkeypatch.setattr("app.nli.service.run_nli", _capture)

    response = _verify(
        active_consultation["consultation_id"], soap_note_id, active_consultation["token"]
    )
    assert response.status_code == 200
    # the original, unmodified negated segment text was passed as premise --
    # never filtered, rewritten, or reversed with the hypothesis
    assert captured_premises == [premise]


def test_no_hallucination_fields_added(active_consultation, monkeypatch):
    premise = "I have had fever for two days."
    transcript_id, segment_ids = _create_transcript_with_segments(
        active_consultation["consultation_id"], [("PATIENT", premise)]
    )
    soap_note_id, claim_ids = _create_soap_note_with_claims(
        active_consultation["consultation_id"],
        transcript_id,
        [("SUBJECTIVE", "Patient reports fever for two days.")],
    )
    _create_evidence_link(claim_ids[0], segment_ids[0])

    monkeypatch.setattr(
        "app.nli.service.run_nli",
        _mock_run_nli_factory(
            {premise: NliScores(entailment=0.94, neutral=0.04, contradiction=0.02, label="ENTAILMENT")}
        ),
    )

    response = _verify(
        active_consultation["consultation_id"], soap_note_id, active_consultation["token"]
    )
    body = response.json()
    assert "hallucination_status" not in body
    assert "hallucination_rate" not in body

    column_names = {c.name for c in EvidenceLink.__table__.columns}
    assert "hallucination_status" not in column_names
    assert "hallucination_rate" not in column_names


def test_verification_failure_is_represented_safely(active_consultation, monkeypatch):
    premise = "I have had fever for two days."
    transcript_id, segment_ids = _create_transcript_with_segments(
        active_consultation["consultation_id"], [("PATIENT", premise)]
    )
    soap_note_id, claim_ids = _create_soap_note_with_claims(
        active_consultation["consultation_id"],
        transcript_id,
        [("SUBJECTIVE", "Patient reports fever for two days.")],
    )
    _create_evidence_link(claim_ids[0], segment_ids[0])

    def _raise(*_args, **_kwargs):
        raise RuntimeError("simulated unexpected NLI failure")

    monkeypatch.setattr("app.nli.service.run_nli", _raise)

    response = _verify(
        active_consultation["consultation_id"], soap_note_id, active_consultation["token"]
    )
    assert response.status_code == 500
    assert "traceback" not in response.text.lower()
    assert str(REPO_ROOT) not in response.text

    db = SessionLocal()
    try:
        note = db.query(SOAPNote).filter(SOAPNote.id == uuid.UUID(soap_note_id)).first()
        assert note.evidence_verification_status == "failed"
        assert note.evidence_verification_error is not None
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Tanglish pipeline tests (mocked -- these test PIPELINE BEHAVIOR, not real
# model accuracy on Tanglish; see manual verification for the real model).
# ---------------------------------------------------------------------------


def test_tanglish_supported_case_pipeline(active_consultation, monkeypatch):
    premise = "Enakku fever rendu naala irukku."
    transcript_id, segment_ids = _create_transcript_with_segments(
        active_consultation["consultation_id"], [("PATIENT", premise)]
    )
    soap_note_id, claim_ids = _create_soap_note_with_claims(
        active_consultation["consultation_id"],
        transcript_id,
        [("SUBJECTIVE", "Patient has had fever for two days.")],
    )
    _create_evidence_link(claim_ids[0], segment_ids[0])

    monkeypatch.setattr(
        "app.nli.service.run_nli",
        _mock_run_nli_factory(
            {premise: NliScores(entailment=0.85, neutral=0.10, contradiction=0.05, label="ENTAILMENT")}
        ),
    )

    response = _verify(
        active_consultation["consultation_id"], soap_note_id, active_consultation["token"]
    )
    body = response.json()
    assert body["per_claim"][0]["verification_status"] == "SUPPORTED"


def test_tanglish_contradiction_case_pipeline(active_consultation, monkeypatch):
    premise = "Enakku fever illa."
    transcript_id, segment_ids = _create_transcript_with_segments(
        active_consultation["consultation_id"], [("PATIENT", premise)]
    )
    soap_note_id, claim_ids = _create_soap_note_with_claims(
        active_consultation["consultation_id"], transcript_id, [("SUBJECTIVE", "Patient has fever.")]
    )
    _create_evidence_link(claim_ids[0], segment_ids[0])

    monkeypatch.setattr(
        "app.nli.service.run_nli",
        _mock_run_nli_factory(
            {premise: NliScores(entailment=0.04, neutral=0.08, contradiction=0.88, label="CONTRADICTION")}
        ),
    )

    response = _verify(
        active_consultation["consultation_id"], soap_note_id, active_consultation["token"]
    )
    body = response.json()
    assert body["per_claim"][0]["verification_status"] == "CONTRADICTED"


def test_tanglish_ungrounded_case_pipeline(active_consultation, monkeypatch):
    premise = "Enakku cough irukku."
    transcript_id, segment_ids = _create_transcript_with_segments(
        active_consultation["consultation_id"], [("PATIENT", premise)]
    )
    soap_note_id, claim_ids = _create_soap_note_with_claims(
        active_consultation["consultation_id"], transcript_id, [("ASSESSMENT", "Patient has pneumonia.")]
    )
    _create_evidence_link(claim_ids[0], segment_ids[0])

    monkeypatch.setattr(
        "app.nli.service.run_nli",
        _mock_run_nli_factory(
            {premise: NliScores(entailment=0.10, neutral=0.80, contradiction=0.10, label="NEUTRAL")}
        ),
    )

    response = _verify(
        active_consultation["consultation_id"], soap_note_id, active_consultation["token"]
    )
    body = response.json()
    assert body["per_claim"][0]["verification_status"] == "UNGROUNDED"


# ---------------------------------------------------------------------------
# Aggregation isolation / status-field semantics
# ---------------------------------------------------------------------------


def test_claim_level_aggregation_across_multiple_claims_is_independent(active_consultation, monkeypatch):
    premise_a = "I have had fever for two days."
    premise_b = "I don't have a headache."
    transcript_id, segment_ids = _create_transcript_with_segments(
        active_consultation["consultation_id"], [("PATIENT", premise_a), ("PATIENT", premise_b)]
    )
    soap_note_id, claim_ids = _create_soap_note_with_claims(
        active_consultation["consultation_id"],
        transcript_id,
        [
            ("SUBJECTIVE", "Patient reports fever for two days."),
            ("SUBJECTIVE", "Patient reports a headache."),
        ],
    )
    _create_evidence_link(claim_ids[0], segment_ids[0])
    _create_evidence_link(claim_ids[1], segment_ids[1])

    monkeypatch.setattr(
        "app.nli.service.run_nli",
        _mock_run_nli_factory(
            {
                premise_a: NliScores(entailment=0.92, neutral=0.05, contradiction=0.03, label="ENTAILMENT"),
                premise_b: NliScores(
                    entailment=0.02, neutral=0.05, contradiction=0.93, label="CONTRADICTION"
                ),
            }
        ),
    )

    response = _verify(
        active_consultation["consultation_id"], soap_note_id, active_consultation["token"]
    )
    body = response.json()
    assert body["supported"] == 1
    assert body["contradicted"] == 1
    statuses_by_claim = {item["claim_id"]: item["verification_status"] for item in body["per_claim"]}
    assert statuses_by_claim[claim_ids[0]] == "SUPPORTED"
    assert statuses_by_claim[claim_ids[1]] == "CONTRADICTED"


def test_verification_status_denormalized_across_all_links_of_a_claim(active_consultation, monkeypatch):
    # Two links (ENTAILMENT + NEUTRAL, no conflict) for the SAME claim must
    # both carry the identical claim-level aggregate verification_status,
    # even though each retains its own individual nli_label/scores.
    premise1 = "I have had fever for two days."
    premise2 = "Yes doctor, the fever began two days back."
    transcript_id, segment_ids = _create_transcript_with_segments(
        active_consultation["consultation_id"], [("PATIENT", premise1), ("PATIENT", premise2)]
    )
    soap_note_id, claim_ids = _create_soap_note_with_claims(
        active_consultation["consultation_id"],
        transcript_id,
        [("SUBJECTIVE", "Patient reports fever for two days.")],
    )
    link_id_1 = _create_evidence_link(claim_ids[0], segment_ids[0])
    link_id_2 = _create_evidence_link(claim_ids[0], segment_ids[1])

    monkeypatch.setattr(
        "app.nli.service.run_nli",
        _mock_run_nli_factory(
            {
                premise1: NliScores(entailment=0.90, neutral=0.07, contradiction=0.03, label="ENTAILMENT"),
                premise2: NliScores(entailment=0.20, neutral=0.75, contradiction=0.05, label="NEUTRAL"),
            }
        ),
    )

    _verify(active_consultation["consultation_id"], soap_note_id, active_consultation["token"])

    db = SessionLocal()
    try:
        link1 = db.query(EvidenceLink).filter(EvidenceLink.id == uuid.UUID(link_id_1)).first()
        link2 = db.query(EvidenceLink).filter(EvidenceLink.id == uuid.UUID(link_id_2)).first()
    finally:
        db.close()
    assert link1.nli_label == "ENTAILMENT"
    assert link2.nli_label == "NEUTRAL"  # individual label differs...
    assert link1.verification_status == link2.verification_status == "SUPPORTED"  # ...aggregate matches


def test_soap_note_operational_status_set_completed_after_verification(active_consultation, monkeypatch):
    premise = "I have had fever for two days."
    transcript_id, segment_ids = _create_transcript_with_segments(
        active_consultation["consultation_id"], [("PATIENT", premise)]
    )
    soap_note_id, claim_ids = _create_soap_note_with_claims(
        active_consultation["consultation_id"],
        transcript_id,
        [("SUBJECTIVE", "Patient reports fever for two days.")],
    )
    _create_evidence_link(claim_ids[0], segment_ids[0])

    monkeypatch.setattr(
        "app.nli.service.run_nli",
        _mock_run_nli_factory(
            {premise: NliScores(entailment=0.94, neutral=0.04, contradiction=0.02, label="ENTAILMENT")}
        ),
    )

    _verify(active_consultation["consultation_id"], soap_note_id, active_consultation["token"])

    db = SessionLocal()
    try:
        note = db.query(SOAPNote).filter(SOAPNote.id == uuid.UUID(soap_note_id)).first()
        # operational status (did the run succeed) is distinct from the
        # semantic per-claim verification_status stored on evidence_links
        assert note.evidence_verification_status == "completed"
        assert note.evidence_verification_error is None
    finally:
        db.close()


def test_verification_scoped_to_requested_soap_note_only(active_consultation, monkeypatch):
    # Two SOAP notes under the same consultation; verifying one must not
    # touch evidence_links belonging to the other.
    premise = "I have had fever for two days."
    transcript_id, segment_ids = _create_transcript_with_segments(
        active_consultation["consultation_id"], [("PATIENT", premise)]
    )
    soap_note_id_1, claim_ids_1 = _create_soap_note_with_claims(
        active_consultation["consultation_id"],
        transcript_id,
        [("SUBJECTIVE", "Patient reports fever for two days.")],
    )
    soap_note_id_2, claim_ids_2 = _create_soap_note_with_claims(
        active_consultation["consultation_id"],
        transcript_id,
        [("SUBJECTIVE", "Patient reports fever for two days.")],
        version=2,
    )
    _create_evidence_link(claim_ids_1[0], segment_ids[0])
    other_link_id = _create_evidence_link(claim_ids_2[0], segment_ids[0])

    monkeypatch.setattr(
        "app.nli.service.run_nli",
        _mock_run_nli_factory(
            {premise: NliScores(entailment=0.94, neutral=0.04, contradiction=0.02, label="ENTAILMENT")}
        ),
    )

    _verify(active_consultation["consultation_id"], soap_note_id_1, active_consultation["token"])

    db = SessionLocal()
    try:
        untouched_link = db.query(EvidenceLink).filter(EvidenceLink.id == uuid.UUID(other_link_id)).first()
    finally:
        db.close()
    assert untouched_link.nli_label is None
    assert untouched_link.verification_status is None


def test_response_shape_matches_documented_contract(active_consultation, monkeypatch):
    premise = "I have had fever for two days."
    transcript_id, segment_ids = _create_transcript_with_segments(
        active_consultation["consultation_id"], [("PATIENT", premise)]
    )
    soap_note_id, claim_ids = _create_soap_note_with_claims(
        active_consultation["consultation_id"],
        transcript_id,
        [("SUBJECTIVE", "Patient reports fever for two days.")],
    )
    _create_evidence_link(claim_ids[0], segment_ids[0])

    monkeypatch.setattr(
        "app.nli.service.run_nli",
        _mock_run_nli_factory(
            {premise: NliScores(entailment=0.94, neutral=0.04, contradiction=0.02, label="ENTAILMENT")}
        ),
    )

    response = _verify(
        active_consultation["consultation_id"], soap_note_id, active_consultation["token"]
    )
    body = response.json()
    assert set(body.keys()) == {
        "consultation_id",
        "soap_note_id",
        "claims_processed",
        "evidence_links_verified",
        "supported",
        "contradicted",
        "ungrounded",
        "status",
        "per_claim",
    }
    assert body["status"] == "completed"
    assert body["consultation_id"] == active_consultation["consultation_id"]
    assert body["soap_note_id"] == soap_note_id
    assert body["claims_processed"] == 1
