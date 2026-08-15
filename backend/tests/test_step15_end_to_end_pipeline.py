"""Step 15: automated end-to-end pipeline integration test.

Exercises the REAL route/service/aggregation/scoring code for every stage
(audio ingestion, Fog normalization, speaker-role heuristic scoring,
medical entity extraction, SOAP generation, evidence retrieval, evidence
verification aggregation, Tanglish-aware preprocessing wiring) chained
together in one flow. Only the two genuinely expensive/model-dependent
calls are mocked for deterministic CI coverage, following this project's
established test patterns (see test_asr.py / test_diarization.py /
test_nli.py):
  - app.asr.service.transcribe_file   (real model: Faster-Whisper)
  - app.diarization.service.diarize_file (real model: PyAnnote, additionally
    gated on a Hugging Face credential this test suite must not require)
  - app.nli.service.run_nli / app.tanglish.service.run_nli (real model:
    mDeBERTa)

Fog audio normalization runs for REAL (local ffmpeg, no model, no gated
credential -- same as test_fog_processing.py already does), so this test
also proves a real WAV byte stream survives real decode/resample/export.

The REAL local manual verification (this automated test's complement, not
a replacement for it) exercised every one of these components against
actual audio with the real models where available -- see Step 15's final
report. This test exists for CI regression protection of the WIRING
between stages, not to re-validate model quality.
"""
import io
import struct
import sys
import uuid
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import or_

from app.database import SessionLocal
from app.database.init_db import init_db
from app.diarization.model import DiarizationTurn
from app.main import app
from app.medical_entities.service import PHI_ENTITY_TYPE_PREFIX
from app.models import (
    AudioRecord,
    Consultation,
    Doctor,
    EvidenceLink,
    MedicalEntity,
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


def _register_and_login(name="Dr. Step15 E2E"):
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
        consultation_ids = [c.id for c in db.query(Consultation).filter(Consultation.doctor_id == doctor.id)]
        if consultation_ids:
            audio_records = db.query(AudioRecord).filter(AudioRecord.consultation_id.in_(consultation_ids)).all()
            for audio_record in audio_records:
                for path_str in (audio_record.storage_path, audio_record.processed_storage_path):
                    if path_str:
                        file_path = REPO_ROOT / path_str
                        if file_path.exists():
                            file_path.unlink()

            transcripts = db.query(Transcript).filter(Transcript.consultation_id.in_(consultation_ids)).all()
            transcript_ids = [t.id for t in transcripts]
            soap_notes = db.query(SOAPNote).filter(SOAPNote.consultation_id.in_(consultation_ids)).all()
            soap_note_ids = [n.id for n in soap_notes]
            if soap_note_ids:
                claim_ids = [c.id for c in db.query(SOAPClaim).filter(SOAPClaim.soap_note_id.in_(soap_note_ids))]
                if claim_ids:
                    db.query(EvidenceLink).filter(EvidenceLink.soap_claim_id.in_(claim_ids)).delete(synchronize_session=False)
                db.query(SOAPClaim).filter(SOAPClaim.soap_note_id.in_(soap_note_ids)).delete(synchronize_session=False)
            db.query(SOAPNote).filter(SOAPNote.consultation_id.in_(consultation_ids)).delete(synchronize_session=False)
            if transcript_ids:
                db.query(MedicalEntity).filter(MedicalEntity.transcript_id.in_(transcript_ids)).delete(synchronize_session=False)
            audio_record_ids = [a.id for a in audio_records]
            segment_conditions = [SpeakerSegment.audio_record_id.in_(audio_record_ids)]
            if transcript_ids:
                segment_conditions.append(SpeakerSegment.transcript_id.in_(transcript_ids))
            db.query(SpeakerSegment).filter(or_(*segment_conditions)).delete(synchronize_session=False)
            db.query(Transcript).filter(Transcript.consultation_id.in_(consultation_ids)).delete(synchronize_session=False)
            db.query(AudioRecord).filter(AudioRecord.consultation_id.in_(consultation_ids)).delete(synchronize_session=False)
        db.query(Consultation).filter(Consultation.doctor_id == doctor.id).delete(synchronize_session=False)
        db.query(Doctor).filter(Doctor.id == doctor.id).delete(synchronize_session=False)
        db.commit()
    finally:
        db.close()


def _generate_wav_bytes(duration_seconds=0.5, sample_rate=44100, channels=1, amplitude=8000):
    """Real, valid WAV bytes -- exercises the actual Fog decode/resample/export path."""
    n_frames = int(duration_seconds * sample_rate)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        frames = bytearray()
        for i in range(n_frames):
            value = int(amplitude * ((i % 100) / 100.0 - 0.5))
            frames += struct.pack("<h", value)
        wav_file.writeframes(bytes(frames))
    return buffer.getvalue()


# Deterministic fake ASR output: cue phrases chosen to make Step 15E's real
# heuristic scoring (app/speaker_roles/scoring.py, NOT mocked) confidently
# classify DOCTOR vs PATIENT, and to exercise negation/duration/measurement/
# medication/assessment/plan entity+claim generation.
_FAKE_SEGMENTS = [
    (0, "DOCTOR", "What brings you in today?"),
    (1, "PATIENT", "I have been feeling unwell. I have a fever and a cough for three days."),
    (2, "DOCTOR", "Any chest pain?"),
    (3, "PATIENT", "I don't have chest pain."),
    (4, "DOCTOR", "Your temperature is 101.2 degrees Fahrenheit."),
    (5, "DOCTOR", "This appears to be a viral infection."),
    (6, "DOCTOR", "I'm prescribing paracetamol for the fever."),
    (7, "DOCTOR", "Please return for a follow up in three days."),
]
_SEGMENT_MS_SPAN = 2000  # each fake segment is 2s, non-overlapping, matching its diarization turn 1:1


def _fake_transcribe_file(_path):
    from app.asr.model import TranscriptionResult, TranscriptionSegment

    return TranscriptionResult(
        language="en",
        segments=[
            TranscriptionSegment(
                sequence_index=idx, text=text,
                start_ms=idx * _SEGMENT_MS_SPAN, end_ms=(idx + 1) * _SEGMENT_MS_SPAN,
            )
            for idx, _speaker, text in _FAKE_SEGMENTS
        ],
    )


def _fake_diarize_file(_path):
    # One diarization turn per fake ASR segment, exact temporal match, so
    # the REAL alignment code (app/diarization/alignment.py) assigns each
    # segment its speaker_label with full confidence -- never mocked.
    speaker_for = {"DOCTOR": "SPEAKER_00", "PATIENT": "SPEAKER_01"}
    return [
        DiarizationTurn(
            speaker_label=speaker_for[speaker],
            start_ms=idx * _SEGMENT_MS_SPAN, end_ms=(idx + 1) * _SEGMENT_MS_SPAN,
        )
        for idx, speaker, _text in _FAKE_SEGMENTS
    ]


def _fake_run_nli(premise, hypothesis):
    """Deterministic stand-in for the real mDeBERTa model: label ENTAILMENT
    whenever the hypothesis's distinctive content words are literally
    present in the premise, else NEUTRAL. Exercises the real aggregation
    decision rule (app.nli.aggregation.aggregate_claim_verification) with
    real threshold comparisons -- only the raw model call is faked.
    """
    premise_l = premise.lower()
    hyp_l = hypothesis.lower()
    # Negation-aware check FIRST (must win over the naive substring check
    # below, since "chest pain" is a literal substring of both the negated
    # premise and the claim -- exactly the kind of trap real NLI is
    # supposed to avoid; this fake model deliberately mirrors that).
    if "chest pain" in hyp_l and "don't have chest pain" in premise_l:
        return NliScores(entailment=0.02, neutral=0.03, contradiction=0.95, label="CONTRADICTION")
    # A question ("Any chest pain?") merely raises a topic, it doesn't
    # assert it -- exclude question premises from the naive entailment
    # check below, same reasoning as the negation check above.
    key_terms = [w for w in ("fever", "cough", "chest pain", "paracetamol", "101.2", "degrees") if w in hyp_l]
    if key_terms and not premise.strip().endswith("?") and all(term in premise_l for term in key_terms):
        return NliScores(entailment=0.95, neutral=0.03, contradiction=0.02, label="ENTAILMENT")
    return NliScores(entailment=0.1, neutral=0.85, contradiction=0.05, label="NEUTRAL")


def test_full_pipeline_flows_end_to_end(monkeypatch):
    monkeypatch.setattr("app.asr.service.transcribe_file", _fake_transcribe_file)
    monkeypatch.setattr("app.diarization.service.diarize_file", _fake_diarize_file)
    monkeypatch.setattr("app.nli.service.run_nli", _fake_run_nli)

    email, token = _register_and_login()
    try:
        # --- consultation ---
        r = client.post("/consultations", json={"patient_reference": "STEP15-CI-001"}, headers=_headers(token))
        assert r.status_code == 201
        consultation_id = r.json()["id"]
        r = client.patch(f"/consultations/{consultation_id}/status", json={"status": "active"}, headers=_headers(token))
        assert r.status_code == 200

        # --- 15A: audio ingestion ---
        audio_bytes = _generate_wav_bytes()
        r = client.post(
            f"/consultations/{consultation_id}/audio",
            files={"file": ("recording.webm", audio_bytes, "audio/webm")},
            headers=_headers(token),
        )
        assert r.status_code == 201
        audio_id = r.json()["id"]
        assert r.json()["storage_path"].startswith("uploads/audio/")

        # --- 15B: Fog (real ffmpeg decode/normalize, no mock) ---
        r = client.post(f"/consultations/{consultation_id}/audio/process", json={"audio_id": audio_id}, headers=_headers(token))
        assert r.status_code == 200
        fog_body = r.json()
        assert fog_body["processing_status"] == "completed"
        assert fog_body["sample_rate"] == 16000
        assert fog_body["channels"] == 1

        # --- 15C: ASR (mocked model, real service/DB wiring) ---
        r = client.post(f"/consultations/{consultation_id}/audio/transcribe", json={"audio_id": audio_id}, headers=_headers(token))
        assert r.status_code == 201
        transcript_id = r.json()["transcript_id"]
        assert r.json()["segment_count"] == len(_FAKE_SEGMENTS)

        db = SessionLocal()
        segments_before_diarization = db.query(SpeakerSegment).filter(SpeakerSegment.transcript_id == uuid.UUID(transcript_id)).all()
        assert all(s.speaker_label is None for s in segments_before_diarization)
        db.close()

        # --- 15D: diarization (mocked model, real alignment logic) ---
        r = client.post(f"/consultations/{consultation_id}/audio/diarize", json={"audio_id": audio_id}, headers=_headers(token))
        assert r.status_code == 201
        assert r.json()["speaker_count"] == 2

        # --- 15E: speaker role identification (fully real heuristic scoring) ---
        r = client.post(f"/consultations/{consultation_id}/audio/identify-speakers", json={"audio_id": audio_id}, headers=_headers(token))
        assert r.status_code == 200
        role_body = r.json()
        assert role_body["status"] == "completed"
        roles_by_label = {s["speaker_label"]: s["role"] for s in role_body["speakers"]}
        assert roles_by_label["SPEAKER_00"] == "DOCTOR"
        assert roles_by_label["SPEAKER_01"] == "PATIENT"

        # --- 15F: medical entity extraction (fully real) ---
        r = client.post(f"/consultations/{consultation_id}/audio/extract-entities", json={"transcript_id": transcript_id}, headers=_headers(token))
        assert r.status_code == 200
        entities_body = r.json()
        assert entities_body["medical_entity_count"] > 0
        entity_types = {e["entity_type"] for e in entities_body["entities"]}
        assert "SYMPTOM" in entity_types
        assert "MEDICATION" in entity_types
        assert "MEASUREMENT" in entity_types
        assert "DURATION" in entity_types
        negated_entities = [e for e in entities_body["entities"] if e["negated"]]
        assert any(e["normalized_text"] == "chest pain" for e in negated_entities)

        # --- 15G: SOAP generation (fully real; run twice for idempotency) ---
        r1 = client.post(f"/consultations/{consultation_id}/soap/generate", json={"transcript_id": transcript_id}, headers=_headers(token))
        assert r1.status_code == 201
        r2 = client.post(f"/consultations/{consultation_id}/soap/generate", json={"transcript_id": transcript_id}, headers=_headers(token))
        assert r2.status_code == 201
        assert r1.json()["soap_note_id"] == r2.json()["soap_note_id"]
        soap_note_id = r2.json()["soap_note_id"]
        assert r2.json()["subjective_claim_count"] > 0
        assert r2.json()["objective_claim_count"] > 0
        assert r2.json()["assessment_claim_count"] > 0
        assert r2.json()["plan_claim_count"] > 0

        db = SessionLocal()
        claim_rows = db.query(SOAPClaim).filter(SOAPClaim.soap_note_id == uuid.UUID(soap_note_id)).all()
        note_rows = db.query(SOAPNote).filter(SOAPNote.consultation_id == uuid.UUID(consultation_id)).all()
        assert len(note_rows) == 1, "repeated generation must not create duplicate SOAP notes"
        db.close()

        # --- 15H: evidence retrieval (fully real; run twice for idempotency) ---
        r1 = client.post(f"/consultations/{consultation_id}/soap/evidence/retrieve", json={"soap_note_id": soap_note_id}, headers=_headers(token))
        assert r1.status_code == 200
        r2 = client.post(f"/consultations/{consultation_id}/soap/evidence/retrieve", json={"soap_note_id": soap_note_id}, headers=_headers(token))
        assert r2.status_code == 200
        assert r1.json()["evidence_links_created"] == r2.json()["evidence_links_created"]

        db = SessionLocal()
        claim_ids = [c.id for c in claim_rows]
        evidence_rows = db.query(EvidenceLink).filter(EvidenceLink.soap_claim_id.in_(claim_ids)).all()
        assert len(evidence_rows) == r2.json()["evidence_links_created"], "repeated retrieval must not duplicate evidence links"
        assert all(e.relationship_type == "candidate" for e in evidence_rows)
        db.close()

        # --- 15I: NLI verification (mocked model, real aggregation; run twice) ---
        r1 = client.post(f"/consultations/{consultation_id}/soap/evidence/verify", json={"soap_note_id": soap_note_id}, headers=_headers(token))
        assert r1.status_code == 200
        r2 = client.post(f"/consultations/{consultation_id}/soap/evidence/verify", json={"soap_note_id": soap_note_id}, headers=_headers(token))
        assert r2.status_code == 200
        assert r1.json()["claims_processed"] == r2.json()["claims_processed"]

        db = SessionLocal()
        evidence_rows_after = db.query(EvidenceLink).filter(EvidenceLink.soap_claim_id.in_(claim_ids)).all()
        assert len(evidence_rows_after) == len(evidence_rows), "repeated verification must update in place, not duplicate"
        assert all(e.relationship_type in ("supports", "contradicts", "insufficient") for e in evidence_rows_after)
        assert any(e.verification_status == "SUPPORTED" for e in evidence_rows_after)
        assert any(e.verification_status == "CONTRADICTED" for e in evidence_rows_after)
        db.close()

        # --- end-to-end integrity: FK validity + no orphans ---
        db = SessionLocal()
        audio_row = db.query(AudioRecord).filter(AudioRecord.id == uuid.UUID(audio_id)).first()
        transcript_row = db.query(Transcript).filter(Transcript.id == uuid.UUID(transcript_id)).first()
        assert transcript_row.audio_record_id == audio_row.id
        note_row = db.query(SOAPNote).filter(SOAPNote.id == uuid.UUID(soap_note_id)).first()
        assert note_row.transcript_id == transcript_row.id
        for claim in claim_rows:
            assert claim.soap_note_id == note_row.id
        segment_ids = {s.id for s in db.query(SpeakerSegment).filter(SpeakerSegment.transcript_id == transcript_row.id)}
        for link in evidence_rows_after:
            assert link.soap_claim_id in {c.id for c in claim_rows}
            assert link.speaker_segment_id in segment_ids
        phi_rows = db.query(MedicalEntity).filter(
            MedicalEntity.transcript_id == transcript_row.id,
            MedicalEntity.entity_type.like(f"{PHI_ENTITY_TYPE_PREFIX}%"),
        ).all()
        for phi_row in phi_rows:
            assert phi_row.entity_text.startswith("[") and phi_row.entity_text.endswith("]")
        db.close()

        # --- ownership boundary ---
        other_email, other_token = _register_and_login(name="Dr. Step15 Other")
        try:
            r = client.get(f"/consultations/{consultation_id}", headers=_headers(other_token))
            assert r.status_code == 404
        finally:
            _cleanup_doctor(other_email)
    finally:
        _cleanup_doctor(email)


def test_tanglish_path_wiring(monkeypatch):
    """Step 15J: the Tanglish-aware path is actually reachable and behaves
    differently from baseline -- mocked model (deterministic CI), real
    normalizer + service wiring.
    """
    from app.tanglish import service as tanglish_service

    captured_premises = []

    def _fake_run_nli(premise, hypothesis):
        captured_premises.append(premise)
        if "does not have chest pain" in premise.lower():
            return NliScores(entailment=0.02, neutral=0.03, contradiction=0.95, label="CONTRADICTION")
        return NliScores(entailment=0.9, neutral=0.05, contradiction=0.05, label="ENTAILMENT")

    monkeypatch.setattr("app.tanglish.service.run_nli", _fake_run_nli)

    raw_text = "Enakku chest pain illa."
    claim = "Patient reports chest pain."

    baseline = tanglish_service.verify_baseline(raw_text, claim)
    tanglish_aware = tanglish_service.verify_tanglish_aware(raw_text, claim)

    assert baseline["normalization"] is None
    assert baseline["nli_premise"] == raw_text  # unchanged premise reached the model
    assert baseline["prediction"] == "SUPPORTED"  # naive lexical entailment, WITHOUT negation understanding

    assert tanglish_aware["normalization"] is not None
    assert tanglish_aware["nli_premise"] == "Patient does not have chest pain."
    assert tanglish_aware["prediction"] == "CONTRADICTED"  # correct once negation is made explicit

    assert captured_premises == [raw_text, "Patient does not have chest pain."]
