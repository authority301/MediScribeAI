from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models import Consultation, MedicalEntity, SOAPClaim, SOAPNote, SpeakerSegment, Transcript
from app.soap.config import SOAP_MODEL_NAME, SOAP_MODEL_PROVIDER
from app.soap.generator import ASSESSMENT, OBJECTIVE, PLAN, SUBJECTIVE, generate_claims

PHI_ENTITY_TYPE_PREFIX = "PHI_"
SECTIONS = (SUBJECTIVE, OBJECTIVE, ASSESSMENT, PLAN)


class SoapGenerationError(Exception):
    """Carries the HTTP status/detail the route should surface for a generation failure."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


def _dispatch_generation(segments: list, entities: list) -> list:
    """Pluggable generation seam: only the local deterministic baseline is
    implemented today. A future local LLM (Llama, Gemma, ...) can be added
    here, selected purely via SOAP_MODEL_PROVIDER/SOAP_MODEL_NAME, without
    changing the API or service architecture above this function.
    """
    if SOAP_MODEL_PROVIDER == "local" and SOAP_MODEL_NAME == "deterministic-baseline":
        return generate_claims(segments, entities)
    raise SoapGenerationError(
        501,
        f"SOAP_MODEL_PROVIDER='{SOAP_MODEL_PROVIDER}' / SOAP_MODEL_NAME='{SOAP_MODEL_NAME}' is not "
        "implemented; use SOAP_MODEL_PROVIDER=local and SOAP_MODEL_NAME=deterministic-baseline",
    )


def _mark_failed(soap_note: SOAPNote, db: Session, error_message: str) -> None:
    soap_note.generation_status = "failed"
    soap_note.generation_error = error_message
    db.commit()


def run_soap_generation(consultation: Consultation, transcript: Transcript, db: Session) -> dict:
    """Generate a SOAP note + individual SOAP claims for one completed,
    entity-extracted transcript. Never runs ASR, diarization, role
    identification, or entity extraction itself -- all must already have
    completed.

    IDEMPOTENCY (documented choice): if a SOAP note already exists for this
    exact (consultation_id, transcript_id) pair AND is still in the raw
    "generated" review state (i.e. the doctor has not edited/approved/
    rejected it), that same note and its claims are replaced in place --
    repeated button presses update the same row rather than accumulating
    versions. If the existing note has already been touched by a doctor
    (status != "generated"), a NEW version is created instead, so
    regeneration never silently overwrites a doctor's edits.
    """
    if transcript.processing_status != "completed":
        raise SoapGenerationError(
            409, "Transcript ASR processing must be completed before SOAP generation"
        )
    if transcript.entity_extraction_status != "completed":
        raise SoapGenerationError(
            409, "Medical entity extraction must be completed before SOAP generation"
        )

    existing = (
        db.query(SOAPNote)
        .filter(
            SOAPNote.consultation_id == consultation.id,
            SOAPNote.transcript_id == transcript.id,
            SOAPNote.status == "generated",
        )
        .order_by(SOAPNote.version.desc())
        .first()
    )

    if existing is not None:
        soap_note = existing
        db.query(SOAPClaim).filter(SOAPClaim.soap_note_id == soap_note.id).delete(
            synchronize_session=False
        )
    else:
        max_version = (
            db.query(func.max(SOAPNote.version))
            .filter(SOAPNote.consultation_id == consultation.id)
            .scalar()
        )
        soap_note = SOAPNote(
            consultation_id=consultation.id,
            transcript_id=transcript.id,
            version=(max_version or 0) + 1,
            status="generated",
        )
        db.add(soap_note)

    soap_note.generation_status = "processing"
    db.commit()
    db.refresh(soap_note)

    try:
        segments = (
            db.query(SpeakerSegment)
            .filter(SpeakerSegment.transcript_id == transcript.id)
            .order_by(SpeakerSegment.sequence_index)
            .all()
        )
        # Clinical entities only -- PHI_* rows are never fed into SOAP text.
        entities = (
            db.query(MedicalEntity)
            .filter(
                MedicalEntity.transcript_id == transcript.id,
                ~MedicalEntity.entity_type.like(f"{PHI_ENTITY_TYPE_PREFIX}%"),
            )
            .all()
        )

        generated = _dispatch_generation(segments, entities)

        sections: dict[str, list] = {section: [] for section in SECTIONS}
        for claim in generated:
            sections[claim.section].append(claim)

        for section in SECTIONS:
            for index, claim in enumerate(sections[section]):
                db.add(
                    SOAPClaim(
                        soap_note_id=soap_note.id,
                        section=section,
                        claim_text=claim.text,
                        sequence_index=index,
                        generation_confidence=claim.confidence,
                    )
                )

        soap_note.subjective = "\n".join(c.text for c in sections[SUBJECTIVE]) or "Not documented."
        soap_note.objective = "\n".join(c.text for c in sections[OBJECTIVE]) or "Not documented."
        soap_note.assessment = "\n".join(c.text for c in sections[ASSESSMENT]) or "Not documented."
        soap_note.plan = "\n".join(c.text for c in sections[PLAN]) or "Not documented."
        soap_note.generated_by = f"{SOAP_MODEL_PROVIDER}:{SOAP_MODEL_NAME}"
        # An evidence-sparse note (many/all sections "Not documented.") is
        # still a successful, completed generation -- not a failure.
        soap_note.generation_status = "completed"
        soap_note.generation_error = None
        db.commit()
        db.refresh(soap_note)

        return {"soap_note": soap_note, "sections": sections}
    except SoapGenerationError:
        raise
    except Exception as exc:
        db.rollback()
        try:
            _mark_failed(soap_note, db, "Unexpected SOAP generation failure")
        except Exception:
            db.rollback()
        raise SoapGenerationError(500, "SOAP generation failed unexpectedly.") from exc
