from sqlalchemy.orm import Session

from app.medical_entities.extraction import extract_entities
from app.medical_entities.phi import deidentify_text, detect_phi
from app.models import MedicalEntity, SpeakerSegment, Transcript

# entity_type namespace: PHI detections are stored in the same table with a
# "PHI_" prefix (e.g. "PHI_PERSON_NAME") rather than a separate table.
PHI_ENTITY_TYPE_PREFIX = "PHI_"


class EntityExtractionError(Exception):
    """Carries the HTTP status/detail the route should surface for an extraction failure."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


def run_entity_extraction(transcript: Transcript, db: Session) -> dict:
    """Extract medical entities and PHI from one transcript's ASR-produced
    speaker segments, and build a de-identified copy of the full transcript.

    This is information extraction, not diagnosis: only terms literally
    present in the transcript are surfaced (see extraction.py). Never runs
    ASR or diarization itself -- both must already have completed.

    Idempotent: any previous extraction for this transcript is replaced
    (deleted then re-inserted) rather than accumulating duplicate rows.
    An empty result (no entities found) is a successful "completed" outcome,
    not a failure.
    """
    if transcript.processing_status != "completed":
        raise EntityExtractionError(
            409, "Transcript ASR processing must be completed before entity extraction"
        )

    try:
        transcript.entity_extraction_status = "processing"
        db.commit()

        segments = (
            db.query(SpeakerSegment)
            .filter(SpeakerSegment.transcript_id == transcript.id)
            .order_by(SpeakerSegment.sequence_index)
            .all()
        )

        clinical_rows = []
        phi_rows = []
        for segment in segments:
            if not segment.segment_text:
                continue
            for entity in extract_entities(segment.segment_text):
                clinical_rows.append((segment.id, entity))
            for detection in detect_phi(segment.segment_text):
                phi_rows.append((segment.id, detection))

        # Idempotent replace: delete any previous extraction for this
        # transcript before inserting the fresh result.
        db.query(MedicalEntity).filter(MedicalEntity.transcript_id == transcript.id).delete(
            synchronize_session=False
        )

        for speaker_segment_id, entity in clinical_rows:
            db.add(
                MedicalEntity(
                    consultation_id=transcript.consultation_id,
                    transcript_id=transcript.id,
                    speaker_segment_id=speaker_segment_id,
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

        for speaker_segment_id, detection in phi_rows:
            db.add(
                MedicalEntity(
                    consultation_id=transcript.consultation_id,
                    transcript_id=transcript.id,
                    speaker_segment_id=speaker_segment_id,
                    entity_type=f"{PHI_ENTITY_TYPE_PREFIX}{detection.phi_type}",
                    # Redaction placeholder only -- the raw PHI value is never stored.
                    entity_text=f"[{detection.phi_type}]",
                    normalized_value=None,
                    start_char=detection.start_offset,
                    end_char=detection.end_offset,
                    confidence_score=detection.confidence,
                    negated=False,
                    historical=False,
                )
            )

        # De-identify the full source transcript as a separate, independent
        # pass -- catches PHI regardless of speaker-segment boundaries.
        # full_text (the original evidence) is never modified.
        transcript.deidentified_text = deidentify_text(transcript.full_text or "")
        transcript.entity_extraction_status = "completed"
        transcript.entity_extraction_error = None
        db.commit()

        entities_summary = [
            {
                "entity_type": entity.entity_type,
                "entity_text": entity.entity_text,
                "normalized_text": entity.normalized_text,
                "negated": entity.negated,
                "historical": entity.historical,
            }
            for _segment_id, entity in clinical_rows
        ] + [
            {
                "entity_type": f"{PHI_ENTITY_TYPE_PREFIX}{detection.phi_type}",
                "entity_text": f"[{detection.phi_type}]",  # never the raw PHI value
                "normalized_text": None,
                "negated": False,
                "historical": False,
            }
            for _segment_id, detection in phi_rows
        ]

        return {
            "medical_entity_count": len(clinical_rows),
            "phi_detected": len(phi_rows) > 0,
            "deidentified_text_available": True,
            "entities": entities_summary,
        }
    except EntityExtractionError:
        raise
    except Exception as exc:
        db.rollback()
        try:
            transcript.entity_extraction_status = "failed"
            transcript.entity_extraction_error = "Unexpected entity extraction failure"
            db.commit()
        except Exception:
            db.rollback()
        raise EntityExtractionError(500, "Entity extraction failed unexpectedly.") from exc
