from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.models import AudioRecord, SpeakerSegment, Transcript
from app.speaker_roles.scoring import (
    RoleClassification,
    apply_two_speaker_consistency,
    classify_speaker,
    score_speaker,
)


class RoleIdentificationError(Exception):
    """Carries the HTTP status/detail the route should surface for a role-ID failure."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


def run_role_identification(
    audio_record: AudioRecord, db: Session
) -> tuple[list[RoleClassification], str]:
    """Identify DOCTOR/PATIENT/UNKNOWN for each anonymous diarization speaker of
    one audio record, using deterministic heuristic scoring over that
    speaker's ASR text (see scoring.py). Never runs ASR or diarization itself;
    both must already have completed.

    Idempotent: re-running finds the same speaker_segments rows by
    speaker_label and updates them in place -- it never inserts new rows, so
    repeated requests never duplicate data.

    Returns (classifications, overall_status) where overall_status is:
      - "completed": every speaker was confidently classified DOCTOR/PATIENT
      - "uncertain": one or more speakers remain UNKNOWN (insufficient
        evidence) -- this is NOT treated as a processing failure
      - never returned as "failed" -- that state is only ever persisted (on
        speaker_segments.role_identification_status) if the identification
        process itself raises an unexpected error, in which case a
        RoleIdentificationError(500, ...) is raised instead of returning
    """
    if audio_record.diarization_status != "completed":
        raise RoleIdentificationError(
            409, "Diarization must be completed before speaker role identification"
        )

    transcript = (
        db.query(Transcript)
        .filter(Transcript.audio_record_id == audio_record.id)
        .order_by(Transcript.created_at.desc())
        .first()
    )

    segment_conditions = [SpeakerSegment.audio_record_id == audio_record.id]
    if transcript is not None:
        segment_conditions.append(SpeakerSegment.transcript_id == transcript.id)

    segments = (
        db.query(SpeakerSegment)
        .filter(or_(*segment_conditions), SpeakerSegment.speaker_label.isnot(None))
        .all()
    )

    if not segments:
        raise RoleIdentificationError(
            409,
            "No speaker segments found for this audio; diarization may not have "
            "produced any speaker-labeled segments",
        )

    try:
        texts_by_speaker: dict[str, list[str]] = {}
        for segment in segments:
            texts_by_speaker.setdefault(segment.speaker_label, [])
            if segment.segment_text:
                texts_by_speaker[segment.speaker_label].append(segment.segment_text)

        evidences = {
            label: score_speaker(label, texts) for label, texts in texts_by_speaker.items()
        }
        classifications = [classify_speaker(evidences[label]) for label in sorted(evidences)]
        classifications = apply_two_speaker_consistency(classifications, evidences)

        # UNKNOWN is a legitimate, honest outcome (insufficient evidence) --
        # it is "uncertain", never treated as a processing failure.
        overall_status = (
            "uncertain" if any(c.role == "UNKNOWN" for c in classifications) else "completed"
        )

        classification_by_label = {c.speaker_label: c for c in classifications}
        for segment in segments:
            classification = classification_by_label.get(segment.speaker_label)
            if classification is None:
                continue
            # Update in place (matched by speaker_label) -- no new rows created,
            # so repeated requests are idempotent rather than duplicating data.
            segment.inferred_role = classification.role
            segment.role_confidence = classification.confidence
            segment.role_identification_status = overall_status
            segment.role_evidence = classification.evidence

        db.commit()
    except RoleIdentificationError:
        raise
    except Exception as exc:
        db.rollback()
        # Best-effort: record the failure against the segments we attempted to
        # process, in a fresh transaction, without exposing internals to the client.
        try:
            for segment in segments:
                segment.role_identification_status = "failed"
            db.commit()
        except Exception:
            db.rollback()
        raise RoleIdentificationError(
            500, "Speaker role identification failed unexpectedly."
        ) from exc

    return classifications, overall_status
