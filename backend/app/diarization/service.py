from sqlalchemy.orm import Session

from app.asr.storage import UnsafeProcessedPathError, resolve_safe_processed_path
from app.diarization.alignment import best_matching_turn_index
from app.diarization.model import DiarizationModelAccessError, diarize_file
from app.models import AudioRecord, SpeakerSegment, Transcript


class DiarizationError(Exception):
    """Carries the HTTP status/detail the route should surface for a diarization failure."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


def _mark_failed(audio_record: AudioRecord, db: Session, error_message: str) -> None:
    audio_record.diarization_status = "failed"
    audio_record.diarization_error = error_message
    db.commit()


def run_diarization(audio_record: AudioRecord, db: Session) -> dict:
    """Run the local PyAnnote pipeline over one Fog-processed audio record,
    then align any existing ASR segments to the detected speaker turns using
    deterministic temporal overlap.

    Updates audio_record.diarization_status through pending -> processing ->
    completed/failed. Never sends audio outside this machine, never fabricates
    Doctor/Patient roles, and never assigns a speaker without real overlap.
    """
    if audio_record.processing_status != "completed":
        raise DiarizationError(
            409, "Audio must be Fog-processed (processing_status=completed) before diarization"
        )

    try:
        source_path = resolve_safe_processed_path(audio_record.processed_storage_path)
    except UnsafeProcessedPathError as exc:
        _mark_failed(audio_record, db, "Processed audio path failed a safety check")
        raise DiarizationError(500, "Processed audio could not be located.") from exc
    except FileNotFoundError as exc:
        _mark_failed(audio_record, db, "Processed audio file was not found on disk")
        raise DiarizationError(500, "Processed audio could not be located.") from exc

    audio_record.diarization_status = "processing"
    db.commit()

    try:
        turns = diarize_file(source_path)
    except DiarizationModelAccessError as exc:
        _mark_failed(audio_record, db, str(exc))
        raise DiarizationError(503, str(exc)) from exc
    except Exception as exc:
        _mark_failed(audio_record, db, "Unexpected diarization failure")
        raise DiarizationError(500, "Diarization failed unexpectedly.") from exc

    # Diarization runs independently of ASR -- align to the latest transcript
    # for this audio if one already exists; otherwise diarization still stands
    # on its own (raw speaker intervals, no text yet).
    transcript = (
        db.query(Transcript)
        .filter(Transcript.audio_record_id == audio_record.id)
        .order_by(Transcript.created_at.desc())
        .first()
    )

    existing_segments = []
    if transcript is not None:
        existing_segments = (
            db.query(SpeakerSegment)
            .filter(SpeakerSegment.transcript_id == transcript.id)
            .order_by(SpeakerSegment.sequence_index)
            .all()
        )

    # Enrich existing ASR segments with a speaker label where overlap exists.
    matched_turn_indices: set[int] = set()
    for segment in existing_segments:
        turn_index, speaker_label, overlap_ratio = best_matching_turn_index(
            segment.start_time_ms, segment.end_time_ms, turns
        )
        if turn_index is not None:
            segment.speaker_label = speaker_label
            segment.diarization_confidence = overlap_ratio
            matched_turn_indices.add(turn_index)
        # else: weak/no overlap -- leave speaker_label as-is (None), never fabricated.

    # Turns that never matched any ASR segment are preserved as standalone
    # speaker intervals, with no text, rather than being silently dropped.
    next_sequence_index = max((s.sequence_index for s in existing_segments), default=-1) + 1
    new_rows: list[SpeakerSegment] = []
    for index, turn in enumerate(turns):
        if index in matched_turn_indices:
            continue
        row = SpeakerSegment(
            transcript_id=transcript.id if transcript is not None else None,
            audio_record_id=audio_record.id,
            sequence_index=next_sequence_index,
            speaker_label=turn.speaker_label,
            start_time_ms=turn.start_ms,
            end_time_ms=turn.end_ms,
            segment_text=None,
        )
        db.add(row)
        new_rows.append(row)
        next_sequence_index += 1

    audio_record.diarization_status = "completed"
    audio_record.diarization_error = None
    db.commit()

    speakers = sorted({turn.speaker_label for turn in turns})
    return {
        "segment_count": len(existing_segments) + len(new_rows),
        "speaker_count": len(speakers),
        "speakers": speakers,
    }
