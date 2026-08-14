from sqlalchemy.orm import Session

from app.asr.config import WHISPER_MODEL_SIZE
from app.asr.model import transcribe_file
from app.asr.storage import UnsafeProcessedPathError, resolve_safe_processed_path
from app.models import AudioRecord, SpeakerSegment, Transcript


class AsrError(Exception):
    """Carries the HTTP status/detail the route should surface for an ASR failure."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


def _mark_failed(transcript: Transcript, db: Session, error_message: str) -> None:
    transcript.processing_status = "failed"
    transcript.processing_error = error_message
    db.commit()


def run_transcription(audio_record: AudioRecord, db: Session) -> Transcript:
    """Run the local Faster-Whisper baseline over one Fog-processed audio record.

    Requires audio_record.processing_status == "completed" (Fog preprocessing);
    the pipeline stages stay separate -- this never triggers Fog processing.
    Updates transcript.processing_status through pending -> processing ->
    completed/failed, and never sends audio outside this machine.
    """
    if audio_record.processing_status != "completed":
        raise AsrError(
            409, "Audio must be Fog-processed (processing_status=completed) before transcription"
        )

    transcript = Transcript(
        consultation_id=audio_record.consultation_id,
        audio_record_id=audio_record.id,
        asr_model=f"faster-whisper-{WHISPER_MODEL_SIZE}",
        processing_status="pending",
    )
    db.add(transcript)
    db.commit()
    db.refresh(transcript)

    try:
        source_path = resolve_safe_processed_path(audio_record.processed_storage_path)
    except UnsafeProcessedPathError as exc:
        _mark_failed(transcript, db, "Processed audio path failed a safety check")
        raise AsrError(500, "Processed audio could not be located.") from exc
    except FileNotFoundError as exc:
        _mark_failed(transcript, db, "Processed audio file was not found on disk")
        raise AsrError(500, "Processed audio could not be located.") from exc

    transcript.processing_status = "processing"
    db.commit()

    try:
        result = transcribe_file(source_path)
    except Exception as exc:
        _mark_failed(transcript, db, "Unexpected transcription failure")
        raise AsrError(500, "Transcription failed unexpectedly.") from exc

    full_text = " ".join(segment.text for segment in result.segments).strip()

    transcript.full_text = full_text
    transcript.language = result.language
    transcript.processing_status = "completed"
    transcript.processing_error = None
    db.commit()

    # Raw ASR segments only -- no speaker identity assigned (that's Step 10).
    for segment in result.segments:
        db.add(
            SpeakerSegment(
                transcript_id=transcript.id,
                sequence_index=segment.sequence_index,
                speaker_label=None,
                inferred_role=None,
                start_time_ms=segment.start_ms,
                end_time_ms=segment.end_ms,
                segment_text=segment.text,
            )
        )
    db.commit()
    db.refresh(transcript)
    return transcript
