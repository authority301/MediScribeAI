from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.fog.audio_processor import (
    TARGET_CHANNELS,
    TARGET_SAMPLE_RATE_HZ,
    AudioDecodeError,
    is_ffmpeg_available,
    normalize_audio,
)
from app.fog.storage import UnsafeSourcePathError, generate_processed_path, resolve_safe_source_path
from app.models import AudioRecord


class FogProcessingError(Exception):
    """Carries the HTTP status/detail the route should surface for a processing failure."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


def _mark_failed(audio_record: AudioRecord, db: Session, error_message: str) -> None:
    audio_record.processing_status = "failed"
    audio_record.processing_error = error_message
    db.commit()


def process_audio_record(audio_record: AudioRecord, db: Session) -> AudioRecord:
    """Run the local Fog normalization pipeline for one audio record.

    Updates audio_record.processing_status through pending -> processing ->
    completed/failed, and never sends audio outside this machine.
    """
    if not is_ffmpeg_available():
        raise FogProcessingError(
            500, "Audio processing is unavailable: FFmpeg was not found on this server."
        )

    try:
        source_path = resolve_safe_source_path(audio_record.storage_path)
    except UnsafeSourcePathError as exc:
        _mark_failed(audio_record, db, "Source audio path failed a safety check")
        raise FogProcessingError(500, "Source audio could not be located.") from exc
    except FileNotFoundError as exc:
        _mark_failed(audio_record, db, "Source audio file was not found on disk")
        raise FogProcessingError(500, "Source audio could not be located.") from exc

    audio_record.processing_status = "processing"
    db.commit()

    destination_path, relative_path = generate_processed_path()

    try:
        normalize_audio(source_path, destination_path)
    except AudioDecodeError as exc:
        destination_path.unlink(missing_ok=True)
        _mark_failed(audio_record, db, "Source audio could not be decoded (corrupt or unsupported)")
        raise FogProcessingError(
            422, "The source audio could not be processed: it may be corrupt or unsupported."
        ) from exc
    except Exception as exc:
        destination_path.unlink(missing_ok=True)
        _mark_failed(audio_record, db, "Unexpected processing failure")
        raise FogProcessingError(500, "Audio processing failed unexpectedly.") from exc

    file_size = destination_path.stat().st_size

    audio_record.processing_status = "completed"
    audio_record.processed_storage_path = relative_path
    audio_record.processed_content_type = "audio/wav"
    audio_record.processed_file_size_bytes = file_size
    audio_record.processed_sample_rate_hz = TARGET_SAMPLE_RATE_HZ
    audio_record.processed_channels = TARGET_CHANNELS
    audio_record.processed_at = datetime.now(timezone.utc)
    audio_record.processing_error = None
    db.commit()
    db.refresh(audio_record)
    return audio_record
