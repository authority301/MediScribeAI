from sqlalchemy import text

from app.database.base import Base
from app.database.session import engine
from app import models  # noqa: F401  (registers all models on Base.metadata)


def _ensure_doctor_auth_columns() -> None:
    """Additively migrate the pre-existing doctors table for authentication.

    create_all() only creates tables that don't exist yet; doctors already
    existed before this step, so its new auth columns are added here via
    idempotent ALTER TABLE statements instead of introducing Alembic.
    """
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE doctors ADD COLUMN IF NOT EXISTS password_hash TEXT NOT NULL DEFAULT ''"))
        conn.execute(text("ALTER TABLE doctors ALTER COLUMN password_hash DROP DEFAULT"))
        conn.execute(text("ALTER TABLE doctors ADD COLUMN IF NOT EXISTS is_active BOOLEAN NOT NULL DEFAULT true"))


def _ensure_consultation_status_default() -> None:
    """Realign the consultations.status default with the approved lifecycle.

    The Step 5A design originally defaulted this column to 'scheduled'. Step 7
    defines the lifecycle as draft/active/completed/cancelled instead. status
    is a plain TEXT column (no CHECK constraint), so only its DEFAULT clause
    needs correcting here; the four allowed values are enforced in the API.
    """
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE consultations ALTER COLUMN status SET DEFAULT 'draft'"))


def _ensure_audio_record_upload_columns() -> None:
    """Additively migrate the pre-existing audio_records table for uploads.

    audio_records already existed from Step 5B without original_filename or
    content_type; Step 8A needs both to describe an uploaded file, so they are
    added here via idempotent ALTER TABLE statements instead of Alembic.
    """
    with engine.begin() as conn:
        conn.execute(
            text(
                "ALTER TABLE audio_records ADD COLUMN IF NOT EXISTS original_filename TEXT NOT NULL DEFAULT ''"
            )
        )
        conn.execute(text("ALTER TABLE audio_records ALTER COLUMN original_filename DROP DEFAULT"))
        conn.execute(
            text(
                "ALTER TABLE audio_records ADD COLUMN IF NOT EXISTS content_type TEXT NOT NULL DEFAULT ''"
            )
        )
        conn.execute(text("ALTER TABLE audio_records ALTER COLUMN content_type DROP DEFAULT"))


def _ensure_audio_record_processing_columns() -> None:
    """Additively migrate audio_records for Step 8B local Fog processing metadata.

    Tracks the pending/processing/completed/failed lifecycle and where the
    normalized WAV output landed, without ever storing audio binary data.
    """
    with engine.begin() as conn:
        conn.execute(
            text(
                "ALTER TABLE audio_records ADD COLUMN IF NOT EXISTS processing_status "
                "TEXT NOT NULL DEFAULT 'pending'"
            )
        )
        conn.execute(
            text("ALTER TABLE audio_records ADD COLUMN IF NOT EXISTS processed_storage_path TEXT")
        )
        conn.execute(
            text("ALTER TABLE audio_records ADD COLUMN IF NOT EXISTS processed_content_type TEXT")
        )
        conn.execute(
            text(
                "ALTER TABLE audio_records ADD COLUMN IF NOT EXISTS processed_file_size_bytes BIGINT"
            )
        )
        conn.execute(
            text(
                "ALTER TABLE audio_records ADD COLUMN IF NOT EXISTS processed_sample_rate_hz INTEGER"
            )
        )
        conn.execute(
            text("ALTER TABLE audio_records ADD COLUMN IF NOT EXISTS processed_channels INTEGER")
        )
        conn.execute(
            text(
                "ALTER TABLE audio_records ADD COLUMN IF NOT EXISTS processed_at TIMESTAMPTZ"
            )
        )
        conn.execute(
            text("ALTER TABLE audio_records ADD COLUMN IF NOT EXISTS processing_error TEXT")
        )


def _ensure_transcript_asr_columns() -> None:
    """Additively migrate transcripts for Step 9A Faster-Whisper baseline ASR.

    full_text is relaxed to nullable because a transcript row now exists
    (pending/processing) before ASR produces any text, mirroring the
    pending/processing/completed/failed pattern used elsewhere.
    """
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE transcripts ALTER COLUMN full_text DROP NOT NULL"))
        conn.execute(text("ALTER TABLE transcripts ADD COLUMN IF NOT EXISTS language TEXT"))
        conn.execute(
            text(
                "ALTER TABLE transcripts ADD COLUMN IF NOT EXISTS processing_status "
                "TEXT NOT NULL DEFAULT 'pending'"
            )
        )
        conn.execute(
            text("ALTER TABLE transcripts ADD COLUMN IF NOT EXISTS processing_error TEXT")
        )


def _ensure_speaker_segment_label_nullable() -> None:
    """Relax speaker_segments.speaker_label to nullable for Step 9A.

    Step 9A writes raw ASR segments with no speaker identity -- diarization is
    Step 10. speaker_label must stay NULL for these rows rather than being
    fabricated just to satisfy a NOT NULL constraint.
    """
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE speaker_segments ALTER COLUMN speaker_label DROP NOT NULL"))


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    _ensure_doctor_auth_columns()
    _ensure_consultation_status_default()
    _ensure_audio_record_upload_columns()
    _ensure_audio_record_processing_columns()
    _ensure_transcript_asr_columns()
    _ensure_speaker_segment_label_nullable()


if __name__ == "__main__":
    init_db()
    print("Database tables created.")
