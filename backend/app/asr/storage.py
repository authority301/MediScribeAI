from pathlib import Path

from app.fog.storage import PROCESSED_ROOT, REPO_ROOT


class UnsafeProcessedPathError(Exception):
    """Raised when an audio record's processed_storage_path is not inside uploads/processed_audio."""


def resolve_safe_processed_path(processed_storage_path: str | None) -> Path:
    """Resolve the DB-stored processed_storage_path, verifying it stays inside
    uploads/processed_audio. The client never supplies this path directly (only
    an audio_id UUID); this is a defense-in-depth check against a tampered value.
    """
    if not processed_storage_path:
        raise FileNotFoundError("No processed audio path is recorded for this audio")

    candidate = (REPO_ROOT / processed_storage_path).resolve()
    processed_root_resolved = PROCESSED_ROOT.resolve()
    if not candidate.is_relative_to(processed_root_resolved):
        raise UnsafeProcessedPathError(
            "Processed audio path is outside the approved processed-audio directory"
        )
    if not candidate.is_file():
        raise FileNotFoundError("Processed audio file was not found")
    return candidate
