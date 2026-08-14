import uuid
from pathlib import Path

from app.audio.storage import UPLOAD_ROOT

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PROCESSED_ROOT = REPO_ROOT / "uploads" / "processed_audio"
PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)


class UnsafeSourcePathError(Exception):
    """Raised when an audio record's stored path does not resolve inside uploads/audio."""


def resolve_safe_source_path(storage_path: str) -> Path:
    """Resolve a DB-stored storage_path, verifying it stays inside uploads/audio.

    The client never supplies this path directly (only an audio_id UUID); this
    is a defense-in-depth check against a malformed/tampered DB value.
    """
    candidate = (REPO_ROOT / storage_path).resolve()
    upload_root_resolved = UPLOAD_ROOT.resolve()
    if not candidate.is_relative_to(upload_root_resolved):
        raise UnsafeSourcePathError("Source audio path is outside the approved upload directory")
    if not candidate.is_file():
        raise FileNotFoundError("Source audio file was not found")
    return candidate


def generate_processed_path() -> tuple[Path, str]:
    """Generate a unique destination path inside uploads/processed_audio/."""
    name = f"{uuid.uuid4().hex}.wav"
    return PROCESSED_ROOT / name, f"uploads/processed_audio/{name}"
