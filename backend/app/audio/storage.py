from pathlib import Path

from fastapi import HTTPException, UploadFile, status

# backend/app/audio/storage.py -> repo root -> uploads/audio
UPLOAD_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "uploads" / "audio"
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

ALLOWED_CONTENT_TYPES: dict[str, str] = {"audio/webm": ".webm"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
_CHUNK_SIZE = 1024 * 1024


async def save_upload_streaming(upload: UploadFile, destination: Path, max_bytes: int) -> int:
    """Stream the upload to disk in chunks, aborting (and cleaning up) if it exceeds max_bytes."""
    total = 0
    try:
        with destination.open("wb") as out_file:
            while True:
                chunk = await upload.read(_CHUNK_SIZE)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        detail="File exceeds the 25 MB upload limit",
                    )
                out_file.write(chunk)
    except HTTPException:
        destination.unlink(missing_ok=True)
        raise
    return total
