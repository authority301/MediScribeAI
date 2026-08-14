"""Local Fog audio normalization: mono, 16 kHz, 16-bit PCM WAV.

Runs entirely on this machine via pydub + a local FFmpeg executable. No audio
is ever sent to an external network service.
"""
import shutil
from pathlib import Path

from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

TARGET_SAMPLE_RATE_HZ = 16000
TARGET_CHANNELS = 1
TARGET_SAMPLE_WIDTH_BYTES = 2  # 16-bit PCM


class AudioDecodeError(Exception):
    """Raised when the source audio cannot be decoded or normalized."""


def is_ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def normalize_audio(source_path: Path, destination_path: Path) -> None:
    """Convert source_path to mono/16kHz/16-bit PCM WAV at destination_path."""
    try:
        audio = AudioSegment.from_file(source_path)
    except CouldntDecodeError as exc:
        raise AudioDecodeError("Source audio could not be decoded") from exc

    normalized = (
        audio.set_channels(TARGET_CHANNELS)
        .set_frame_rate(TARGET_SAMPLE_RATE_HZ)
        .set_sample_width(TARGET_SAMPLE_WIDTH_BYTES)
    )
    normalized.export(destination_path, format="wav")
