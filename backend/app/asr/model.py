"""Lazily-loaded, reusable local Faster-Whisper model + the raw transcription call.

Runs entirely on this machine via CTranslate2 -- no audio or model data is ever
sent to an external network service. Language is auto-detected (never forced).
"""
import threading
from pathlib import Path

from faster_whisper import WhisperModel

from app.asr.config import (
    WHISPER_BEAM_SIZE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_MODEL_SIZE,
    WHISPER_VAD_FILTER,
)

_model_lock = threading.Lock()
_model: WhisperModel | None = None


def get_model() -> WhisperModel:
    """Return the process-wide WhisperModel instance, loading it on first use only."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = WhisperModel(
                    WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE
                )
    return _model


class TranscriptionSegment:
    __slots__ = ("sequence_index", "text", "start_ms", "end_ms")

    def __init__(self, sequence_index: int, text: str, start_ms: int, end_ms: int):
        self.sequence_index = sequence_index
        self.text = text
        self.start_ms = start_ms
        self.end_ms = end_ms


class TranscriptionResult:
    __slots__ = ("language", "segments")

    def __init__(self, language: str | None, segments: list[TranscriptionSegment]):
        self.language = language
        self.segments = segments


def transcribe_file(path: Path) -> TranscriptionResult:
    """Run the local model over path; language is auto-detected, never forced."""
    model = get_model()
    segments_iter, info = model.transcribe(
        str(path), beam_size=WHISPER_BEAM_SIZE, vad_filter=WHISPER_VAD_FILTER
    )
    segments = [
        TranscriptionSegment(
            sequence_index=idx,
            text=segment.text.strip(),
            start_ms=int(segment.start * 1000),
            end_ms=int(segment.end * 1000),
        )
        for idx, segment in enumerate(segments_iter)
    ]
    return TranscriptionResult(language=info.language, segments=segments)
