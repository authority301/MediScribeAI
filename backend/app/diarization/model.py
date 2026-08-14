"""Lazily-loaded, reusable local PyAnnote speaker-diarization pipeline.

Runs entirely on this machine via pyannote.audio + a local PyTorch backend.
No audio is ever sent to an external network service. Model weights are
fetched once from Hugging Face Hub -- a gated pretrained model, so a valid
HF_TOKEN (and having accepted the model's terms on Hugging Face) is required
for that one-time download, same as any other private Hugging Face model.

Speaker labels come directly from pyannote's own output (SPEAKER_00,
SPEAKER_01, ...) -- never invented or renamed here.
"""
import threading
from pathlib import Path

from app.diarization.config import DIARIZATION_DEVICE, DIARIZATION_MODEL, HF_TOKEN

_pipeline_lock = threading.Lock()
_pipeline = None


class DiarizationModelAccessError(Exception):
    """Raised when the pretrained model cannot be loaded: missing/invalid HF
    token, or the model's terms have not been accepted on Hugging Face."""


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                if not HF_TOKEN:
                    raise DiarizationModelAccessError(
                        f"HF_TOKEN is not set. {DIARIZATION_MODEL} is a gated model on Hugging "
                        f"Face: create an account, accept its terms at "
                        f"https://huggingface.co/{DIARIZATION_MODEL}, generate a personal access "
                        f"token, and set HF_TOKEN in backend/.env."
                    )

                import torch
                from pyannote.audio import Pipeline

                try:
                    pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, token=HF_TOKEN)
                except Exception as exc:
                    raise DiarizationModelAccessError(
                        f"Could not load the pretrained diarization model '{DIARIZATION_MODEL}'. "
                        "Confirm HF_TOKEN is valid and that the model's terms have been accepted "
                        "on Hugging Face."
                    ) from exc

                if pipeline is None:
                    raise DiarizationModelAccessError(
                        f"Could not load the pretrained diarization model '{DIARIZATION_MODEL}'."
                    )

                pipeline.to(torch.device(DIARIZATION_DEVICE))
                _pipeline = pipeline
    return _pipeline


class DiarizationTurn:
    __slots__ = ("speaker_label", "start_ms", "end_ms")

    def __init__(self, speaker_label: str, start_ms: int, end_ms: int):
        self.speaker_label = speaker_label
        self.start_ms = start_ms
        self.end_ms = end_ms


def diarize_file(path: Path) -> list[DiarizationTurn]:
    """Run the local pipeline over path; returns anonymous SPEAKER_XX turns only."""
    pipeline = get_pipeline()
    diarization = pipeline(str(path))
    return [
        DiarizationTurn(
            speaker_label=speaker,
            start_ms=int(turn.start * 1000),
            end_ms=int(turn.end * 1000),
        )
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]
