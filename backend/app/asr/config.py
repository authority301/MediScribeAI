import os
from pathlib import Path

from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(ENV_PATH)

# Configurable so the same code runs tiny/base/small/medium/large-v3 without changes.
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "small")

# Safe CPU defaults; "cuda" only if the operator has a working GPU setup -- never assumed.
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")

WHISPER_BEAM_SIZE = int(os.environ.get("WHISPER_BEAM_SIZE", "5"))
WHISPER_VAD_FILTER = os.environ.get("WHISPER_VAD_FILTER", "true").strip().lower() in (
    "1",
    "true",
    "yes",
)
