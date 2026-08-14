import os
from pathlib import Path

from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(ENV_PATH)

# Gated pretrained model on Hugging Face Hub -- requires an account, accepting
# the model's terms, and a personal access token. Never hard-coded.
HF_TOKEN = os.environ.get("HF_TOKEN")

# Configurable so the pipeline can be swapped without code changes.
DIARIZATION_MODEL = os.environ.get("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")

# Safe CPU default; "cuda" only if the operator has a working GPU setup -- never assumed.
DIARIZATION_DEVICE = os.environ.get("DIARIZATION_DEVICE", "cpu")
