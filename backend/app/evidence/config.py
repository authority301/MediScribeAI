import os
from pathlib import Path

from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(ENV_PATH)

EVIDENCE_TOP_K = int(os.environ.get("EVIDENCE_TOP_K", "3"))
EVIDENCE_MIN_SCORE = float(os.environ.get("EVIDENCE_MIN_SCORE", "0.15"))
