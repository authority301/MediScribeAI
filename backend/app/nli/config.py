import os
from pathlib import Path

from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(ENV_PATH)

NLI_MODEL_PROVIDER = os.environ.get("NLI_MODEL_PROVIDER", "local")
NLI_MODEL_NAME = os.environ.get(
    "NLI_MODEL_NAME", "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
)
# Safe CPU default; "cuda" only if the operator has a working GPU setup -- never assumed.
NLI_DEVICE = os.environ.get("NLI_DEVICE", "cpu")

# Baseline experimental thresholds, NOT calibrated probabilities and NOT a
# claim of medical certainty. See app/nli/aggregation.py.
NLI_ENTAILMENT_THRESHOLD = float(os.environ.get("NLI_ENTAILMENT_THRESHOLD", "0.70"))
NLI_CONTRADICTION_THRESHOLD = float(os.environ.get("NLI_CONTRADICTION_THRESHOLD", "0.70"))
