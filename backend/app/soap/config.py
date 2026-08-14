import os
from pathlib import Path

from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(ENV_PATH)

# Configurable so a local LLM (e.g. Llama, Gemma) can be plugged in later
# without changing the API/service architecture. "deterministic-baseline" is
# the only implemented model name today -- see service.py's dispatch.
SOAP_MODEL_PROVIDER = os.environ.get("SOAP_MODEL_PROVIDER", "local")
SOAP_MODEL_NAME = os.environ.get("SOAP_MODEL_NAME", "deterministic-baseline")
