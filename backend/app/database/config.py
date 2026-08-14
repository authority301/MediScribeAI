import os
from pathlib import Path

from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(ENV_PATH)

DATABASE_URL = os.environ["DATABASE_URL"]
