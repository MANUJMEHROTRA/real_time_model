import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

FRONTEND_DIR = ROOT / "frontend"
CATALOG_DIR = ROOT / "catalog"
OUTPUT_DIR = ROOT / "outputs"

TRYON_BACKEND = os.getenv("TRYON_BACKEND", "mock").strip().lower()
TRYON_TIMEOUT = float(os.getenv("TRYON_TIMEOUT", "90"))
SAVE_RESULTS = os.getenv("SAVE_RESULTS", "false").strip().lower() in {"1", "true", "yes"}

FASHN_API_KEY = os.getenv("FASHN_API_KEY", "")
FASHN_MODE = os.getenv("FASHN_MODE", "balanced")

HF_TOKEN = os.getenv("HF_TOKEN", "")

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "")
REPLICATE_MODEL = os.getenv("REPLICATE_MODEL", "cuuupid/idm-vton")
