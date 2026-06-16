import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"
SVD_DIR = MODELS_DIR / "svd_artifacts"
IALS_DIR = MODELS_DIR / "ials_artifacts"
VAE_DIR = MODELS_DIR / "vae_artifacts"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Database
_db_host = os.getenv("DB_HOST", "localhost")
_pg_user = os.getenv("POSTGRES_USER") or ""
_pg_pass = os.getenv("POSTGRES_PASSWORD") or ""
_pg_db = os.getenv("POSTGRES_DB") or ""
DATABASE_URL = f"postgresql+psycopg2://{_pg_user}:{_pg_pass}@{_db_host}:5444/{_pg_db}"

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN must be set in the environment variables.")

SECRET_KEY = os.getenv("SECRET_KEY")
if SECRET_KEY is None:
    raise ValueError("SECRET_KEY must be set in the environment variables.")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
