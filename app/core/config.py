"""
애플리케이션 설정 관리
"""
import os
from pydantic_settings import BaseSettings
from typing import List

from app.core.constants import PII_CATEGORIES, PII_NAMES, CONFIDENCE_THRESHOLDS


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "PIILOT"
    APP_ENV: str = "development"
    DEBUG: bool = True
    SECRET_KEY: str = "your-secret-key-change-in-production"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Database
    DATABASE_URL: str = "postgresql+psycopg://ledu1017:1q2w3e4r@localhost/kt_db"

    # OpenAI
    OPENAI_API_KEY: str = ""

    # Vector Database (PostgreSQL + pgvector)
    PGVECTOR_DATABASE_URL: str = ""
    PGVECTOR_TABLE_NAME: str = "regulations"
    PGVECTOR_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    PGVECTOR_EMBEDDING_DIM: int = 384

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # File Upload
    MAX_UPLOAD_SIZE: int = 1073741824  # 1GB
    UPLOAD_DIR: str = "./data/uploads"

    # Scanning
    SCAN_BATCH_SIZE: int = 100
    SCAN_INTERVAL_HOURS: int = 24

    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]

    # --- PII Detector Settings ---
    PII_MODEL_PATH: str = "ParkJunSeong/PIILOT_NER_Model"
    PII_OUTPUT_DIR: str = "./output_file"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

MODEL_PATH = settings.PII_MODEL_PATH
OUTPUT_DIR = settings.PII_OUTPUT_DIR

# 상대 경로를 절대 경로로 변환 (프로젝트 루트 기준)
if not os.path.isabs(OUTPUT_DIR):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    OUTPUT_DIR = os.path.join(project_root, OUTPUT_DIR.lstrip('./'))

# 절대 경로로 변환
OUTPUT_DIR = os.path.abspath(OUTPUT_DIR)

os.makedirs(OUTPUT_DIR, exist_ok=True)
