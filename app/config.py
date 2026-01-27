"""
애플리케이션 설정 관리
"""
import os
import sys
from pydantic_settings import BaseSettings
from typing import List

# Import PII constants from models
try:
    from app.models.personal_info import PII_CATEGORIES, PII_NAMES, CONFIDENCE_THRESHOLDS
except ImportError:
    # Fallback/Safety check
    PII_CATEGORIES = []
    PII_NAMES = {}
    CONFIDENCE_THRESHOLDS = {}

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
    # - 기본값은 DATABASE_URL을 그대로 사용
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

os.makedirs(OUTPUT_DIR, exist_ok=True)