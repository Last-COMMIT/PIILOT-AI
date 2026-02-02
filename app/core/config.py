"""
애플리케이션 설정 관리
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import List


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

    # DB 비밀번호 복호화 (백엔드 AES-256-GCM과 동일 키. 환경 변수 ENCRYPTION_AES_KEY, 32바이트 UTF-8)
    ENCRYPTION_AES_KEY: str = "piilot-aes-256-secret-key-32by!!"

    # --- PII Detector Settings ---
    PII_MODEL_PATH: str = "ParkJunSeong/PIILOT_NER_Model"

    # --- 영상 화면 텍스트 PII (OCR) ---
    VIDEO_KEYFRAME_INTERVAL: int = 20  # N프레임마다 OCR (작을수록 탐지 밀도↑, 비용↑)
    VIDEO_TEXT_PII_PADDING_PX: int = 4   # 텍스트 bbox 주변 패딩(px), 0이면 OCR bbox 그대로 (범위 작게)
    VIDEO_TEXT_PII_EXTEND_HALF: int = 25  # 키프레임 기준 전후 N프레임까지 동일 영역 적용 (겹침으로 끊김 방지)
    
    # Output Directories (모든 출력 파일 경로 통일)
    OUTPUT_BASE_DIR: str = "./output_file"
    OUTPUT_DIR: str = "./output_file/documents"  # 문서 마스킹 결과
    IMAGE_OUTPUT_DIR: str = "./output_file/images"  # 이미지 마스킹 결과
    VIDEO_OUTPUT_DIR: str = "./output_file/videos"  # 비디오 마스킹 결과
    AUDIO_OUTPUT_DIR: str = "./output_file/audio"  # 오디오 마스킹 결과

    class Config:
        env_file = ".env"
        case_sensitive = True


def get_project_root() -> Path:
    """프로젝트 루트 경로 반환 (requirements.txt 기준)"""
    current = Path(__file__).resolve().parent
    
    # requirements.txt가 있는 디렉토리를 찾을 때까지 상위로 이동
    while current != current.parent:
        if (current / 'requirements.txt').exists():
            return current
        current = current.parent
    
    # requirements.txt를 찾지 못한 경우 fallback
    raise RuntimeError("프로젝트 루트(requirements.txt)를 찾을 수 없습니다.")


def _resolve_output_path(relative_path: str) -> str:
    """상대 경로를 절대 경로로 변환 (프로젝트 루트 기준)"""
    if os.path.isabs(relative_path):
        return relative_path
    
    project_root = get_project_root()
    # './' 제거
    clean_path = relative_path.lstrip('./')
    absolute_path = project_root / clean_path
    return str(absolute_path.resolve())


settings = Settings()

# 모델 경로
MODEL_PATH = settings.PII_MODEL_PATH

# 출력 디렉토리 (절대 경로로 변환)
OUTPUT_BASE_DIR = _resolve_output_path(settings.OUTPUT_BASE_DIR)
OUTPUT_DIR = _resolve_output_path(settings.OUTPUT_DIR)
IMAGE_OUTPUT_DIR = _resolve_output_path(settings.IMAGE_OUTPUT_DIR)
VIDEO_OUTPUT_DIR = _resolve_output_path(settings.VIDEO_OUTPUT_DIR)
AUDIO_OUTPUT_DIR = _resolve_output_path(settings.AUDIO_OUTPUT_DIR)

# 출력 디렉토리 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)
