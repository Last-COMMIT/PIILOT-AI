"""
애플리케이션 설정 관리
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """애플리케이션 설정 (환경 변수 및 .env 파일에서 자동 로드)"""
    
    # ==================== Database ====================
    DATABASE_URL: str = ""  # .env 파일에서 로드
    
    # ==================== Vector Database ====================
    PGVECTOR_DATABASE_URL: str = ""
    PGVECTOR_TABLE_NAME: str = "regulations"
    PGVECTOR_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    PGVECTOR_EMBEDDING_DIM: int = 384
    
    # ==================== External APIs ====================
    OPENAI_API_KEY: str = ""  # .env 파일에서 로드
    
    # ==================== Security ====================
    ENCRYPTION_AES_KEY: str = ""  # .env 파일에서 로드 (32바이트 UTF-8)
    
    # ==================== AI Models ====================
    PII_MODEL_PATH: str = "ParkJunSeong/PIILOT_NER_Model"
    
    # ==================== Video Processing ====================
    VIDEO_KEYFRAME_INTERVAL: int = 20  # N프레임마다 OCR (작을수록 탐지 밀도↑, 비용↑)
    VIDEO_TEXT_PII_PADDING_PX: int = 4  # 텍스트 bbox 주변 패딩(px)
    VIDEO_TEXT_PII_EXTEND_HALF: int = 25  # 키프레임 기준 전후 N프레임까지 확장 적용
    
    # ==================== Output Directories ====================
    OUTPUT_BASE_DIR: str = "./output_file"
    OUTPUT_DIR: str = "./output_file/documents"
    IMAGE_OUTPUT_DIR: str = "./output_file/images"
    VIDEO_OUTPUT_DIR: str = "./output_file/videos"
    AUDIO_OUTPUT_DIR: str = "./output_file/audio"
    
    # ==================== Masking ====================
    SAVE_MASKED_OUTPUT: bool = False  # True: output_file에 저장, False: base64만 반환
    
    class Config:
        env_file = ".env"
        case_sensitive = True


def get_project_root() -> Path:
    """프로젝트 루트 경로 반환 (requirements.txt 기준)"""
    current = Path(__file__).resolve().parent
    
    while current != current.parent:
        if (current / 'requirements.txt').exists():
            return current
        current = current.parent
    
    raise RuntimeError("프로젝트 루트(requirements.txt)를 찾을 수 없습니다.")


def _resolve_output_path(relative_path: str) -> str:
    """상대 경로를 절대 경로로 변환 (프로젝트 루트 기준)"""
    if os.path.isabs(relative_path):
        return relative_path
    
    project_root = get_project_root()
    clean_path = relative_path.lstrip('./')
    absolute_path = project_root / clean_path
    return str(absolute_path.resolve())


# ==================== Settings Instance ====================
settings = Settings()

# ==================== Resolved Paths ====================
OUTPUT_BASE_DIR = _resolve_output_path(settings.OUTPUT_BASE_DIR)
OUTPUT_DIR = _resolve_output_path(settings.OUTPUT_DIR)
IMAGE_OUTPUT_DIR = _resolve_output_path(settings.IMAGE_OUTPUT_DIR)
VIDEO_OUTPUT_DIR = _resolve_output_path(settings.VIDEO_OUTPUT_DIR)
AUDIO_OUTPUT_DIR = _resolve_output_path(settings.AUDIO_OUTPUT_DIR)

# ==================== Directory Initialization ====================
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)

# ==================== Legacy Exports (하위 호환성) ====================
MODEL_PATH = settings.PII_MODEL_PATH
SAVE_MASKED_OUTPUT = settings.SAVE_MASKED_OUTPUT
