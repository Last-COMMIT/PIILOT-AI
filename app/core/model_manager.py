"""
모델 다운로드 및 관리
FastAPI 시작 시 모든 모델을 다운로드하여 준비
"""
import os
from pathlib import Path
from app.core.logging import logger
from app.core.config import get_project_root
from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification


class ModelManager:
    """모델 다운로드 및 관리"""
    
    CACHE_DIR: str = None  # setup_cache_dir()에서 초기화됨
    WHISPER_CACHE_DIR: str = None  # setup_cache_dir()에서 초기화됨
    FLASHRANK_CACHE_DIR: str = None  # setup_cache_dir()에서 초기화됨
    
    # 사용하는 모든 HuggingFace 모델 목록
    HUGGINGFACE_MODELS = {
        "pii_ner": {
            "name": "ParkJunSeong/PIILOT_NER_Model",
            "type": "token_classification"
        },
        "embedding": {
            "name": "intfloat/multilingual-e5-large-instruct",
            "type": "auto"
        },
        "llm": {
            "name": "K-intelligence/Midm-2.0-Mini-Instruct",
            "type": "auto"
        },
    }
    
    # Whisper 모델 설정
    WHISPER_MODEL_SIZE = "large-v3"
    
    # 로컬 모델 경로 (상대 경로, 프로젝트 루트 기준)
    LOCAL_MODELS = {
        "yolo_face": "models/vision/yolov12n-face.pt",
        "xgboost": "models/encryption_classifier/pii_xgboost_model.pkl",
        "xgboost_acn": "models/encryption_classifier/pii_acn_binary_model.pkl",
    }
    
    @classmethod
    def setup_cache_dir(cls):
        """캐시 디렉토리 설정 및 환경 변수 설정"""
        project_root = get_project_root()
        if cls.CACHE_DIR is None:
            cache_dir = project_root / "models" / "huggingface"
            cls.CACHE_DIR = str(cache_dir)
        if cls.WHISPER_CACHE_DIR is None:
            whisper_dir = project_root / "models" / "whisper"
            cls.WHISPER_CACHE_DIR = str(whisper_dir)
        if cls.FLASHRANK_CACHE_DIR is None:
            flashrank_dir = project_root / "models" / "flashrank"
            cls.FLASHRANK_CACHE_DIR = str(flashrank_dir)
        
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.WHISPER_CACHE_DIR, exist_ok=True)
        os.makedirs(cls.FLASHRANK_CACHE_DIR, exist_ok=True)
        os.environ['HF_HOME'] = cls.CACHE_DIR
        os.environ['TRANSFORMERS_CACHE'] = cls.CACHE_DIR
        # Flashrank 모델 캐시 디렉토리 환경변수 설정
        os.environ['FLASHRANK_CACHE_DIR'] = cls.FLASHRANK_CACHE_DIR
        logger.info(f"모델 캐시 디렉토리 설정: {cls.CACHE_DIR}")
        logger.info(f"Whisper 캐시 디렉토리 설정: {cls.WHISPER_CACHE_DIR}")
        logger.info(f"Flashrank 캐시 디렉토리 설정: {cls.FLASHRANK_CACHE_DIR}")
    
    @classmethod
    def download_huggingface_model(cls, model_name: str, model_type: str = "auto"):
        """HuggingFace 모델 다운로드"""
        try:
            logger.info(f"모델 다운로드 중: {model_name}")
            
            if model_type == "tokenizer":
                AutoTokenizer.from_pretrained(model_name, cache_dir=cls.CACHE_DIR)
            elif model_type == "model":
                AutoModel.from_pretrained(model_name, cache_dir=cls.CACHE_DIR)
            elif model_type == "token_classification":
                # Tokenizer와 Model 모두 필요
                AutoTokenizer.from_pretrained(model_name, cache_dir=cls.CACHE_DIR)
                AutoModelForTokenClassification.from_pretrained(
                    model_name, cache_dir=cls.CACHE_DIR
                )
            else:  # auto - tokenizer와 model 모두
                AutoTokenizer.from_pretrained(model_name, cache_dir=cls.CACHE_DIR)
                AutoModel.from_pretrained(model_name, cache_dir=cls.CACHE_DIR)
            
            logger.info(f"✓ 모델 다운로드 완료: {model_name}")
        except Exception as e:
            logger.error(f"✗ 모델 다운로드 실패: {model_name} - {e}")
            raise
    
    @classmethod
    def download_whisper_model(cls, model_size: str = "large-v3"):
        """Whisper 모델 다운로드 (faster-whisper). 프로젝트 models/whisper에 저장."""
        try:
            cls.setup_cache_dir()
            download_root = cls.WHISPER_CACHE_DIR
            # 지연 import: faster-whisper은 선택적 의존성
            from faster_whisper import WhisperModel
            
            logger.info(f"Whisper 모델 다운로드 중: {model_size} -> {download_root}")
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "int8"  # 다운로드용으로는 int8 사용
            model = WhisperModel(
                model_size, device=device, compute_type=compute_type,
                download_root=download_root,
            )
            logger.info(f"✓ Whisper 모델 다운로드 완료: {model_size}")
            del model  # 메모리 해제
        except ImportError as e:
            logger.warning(f"faster-whisper이 설치되지 않았습니다. Whisper 기능을 사용할 수 없습니다: {e}")
            raise
        except Exception as e:
            logger.error(f"✗ Whisper 모델 다운로드 실패: {model_size} - {e}")
            raise
    
    @classmethod
    def get_local_model_path(cls, model_key: str) -> str:
        """로컬 모델의 절대 경로 반환"""
        if model_key not in cls.LOCAL_MODELS:
            raise ValueError(f"알 수 없는 모델 키: {model_key}")
        
        relative_path = cls.LOCAL_MODELS[model_key]
        project_root = get_project_root()
        return str(project_root / relative_path)
    
    @classmethod
    def check_local_model(cls, model_path: str) -> bool:
        """로컬 모델 파일 존재 확인 (프로젝트 루트 기준)"""
        # 절대 경로가 아니면 프로젝트 루트 기준으로 변환
        if not os.path.isabs(model_path):
            project_root = get_project_root()
            model_path = str(project_root / model_path)
        return os.path.exists(model_path)
    
    @classmethod
    def ensure_all_models(cls):
        """모든 모델 다운로드 확인 및 다운로드"""
        logger.info("=" * 60)
        logger.info("모델 준비 상태 확인 시작")
        logger.info("=" * 60)
        
        # 1. 캐시 디렉토리 설정
        cls.setup_cache_dir()
        
        # 2. HuggingFace 모델들 다운로드
        logger.info("\n[HuggingFace 모델 다운로드]")
        for model_key, model_info in cls.HUGGINGFACE_MODELS.items():
            model_name = model_info["name"]
            model_type = model_info["type"]
            try:
                cls.download_huggingface_model(model_name, model_type)
            except Exception as e:
                logger.warning(
                    f"{model_key} 모델 다운로드 실패 (서비스는 계속 실행됩니다): {e}"
                )
        
        # 3. Whisper 모델 다운로드
        logger.info("\n[Whisper 모델 다운로드]")
        try:
            cls.download_whisper_model(cls.WHISPER_MODEL_SIZE)
        except Exception as e:
            logger.warning(
                f"Whisper 모델 다운로드 실패 (서비스는 계속 실행됩니다): {e}"
            )
        
        # 4. 로컬 모델 확인
        logger.info("\n[로컬 모델 확인]")
        for model_key, model_path in cls.LOCAL_MODELS.items():
            if cls.check_local_model(model_path):
                logger.info(f"✓ 로컬 모델 확인: {model_key} ({model_path})")
            else:
                logger.warning(f"⚠ 로컬 모델 없음: {model_key} ({model_path})")
        
        logger.info("=" * 60)
        logger.info("모델 준비 완료")
        logger.info("=" * 60)
    
    @classmethod
    def get_cache_dir(cls) -> str:
        """캐시 디렉토리 경로 반환"""
        return cls.CACHE_DIR

    @classmethod
    def get_whisper_cache_dir(cls) -> str:
        """Whisper 모델 캐시 디렉토리 경로 반환 (다운로드/로드 시 download_root로 사용)"""
        if cls.WHISPER_CACHE_DIR is None:
            cls.setup_cache_dir()
        return cls.WHISPER_CACHE_DIR
    
    @classmethod
    def get_flashrank_cache_dir(cls) -> str:
        """Flashrank 모델 캐시 디렉토리 경로 반환"""
        if cls.FLASHRANK_CACHE_DIR is None:
            cls.setup_cache_dir()
        return cls.FLASHRANK_CACHE_DIR
