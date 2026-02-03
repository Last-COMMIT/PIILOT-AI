"""
FastAPI 의존성 주입 (Depends)
서비스 인스턴스를 싱글톤으로 관리
"""
import sys
from pathlib import Path
from functools import lru_cache
from app.core.logging import logger


@lru_cache
def get_column_detector():
    from app.services.db.column_detection_service import ColumnDetector
    logger.info("ColumnDetector 의존성 생성")
    return ColumnDetector()


@lru_cache
def get_encryption_classifier():
    from app.services.db.encryption_service import EncryptionClassifier
    logger.info("EncryptionClassifier 의존성 생성")
    return EncryptionClassifier()


@lru_cache
def get_document_detector():
    from app.services.file.document_service import DocumentDetector
    from app.core.config import MODEL_PATH
    logger.info("DocumentDetector 의존성 생성")
    return DocumentDetector(model_path=MODEL_PATH)


@lru_cache
def get_image_detector():
    from app.ml.image_detector import ImageDetector
    logger.info("ImageDetector 의존성 생성")
    return ImageDetector()


@lru_cache
def get_audio_detector():
    from app.services.file.audio_service import AudioDetector
    logger.info("AudioDetector 의존성 생성")
    return AudioDetector()


@lru_cache
def get_video_detector():
    from app.services.file.video_service import VideoDetector
    logger.info("VideoDetector 의존성 생성")
    return VideoDetector()


@lru_cache
def get_masker():
    from app.services.file.masking_service import Masker
    logger.info("Masker 의존성 생성")
    return Masker()


@lru_cache
def get_document_file_processor():
    """PDF/DOCX/TXT 파일 단위 탐지·마스킹용 (process_file)"""
    from app.services.file.document_detector import DocumentDetector as DocumentFileProcessor
    from app.core.config import MODEL_PATH
    logger.info("DocumentFileProcessor 의존성 생성")
    return DocumentFileProcessor(model_path=MODEL_PATH)


@lru_cache
def get_assistant():
    from app.services.chat.assistant_service import AIAssistant
    logger.info("AIAssistant 의존성 생성")
    return AIAssistant()


@lru_cache
def get_regulation_search():
    from app.services.chat.regulation_service import RegulationSearch
    logger.info("RegulationSearch 의존성 생성")
    return RegulationSearch()


@lru_cache
def get_pii_column_classifier():
    from app.services.db.pii_column_classifier import PIIColumnClassifier
    logger.info("PIIColumnClassifier 의존성 생성")
    return PIIColumnClassifier()


@lru_cache
def get_langgraph_chatbot():
    """LangGraph 챗봇 앱 의존성"""
    from app.services.chat.langgraph_chatbot import create_chatbot_app
    logger.info("LangGraph 챗봇 의존성 생성")
    return create_chatbot_app()


def get_regulation_upload():
    """법령 PDF 업로드 의존성"""
    # scripts를 찾을 수 있도록 프로젝트 루트를 sys.path 맨 앞에 항상 추가
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root))
    from scripts.setup_vector_db import law_pdf_to_vector
    logger.info("법령 PDF 업로드 의존성 생성")
    return law_pdf_to_vector

def get_column_dictionary_upload():
    """DB 단어사전 PDF 업로드 의존성"""
    from app.services.db.column_dictionary_vector_embedding import column_dictionary_to_vector
    logger.info("DB 단어사전 PDF 업로드 의존성 생성")
    return column_dictionary_to_vector