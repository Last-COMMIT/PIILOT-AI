"""
요청 모델 (Spring Boot → AI 서비스)
"""
from pydantic import BaseModel
from typing import List, Dict, Optional


# ========== DB AI 요청 ==========

class ColumnDetectionRequest(BaseModel):
    """개인정보 컬럼 탐지 요청"""
    schema_info: Dict  # 스키마 정보
    # 예: {
    #   "table_name": "users",
    #   "columns": [
    #     {"name": "id", "type": "integer"},
    #     {"name": "user_name", "type": "varchar"},
    #     ...
    #   ]
    # }


class EncryptionCheckRequest(BaseModel):
    """암호화 여부 확인 요청"""
    data_samples: List[Dict]
    # 예: [
    #   {"connection_id": "1", "table_name": "sdb4_a_counsel_log", "column_name": "counselor_nm"},
    #   {"connection_id": "1", "table_name": "sdb4_a_cust_detail", "column_name": "cust_nm"},
    # ...
    # ]

# ========== File AI 요청 ==========

class DocumentDetectionRequest(BaseModel):
    """문서 개인정보 탐지 요청"""
    file_content: str  # 텍스트 내용
    file_type: Optional[str] = "txt"  # 파일 타입


class ImageDetectionRequest(BaseModel):
    """이미지 얼굴 탐지 요청"""
    image_data: str  # base64 인코딩된 이미지 또는 파일 경로
    image_format: Optional[str] = "base64"  # "base64" or "path"


class AudioDetectionRequest(BaseModel):
    """음성 개인정보 탐지 요청"""
    audio_data: str  # base64 인코딩된 오디오 또는 파일 경로
    audio_format: Optional[str] = "base64"  # "base64" or "path"


class VideoDetectionRequest(BaseModel):
    """영상 개인정보 탐지 요청"""
    video_data: str  # base64 인코딩된 비디오 또는 파일 경로
    video_format: Optional[str] = "base64"  # "base64" or "path"


class MaskingRequest(BaseModel):
    """마스킹 처리 요청"""
    file_type: str  # "document", "image", "audio", "video"
    file_data: str  # 원본 파일 데이터
    detected_items: List[Dict]  # 탐지된 개인정보 리스트
    # document: [{"type": "name", "value": "홍길동", "start": 0, "end": 3}, ...]
    # image: [{"x": 100, "y": 200, "width": 50, "height": 50}, ...]
    # audio: [{"type": "name", "start_time": 1.5, "end_time": 2.0}, ...]
    # video: {"faces": [...], "audio_items": [...]}


# ========== Chat AI 요청 ==========

class ChatRequest(BaseModel):
    """AI 어시스턴트 질의 요청"""
    query: str  # 사용자 질의
    context: Optional[Dict] = None  # 추가 컨텍스트
    # 예: {"encryption_rate": 75.5, "total_columns": 100, ...}


class RegulationSearchRequest(BaseModel):
    """법령 검색 요청"""
    query: str  # 검색 쿼리
    n_results: Optional[int] = 10  # 검색할 결과 수
    top_n: Optional[int] = 5  # Reranking 후 상위 N개 문서 선택