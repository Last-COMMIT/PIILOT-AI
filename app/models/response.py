"""
응답 모델 (AI 서비스 → Spring Boot)
"""
from pydantic import BaseModel
from typing import List, Dict, Optional


# ========== DB AI 응답 ==========

class DetectedColumn(BaseModel):
    """탐지된 개인정보 컬럼"""
    table_name: str
    column_name: str
    personal_info_types: List[str]  # ["name", "phone", ...]
    confidence: float


class ColumnDetectionResponse(BaseModel):
    """개인정보 컬럼 탐지 응답"""
    detected_columns: List[DetectedColumn]


class EncryptionCheckResult(BaseModel):
    """암호화 여부 확인 결과"""
    column: str
    total_records: int
    encrypted_records: int


class EncryptionCheckResponse(BaseModel):
    """암호화 여부 확인 응답"""
    results: List[EncryptionCheckResult]


# ========== File AI 응답 ==========

class DetectedPersonalInfo(BaseModel):
    """탐지된 개인정보"""
    type: str  # 개인정보 타입
    value: Optional[str] = None  # 탐지된 값 (문서/음성)
    start: Optional[int] = None  # 시작 위치 (문서)
    end: Optional[int] = None  # 끝 위치 (문서)
    start_time: Optional[float] = None  # 시작 시간 (음성/영상)
    end_time: Optional[float] = None  # 끝 시간 (음성/영상)
    confidence: float


class DetectedFace(BaseModel):
    """탐지된 얼굴"""
    x: int
    y: int
    width: int
    height: int
    confidence: float
    frame_number: Optional[int] = None  # 영상의 경우


class DocumentDetectionResponse(BaseModel):
    """문서 탐지 응답"""
    detected_items: List[DetectedPersonalInfo]
    is_masked: bool = False


class ImageDetectionResponse(BaseModel):
    """이미지 탐지 응답"""
    detected_faces: List[DetectedFace]


class AudioDetectionResponse(BaseModel):
    """음성 탐지 응답"""
    detected_items: List[DetectedPersonalInfo]


class VideoDetectionResponse(BaseModel):
    """영상 탐지 응답"""
    faces: List[DetectedFace]
    personal_info_in_audio: List[DetectedPersonalInfo]


class MaskingResponse(BaseModel):
    """마스킹 처리 응답"""
    masked_file: str  # base64 인코딩된 마스킹된 파일
    file_type: str  # "document", "image", "audio", "video"


# ========== Chat AI 응답 ==========

class ChatResponse(BaseModel):
    """AI 어시스턴트 응답"""
    answer: str  # AI 응답
    sources: List[str] = []  # 참조한 법령 출처


class RegulationSearchResult(BaseModel):
    """법령 검색 결과"""
    text: str  # 법령 텍스트
    source: str  # 출처
    score: float  # 유사도 점수


class RegulationSearchResponse(BaseModel):
    """법령 검색 응답"""
    results: List[RegulationSearchResult]

