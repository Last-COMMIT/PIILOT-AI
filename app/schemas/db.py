"""
DB AI 요청/응답 스키마
"""
from pydantic import BaseModel
from typing import List, Dict


# ========== 요청 ==========

class ColumnDetectionRequest(BaseModel):
    """개인정보 컬럼 탐지 요청"""
    schema_info: Dict


class EncryptionCheckRequest(BaseModel):
    """암호화 여부 확인 요청"""
    data_samples: List[Dict]


# ========== 응답 ==========

class DetectedColumn(BaseModel):
    """탐지된 개인정보 컬럼"""
    table_name: str
    column_name: str
    personal_info_types: List[str]
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
