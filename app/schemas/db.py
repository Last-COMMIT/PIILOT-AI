"""
DB AI 요청/응답 스키마
"""
from pydantic import BaseModel
from typing import List, Dict, Any, Optional


# ========== 요청 ==========

class ColumnDetectionRequest(BaseModel):
    """개인정보 컬럼 탐지 요청"""
    schema_info: Dict


class PiiColumnItem(BaseModel):
    """암호화 확인 대상 컬럼 (tableName, columnName, piiType, keyColumn)"""
    tableName: str
    columnName: str
    piiType: str  # 표준약어 NM, EM, PH, ADD 등
    keyColumn: Optional[str] = None  # PK 컬럼명. None이면 DB 메타데이터에서 자동 조회


class EncryptionCheckRequest(BaseModel):
    """암호화 여부 확인 요청"""
    connectionId: int
    piiColumns: List[PiiColumnItem]

class TableColumns(BaseModel):
    """테이블과 컬럼 정보"""
    tableName: str
    columns: List[str]

class PIIColumnDetectRequest(BaseModel):
    """PII 컬럼 탐지 요청"""
    tables: List[TableColumns]

class ColumnDictionaryUploadRequest(BaseModel):
    """DB 단어사전 업로드 요청"""
    file_path: str


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


class PIIColumnResult(BaseModel):
    """개별 PII 컬럼 결과"""
    tableName: str
    columnName: str
    piiType: str # 표준약어 (NM, EM, PH, ADD, etc)

class PIIColumnDetectResponse(BaseModel):
    """PII 컬럼 탐지 응답"""
    piiColumns: List[PIIColumnResult]


class EncryptionCheckResult(BaseModel):
    """암호화 여부 확인 결과 (camelCase)"""
    tableName: str
    columnName: str
    piiType: str
    keyColumn: str  # 암호화 체크에 사용된 PK 컬럼명
    totalRecordsCount: int
    encRecordsCount: int
    unencRecordsKeys: List[Any]  # 미암호화(PII 감지) 행의 PK 값 목록


class EncryptionCheckResponse(BaseModel):
    """암호화 여부 확인 응답"""
    results: List[EncryptionCheckResult]


class ColumnDictionaryUploadResponse(BaseModel):
    """DB 단어사전 업로드 응답"""
    status: str
class SupportedDbmsItem(BaseModel):
    """지원 DBMS 항목 (connection 등록 시 dbms_type_id 참고용)"""
    id: str
    name: str


class SupportedDbmsResponse(BaseModel):
    """접속 가능한 DBMS 목록 응답"""
    dbms_list: List[SupportedDbmsItem]
