"""
DB 관련 AI 처리 API
"""
from typing import List
from fastapi import APIRouter, Depends
from app.schemas.db import (
    ColumnDetectionRequest,
    ColumnDetectionResponse,
    EncryptionCheckRequest,
    EncryptionCheckResponse,
    EncryptionCheckResult,
    PIIColumnDetectRequest,
    PIIColumnDetectResponse,
    PIIColumnResult,
    SupportedDbmsItem,
    SupportedDbmsResponse,
)
from app.api.deps import get_column_detector, get_encryption_classifier, get_pii_column_classifier
from app.core.logging import logger

router = APIRouter()

# 암호화 확인 등에서 접속 가능한 DBMS 목록 (dbms_types.name과 매칭 시 참고)
SUPPORTED_DBMS_LIST = [
    SupportedDbmsItem(id="postgresql", name="PostgreSQL"),
    SupportedDbmsItem(id="mysql", name="MySQL"),
    SupportedDbmsItem(id="oracle", name="Oracle"),
]


@router.get("/supported-dbms", response_model=SupportedDbmsResponse)
async def get_supported_dbms():
    """접속 가능한 DBMS 목록 (connection 등록 시 dbms_type_id 참고용)"""
    return SupportedDbmsResponse(dbms_list=SUPPORTED_DBMS_LIST)


@router.post("/detect-columns", response_model=ColumnDetectionResponse)
async def detect_personal_info_columns(
    request: ColumnDetectionRequest,
    column_detector=Depends(get_column_detector),
):
    """개인정보 컬럼 탐지"""
    logger.info(f"컬럼 탐지 요청: {request.schema_info.get('table_name', 'unknown')}")

    detected_columns = column_detector.detect_personal_info_columns(
        request.schema_info
    )

    return ColumnDetectionResponse(detected_columns=detected_columns)


@router.post("/check-encryption", response_model=EncryptionCheckResponse)
async def check_encryption(
    request: EncryptionCheckRequest,
    encryption_classifier=Depends(get_encryption_classifier),
):
    """암호화 여부 확인 (connectionId + piiColumns, 전체 행 조회, 미암호화 행 PK 목록 반환)"""
    logger.info(f"암호화 확인 요청: connectionId={request.connectionId}, {len(request.piiColumns)}개 컬럼")

    results: List[EncryptionCheckResult] = []
    connection_id = str(request.connectionId)

    for item in request.piiColumns:
        # keyColumn 생략 시 None 전달 → 서비스에서 PK 자동 조회
        key_column = item.keyColumn
        classification_result = encryption_classifier.classify(
            connection_id=connection_id,
            table_name=item.tableName,
            column_name=item.columnName,
            key_column=key_column,
        )
        results.append(
            EncryptionCheckResult(
                tableName=item.tableName,
                columnName=item.columnName,
                piiType=item.piiType,
                totalRecordsCount=classification_result.get("total_records", 0),
                encRecordsCount=classification_result.get("encrypted_records", 0),
                unencRecordsKeys=classification_result.get("unenc_record_keys", []),
            )
        )

    return EncryptionCheckResponse(results=results)


@router.post("/detect-pii-columns", response_model=PIIColumnDetectResponse)
async def detect_pii_columns(
    request: PIIColumnDetectRequest,
    classifier=Depends(get_pii_column_classifier),
):
    """PII 컬럼 유형 탐지 (RAG 기반)"""
    logger.info(f"PII 컬럼 탐지 요청: {len(request.tables)}개 테이블")

    # 요청 데이터를 Dict 형태로 변환
    tables_dict = [
        {"tableName": t.tableName, "columns": t.columns}
        for t in request.tables
    ]

    # PII 분류 실행
    pii_results = classifier.classify(tables=tables_dict)

    # 응답 생성
    pii_columns = [PIIColumnResult(**r) for r in pii_results]

    return PIIColumnDetectResponse(piiColumns=pii_columns)
