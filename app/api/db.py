"""
DB 관련 AI 처리 API
"""
from fastapi import APIRouter, Depends
from app.schemas.db import (
    ColumnDetectionRequest,
    ColumnDetectionResponse,
    EncryptionCheckRequest,
    EncryptionCheckResponse,
    PIIColumnDetectRequest,
    PIIColumnDetectResponse,
    PIIColumnResult,
)
from app.api.deps import get_column_detector, get_encryption_classifier, get_pii_column_classifier
from app.core.logging import logger

router = APIRouter()


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
    """암호화 여부 확인"""
    logger.info(f"암호화 확인 요청: {len(request.data_samples)}개 컬럼 암호화 여부 확인")

    results = []
    for db_info in request.data_samples:
        connection_id = str(db_info.get("connection_id"))
        table_name = db_info.get("table_name")
        column_name = db_info.get("column_name")

        if not all([connection_id, table_name, column_name]):
            logger.warning(f"필수 파라미터 누락: {db_info}")
            continue

        classification_result = encryption_classifier.classify(
            connection_id=connection_id,
            table_name=table_name,
            column_name=column_name,
        )

        results.append({
            "column": f"{table_name}.{column_name}",
            "total_records": classification_result.get("total_records", 0),
            "encrypted_records": classification_result.get("encrypted_records", 0),
        })

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
