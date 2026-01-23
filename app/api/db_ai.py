"""
DB 관련 AI 처리 API
"""
from fastapi import APIRouter, HTTPException
from app.models.request import ColumnDetectionRequest, EncryptionCheckRequest
from app.models.response import (
    ColumnDetectionResponse,
    EncryptionCheckResponse
)
from app.services.db.column_detector import ColumnDetector
from app.services.db.encryption_classifier import EncryptionClassifier
from app.utils.logger import logger

router = APIRouter()

# 서비스 인스턴스 (싱글톤 패턴 고려)
column_detector = ColumnDetector()
encryption_classifier = EncryptionClassifier()


@router.post("/detect-columns", response_model=ColumnDetectionResponse)
async def detect_personal_info_columns(request: ColumnDetectionRequest):
    """
    개인정보 컬럼 탐지
    
    Spring Boot에서 스키마 정보를 전달받아
    LLM + LangChain으로 개인정보 컬럼을 탐지
    """
    try:
        logger.info(f"컬럼 탐지 요청: {request.schema_info.get('table_name', 'unknown')}")
        
        detected_columns = column_detector.detect_personal_info_columns(
            request.schema_info
        )
        
        return ColumnDetectionResponse(detected_columns=detected_columns)
    
    except Exception as e:
        logger.error(f"컬럼 탐지 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/check-encryption", response_model=EncryptionCheckResponse)
async def check_encryption(request: EncryptionCheckRequest):
    """
    암호화 여부 확인
    
    Spring Boot에서 데이터 샘플을 전달받아
    분류 모델로 암호화 여부를 판단
    """
    try:
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
                column_name=column_name
            )
            
            results.append({
                "column": f"{table_name}.{column_name}",
                "total_records": classification_result.get("total_records", 0),
                "encrypted_records": classification_result.get("encrypted_records", 0)
            })
        
        return EncryptionCheckResponse(results=results)
    
    except Exception as e:
        logger.error(f"암호화 확인 오류: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

