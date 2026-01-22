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
        logger.info(f"암호화 확인 요청: {len(request.data_samples)}개 샘플")
        
        results = []
        for sample in request.data_samples:
            column = sample.get("column")
            data_sample = sample.get("sample", "")
            
            is_encrypted = encryption_classifier.is_encrypted(data_sample)
            classification_result = encryption_classifier.classify(data_sample)
            confidence = classification_result.get("encrypted", 0.0) if is_encrypted else classification_result.get("plain", 0.0)
            
            results.append({
                "column": column,
                "is_encrypted": is_encrypted,
                "confidence": confidence
            })
        
        return EncryptionCheckResponse(results=results)
    
    except Exception as e:
        logger.error(f"암호화 확인 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

