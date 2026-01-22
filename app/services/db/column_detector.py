"""
개인정보 컬럼 탐지 (LLM + LangChain)
"""
from typing import List, Dict
from app.utils.logger import logger


class ColumnDetector:
    """LLM + LangChain을 사용한 개인정보 컬럼 탐지"""
    
    def __init__(self):
        # TODO: LangChain 설정
        # - LLM 초기화
        # - 프롬프트 템플릿 설정
        logger.info("ColumnDetector 초기화")
        pass
    
    def detect_personal_info_columns(
        self, 
        schema_info: Dict
    ) -> List[Dict]:
        """
        스키마 정보를 기반으로 개인정보 컬럼 탐지
        
        Args:
            schema_info: {
                "table_name": str,
                "columns": [
                    {"name": str, "type": str, ...},
                    ...
                ]
            }
            
        Returns:
            [
                {
                    "table_name": str,
                    "column_name": str,
                    "personal_info_types": List[str],  # ["name", "phone", ...]
                    "confidence": float
                },
                ...
            ]
        """
        # TODO: 구현 필요
        # - LLM에 스키마 정보 전달
        # - 개인정보 8종류 중 어떤 것이 포함되어 있는지 판단
        logger.info(f"컬럼 탐지 시작: {schema_info.get('table_name')}")
        pass

