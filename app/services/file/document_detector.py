"""
문서 개인정보 탐지 (BERT + NER)
"""
from typing import List, Dict
from app.utils.logger import logger


class DocumentDetector:
    """BERT + NER 기반 문서 개인정보 탐지"""
    
    def __init__(self, model_path: str = None):
        """
        Args:
            model_path: 학습된 BERT + NER 모델 경로
        """
        # TODO: 모델 로드
        # - BERT 모델 초기화
        # - NER 모델 초기화
        self.model_path = model_path
        logger.info(f"DocumentDetector 초기화: {model_path}")
        pass
    
    def detect(self, text: str) -> List[Dict]:
        """
        문서에서 개인정보 탐지
        
        Args:
            text: 문서 텍스트
            
        Returns:
            [
                {
                    "type": "name",  # 개인정보 타입
                    "value": "홍길동",  # 탐지된 값
                    "start": 0,  # 시작 위치
                    "end": 3,  # 끝 위치
                    "confidence": 0.98  # 신뢰도
                },
                ...
            ]
        """
        # TODO: 구현 필요
        # - 텍스트 전처리
        # - 토큰화
        # - NER 수행
        # - 개인정보 8종류 필터링
        logger.info(f"문서 탐지 시작: 텍스트 길이 {len(text)}")
        pass

