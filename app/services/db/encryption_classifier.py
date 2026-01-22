"""
암호화 여부 판단 (분류 모델)
"""
from typing import Dict
from app.utils.logger import logger


class EncryptionClassifier:
    """암호화 여부 판단 분류 모델"""
    
    def __init__(self, model_path: str = None):
        """
        Args:
            model_path: 학습된 모델 경로 (없으면 기본 모델 사용)
        """
        # TODO: 모델 로드
        # - 분류 모델 초기화
        # - 토크나이저 로드
        self.model_path = model_path
        logger.info(f"EncryptionClassifier 초기화: {model_path}")
        pass
    
    def classify(self, data_sample: str) -> Dict[str, float]:
        """
        데이터 샘플의 암호화 여부 판단
        
        Args:
            data_sample: 컬럼의 데이터 샘플
            
        Returns:
            {"encrypted": 0.8, "plain": 0.2} 형태의 확률
        """
        # TODO: 구현 필요
        # - 데이터 샘플 전처리
        # - 모델 추론
        # - 확률 반환
        logger.debug(f"암호화 여부 판단: 샘플 길이 {len(data_sample)}")
        pass
    
    def is_encrypted(self, data_sample: str, threshold: float = 0.5) -> bool:
        """
        암호화 여부를 boolean으로 반환
        
        Args:
            data_sample: 데이터 샘플
            threshold: 판단 임계값 (기본 0.5)
            
        Returns:
            암호화 여부
        """
        result = self.classify(data_sample)
        return result.get("encrypted", 0.0) > threshold

