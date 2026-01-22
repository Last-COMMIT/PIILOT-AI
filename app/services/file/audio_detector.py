"""
음성 개인정보 탐지 (LLM)
"""
from typing import List, Dict
from app.utils.logger import logger


class AudioDetector:
    """LLM 기반 음성 개인정보 탐지"""
    
    def __init__(self):
        # TODO: Whisper 모델 로드 (음성 → 텍스트)
        # TODO: LLM 초기화 (텍스트에서 개인정보 탐지)
        logger.info("AudioDetector 초기화")
        pass
    
    def detect(self, audio_path: str) -> List[Dict]:
        """
        음성 파일에서 개인정보 탐지
        
        Args:
            audio_path: 음성 파일 경로 또는 base64 인코딩된 오디오
            
        Returns:
            [
                {
                    "type": "name",  # 개인정보 타입
                    "value": "홍길동",  # 탐지된 값
                    "start_time": 1.5,  # 시작 시간 (초)
                    "end_time": 2.0,  # 끝 시간 (초)
                    "confidence": 0.95  # 신뢰도
                },
                ...
            ]
        """
        # TODO: 구현 필요
        # - 음성 → 텍스트 변환 (Whisper)
        # - 텍스트에서 개인정보 탐지 (LLM)
        logger.info(f"음성 탐지 시작: {audio_path}")
        pass

