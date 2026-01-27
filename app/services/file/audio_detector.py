"""
음성 개인정보 탐지 (LLM)
"""
from typing import List, Dict
from app.utils.logger import logger
from app.services.file.audio_masking import audio_pii_service

class AudioDetector:
    """LLM 기반 음성 개인정보 탐지"""
    
    def __init__(self):
        logger.info("AudioDetector 초기화")
        # 모델은 실제로 사용할 때 로드됩니다 (Lazy Loading)
        pass
    
    def detect(self, audio_data: str) -> List[Dict]:
        """
        음성 파일에서 개인정보 탐지
        
        Args:
            audio_data: 음성 파일 경로 또는 base64 인코딩된 오디오 데이터
            
        Returns:
            [
                {
                    "type": "name",
                    "value": "홍길동",
                    "start_time": 1.5,
                    "end_time": 2.0,
                    "confidence": 0.95
                },
                ...
            ]
        """
        try:
            logger.info("음성 탐지 시작")
            
            # 오디오 마스킹 서비스의 탐지 기능 사용
            result = audio_pii_service.transcribe_and_detect(audio_data)
            
            detected_items = result.get("detected_items", [])
            logger.info(f"음성 탐지 완료: {len(detected_items)}개 항목 발견")
            
            return detected_items
            
        except Exception as e:
            logger.error(f"음성 탐지 중 오류 발생: {str(e)}")
            raise e

