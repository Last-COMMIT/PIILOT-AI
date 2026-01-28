"""
오디오 PII 탐지 서비스
(기존 audio_detector.py 리팩토링)
"""
from typing import List, Dict
from app.core.logging import logger
from app.services.file.processors.audio_masker import AudioMasker


class AudioDetector:
    """오디오 PII 탐지"""

    def __init__(self):
        logger.info("AudioDetector 초기화")
        self._masker = AudioMasker()

    def detect(self, audio_data: str) -> List[Dict]:
        """음성 파일에서 개인정보 탐지"""
        try:
            logger.info("음성 탐지 시작")
            result = self._masker.transcribe_and_detect(audio_data)
            detected_items = result.get("detected_items", [])
            logger.info(f"음성 탐지 완료: {len(detected_items)}개 항목 발견")
            return detected_items
        except Exception as e:
            logger.error(f"음성 탐지 중 오류 발생: {str(e)}")
            raise e
