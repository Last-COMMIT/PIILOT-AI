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

    def detect(self, audio_data: str, audio_format: str = "base64") -> List[Dict]:
        """
        음성 파일에서 개인정보 탐지
        
        Args:
            audio_data: 음성 파일 경로 또는 base64 인코딩된 오디오 데이터
            audio_format: "base64" 또는 "path" (기본값: "base64")
            
        Returns:
            탐지된 개인정보 항목 리스트
        """
        try:
            logger.info(f"음성 탐지 시작 (포맷: {audio_format})")
            
            # 포맷에 따라 처리
            if audio_format == "base64":
                # base64 디코딩하여 bytes로 변환
                from app.utils.base64_utils import decode_base64_data
                audio_bytes = decode_base64_data(audio_data)
                audio_input = audio_bytes
            else:  # "path"
                # 파일 경로 그대로 사용
                import os
                if not os.path.exists(audio_data):
                    raise FileNotFoundError(f"오디오 파일을 찾을 수 없습니다: {audio_data}")
                audio_input = audio_data
            
            result = self._masker.transcribe_and_detect(audio_input)
            detected_items = result.get("detected_items", [])
            logger.info(f"음성 탐지 완료: {len(detected_items)}개 항목 발견")
            return detected_items
        except Exception as e:
            logger.error(f"음성 탐지 중 오류 발생: {str(e)}", exc_info=True)
            # 예외를 다시 raise하지 않고 빈 리스트 반환 (서비스 계속 유지)
            logger.warning("음성 탐지 실패로 빈 리스트 반환")
            return []
