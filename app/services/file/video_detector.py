"""
영상 개인정보 탐지 (Vision + LLM)
"""
from typing import Dict, List
from app.services.file.image_detector import ImageDetector
from app.services.file.audio_detector import AudioDetector
from app.utils.logger import logger


class VideoDetector:
    """Vision + LLM 기반 영상 개인정보 탐지"""
    
    def __init__(self):
        self.image_detector = ImageDetector()
        self.audio_detector = AudioDetector()
        logger.info("VideoDetector 초기화")
    
    def detect(self, video_path: str) -> Dict:
        """
        영상에서 개인정보 탐지
        - 화면에서 얼굴 탐지 (Vision)
        - 오디오에서 개인정보 탐지 (LLM)
        
        Args:
            video_path: 영상 파일 경로 또는 base64 인코딩된 비디오
            
        Returns:
            {
                "faces": [
                    {
                        "frame_number": int,  # 프레임 번호
                        "x": int,
                        "y": int,
                        "width": int,
                        "height": int,
                        "confidence": float
                    },
                    ...
                ],
                "personal_info_in_audio": [
                    {
                        "type": "name",
                        "value": "홍길동",
                        "start_time": 1.5,
                        "end_time": 2.0,
                        "confidence": 0.95
                    },
                    ...
                ]
            }
        """
        # TODO: 구현 필요
        # - 영상 프레임 추출
        # - 각 프레임에서 얼굴 탐지
        # - 오디오 추출 및 개인정보 탐지
        logger.info(f"영상 탐지 시작: {video_path}")
        pass

