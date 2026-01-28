"""
이미지 detect+mask 오케스트레이션
"""
from typing import List, Dict
from app.ml.image_detector import ImageDetector
from app.services.file.processors.image_masker import ImageMasker
from app.core.logging import logger


class ImageService:
    """이미지 얼굴 탐지 및 마스킹 서비스"""

    def __init__(self):
        self.detector = ImageDetector()
        self.masker = ImageMasker()
        logger.info("ImageService 초기화")

    def detect_faces(self, image_data: str) -> List[Dict]:
        """이미지에서 얼굴 탐지"""
        return self.detector.detect_faces(image_data)

    def detect_and_mask(self, image_data: str) -> bytes:
        """얼굴 탐지 후 마스킹"""
        faces = self.detector.detect_faces(image_data)
        if faces:
            return self.masker.mask_image(image_data, faces)
        return b""

    def mask(self, image_data: str, faces: List[Dict]) -> bytes:
        """주어진 얼굴 정보로 마스킹"""
        return self.masker.mask_image(image_data, faces)
