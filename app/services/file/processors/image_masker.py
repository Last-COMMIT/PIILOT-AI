"""
이미지 마스킹 처리
(기존 masker.py에서 분리)
"""
import cv2
from typing import List, Dict
from app.core.logging import logger
from app.utils.image_utils import load_image


class ImageMasker:
    """이미지 마스킹 처리 (얼굴 모자이크)"""

    def mask_image(self, image_path: str, faces: List[Dict]) -> bytes:
        """
        이미지 마스킹 (얼굴 블러 처리)

        Args:
            image_path: 이미지 파일 경로 또는 base64 인코딩된 이미지
            faces: 얼굴 위치 리스트

        Returns:
            마스킹된 이미지 (bytes)
        """
        logger.info(f"이미지 마스킹: {len(faces)}개 얼굴")
        img = load_image(image_path)
        if img is None:
            logger.error(f"이미지 로드 실패: {image_path[:50]}")
            return b""

        for face in faces:
            x = face.get("x", 0)
            y = face.get("y", 0)
            width = face.get("width", 0)
            height = face.get("height", 0)
            if x < 0 or y < 0 or width <= 0 or height <= 0:
                continue

            img_height, img_width = img.shape[:2]
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(img_width, x + width)
            y2 = min(img_height, y + height)

            if x2 > x1 and y2 > y1:
                face_region = img[y1:y2, x1:x2]
                if face_region.size > 0:
                    blurred_face = cv2.blur(face_region, (100, 100))
                    img[y1:y2, x1:x2] = blurred_face

        _, encoded_img = cv2.imencode('.jpg', img)
        return encoded_img.tobytes()
