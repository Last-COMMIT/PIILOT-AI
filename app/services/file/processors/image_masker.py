"""
이미지 마스킹 처리
(기존 masker.py에서 분리)
"""
import cv2
import os
import numpy as np
from pathlib import Path
from typing import List, Dict
from app.core.logging import logger
from app.core.config import IMAGE_OUTPUT_DIR
from app.utils.image_utils import load_image


class ImageMasker:
    """이미지 마스킹 처리 (얼굴 모자이크)"""

    def mask_image(self, image_path: str | bytes, faces: List[Dict]) -> bytes:
        """
        이미지 마스킹 (얼굴 블러 처리)

        Args:
            image_path: 이미지 파일 경로, base64 문자열, 또는 이미지 bytes
            faces: 얼굴 위치 리스트

        Returns:
            마스킹된 이미지 (bytes)
        """
        try:
            logger.info(f"이미지 마스킹: {len(faces)}개 얼굴")
            
            if isinstance(image_path, bytes):
                nparr = np.frombuffer(image_path, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                img = load_image(image_path)

            if img is None:
                logger.error(f"이미지 로드 실패: {image_path[:50]}")
                return b""

            for face in faces:
                try:
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
                except Exception as face_error:
                    logger.warning(f"얼굴 마스킹 중 오류 발생 (다음 얼굴 계속 처리): {face_error}")
                    continue

            # IMAGE_OUTPUT_DIR에 저장
            import datetime
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = Path(IMAGE_OUTPUT_DIR) / f"masked_image_{timestamp}.jpg"
            
            # 파일로 저장
            cv2.imwrite(str(output_path), img)
            logger.info(f"마스킹된 이미지가 저장되었습니다: {output_path}")
            
            # Bytes로도 반환 (API 호환성 유지)
            with open(output_path, "rb") as f:
                masked_bytes = f.read()
            
            return masked_bytes
        except Exception as e:
            logger.error(f"이미지 마스킹 중 오류 발생: {str(e)}", exc_info=True)
            # 원본 이미지를 반환하거나 빈 bytes 반환
            try:
                # 원본 이미지를 로드하여 반환 시도
                img = load_image(image_path)
                if img is not None:
                    _, encoded_img = cv2.imencode('.jpg', img)
                    logger.warning("마스킹 실패로 원본 이미지 반환")
                    return encoded_img.tobytes()
            except:
                pass
            logger.error("원본 이미지 로드도 실패하여 빈 bytes 반환")
            return b""
