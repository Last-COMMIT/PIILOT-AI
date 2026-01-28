"""
비디오 프레임 블러 처리
"""
import cv2
import numpy as np
from app.core.logging import logger


class BlurCensor:
    """비디오 프레임에 블러를 적용하는 클래스"""

    def __init__(self, blur_factor=99):
        if blur_factor < 1 or blur_factor % 2 == 0:
            raise ValueError("blur_factor must be a positive and odd number.")
        self.blur_factor = blur_factor
        logger.info(f"BlurCensor 초기화: blur_factor={blur_factor}")

    def apply(self, frame, bbox):
        """프레임의 특정 영역에 블러 적용"""
        x1, y1, x2, y2 = bbox[:4]
        if x2 <= x1 or y2 <= y1:
            return frame
        frame_height, frame_width = frame.shape[:2]
        x1 = max(0, min(x1, frame_width - 1))
        y1 = max(0, min(y1, frame_height - 1))
        x2 = max(0, min(x2, frame_width - 1))
        y2 = max(0, min(y2, frame_height - 1))
        if x2 <= x1 or y2 <= y1:
            return frame
        try:
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                return frame
            blurred = cv2.GaussianBlur(roi, (self.blur_factor, self.blur_factor), 0)
            frame[y1:y2, x1:x2] = blurred
        except Exception as e:
            logger.warning(f"블러 적용 중 오류 발생 (무시하고 계속): {e}")
        return frame
