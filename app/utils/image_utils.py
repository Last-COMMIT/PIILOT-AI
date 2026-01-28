"""
이미지 로드 유틸리티 (공통)
"""
import cv2
import numpy as np
import base64
import os
from app.core.logging import logger


def load_image(image_path: str) -> np.ndarray:
    """
    이미지 로드 (파일 경로 또는 base64)

    Args:
        image_path: 이미지 파일 경로 또는 base64 인코딩된 이미지

    Returns:
        OpenCV 이미지 배열

    Raises:
        ValueError: 이미지 디코딩 실패
        FileNotFoundError: 이미지 파일을 찾을 수 없음
    """
    # base64 인코딩된 이미지인지 확인
    if image_path.startswith("data:image") or len(image_path) > 1000:
        # base64 디코딩
        try:
            if "," in image_path:
                base64_data = image_path.split(",")[1]
            else:
                base64_data = image_path

            image_bytes = base64.b64decode(base64_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Base64 이미지를 디코딩할 수 없습니다.")
            return img
        except Exception as e:
            logger.error(f"Base64 이미지 디코딩 오류: {e}")
            raise ValueError(f"이미지 디코딩 실패: {e}")
    else:
        # 파일 경로로 로드
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
        return img
