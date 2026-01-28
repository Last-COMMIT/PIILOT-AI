"""
비디오 프레임 블러 처리
"""
import cv2
import numpy as np
from app.utils.logger import logger


class BlurCensor:
    """비디오 프레임에 블러를 적용하는 클래스"""
    
    def __init__(self, blur_factor=99):
        """
        Args:
            blur_factor: 블러 강도 (양수이고 홀수여야 함)
        """
        if blur_factor < 1 or blur_factor % 2 == 0:
            raise ValueError("blur_factor must be a positive and odd number.")
        
        self.blur_factor = blur_factor
        logger.info(f"BlurCensor 초기화: blur_factor={blur_factor}")
    
    def apply(self, frame, bbox):
        """
        프레임의 특정 영역에 블러 적용
        
        Args:
            frame: 입력 프레임 (BGR 형식)
            bbox: 바운딩 박스 [x1, y1, x2, y2, conf] 또는 [x1, y1, x2, y2]
            
        Returns:
            블러가 적용된 프레임
        """
        x1, y1, x2, y2 = bbox[:4]
        
        # 좌표 유효성 검사
        if x2 <= x1 or y2 <= y1:
            return frame
        
        # 프레임 크기 확인
        frame_height, frame_width = frame.shape[:2]
        
        # 좌표를 프레임 범위 내로 제한
        x1 = max(0, min(x1, frame_width - 1))
        y1 = max(0, min(y1, frame_height - 1))
        x2 = max(0, min(x2, frame_width - 1))
        y2 = max(0, min(y2, frame_height - 1))
        
        if x2 <= x1 or y2 <= y1:
            return frame
        
        try:
            # 얼굴 영역 추출
            roi = frame[y1:y2, x1:x2]
            
            if roi.size == 0:
                return frame
            
            # Gaussian Blur 적용
            blurred = cv2.GaussianBlur(roi, (self.blur_factor, self.blur_factor), 0)
            
            # 원본 프레임에 블러 적용된 영역 복사
            frame[y1:y2, x1:x2] = blurred
            
        except Exception as e:
            logger.warning(f"블러 적용 중 오류 발생 (무시하고 계속): {e}")
        
        return frame
