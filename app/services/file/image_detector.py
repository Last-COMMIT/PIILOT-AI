"""
이미지 얼굴 탐지 (Vision)
"""
from typing import List, Dict
import numpy as np
from app.utils.logger import logger
from app.utils.image_loader import load_image
from app.core.model_manager import ModelManager
import os


class ImageDetector:
    """Vision 기반 얼굴 탐지"""
    
    def __init__(self, model_path: str = None):
        """
        Args:
            model_path: Vision 모델 경로 (None이면 기본 모델 사용, 직접 전달 시 절대 경로)
        """
        try:
            from ultralytics import YOLO  # lazy import
        except ImportError:
            raise ImportError("ultralytics 모듈이 설치되지 않았습니다. 'pip install ultralytics'를 실행하세요.")
        
        # 기본 모델 경로 설정 (중앙 관리)
        if model_path is None:
            model_path = ModelManager.get_local_model_path("yolo_face")
        # 사용자가 직접 경로를 전달한 경우 절대 경로로 가정
        
        self.model_path = model_path
        
        # 모델 파일 존재 확인
        if not os.path.exists(model_path):
            logger.warning(f"모델 파일을 찾을 수 없습니다: {model_path}")
            logger.warning("기본 YOLO 모델을 사용합니다.")
            self.model = YOLO("yolov8n.pt")  # 기본 YOLO 모델 (얼굴 탐지 전용이 아님)
        else:
            # YOLO 얼굴 탐지 모델 로드
            self.model = YOLO(model_path)
        
        # 마스킹된 이미지 판단 기준 (신뢰도가 낮으면 이미 마스킹된 것으로 판단)
        self.confidence_threshold = 0.3
        
        logger.info(f"ImageDetector 초기화 완료: {model_path}")
    
    def detect_faces(self, image_path: str) -> List[Dict]:
        """
        이미지에서 얼굴 탐지
        
        Args:
            image_path: 이미지 파일 경로 또는 base64 인코딩된 이미지
            
        Returns:
            [
                {
                    "x": int,  # 얼굴 위치 x 좌표
                    "y": int,  # 얼굴 위치 y 좌표
                    "width": int,  # 얼굴 너비
                    "height": int,  # 얼굴 높이
                    "confidence": float  # 신뢰도
                },
                ...
            ]
        """
        logger.info(f"이미지 얼굴 탐지 시작: {image_path[:50]}...")
        
        try:
            # 이미지 로드 (공통 유틸리티 사용)
            img = load_image(image_path)
            
            # YOLO 모델로 얼굴 탐지
            results = self.model(img)
            
            detected_faces = []
            
            for r in results:
                face_count = len(r.boxes.xyxy)
                if face_count > 0:
                    # 신뢰도 확인
                    confidences = r.boxes.conf.cpu().numpy()
                    min_confidence = float(np.min(confidences))
                    
                    # 신뢰도가 낮으면 이미 마스킹된 이미지로 판단
                    if min_confidence < self.confidence_threshold:
                        logger.info(f"이미 마스킹된 이미지로 판단됨 (최소 신뢰도: {min_confidence:.2f})")
                        return []  # 마스킹된 이미지는 얼굴이 없다고 반환
                    
                    # 얼굴 좌표 추출 및 변환
                    for i, box in enumerate(r.boxes.xyxy):
                        x1, y1, x2, y2 = map(int, box)
                        confidence = float(confidences[i])
                        
                        # x, y, width, height 형식으로 변환
                        detected_faces.append({
                            "x": x1,
                            "y": y1,
                            "width": x2 - x1,
                            "height": y2 - y1,
                            "confidence": confidence
                        })
            
            if detected_faces:
                logger.info(f"얼굴 탐지 성공: {len(detected_faces)}개의 얼굴이 탐지되었습니다.")
                for i, face in enumerate(detected_faces, 1):
                    logger.debug(
                        f"  얼굴 {i}: 위치 ({face['x']}, {face['y']}) "
                        f"크기 ({face['width']}x{face['height']}), "
                        f"신뢰도: {face['confidence']:.2f}"
                    )
            else:
                logger.info("얼굴 탐지 실패: 이미지에서 얼굴을 찾을 수 없습니다.")
            
            return detected_faces
            
        except Exception as e:
            logger.error(f"얼굴 탐지 중 오류 발생: {e}", exc_info=True)
            # 예외를 다시 raise하지 않고 빈 리스트 반환 (서비스 계속 유지)
            logger.warning("얼굴 탐지 실패로 빈 리스트 반환")
            return []

