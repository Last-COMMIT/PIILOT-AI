"""
이미지 얼굴 탐지 (Vision)
"""
from typing import List, Dict
import numpy as np
from app.core.logging import logger
from app.utils.image_utils import load_image
from app.core.model_manager import ModelManager
import os


class ImageDetector:
    """Vision 기반 얼굴 탐지"""

    def __init__(self, model_path: str = None):
        try:
            from ultralytics import YOLO  # lazy import
        except ImportError:
            raise ImportError("ultralytics 모듈이 설치되지 않았습니다. 'pip install ultralytics'를 실행하세요.")
        
        # 기본 모델 경로 설정 (중앙 관리)
        if model_path is None:
            model_path = ModelManager.get_local_model_path("yolo_face")
        # 사용자가 직접 경로를 전달한 경우 절대 경로로 가정

        self.model_path = model_path

        if not os.path.exists(model_path):
            logger.warning(f"모델 파일을 찾을 수 없습니다: {model_path}")
            logger.warning("기본 YOLO 모델을 사용합니다.")
            self.model = YOLO("yolov8n.pt")
        else:
            self.model = YOLO(model_path)

        self.confidence_threshold = 0.3
        logger.info(f"ImageDetector 초기화 완료: {model_path}")

    def detect_faces(self, image_path: str) -> List[Dict]:
        """이미지에서 얼굴 탐지"""
        logger.info(f"이미지 얼굴 탐지 시작: {image_path[:50]}...")

        try:
            img = load_image(image_path)
            results = self.model(img)
            detected_faces = []

            for r in results:
                face_count = len(r.boxes.xyxy)
                if face_count > 0:
                    confidences = r.boxes.conf.cpu().numpy()
                    min_confidence = float(np.min(confidences))

                    if min_confidence < self.confidence_threshold:
                        logger.info(f"이미 마스킹된 이미지로 판단됨 (최소 신뢰도: {min_confidence:.2f})")
                        return []

                    for i, box in enumerate(r.boxes.xyxy):
                        x1, y1, x2, y2 = map(int, box)
                        confidence = float(confidences[i])
                        detected_faces.append({
                            "x": x1,
                            "y": y1,
                            "width": x2 - x1,
                            "height": y2 - y1,
                            "confidence": confidence,
                        })

            if detected_faces:
                logger.info(f"얼굴 탐지 성공: {len(detected_faces)}개의 얼굴이 탐지되었습니다.")
            else:
                logger.info("얼굴 탐지 실패: 이미지에서 얼굴을 찾을 수 없습니다.")

            return detected_faces

        except Exception as e:
            logger.error(f"얼굴 탐지 중 오류 발생: {e}")
            raise
