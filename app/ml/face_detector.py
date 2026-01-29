"""
YOLO 얼굴 탐지 모델
"""
import cv2
import numpy as np
import os
from app.core.model_manager import ModelManager


class YOLOFaceDetector:
    def __init__(self, model_path=None,
                 conf_threshold=0.25,
                 iou_threshold=0.45,
                 imgsz=640,
                 enhance_image=True):
        try:
            from ultralytics import YOLO  # lazy import
        except ImportError:
            raise ImportError("ultralytics 모듈이 설치되지 않았습니다. 'pip install ultralytics'를 실행하세요.")
        
        # 기본 모델 경로 설정 (중앙 관리)
        if model_path is None:
            model_path = ModelManager.get_local_model_path("yolo_face")
        # 사용자가 직접 경로를 전달한 경우 절대 경로로 가정
        
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.enhance_image = enhance_image

    def _enhance_image(self, frame):
        """이미지 전처리로 감지 정확도 향상"""
        if not self.enhance_image:
            return frame
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        return enhanced

    def detect(self, frame, conf_threshold=None, iou_threshold=None):
        """얼굴 감지"""
        processed_frame = self._enhance_image(frame)
        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        iou = iou_threshold if iou_threshold is not None else self.iou_threshold

        results = self.model(
            processed_frame,
            conf=conf,
            iou=iou,
            imgsz=self.imgsz,
            verbose=False,
            agnostic_nms=False,
            max_det=300,
        )[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            detections.append([int(x1), int(y1), int(x2), int(y2), conf])
        return detections
