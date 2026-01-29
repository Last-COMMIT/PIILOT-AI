import cv2
import numpy as np
import os
from pathlib import Path
from app.core.model_manager import ModelManager
# from .base import DetectionModel

# class YOLOFaceDetector(DetectionModel):
class YOLOFaceDetector():
    # def __init__(self, model_path="src/data/models/face_detection_model.pt"):
    def __init__(self, model_path=None, 
                 conf_threshold=0.25, 
                 iou_threshold=0.45,
                 imgsz=640,
                 enhance_image=True):
        """
        Args:
            model_path: 모델 파일 경로 (None이면 기본 모델 사용, 직접 전달 시 절대 경로)
            conf_threshold: Confidence threshold (낮을수록 더 많은 얼굴 감지)
            iou_threshold: IoU threshold for NMS (낮을수록 더 많은 얼굴 감지)
            imgsz: 입력 이미지 크기 (큰 값일수록 정확하지만 느림)
            enhance_image: 이미지 전처리 적용 여부
        """
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
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization) 적용
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
        
    def detect(self, frame, conf_threshold=None, iou_threshold=None):
        """
        얼굴 감지
        
        Args:
            frame: 입력 프레임 (BGR)
            conf_threshold: Confidence threshold (None이면 기본값 사용)
            iou_threshold: IoU threshold (None이면 기본값 사용)
        """
        # 이미지 전처리
        processed_frame = self._enhance_image(frame)
        
        # 파라미터 설정
        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        iou = iou_threshold if iou_threshold is not None else self.iou_threshold
        
        # YOLO 감지 (더 정확한 설정)
        results = self.model(
            processed_frame,
            conf=conf,
            iou=iou,
            imgsz=self.imgsz,
            verbose=False,
            agnostic_nms=False,  # 클래스별 NMS
            max_det=300,  # 최대 감지 수 증가
        )[0]
        
        detections = []
        
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            detections.append([int(x1), int(y1), int(x2), int(y2), conf])
        
        return detections