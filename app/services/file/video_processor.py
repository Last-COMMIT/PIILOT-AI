"""
향상된 얼굴 추적 기능이 있는 비디오 프로세서 (Kalman Filter 사용)
"""
import cv2
import numpy as np
from collections import deque
from typing import List
from app.services.file.face_detector import YOLOFaceDetector
from app.services.file.blur_censor import BlurCensor
from app.utils.logger import logger


class VideoProcessorEnhanced:
    """향상된 얼굴 추적 기능이 있는 미디어 프로세서 (Kalman Filter 사용)"""
    
    def __init__(self, detector: YOLOFaceDetector, censor: BlurCensor, 
                 max_track_frames=10, iou_threshold=0.3, use_kalman=True):
        """
        Args:
            detector: 얼굴 감지기
            censor: 블러/마스킹 적용기
            max_track_frames: 얼굴을 추적할 최대 프레임 수
            iou_threshold: 얼굴 매칭을 위한 IoU 임계값
            use_kalman: Kalman Filter 사용 여부
        """
        self.detector = detector
        self.censor = censor
        self.max_track_frames = max_track_frames
        self.iou_threshold = iou_threshold
        self.use_kalman = use_kalman
        self.tracked_faces = []  # [{'bbox': bbox, 'age': age, 'kalman': kalman}, ...]
        logger.info(f"VideoProcessorEnhanced 초기화: max_track_frames={max_track_frames}, use_kalman={use_kalman}")
        
    def _init_kalman_filter(self, bbox):
        """Kalman Filter 초기화"""
        kalman = cv2.KalmanFilter(8, 4)  # 8 states, 4 measurements
        
        # 상태: [cx, cy, w, h, vx, vy, vw, vh]
        # 측정: [cx, cy, w, h]
        
        # 전이 행렬 (상태 전이)
        kalman.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],  # cx = cx + vx
            [0, 1, 0, 0, 0, 1, 0, 0],  # cy = cy + vy
            [0, 0, 1, 0, 0, 0, 1, 0],  # w = w + vw
            [0, 0, 0, 1, 0, 0, 0, 1],  # h = h + vh
            [0, 0, 0, 0, 1, 0, 0, 0],  # vx = vx
            [0, 0, 0, 0, 0, 1, 0, 0],  # vy = vy
            [0, 0, 0, 0, 0, 0, 1, 0],  # vw = vw
            [0, 0, 0, 0, 0, 0, 0, 1],  # vh = vh
        ], dtype=np.float32)
        
        # 측정 행렬
        kalman.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ], dtype=np.float32)
        
        # 프로세스 노이즈 공분산
        kalman.processNoiseCov = np.eye(8, dtype=np.float32) * 0.03
        
        # 측정 노이즈 공분산
        kalman.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.5
        
        # 오차 공분산
        kalman.errorCovPost = np.eye(8, dtype=np.float32) * 0.1
        
        # 초기 상태 설정
        x1, y1, x2, y2 = bbox[:4]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        
        kalman.statePre = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32)
        kalman.statePost = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32)
        
        return kalman
    
    def _bbox_to_center(self, bbox):
        """박스를 중심점과 크기로 변환"""
        x1, y1, x2, y2 = bbox[:4]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return np.array([cx, cy, w, h], dtype=np.float32)
    
    def _center_to_bbox(self, center, conf=0.5, frame_width=None, frame_height=None):
        """중심점과 크기를 박스로 변환"""
        cx, cy, w, h = center
        
        # 크기가 유효한지 확인
        if w <= 0 or h <= 0:
            return None
        
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)
        
        # 프레임 경계 내로 제한
        if frame_width is not None and frame_height is not None:
            x1 = max(0, min(x1, frame_width - 1))
            y1 = max(0, min(y1, frame_height - 1))
            x2 = max(0, min(x2, frame_width - 1))
            y2 = max(0, min(y2, frame_height - 1))
            
            # 유효한 박스인지 확인
            if x2 <= x1 or y2 <= y1:
                return None
        
        return [x1, y1, x2, y2, conf]
    
    def _calculate_iou(self, box1, box2):
        """두 박스 간의 IoU 계산"""
        x1_1, y1_1, x2_1, y2_1 = box1[:4]
        x1_2, y1_2, x2_2, y2_2 = box2[:4]
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _match_detections_to_tracks(self, detections, frame_width, frame_height):
        """새로운 감지 결과를 기존 추적 얼굴과 매칭 (Kalman Filter 사용)"""
        matched = [False] * len(detections)
        updated_tracks = []
        
        # 각 추적 얼굴에 대해 예측 및 매칭
        for track in self.tracked_faces:
            track_bbox = track['bbox']
            track_age = track['age']
            kalman = track.get('kalman')
            
            if self.use_kalman and kalman is not None:
                # Kalman Filter로 다음 위치 예측
                predicted = kalman.predict()
                predicted_bbox = self._center_to_bbox(predicted[:4], track_bbox[4], frame_width, frame_height)
                # 예측 실패 시 원래 박스 사용
                if predicted_bbox is None:
                    predicted_bbox = track_bbox
            else:
                predicted_bbox = track_bbox
            
            # 가장 가까운 감지 결과 찾기
            best_iou = 0
            best_idx = -1
            
            for i, det_bbox in enumerate(detections):
                if matched[i]:
                    continue
                
                iou = self._calculate_iou(predicted_bbox, det_bbox)
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_idx = i
            
            if best_idx >= 0:
                # 매칭 성공: 새로운 감지 결과로 업데이트
                matched[best_idx] = True
                new_bbox = detections[best_idx]
                
                if self.use_kalman and kalman is not None:
                    # 측정값으로 Kalman Filter 업데이트
                    measurement = self._bbox_to_center(new_bbox)
                    kalman.correct(measurement)
                    # 업데이트된 상태로 박스 재생성
                    updated_state = kalman.statePost
                    updated_bbox = self._center_to_bbox(updated_state[:4], new_bbox[4], frame_width, frame_height)
                    # 유효한 경우에만 사용
                    if updated_bbox is not None:
                        new_bbox = updated_bbox
                
                updated_tracks.append({
                    'bbox': new_bbox,
                    'age': 0,
                    'kalman': kalman if self.use_kalman else None
                })
            else:
                # 매칭 실패: 예측 위치 사용 (age 증가)
                if self.use_kalman and kalman is not None:
                    # 예측 위치 사용
                    predicted_bbox = self._center_to_bbox(predicted[:4], track_bbox[4], frame_width, frame_height)
                    # 예측이 유효하지 않으면 원래 박스 사용
                    if predicted_bbox is None:
                        predicted_bbox = track_bbox
                    updated_tracks.append({
                        'bbox': predicted_bbox,
                        'age': track_age + 1,
                        'kalman': kalman
                    })
                else:
                    updated_tracks.append({
                        'bbox': track_bbox,
                        'age': track_age + 1,
                        'kalman': None
                    })
        
        # 매칭되지 않은 새로운 감지 결과 추가
        for i, det_bbox in enumerate(detections):
            if not matched[i]:
                kalman = self._init_kalman_filter(det_bbox) if self.use_kalman else None
                updated_tracks.append({
                    'bbox': det_bbox,
                    'age': 0,
                    'kalman': kalman
                })
        
        # 너무 오래된 추적 제거
        self.tracked_faces = [track for track in updated_tracks 
                             if track['age'] < self.max_track_frames]
        
        return [track['bbox'] for track in self.tracked_faces]
    
    def process_video(self, video_path, output_path, conf_thresh=0.25):
        """
        영상 처리 (향상된 얼굴 추적 기능 포함)
        
        Args:
            video_path: 입력 비디오 경로
            output_path: 출력 비디오 경로
            conf_thresh: Confidence threshold
            
        Returns:
            출력 비디오 경로
        """
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise ValueError(f"Could not open video at {video_path}")
        
        fps = int(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # FPS가 0이면 기본값 설정
        if fps == 0:
            fps = 30
            logger.warning(f"FPS를 감지할 수 없어 기본값 {fps}를 사용합니다.")
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        try:
            while capture.isOpened():
                ret, frame = capture.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 얼굴 감지 (낮은 threshold로 더 많은 얼굴 감지)
                detections = self.detector.detect(frame, conf_threshold=conf_thresh)
                
                # 추적 얼굴과 매칭
                tracked_bboxes = self._match_detections_to_tracks(detections, width, height)
                
                # 블러 적용 (유효한 박스만)
                for bbox in tracked_bboxes:
                    if bbox is None:
                        continue
                    x1, y1, x2, y2 = bbox[:4]
                    # 유효한 박스인지 확인
                    if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0:
                        try:
                            frame = self.censor.apply(frame, bbox)
                        except Exception as e:
                            # 블러 적용 실패 시 스킵
                            logger.debug(f"블러 적용 실패 (프레임 {frame_count}): {e}")
                
                out.write(frame)
                
                # 진행 상황 출력 (50프레임마다)
                if frame_count % 50 == 0:
                    logger.info(f"비디오 처리 중... {frame_count}프레임 (추적 중: {len(self.tracked_faces)}개 얼굴)")
        
        finally:
            capture.release()
            out.release()
        
        logger.info(f"비디오 처리 완료: {frame_count}프레임 처리됨")
        return output_path
