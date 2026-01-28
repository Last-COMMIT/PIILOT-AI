"""
향상된 얼굴 추적 기능이 있는 비디오 프로세서 (Kalman Filter 사용)
"""
import cv2
import numpy as np
from typing import List
from app.ml.face_detector import YOLOFaceDetector
from app.services.file.processors.blur_censor import BlurCensor
from app.core.logging import logger


class VideoProcessorEnhanced:
    """향상된 얼굴 추적 기능이 있는 미디어 프로세서 (Kalman Filter 사용)"""

    def __init__(self, detector: YOLOFaceDetector, censor: BlurCensor,
                 max_track_frames=10, iou_threshold=0.3, use_kalman=True):
        self.detector = detector
        self.censor = censor
        self.max_track_frames = max_track_frames
        self.iou_threshold = iou_threshold
        self.use_kalman = use_kalman
        self.tracked_faces = []
        logger.info(f"VideoProcessorEnhanced 초기화: max_track_frames={max_track_frames}, use_kalman={use_kalman}")

    def _init_kalman_filter(self, bbox):
        """Kalman Filter 초기화"""
        kalman = cv2.KalmanFilter(8, 4)
        kalman.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ], dtype=np.float32)
        kalman.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ], dtype=np.float32)
        kalman.processNoiseCov = np.eye(8, dtype=np.float32) * 0.03
        kalman.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.5
        kalman.errorCovPost = np.eye(8, dtype=np.float32) * 0.1
        x1, y1, x2, y2 = bbox[:4]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        kalman.statePre = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32)
        kalman.statePost = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32)
        return kalman

    def _bbox_to_center(self, bbox):
        x1, y1, x2, y2 = bbox[:4]
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dtype=np.float32)

    def _center_to_bbox(self, center, conf=0.5, frame_width=None, frame_height=None):
        cx, cy, w, h = center
        if w <= 0 or h <= 0:
            return None
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)
        if frame_width is not None and frame_height is not None:
            x1 = max(0, min(x1, frame_width - 1))
            y1 = max(0, min(y1, frame_height - 1))
            x2 = max(0, min(x2, frame_width - 1))
            y2 = max(0, min(y2, frame_height - 1))
            if x2 <= x1 or y2 <= y1:
                return None
        return [x1, y1, x2, y2, conf]

    def _calculate_iou(self, box1, box2):
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
        matched = [False] * len(detections)
        updated_tracks = []
        for track in self.tracked_faces:
            track_bbox = track['bbox']
            track_age = track['age']
            kalman = track.get('kalman')
            if self.use_kalman and kalman is not None:
                predicted = kalman.predict()
                predicted_bbox = self._center_to_bbox(predicted[:4], track_bbox[4], frame_width, frame_height)
                if predicted_bbox is None:
                    predicted_bbox = track_bbox
            else:
                predicted_bbox = track_bbox
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
                matched[best_idx] = True
                new_bbox = detections[best_idx]
                if self.use_kalman and kalman is not None:
                    measurement = self._bbox_to_center(new_bbox)
                    kalman.correct(measurement)
                    updated_state = kalman.statePost
                    updated_bbox = self._center_to_bbox(updated_state[:4], new_bbox[4], frame_width, frame_height)
                    if updated_bbox is not None:
                        new_bbox = updated_bbox
                updated_tracks.append({'bbox': new_bbox, 'age': 0, 'kalman': kalman if self.use_kalman else None})
            else:
                if self.use_kalman and kalman is not None:
                    predicted_bbox2 = self._center_to_bbox(predicted[:4], track_bbox[4], frame_width, frame_height)
                    if predicted_bbox2 is None:
                        predicted_bbox2 = track_bbox
                    updated_tracks.append({'bbox': predicted_bbox2, 'age': track_age + 1, 'kalman': kalman})
                else:
                    updated_tracks.append({'bbox': track_bbox, 'age': track_age + 1, 'kalman': None})
        for i, det_bbox in enumerate(detections):
            if not matched[i]:
                kalman = self._init_kalman_filter(det_bbox) if self.use_kalman else None
                updated_tracks.append({'bbox': det_bbox, 'age': 0, 'kalman': kalman})
        self.tracked_faces = [track for track in updated_tracks if track['age'] < self.max_track_frames]
        return [track['bbox'] for track in self.tracked_faces]

    def process_video(self, video_path, output_path, conf_thresh=0.25):
        """영상 처리 (향상된 얼굴 추적)"""
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise ValueError(f"Could not open video at {video_path}")
        fps = int(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
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
                detections = self.detector.detect(frame, conf_threshold=conf_thresh)
                tracked_bboxes = self._match_detections_to_tracks(detections, width, height)
                for bbox in tracked_bboxes:
                    if bbox is None:
                        continue
                    x1, y1, x2, y2 = bbox[:4]
                    if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0:
                        try:
                            frame = self.censor.apply(frame, bbox)
                        except Exception as e:
                            logger.debug(f"블러 적용 실패 (프레임 {frame_count}): {e}")
                out.write(frame)
                if frame_count % 50 == 0:
                    logger.info(f"비디오 처리 중... {frame_count}프레임 (추적 중: {len(self.tracked_faces)}개 얼굴)")
        finally:
            capture.release()
            out.release()
        logger.info(f"비디오 처리 완료: {frame_count}프레임 처리됨")
        return output_path
