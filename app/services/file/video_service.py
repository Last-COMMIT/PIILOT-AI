"""
비디오 PII 탐지 서비스
(기존 video_detector.py 리팩토링)
"""
import os
import subprocess
import tempfile
from typing import Dict, List
from app.ml.image_detector import ImageDetector
from app.services.file.audio_service import AudioDetector
from app.ml.face_detector import YOLOFaceDetector
from app.utils.base64_utils import is_base64, decode_base64_to_temp_file
from app.utils.temp_file import TempFileManager
from app.core.logging import logger
import cv2


class VideoDetector:
    """Vision + LLM 기반 영상 개인정보 탐지"""

    def __init__(self):
        self.image_detector = ImageDetector()
        self.audio_detector = AudioDetector()
        logger.info("VideoDetector 초기화")

    def detect(self, video_path: str) -> Dict:
        """영상에서 개인정보 탐지"""
        logger.info(f"영상 탐지 시작: {video_path[:50]}...")

        with TempFileManager() as temp:
            if is_base64(video_path):
                input_video_path = decode_base64_to_temp_file(video_path, suffix='.mp4')
                temp.add(input_video_path)
            else:
                if not os.path.exists(video_path):
                    raise FileNotFoundError(f"비디오 파일을 찾을 수 없습니다: {video_path}")
                input_video_path = video_path

            # 얼굴 감지기 초기화
            face_detector = YOLOFaceDetector(
                conf_threshold=0.25, iou_threshold=0.45, imgsz=640, enhance_image=True
            )

            # 비디오 열기
            capture = cv2.VideoCapture(input_video_path)
            if not capture.isOpened():
                raise ValueError(f"비디오를 열 수 없습니다: {input_video_path}")

            frame_count = 0
            detected_faces = []

            logger.info("프레임별 얼굴 감지 중...")
            while capture.isOpened():
                ret, frame = capture.read()
                if not ret:
                    break
                frame_count += 1
                detections = face_detector.detect(frame, conf_threshold=0.25)
                for bbox in detections:
                    x1, y1, x2, y2, conf = bbox
                    detected_faces.append({
                        "frame_number": frame_count,
                        "x": int(x1),
                        "y": int(y1),
                        "width": int(x2 - x1),
                        "height": int(y2 - y1),
                        "confidence": float(conf),
                    })
                if frame_count % 100 == 0:
                    logger.info(f"프레임 처리 중... {frame_count}프레임 ({len(detected_faces)}개 얼굴 감지)")

            capture.release()

            # 오디오 개인정보 탐지
            logger.info("오디오 개인정보 탐지 중...")
            personal_info_in_audio = []

            try:
                temp_audio_path = temp.create(suffix='.mp3')
                extract_cmd = [
                    'ffmpeg', '-i', input_video_path,
                    '-vn', '-acodec', 'libmp3lame', '-y', temp_audio_path,
                ]
                try:
                    subprocess.run(extract_cmd, check=True, capture_output=True)
                    logger.info("비디오에서 오디오 추출 완료 (탐지용)")
                    audio_detections = self.audio_detector.detect(temp_audio_path)
                    if isinstance(audio_detections, dict):
                        personal_info_in_audio = audio_detections.get("personal_info", [])
                    elif isinstance(audio_detections, list):
                        personal_info_in_audio = audio_detections
                except subprocess.CalledProcessError as e:
                    logger.warning(f"오디오 추출 중 오류 발생 (계속 진행): {e}")
                except FileNotFoundError:
                    logger.warning("ffmpeg가 설치되어 있지 않습니다. 오디오 탐지를 건너뜁니다.")
            except Exception as e:
                logger.warning(f"오디오 탐지 중 오류 발생 (계속 진행): {e}")

            logger.info(f"영상 탐지 완료: {len(detected_faces)}개 얼굴, {len(personal_info_in_audio)}개 오디오 항목")

            return {
                "faces": detected_faces,
                "personal_info_in_audio": personal_info_in_audio,
            }
