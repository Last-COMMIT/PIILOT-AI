"""
비디오 PII 탐지 서비스
(기존 video_detector.py 리팩토링)
- 얼굴, 오디오 8종 PII, 화면 텍스트 8종 PII(OCR+NER/정규식) 탐지
"""
import os
import subprocess
import tempfile
from typing import Dict, List, Any, Optional
from app.ml.image_detector import ImageDetector
from app.services.file.audio_service import AudioDetector
from app.ml.face_detector import YOLOFaceDetector
from app.utils.base64_utils import is_base64, decode_base64_to_temp_file
from app.utils.temp_file import TempFileManager
from app.core.logging import logger
from app.core.config import settings
import cv2


class VideoDetector:
    """Vision + LLM 기반 영상 개인정보 탐지 (얼굴, 오디오, 화면 텍스트 PII)"""

    def __init__(self):
        self.image_detector = ImageDetector()
        self.audio_detector = AudioDetector()
        self._text_extractor: Optional[Any] = None
        self._pii_detector: Optional[Any] = None
        logger.info("VideoDetector 초기화")

    def _get_text_extractor(self):
        """OCR용 TextExtractor 지연 초기화"""
        if self._text_extractor is None:
            from app.services.file.extractors.text_extractor import TextExtractor
            self._text_extractor = TextExtractor(use_gpu=False)
            logger.info("TextExtractor(EasyOCR) 초기화 완료 (화면 텍스트 PII 탐지용)")
        return self._text_extractor

    def _get_pii_detector(self):
        """화면 텍스트 PII 탐지용 HybridPIIDetector 지연 초기화"""
        if self._pii_detector is None:
            from app.ml.pii_detectors.hybrid_detector import HybridPIIDetector
            self._pii_detector = HybridPIIDetector(settings.PII_MODEL_PATH)
            logger.info("HybridPIIDetector 초기화 완료 (화면 텍스트 PII 탐지용)")
        return self._pii_detector

    def _detect_text_pii_on_frame(self, frame, frame_number: int, temp_manager, extractor, pii_detector) -> List[Dict]:
        """한 프레임에서 OCR(EasyOCR) 후 8종 PII 탐지, 마스킹할 영역(bbox, 패딩 적용) 반환"""
        regions = []
        try:
            frame_path = temp_manager.create(suffix=".jpg")
            cv2.imwrite(frame_path, frame)
            blocks = extractor.extract_from_image(frame_path)
        except Exception as e:
            logger.debug(f"프레임 {frame_number} OCR 실패: {e}")
            return regions
        h, w = frame.shape[:2]
        pad = max(0, getattr(settings, "VIDEO_TEXT_PII_PADDING_PX", 4))
        for block in blocks:
            text = (block.get("text") or "").strip()
            if not text or len(text) < 2:
                continue
            bbox_points = block.get("bbox")
            if not bbox_points or len(bbox_points) < 4:
                continue
            try:
                entities = pii_detector.detect_pii(text)
            except Exception as e:
                logger.debug(f"PII 탐지 실패 (텍스트 일부): {e}")
                continue
            if not entities:
                continue
            xs = [p[0] for p in bbox_points]
            ys = [p[1] for p in bbox_points]
            x1 = max(0, int(min(xs)) - pad)
            y1 = max(0, int(min(ys)) - pad)
            x2 = min(w, int(max(xs)) + pad)
            y2 = min(h, int(max(ys)) + pad)
            if x2 <= x1 or y2 <= y1:
                continue
            label = entities[0].get("label", "p_nm")
            regions.append({
                "frame_number": frame_number,
                "x": x1, "y": y1,
                "width": x2 - x1, "height": y2 - y1,
                "label": label,
            })
        return regions

    def detect(self, video_data: str, video_format: str = "base64") -> Dict:
        """
        영상에서 개인정보 탐지
        
        Args:
            video_data: 영상 파일 경로 또는 base64 인코딩된 비디오
            video_format: "base64" 또는 "path" (기본값: "base64")
        """
        logger.info(f"영상 탐지 시작 (포맷: {video_format}): {video_data[:50] if len(video_data) > 50 else video_data}...")

        try:
            with TempFileManager() as temp:
                # video_format에 따라 처리
                if video_format == "base64":
                    input_video_path = decode_base64_to_temp_file(video_data, suffix='.mp4')
                    temp.add(input_video_path)
                else:  # "path"
                    if not os.path.exists(video_data):
                        raise FileNotFoundError(f"비디오 파일을 찾을 수 없습니다: {video_data}")
                    input_video_path = video_data

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
                    # 홀수 프레임(1, 3, 5, ...)에서만 얼굴 탐지 (성능 절반 절감)
                    if frame_count % 2 == 0:
                        continue
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

                # 화면 텍스트 PII 탐지 (키프레임 OCR + 8종 PII)
                text_pii_regions: List[Dict] = []
                try:
                    extractor = self._get_text_extractor()
                    pii_detector = self._get_pii_detector()
                    capture2 = cv2.VideoCapture(input_video_path)
                    if capture2.isOpened():
                        logger.info("화면 텍스트 PII 탐지 중 (키프레임 OCR)...")
                        frame_idx = 0
                        while True:
                            ret, frame = capture2.read()
                            if not ret:
                                break
                            frame_idx += 1
                            interval = getattr(settings, "VIDEO_KEYFRAME_INTERVAL", 20)
                            if frame_idx % interval != 0:
                                continue
                            regions = self._detect_text_pii_on_frame(
                                frame, frame_idx, temp, extractor, pii_detector
                            )
                            text_pii_regions.extend(regions)
                            if frame_idx % (interval * 10) == 0 and frame_idx > 0:
                                logger.info(f"키프레임 OCR 진행 중... {frame_idx}프레임 ({len(text_pii_regions)}개 텍스트 PII)")
                        capture2.release()
                        logger.info(f"화면 텍스트 PII 탐지 완료: {len(text_pii_regions)}개 영역")
                except Exception as e:
                    logger.warning(f"화면 텍스트 PII 탐지 중 오류 (계속 진행): {e}")

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
                        audio_detections = self.audio_detector.detect(temp_audio_path, "path")
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

                logger.info(
                    f"영상 탐지 완료: {len(detected_faces)}개 얼굴, {len(personal_info_in_audio)}개 오디오, {len(text_pii_regions)}개 화면 텍스트 PII"
                )

                return {
                    "faces": detected_faces,
                    "personal_info_in_audio": personal_info_in_audio,
                    "text_pii_regions": text_pii_regions,
                }
        except Exception as e:
            logger.error(f"영상 탐지 중 오류 발생: {str(e)}", exc_info=True)
            logger.warning("영상 탐지 실패로 빈 결과 반환")
            return {
                "faces": [],
                "personal_info_in_audio": [],
                "text_pii_regions": [],
            }
