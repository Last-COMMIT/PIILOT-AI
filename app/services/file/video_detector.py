"""
영상 개인정보 탐지 (Vision + LLM)
"""
from typing import Dict, List
from app.services.file.image_detector import ImageDetector
from app.services.file.audio_service import AudioDetector
from app.utils.logger import logger
import cv2


class VideoDetector:
    """Vision + LLM 기반 영상 개인정보 탐지"""
    
    def __init__(self):
        self.image_detector = ImageDetector()
        self.audio_detector = AudioDetector()
        logger.info("VideoDetector 초기화")
    
    def detect(self, video_data: str, video_format: str = "base64") -> Dict:
        """
        영상에서 개인정보 탐지
        - 화면에서 얼굴 탐지 (Vision)
        - 오디오에서 개인정보 탐지 (LLM)
        
        Args:
            video_data: 영상 파일 경로 또는 base64 인코딩된 비디오
            video_format: "base64" 또는 "path" (기본값: "base64")
            
        Returns:
            {
                "faces": [
                    {
                        "frame_number": int,  # 프레임 번호
                        "x": int,
                        "y": int,
                        "width": int,
                        "height": int,
                        "confidence": float
                    },
                    ...
                ],
                "personal_info_in_audio": [
                    {
                        "type": "name",
                        "value": "홍길동",
                        "start_time": 1.5,
                        "end_time": 2.0,
                        "confidence": 0.95
                    },
                    ...
                ]
            }
        """
        import os
        import subprocess
        from app.services.file.face_detector import YOLOFaceDetector
        
        logger.info(f"영상 탐지 시작 (포맷: {video_format})")
        
        # 포맷에 따라 처리
        if video_format == "base64":
            # base64 디코딩하여 임시 파일로 저장
            try:
                from app.utils.base64_utils import decode_base64_to_temp_file
                input_video_path = decode_base64_to_temp_file(video_data, suffix='.mp4')
                is_base64 = True
            except Exception as e:
                logger.error(f"Base64 비디오 디코딩 오류: {e}")
                raise ValueError(f"비디오 디코딩 실패: {e}")
        else:  # "path"
            # 파일 경로
            if not os.path.exists(video_data):
                raise FileNotFoundError(f"비디오 파일을 찾을 수 없습니다: {video_data}")
            input_video_path = video_data
            is_base64 = False
        
        try:
            # 얼굴 감지기 초기화
            face_detector = YOLOFaceDetector(
                conf_threshold=0.25,
                iou_threshold=0.45,
                imgsz=640,
                enhance_image=True
            )
            
            # 비디오 열기
            capture = cv2.VideoCapture(input_video_path)
            if not capture.isOpened():
                raise ValueError(f"비디오를 열 수 없습니다: {input_video_path}")
            
            fps = capture.get(cv2.CAP_PROP_FPS)
            frame_count = 0
            detected_faces = []
            
            # 프레임별 얼굴 감지
            logger.info("프레임별 얼굴 감지 중...")
            while capture.isOpened():
                ret, frame = capture.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 얼굴 감지
                detections = face_detector.detect(frame, conf_threshold=0.25)
                
                # 감지 결과를 API 형식으로 변환
                for bbox in detections:
                    x1, y1, x2, y2, conf = bbox
                    detected_faces.append({
                        "frame_number": frame_count,
                        "x": int(x1),
                        "y": int(y1),
                        "width": int(x2 - x1),
                        "height": int(y2 - y1),
                        "confidence": float(conf)
                    })
                
                # 진행 상황 출력 (100프레임마다)
                if frame_count % 100 == 0:
                    logger.info(f"프레임 처리 중... {frame_count}프레임 ({len(detected_faces)}개 얼굴 감지)")
            
            capture.release()
            
            # 오디오 개인정보 탐지
            logger.info("오디오 개인정보 탐지 중...")
            temp_audio_path = None
            try:
                # 비디오에서 오디오 추출 (마스킹 단계와 동일한 방식으로 추출하여 타임스탬프 일치 보장)
                temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                temp_audio.close()
                audio_extract_path = temp_audio.name
                temp_audio_path = audio_extract_path
                
                # ffmpeg로 오디오 추출
                extract_cmd = [
                    'ffmpeg', '-i', input_video_path,
                    '-vn', '-acodec', 'libmp3lame',
                    '-y', audio_extract_path
                ]
                try:
                    subprocess.run(extract_cmd, check=True, capture_output=True)
                    logger.info("비디오에서 오디오 추출 완료 (탐지용)")
                    
                    # 추출된 오디오 파일로 탐지 수행
                    audio_detections = self.audio_detector.detect(audio_extract_path)
                    # audio_detector.detect()는 List[Dict]를 반환하지만, 
                    # 혹시 딕셔너리를 반환하는 경우도 대비하여 분기 처리
                    if isinstance(audio_detections, dict):
                        personal_info_in_audio = audio_detections.get("personal_info", [])
                    elif isinstance(audio_detections, list):
                        personal_info_in_audio = audio_detections
                    else:
                        personal_info_in_audio = []
                except subprocess.CalledProcessError as e:
                    logger.warning(f"오디오 추출 중 오류 발생 (계속 진행): {e}")
                    personal_info_in_audio = []
                except FileNotFoundError:
                    logger.warning("ffmpeg가 설치되어 있지 않습니다. 오디오 탐지를 건너뜁니다.")
                    personal_info_in_audio = []
            except Exception as e:
                logger.warning(f"오디오 탐지 중 오류 발생 (계속 진행): {e}")
                personal_info_in_audio = []
            
            logger.info(f"영상 탐지 완료: {len(detected_faces)}개 얼굴, {len(personal_info_in_audio)}개 오디오 항목")
            
            return {
                "faces": detected_faces,
                "personal_info_in_audio": personal_info_in_audio
            }
            
        finally:
            # 임시 파일 정리
            if is_base64 and input_video_path and os.path.exists(input_video_path):
                try:
                    os.unlink(input_video_path)
                except:
                    pass
            # 임시 오디오 파일 정리
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.unlink(temp_audio_path)
                except:
                    pass

