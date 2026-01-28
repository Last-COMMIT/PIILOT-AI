"""
영상 마스킹 처리 (얼굴 블러 + 오디오 마스킹)
(기존 masker.py에서 분리)
"""
import os
import subprocess
import shutil
from pathlib import Path
from typing import List
from app.core.logging import logger
from app.utils.base64_utils import is_base64, decode_base64_to_temp_file
from app.utils.temp_file import TempFileManager
from app.ml.face_detector import YOLOFaceDetector
from app.services.file.processors.blur_censor import BlurCensor
from app.services.file.processors.video_processor import VideoProcessorEnhanced


class VideoMasker:
    """영상 마스킹 처리"""

    def mask_video(self, video_path: str, faces: list, audio_items: list,
                   save_path: str = None) -> bytes:
        """
        영상 마스킹 (얼굴 모자이크 + 오디오 마스킹)
        """
        logger.info(f"영상 마스킹 시작: {len(faces)}개 얼굴 정보, {len(audio_items)}개 오디오 항목")

        with TempFileManager() as temp:
            # 비디오 파일 경로 처리
            if is_base64(video_path):
                input_video_path = decode_base64_to_temp_file(video_path, suffix='.mp4')
                temp.add(input_video_path)
            else:
                if not os.path.exists(video_path):
                    raise FileNotFoundError(f"비디오 파일을 찾을 수 없습니다: {video_path}")
                input_video_path = video_path

            # 출력 경로 설정
            if save_path is None:
                project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
                tests_dir = project_root / "tests"
                tests_dir.mkdir(exist_ok=True)
                import datetime
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                save_path = str(tests_dir / f"masked_video_{timestamp}.mp4")
            output_dir = os.path.dirname(os.path.abspath(save_path))
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            # 얼굴 감지기/블러/프로세서 초기화
            detector = YOLOFaceDetector(
                conf_threshold=0.25, iou_threshold=0.45, imgsz=640, enhance_image=True
            )
            blur_censor = BlurCensor(blur_factor=99)
            processor = VideoProcessorEnhanced(
                detector=detector, censor=blur_censor,
                max_track_frames=10, iou_threshold=0.3, use_kalman=True
            )

            # 비디오 처리 (얼굴 블러)
            logger.info("비디오 얼굴 마스킹 처리 중...")
            processor.process_video(input_video_path, save_path, conf_thresh=0.25)

            # 오디오 처리
            final_video_path = save_path
            temp_audio_path = temp.create(suffix='.mp3')

            try:
                extract_cmd = [
                    'ffmpeg', '-i', input_video_path,
                    '-vn', '-acodec', 'libmp3lame', '-y', temp_audio_path
                ]
                subprocess.run(extract_cmd, check=True, capture_output=True)
                logger.info("비디오에서 오디오 추출 완료")

                if audio_items:
                    from app.services.file.processors.audio_masker import AudioMasker
                    audio_masker = AudioMasker()
                    masked_audio_bytes = audio_masker.mask_audio(temp_audio_path, audio_items)

                    temp_masked_audio = temp.create(suffix='.mp3')
                    with open(temp_masked_audio, 'wb') as f:
                        f.write(masked_audio_bytes)
                    masked_audio_path = temp_masked_audio
                else:
                    masked_audio_path = temp_audio_path

                # 오디오 합성
                temp_final_video = temp.create(suffix='.mp4')
                merge_cmd = [
                    'ffmpeg', '-i', save_path,
                    '-i', masked_audio_path,
                    '-c:v', 'copy', '-c:a', 'aac',
                    '-map', '0:v:0', '-map', '1:a:0',
                    '-shortest', '-y', temp_final_video,
                ]
                subprocess.run(merge_cmd, check=True, capture_output=True)
                logger.info("오디오 합성 완료")
                shutil.copy2(temp_final_video, save_path)
                final_video_path = save_path

            except subprocess.CalledProcessError as e:
                logger.warning(f"오디오 처리 중 오류 발생 (비디오만 저장): {e}")
                final_video_path = save_path
            except FileNotFoundError:
                logger.warning("ffmpeg가 설치되어 있지 않습니다. 오디오 없이 비디오만 저장됩니다.")
                final_video_path = save_path
            except Exception as e:
                logger.warning(f"오디오 처리 중 예상치 못한 오류 발생 (비디오만 저장): {e}")
                final_video_path = save_path

            # 결과 읽기
            with open(final_video_path, 'rb') as f:
                masked_video_bytes = f.read()

            logger.info(f"영상 마스킹 완료: {len(masked_video_bytes)} bytes")
            return masked_video_bytes
