"""
영상 마스킹 처리 (얼굴 블러 + 화면 텍스트 PII 블러 + 오디오 마스킹)
(기존 masker.py에서 분리)
"""
import os
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Any
from app.core.logging import logger
from app.core.config import VIDEO_OUTPUT_DIR, get_project_root
from app.utils.base64_utils import is_base64, decode_base64_to_temp_file
from app.utils.temp_file import TempFileManager
from app.ml.face_detector import YOLOFaceDetector
from app.services.file.processors.blur_censor import BlurCensor
from app.services.file.processors.video_processor import VideoProcessorEnhanced
from app.core.config import settings


def _text_pii_regions_to_by_frame(text_pii_regions: List[Dict[str, Any]]) -> Dict[int, List]:
    """
    text_pii_regions -> {frame_number: [(x1,y1,x2,y2), ...]}
    키프레임에서 탐지된 영역을 전후 EXTEND_HALF 프레임까지 확장 적용해
    구간이 겹치고 끊김 없이 마스킹되도록 함.
    """
    extend_half = max(1, getattr(settings, "VIDEO_TEXT_PII_EXTEND_HALF", 25))
    by_frame: Dict[int, List] = {}
    for r in text_pii_regions or []:
        fn = r.get("frame_number")
        if fn is None:
            continue
        x = int(r.get("x", 0))
        y = int(r.get("y", 0))
        w = int(r.get("width", 0))
        h = int(r.get("height", 0))
        if w <= 0 or h <= 0:
            continue
        bbox = (x, y, x + w, y + h)
        # 키프레임 fn 기준 전후 extend_half 프레임까지 적용 (겹침으로 끊김 방지)
        for f in range(max(1, fn - extend_half), fn + extend_half + 1):
            by_frame.setdefault(f, []).append(bbox)
    return by_frame


class VideoMasker:
    """영상 마스킹 처리 (얼굴 + 화면 텍스트 PII + 오디오)"""

    @staticmethod
    def _reencode_to_h264(video_path: str, temp: "TempFileManager") -> str:
        """
        OpenCV mp4v 코덱 영상을 H.264(libx264)로 재인코딩.
        브라우저(Chrome 등)는 mp4v를 재생할 수 없으므로,
        오디오가 없는 영상도 반드시 H.264로 변환해야 합니다.
        """
        try:
            h264_path = temp.create(suffix='.mp4')
            reencode_cmd = [
                'ffmpeg', '-i', video_path,
                '-c:v', 'libx264', '-preset', 'fast',
                '-an',  # 오디오 없음
                '-y', h264_path,
            ]
            subprocess.run(reencode_cmd, check=True, capture_output=True)
            shutil.copy2(h264_path, video_path)
            logger.info("mp4v → H.264 재인코딩 완료 (오디오 없는 영상)")
            return video_path
        except Exception as e2:
            logger.warning(f"H.264 재인코딩 실패, mp4v 그대로 반환: {e2}")
            return video_path

    def mask_video(
        self,
        video_path: str,
        faces: list,
        audio_items: list,
        save_path: str = None,
        text_pii_regions: list = None,
    ) -> bytes:
        """
        영상 마스킹 (얼굴 모자이크 + 화면 텍스트 PII 블러 + 오디오 마스킹)
        text_pii_regions: [{frame_number, x, y, width, height, label?}, ...]
        """
        text_pii_regions = text_pii_regions or []
        try:
            logger.info(
                f"영상 마스킹 시작: {len(faces)}개 얼굴, {len(audio_items)}개 오디오, {len(text_pii_regions)}개 화면 텍스트 PII"
            )

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
                    import datetime
                    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    save_path = str(Path(VIDEO_OUTPUT_DIR) / f"masked_video_{timestamp}.mp4")
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

                # 비디오 처리 (얼굴 블러 + 화면 텍스트 PII 블러)
                text_pii_by_frame = _text_pii_regions_to_by_frame(text_pii_regions)
                try:
                    logger.info("비디오 얼굴·화면 텍스트 PII 마스킹 처리 중...")
                    processor.process_video(
                        input_video_path,
                        save_path,
                        conf_thresh=0.25,
                        text_pii_regions_by_frame=text_pii_by_frame,
                    )
                except Exception as e:
                    logger.error(f"비디오 얼굴 마스킹 중 오류 발생: {str(e)}", exc_info=True)
                    # 원본 비디오를 복사하여 반환 시도
                    try:
                        shutil.copy2(input_video_path, save_path)
                        logger.warning("얼굴 마스킹 실패로 원본 비디오 반환")
                    except Exception as copy_error:
                        logger.error(f"원본 비디오 복사도 실패: {str(copy_error)}")
                        raise

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

                    # 오디오 합성 (H.264로 재인코딩하여 Chrome 호환성 확보)
                    temp_final_video = temp.create(suffix='.mp4')
                    merge_cmd = [
                        'ffmpeg', '-i', save_path,
                        '-i', masked_audio_path,
                        '-c:v', 'libx264', '-preset', 'fast', '-c:a', 'aac',
                        '-map', '0:v:0', '-map', '1:a:0',
                        '-shortest', '-y', temp_final_video,
                    ]
                    subprocess.run(merge_cmd, check=True, capture_output=True)
                    logger.info("오디오 합성 완료")
                    shutil.copy2(temp_final_video, save_path)
                    final_video_path = save_path
                except subprocess.CalledProcessError as e:
                    logger.warning(f"오디오 추출/합성 중 오류 발생 (비디오만 H.264 재인코딩): {e}")
                    final_video_path = self._reencode_to_h264(save_path, temp)
                except FileNotFoundError:
                    logger.warning("ffmpeg가 설치되어 있지 않습니다. 비디오만 저장됩니다.")
                    final_video_path = save_path
                except Exception as e:
                    logger.warning(f"오디오 처리 중 예상치 못한 오류 발생 (비디오만 H.264 재인코딩): {e}", exc_info=True)
                    final_video_path = self._reencode_to_h264(save_path, temp)

                # 결과 읽기
                with open(final_video_path, 'rb') as f:
                    masked_video_bytes = f.read()

                logger.info(f"영상 마스킹 완료: {len(masked_video_bytes)} bytes")
                return masked_video_bytes
        except Exception as e:
            logger.error(f"영상 마스킹 중 예상치 못한 오류 발생: {str(e)}", exc_info=True)
            # 원본 비디오를 반환 시도
            try:
                if os.path.exists(video_path) and not is_base64(video_path):
                    with open(video_path, 'rb') as f:
                        return f.read()
            except:
                pass
            # 모든 시도 실패 시 빈 bytes 반환
            logger.error("영상 마스킹 실패로 빈 bytes 반환")
            return b""
