"""
파일 서버 마스킹: connectionId + filePath + fileCategory → fetch → detect → mask → base64
로컬(SAVE_MASKED_OUTPUT=True): output_file에 저장. 서버(False): 저장 없이 base64만 반환.
"""
import base64
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from app.crud.file_server_connection import get_file_server_connection
from app.services.file.fetchers import fetch_file_bytes
from app.core.config import (
    SAVE_MASKED_OUTPUT,
    OUTPUT_DIR,
    IMAGE_OUTPUT_DIR,
    AUDIO_OUTPUT_DIR,
    VIDEO_OUTPUT_DIR,
)
from app.core.logging import logger


def mask_file_from_server(
    connection_id: int,
    file_path: str,
    file_category: str,
    document_file_processor,
    image_detector,
    audio_detector,
    video_detector,
    masker,
) -> Dict[str, Any]:
    """
    파일 서버에서 파일을 가져와 detect → mask 후 base64 반환.

    Returns:
        {"success": bool, "maskedFileBase64": str}
    """
    try:
        connection_info = get_file_server_connection(connection_id)
    except Exception as e:
        logger.warning(f"파일 서버 연결 조회 실패: {e}")
        return {"success": False, "maskedFileBase64": ""}

    try:
        data = fetch_file_bytes(connection_info, file_path)
    except Exception as e:
        logger.warning("파일 다운로드 실패 path=%s error=%s", file_path, e)
        return {"success": False, "maskedFileBase64": ""}

    ext = Path(file_path).suffix.lower()
    category = (file_category or "").upper()

    if category == "DOCUMENT":
        return _mask_document(
            data, ext, document_file_processor, masker
        )
    if category == "PHOTO":
        return _mask_photo(data, image_detector, masker)
    if category == "AUDIO":
        return _mask_audio(data, audio_detector, masker)
    if category == "VIDEO":
        return _mask_video(data, ext, video_detector, masker)

    logger.warning("지원하지 않는 fileCategory: %s", file_category)
    return {"success": False, "maskedFileBase64": ""}


def _mask_document(data: bytes, ext: str, doc_processor, masker) -> Dict[str, Any]:
    if ext not in (".pdf", ".docx", ".txt"):
        logger.warning("문서 확장자 미지원: %s", ext)
        return {"success": False, "maskedFileBase64": ""}

    fd_in, input_path = tempfile.mkstemp(suffix=ext)
    try:
        os.write(fd_in, data)
        os.close(fd_in)
        fd_in = None

        if SAVE_MASKED_OUTPUT:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(OUTPUT_DIR, f"masked_document_{timestamp}{ext}")
        else:
            fd_out, output_path = tempfile.mkstemp(suffix=ext)
            os.close(fd_out)

        doc_processor.process_file(input_path, output_path)
        with open(output_path, "rb") as f:
            masked_b64 = base64.b64encode(f.read()).decode()

        if not SAVE_MASKED_OUTPUT and os.path.exists(output_path):
            try:
                os.unlink(output_path)
            except Exception:
                pass

        return {"success": True, "maskedFileBase64": masked_b64}
    except Exception as e:
        logger.error("문서 마스킹 실패: %s", e, exc_info=True)
        return {"success": False, "maskedFileBase64": ""}
    finally:
        if input_path and os.path.exists(input_path):
            try:
                os.unlink(input_path)
            except Exception:
                pass


def _mask_photo(data: bytes, image_detector, masker) -> Dict[str, Any]:
    try:
        b64 = base64.b64encode(data).decode("ascii")
        faces = image_detector.detect_faces(b64)
        if faces:
            masked_bytes = masker.mask_image(b64, faces)
        else:
            masked_bytes = data

        if SAVE_MASKED_OUTPUT and masked_bytes:
            os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(IMAGE_OUTPUT_DIR, f"masked_image_{timestamp}.jpg")
            with open(out_path, "wb") as f:
                f.write(masked_bytes)

        masked_b64 = base64.b64encode(masked_bytes).decode() if masked_bytes else ""
        return {"success": True, "maskedFileBase64": masked_b64}
    except Exception as e:
        logger.error("이미지 마스킹 실패: %s", e, exc_info=True)
        return {"success": False, "maskedFileBase64": ""}


def _mask_audio(data: bytes, audio_detector, masker) -> Dict[str, Any]:
    fd_in, input_path = tempfile.mkstemp(suffix=".mp3")
    try:
        os.write(fd_in, data)
        os.close(fd_in)
        fd_in = None

        b64 = base64.b64encode(data).decode("ascii")
        detected_items = audio_detector.detect(b64, "base64")
        if isinstance(detected_items, dict):
            detected_items = detected_items.get("personal_info", []) or []
        masked_bytes = masker.mask_audio(input_path, detected_items)

        if SAVE_MASKED_OUTPUT and masked_bytes:
            os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(AUDIO_OUTPUT_DIR, f"masked_audio_{timestamp}.mp3")
            with open(out_path, "wb") as f:
                f.write(masked_bytes)

        masked_b64 = base64.b64encode(masked_bytes).decode() if masked_bytes else ""
        return {"success": True, "maskedFileBase64": masked_b64}
    except Exception as e:
        logger.error("오디오 마스킹 실패: %s", e, exc_info=True)
        return {"success": False, "maskedFileBase64": ""}
    finally:
        if input_path and os.path.exists(input_path):
            try:
                os.unlink(input_path)
            except Exception:
                pass


def _mask_video(data: bytes, ext: str, video_detector, masker) -> Dict[str, Any]:
    fd_in, input_path = tempfile.mkstemp(suffix=ext or ".mp4")
    try:
        os.write(fd_in, data)
        os.close(fd_in)
        fd_in = None

        detection_result = video_detector.detect(input_path, "path")
        faces = detection_result.get("faces", [])
        audio_items = detection_result.get("personal_info_in_audio", [])
        text_pii_regions = detection_result.get("text_pii_regions", [])

        if SAVE_MASKED_OUTPUT:
            os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(VIDEO_OUTPUT_DIR, f"masked_video_{timestamp}.mp4")
        else:
            fd_out, save_path = tempfile.mkstemp(suffix=".mp4")
            os.close(fd_out)

        masked_bytes = masker.mask_video(
            input_path, faces, audio_items,
            save_path=save_path,
            text_pii_regions=text_pii_regions,
        )

        if not SAVE_MASKED_OUTPUT and save_path and os.path.exists(save_path):
            try:
                os.unlink(save_path)
            except Exception:
                pass

        masked_b64 = base64.b64encode(masked_bytes).decode() if masked_bytes else ""
        return {"success": True, "maskedFileBase64": masked_b64}
    except Exception as e:
        logger.error("비디오 마스킹 실패: %s", e, exc_info=True)
        return {"success": False, "maskedFileBase64": ""}
    finally:
        if input_path and os.path.exists(input_path):
            try:
                os.unlink(input_path)
            except Exception:
                pass
