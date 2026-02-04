"""
파일 관련 AI 처리 API
"""
from fastapi import APIRouter, HTTPException, Depends
from app.schemas.file import (
    DocumentDetectionRequest,
    DocumentDetectionResponse,
    ImageDetectionRequest,
    ImageDetectionResponse,
    AudioDetectionRequest,
    AudioDetectionResponse,
    VideoDetectionRequest,
    VideoDetectionResponse,
    FileDetectRequest,
    FileDetectResponse,
    FileMaskRequest,
    FileMaskResponse,
)
from app.api.deps import (
    get_document_detector,
    get_document_file_processor,
    get_image_detector,
    get_audio_detector,
    get_video_detector,
    get_masker,
)
from app.core.logging import logger
from app.core.async_utils import run_in_thread

router = APIRouter()


@router.post("/scan", response_model=FileDetectResponse)
async def file_server_detect(
    request: FileDetectRequest,
    document_detector=Depends(get_document_detector),
    image_detector=Depends(get_image_detector),
    audio_detector=Depends(get_audio_detector),
    video_detector=Depends(get_video_detector),
):
    """파일 서버 스캔: connectionId + piiFiles 경로로 파일 다운로드 후 개인정보 탐지"""
    try:
        from app.services.file.file_detect_service import scan_files
        results = await run_in_thread(
            scan_files,
            connection_id=request.connectionId,
            pii_files=request.piiFiles,
            document_detector=document_detector,
            image_detector=image_detector,
            audio_detector=audio_detector,
            video_detector=video_detector,
        )
        return FileDetectResponse(results=results)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("파일 서버 스캔 오류: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/document/detect", response_model=DocumentDetectionResponse)
async def detect_document_personal_info(
    request: DocumentDetectionRequest,
    document_detector=Depends(get_document_detector),
):
    """문서 개인정보 탐지 (BERT + NER)"""
    try:
        logger.info(f"문서 탐지 요청: 타입={request.file_type}")

        detected_items = await run_in_thread(
            document_detector.detect,
            request.file_content
        )

        return DocumentDetectionResponse(
            detected_items=detected_items,
            is_masked=False,
        )
    except Exception as e:
        logger.error(f"문서 탐지 중 오류 발생: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"문서 탐지 중 오류가 발생했습니다: {str(e)}"
        )


@router.post("/image/detect", response_model=ImageDetectionResponse)
async def detect_image_faces(
    request: ImageDetectionRequest,
    image_detector=Depends(get_image_detector),
):
    """이미지 얼굴 탐지 (Vision)"""
    try:
        logger.info(f"이미지 탐지 요청: 포맷={request.image_format}")

        detected_faces = await run_in_thread(
            image_detector.detect_faces,
            request.image_data
        )

        return ImageDetectionResponse(detected_faces=detected_faces)
    except Exception as e:
        logger.error(f"이미지 탐지 중 오류 발생: {str(e)}", exc_info=True)
        # 서비스 레이어에서 이미 빈 리스트를 반환하므로 여기서도 안전하게 처리
        return ImageDetectionResponse(detected_faces=[])


@router.post("/audio/detect", response_model=AudioDetectionResponse)
async def detect_audio_personal_info(
    request: AudioDetectionRequest,
    audio_detector=Depends(get_audio_detector),
):
    """음성 개인정보 탐지 (LLM)"""
    try:
        logger.info(f"음성 탐지 요청: 포맷={request.audio_format}")

        detected_items = await run_in_thread(
            audio_detector.detect,
            request.audio_data,
            request.audio_format
        )

        return AudioDetectionResponse(detected_items=detected_items)
    except Exception as e:
        logger.error(f"음성 탐지 중 오류 발생: {str(e)}", exc_info=True)
        # 서비스 레이어에서 이미 빈 리스트를 반환하므로 여기서도 안전하게 처리
        return AudioDetectionResponse(detected_items=[])


@router.post("/video/detect", response_model=VideoDetectionResponse)
async def detect_video_personal_info(
    request: VideoDetectionRequest,
    video_detector=Depends(get_video_detector),
):
    """영상 개인정보 탐지 (Vision + LLM)"""
    try:
        logger.info(f"영상 탐지 요청: 포맷={request.video_format}")

        result = await run_in_thread(
            video_detector.detect,
            request.video_data,
            request.video_format
        )

        faces = result.get("faces", [])
        personal_info_in_audio = result.get("personal_info_in_audio", [])
        text_pii_regions = result.get("text_pii_regions", [])
        has_pii = (
            len(faces) > 0
            or len(personal_info_in_audio) > 0
            or len(text_pii_regions) > 0
        )

        if has_pii:
            status = "detected"
            message = (
                f"개인정보가 탐지되었습니다. "
                f"(얼굴: {len(faces)}개, 오디오: {len(personal_info_in_audio)}개, 화면 텍스트 PII: {len(text_pii_regions)}개)"
            )
        else:
            status = "no_pii"
            message = "개인정보가 탐지되지 않았습니다."

        logger.info(f"영상 탐지 결과: {status} - {message}")

        return VideoDetectionResponse(
            success=True,
            status=status,
            message=message,
        )
    except Exception as e:
        logger.error(f"영상 탐지 중 오류 발생: {str(e)}", exc_info=True)
        # 서비스 레이어에서 이미 빈 결과를 반환하므로 여기서도 안전하게 처리
        return VideoDetectionResponse(
            success=False,
            status="error",
            message=f"영상 탐지 중 오류가 발생했습니다: {str(e)}"
        )


@router.post("/mask", response_model=FileMaskResponse)
async def apply_masking(
    request: FileMaskRequest,
    document_file_processor=Depends(get_document_file_processor),
    image_detector=Depends(get_image_detector),
    audio_detector=Depends(get_audio_detector),
    video_detector=Depends(get_video_detector),
    masker=Depends(get_masker),
):
    """파일 서버 마스킹: connectionId + filePath + fileCategory → 마스킹 결과 base64. 로컬(SAVE_MASKED_OUTPUT=True)이면 output_file에 저장."""
    try:
        from app.services.file.file_mask_service import mask_file_from_server
        logger.info(f"마스킹 요청: connectionId={request.connectionId}, filePath={request.filePath}, fileCategory={request.fileCategory}")
        result = await run_in_thread(
            mask_file_from_server,
            connection_id=request.connectionId,
            file_path=request.filePath,
            file_category=request.fileCategory,
            document_file_processor=document_file_processor,
            image_detector=image_detector,
            audio_detector=audio_detector,
            video_detector=video_detector,
            masker=masker,
        )
        return FileMaskResponse(
            success=result["success"],
            maskedFileBase64=result.get("maskedFileBase64", ""),
        )
    except Exception as e:
        logger.error("마스킹 처리 중 오류: %s", e, exc_info=True)
        return FileMaskResponse(success=False, maskedFileBase64="")
