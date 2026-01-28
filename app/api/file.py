"""
파일 관련 AI 처리 API
"""
from fastapi import APIRouter, HTTPException, Response, Depends
import base64
from app.schemas.file import (
    DocumentDetectionRequest,
    DocumentDetectionResponse,
    ImageDetectionRequest,
    ImageDetectionResponse,
    AudioDetectionRequest,
    AudioDetectionResponse,
    VideoDetectionRequest,
    VideoDetectionResponse,
    MaskingRequest,
    MaskingResponse,
)
from app.api.deps import (
    get_document_detector,
    get_image_detector,
    get_audio_detector,
    get_video_detector,
    get_masker,
)
from app.utils.base64_utils import get_original_image_bytes, get_original_image_base64
from app.core.logging import logger

router = APIRouter()


@router.post("/document/detect", response_model=DocumentDetectionResponse)
async def detect_document_personal_info(
    request: DocumentDetectionRequest,
    document_detector=Depends(get_document_detector),
):
    """문서 개인정보 탐지 (BERT + NER)"""
    logger.info(f"문서 탐지 요청: 타입={request.file_type}")

    detected_items = document_detector.detect(request.file_content)

    return DocumentDetectionResponse(
        detected_items=detected_items,
        is_masked=False,
    )


@router.post("/image/detect", response_model=ImageDetectionResponse)
async def detect_image_faces(
    request: ImageDetectionRequest,
    image_detector=Depends(get_image_detector),
):
    """이미지 얼굴 탐지 (Vision)"""
    logger.info(f"이미지 탐지 요청: 포맷={request.image_format}")

    detected_faces = image_detector.detect_faces(request.image_data)

    return ImageDetectionResponse(detected_faces=detected_faces)


@router.post("/image/detect-and-mask", response_model=MaskingResponse)
async def detect_and_mask_image(
    request: ImageDetectionRequest,
    image_detector=Depends(get_image_detector),
    masker=Depends(get_masker),
):
    """이미지 얼굴 탐지 및 마스킹 (한 번에 처리)"""
    logger.info(f"이미지 탐지 및 마스킹 요청: 포맷={request.image_format}")

    image_data = request.image_data
    detected_faces = image_detector.detect_faces(image_data)

    if detected_faces:
        logger.info(f"{len(detected_faces)}개의 얼굴 탐지됨, 마스킹 처리 시작")
        masked_image_bytes = masker.mask_image(image_data, detected_faces)
        masked_image_base64 = base64.b64encode(masked_image_bytes).decode()
    else:
        logger.info("얼굴이 탐지되지 않음, 원본 이미지 반환")
        masked_image_base64 = get_original_image_base64(image_data, request.image_format)

    return MaskingResponse(
        masked_file=masked_image_base64,
        file_type="image",
    )


@router.post("/image/detect-and-mask/file")
async def detect_and_mask_image_file(
    request: ImageDetectionRequest,
    image_detector=Depends(get_image_detector),
    masker=Depends(get_masker),
):
    """이미지 얼굴 탐지 및 마스킹 (이미지 파일로 직접 반환)"""
    logger.info(f"이미지 탐지 및 마스킹 요청 (파일 반환): 포맷={request.image_format}")

    image_data = request.image_data
    detected_faces = image_detector.detect_faces(image_data)

    if detected_faces:
        logger.info(f"{len(detected_faces)}개의 얼굴 탐지됨, 마스킹 처리 시작")
        masked_image_bytes = masker.mask_image(image_data, detected_faces)
    else:
        logger.info("얼굴이 탐지되지 않음, 원본 이미지 반환")
        masked_image_bytes = get_original_image_bytes(image_data, request.image_format)

    return Response(
        content=masked_image_bytes,
        media_type="image/jpeg",
        headers={"Content-Disposition": "inline; filename=masked_image.jpg"},
    )


@router.post("/audio/detect", response_model=AudioDetectionResponse)
async def detect_audio_personal_info(
    request: AudioDetectionRequest,
    audio_detector=Depends(get_audio_detector),
):
    """음성 개인정보 탐지 (LLM)"""
    logger.info(f"음성 탐지 요청: 포맷={request.audio_format}")

    detected_items = audio_detector.detect(request.audio_data)

    return AudioDetectionResponse(detected_items=detected_items)


@router.post("/video/detect", response_model=VideoDetectionResponse)
async def detect_video_personal_info(
    request: VideoDetectionRequest,
    video_detector=Depends(get_video_detector),
):
    """영상 개인정보 탐지 (Vision + LLM)"""
    logger.info(f"영상 탐지 요청: 포맷={request.video_format}")

    result = video_detector.detect(request.video_data)

    faces = result.get("faces", [])
    personal_info_in_audio = result.get("personal_info_in_audio", [])
    has_pii = len(faces) > 0 or len(personal_info_in_audio) > 0

    if has_pii:
        status = "detected"
        message = f"개인정보가 탐지되었습니다. (얼굴: {len(faces)}개, 오디오: {len(personal_info_in_audio)}개)"
    else:
        status = "no_pii"
        message = "개인정보가 탐지되지 않았습니다."

    logger.info(f"영상 탐지 결과: {status} - {message}")

    return VideoDetectionResponse(
        success=True,
        status=status,
        message=message,
    )


@router.post("/mask", response_model=MaskingResponse)
async def apply_masking(
    request: MaskingRequest,
    masker=Depends(get_masker),
    audio_detector=Depends(get_audio_detector),
    video_detector=Depends(get_video_detector),
):
    """마스킹 처리"""
    detected_items_count = len(request.detected_items) if request.detected_items else 0
    logger.info(f"마스킹 요청: 타입={request.file_type}, 항목 수={detected_items_count}")

    masked_data = None

    if request.file_type == "document":
        masked_text = masker.mask_document(request.file_data, request.detected_items)
        masked_data = base64.b64encode(masked_text.encode()).decode()

    elif request.file_type == "image":
        masked_bytes = masker.mask_image(request.file_data, request.detected_items)
        masked_data = base64.b64encode(masked_bytes).decode()

    elif request.file_type == "audio":
        if not request.detected_items:
            logger.info("오디오 detected_items가 비어있어 자동 탐지를 수행합니다.")
            detected_items = audio_detector.detect(request.file_data)
            logger.info(f"자동 탐지 완료: {len(detected_items)}개 항목 발견")
        else:
            detected_items = request.detected_items
            logger.info(f"제공된 detected_items 사용: {len(detected_items)}개 항목")

        masked_bytes = masker.mask_audio(request.file_data, detected_items)
        masked_data = base64.b64encode(masked_bytes).decode()

    elif request.file_type == "video":
        if not request.detected_items:
            logger.info("비디오 detected_items가 비어있어 자동 탐지를 수행합니다.")
            detection_result = video_detector.detect(request.file_data)
            faces = detection_result.get("faces", [])
            audio_items = detection_result.get("personal_info_in_audio", [])
            logger.info(f"자동 탐지 완료: 얼굴 {len(faces)}개, 오디오 항목 {len(audio_items)}개 발견")
        else:
            faces = [item for item in request.detected_items if "x" in item]
            audio_items = [item for item in request.detected_items if "start_time" in item]
            logger.info(f"제공된 detected_items 사용: 얼굴 {len(faces)}개, 오디오 항목 {len(audio_items)}개")

        masked_bytes = masker.mask_video(request.file_data, faces, audio_items)
        masked_data = base64.b64encode(masked_bytes).decode()

    else:
        raise HTTPException(status_code=400, detail=f"지원하지 않는 파일 타입: {request.file_type}")

    return MaskingResponse(
        masked_file=masked_data,
        file_type=request.file_type,
    )
