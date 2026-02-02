"""
파일 관련 AI 처리 API
"""
from fastapi import APIRouter, HTTPException, Depends
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
    FileDetectRequest,
    FileDetectResponse,
)
from app.api.deps import (
    get_document_detector,
    get_image_detector,
    get_audio_detector,
    get_video_detector,
    get_masker,
)
from app.core.logging import logger

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
        results = scan_files(
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

        detected_items = document_detector.detect(request.file_content)

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

        detected_faces = image_detector.detect_faces(request.image_data)

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

        detected_items = audio_detector.detect(request.audio_data, request.audio_format)

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

        result = video_detector.detect(request.video_data, request.video_format)

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


@router.post("/mask", response_model=MaskingResponse)
async def apply_masking(
    request: MaskingRequest,
    masker=Depends(get_masker),
):
    """마스킹 처리"""
    try:
        detected_items_count = len(request.detected_items) if request.detected_items else 0
        logger.info(f"마스킹 요청: 타입={request.file_type}, 포맷={request.file_format}, 항목 수={detected_items_count}")

        masked_data = None

        if request.file_type == "document":
            try:
                # 기존 로직: 텍스트 내용(String)만 마스킹 -> Base64 반환
                # (로컬 파일 경로 테스트 로직 제거됨)
                detected_items = request.detected_items if request.detected_items else []
                masked_text = masker.mask_document(request.file_data, detected_items)
                masked_data = base64.b64encode(masked_text.encode()).decode()
                
            except Exception as e:
                logger.error(f"문서 마스킹 중 오류 발생: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"문서 마스킹 중 오류가 발생했습니다: {str(e)}")

        elif request.file_type == "image":
            try:
                # 이미지 타입일 때만 ImageDetector 필요
                if not request.detected_items:
                    logger.info("이미지 detected_items가 비어있어 자동 탐지를 수행합니다.")
                    from app.api.deps import get_image_detector
                    image_detector = get_image_detector()
                    detected_faces = image_detector.detect_faces(request.file_data)
                    logger.info(f"자동 탐지 완료: {len(detected_faces)}개 얼굴 발견")
                else:
                    detected_faces = request.detected_items
                    logger.info(f"제공된 detected_items 사용: {len(detected_faces)}개 항목")

                if detected_faces:
                    logger.info(f"{len(detected_faces)}개의 얼굴 탐지됨, 마스킹 처리 시작")
                    masked_bytes = masker.mask_image(request.file_data, detected_faces)
                    if masked_bytes:
                        masked_data = base64.b64encode(masked_bytes).decode()
                    else:
                        logger.warning("마스킹된 이미지가 비어있어 원본 이미지 반환")
                        from app.utils.base64_utils import get_original_image_base64
                        masked_data = get_original_image_base64(request.file_data, request.file_format)
                else:
                    logger.info("얼굴이 탐지되지 않음, 원본 이미지 반환")
                    # 원본 이미지를 base64로 반환
                    from app.utils.base64_utils import get_original_image_base64
                    masked_data = get_original_image_base64(request.file_data, request.file_format)
            except Exception as e:
                logger.error(f"이미지 마스킹 중 오류 발생: {str(e)}", exc_info=True)
                # 원본 이미지 반환 시도
                try:
                    from app.utils.base64_utils import get_original_image_base64
                    masked_data = get_original_image_base64(request.file_data, request.file_format)
                    logger.warning("마스킹 실패로 원본 이미지 반환")
                except:
                    raise HTTPException(status_code=500, detail=f"이미지 마스킹 중 오류가 발생했습니다: {str(e)}")

        elif request.file_type == "audio":
            try:
                # 오디오 타입일 때만 AudioDetector 필요
                if not request.detected_items:
                    logger.info("오디오 detected_items가 비어있어 자동 탐지를 수행합니다.")
                    from app.api.deps import get_audio_detector
                    audio_detector = get_audio_detector()
                    # file_format 사용
                    detected_items = audio_detector.detect(request.file_data, request.file_format)
                    logger.info(f"자동 탐지 완료: {len(detected_items)}개 항목 발견")
                else:
                    detected_items = request.detected_items
                    logger.info(f"제공된 detected_items 사용: {len(detected_items)}개 항목")

                # file_format에 따라 처리
                if request.file_format == "base64":
                    # base64 디코딩하여 bytes로 변환
                    from app.utils.base64_utils import decode_base64_data
                    audio_input = decode_base64_data(request.file_data)
                else:  # "path"
                    # 파일 경로 그대로 사용
                    import os
                    if not os.path.exists(request.file_data):
                        raise FileNotFoundError(f"오디오 파일을 찾을 수 없습니다: {request.file_data}")
                    audio_input = request.file_data

                masked_bytes = masker.mask_audio(audio_input, detected_items)
                if masked_bytes:
                    masked_data = base64.b64encode(masked_bytes).decode()
                else:
                    raise ValueError("오디오 마스킹 결과가 비어있습니다.")
            except Exception as e:
                logger.error(f"오디오 마스킹 중 오류 발생: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"오디오 마스킹 중 오류가 발생했습니다: {str(e)}")

        elif request.file_type == "video":
            try:
                # 비디오 타입일 때만 VideoDetector 필요
                if not request.detected_items:
                    logger.info("비디오 detected_items가 비어있어 자동 탐지를 수행합니다.")
                    from app.api.deps import get_video_detector
                    video_detector = get_video_detector()
                    # file_format 사용
                    detection_result = video_detector.detect(request.file_data, request.file_format)
                    faces = detection_result.get("faces", [])
                    audio_items = detection_result.get("personal_info_in_audio", [])
                    text_pii_regions = detection_result.get("text_pii_regions", [])
                    logger.info(
                        f"자동 탐지 완료: 얼굴 {len(faces)}개, 오디오 {len(audio_items)}개, 화면 텍스트 PII {len(text_pii_regions)}개"
                    )
                else:
                    faces = [item for item in request.detected_items if "x" in item]
                    audio_items = [item for item in request.detected_items if "start_time" in item]
                    text_pii_regions = [item for item in request.detected_items if "frame_number" in item and "width" in item]
                    logger.info(
                        f"제공된 detected_items 사용: 얼굴 {len(faces)}개, 오디오 {len(audio_items)}개, 화면 텍스트 PII {len(text_pii_regions)}개"
                    )

                # mask_video에 화면 텍스트 PII 영역 전달
                masked_bytes = masker.mask_video(
                    request.file_data, faces, audio_items, text_pii_regions=text_pii_regions
                )
                if masked_bytes:
                    masked_data = base64.b64encode(masked_bytes).decode()
                else:
                    raise ValueError("비디오 마스킹 결과가 비어있습니다.")
            except Exception as e:
                logger.error(f"비디오 마스킹 중 오류 발생: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"비디오 마스킹 중 오류가 발생했습니다: {str(e)}")

        else:
            raise HTTPException(status_code=400, detail=f"지원하지 않는 파일 타입: {request.file_type}")

        if masked_data is None:
            raise ValueError("마스킹 결과가 생성되지 않았습니다.")

        return MaskingResponse(
            masked_file=masked_data,
            file_type=request.file_type,
        )
    except HTTPException:
        # HTTPException은 그대로 전파
        raise
    except Exception as e:
        logger.error(f"마스킹 처리 중 예상치 못한 오류 발생: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"마스킹 처리 중 오류가 발생했습니다: {str(e)}"
        )
