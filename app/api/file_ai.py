"""
파일 관련 AI 처리 API
"""
from fastapi import APIRouter, HTTPException, Response
import cv2
import base64
from app.models.request import (
    DocumentDetectionRequest,
    ImageDetectionRequest,
    AudioDetectionRequest,
    VideoDetectionRequest,
    MaskingRequest
)
from app.models.response import (
    DocumentDetectionResponse,
    ImageDetectionResponse,
    AudioDetectionResponse,
    VideoDetectionResponse,
    MaskingResponse
)
from app.services.file.document_detector import DocumentDetector
from app.services.file.image_detector import ImageDetector
from app.services.file.audio_detector import AudioDetector
from app.services.file.video_detector import VideoDetector
from app.services.file.masker import Masker
from app.utils.logger import logger
from app.utils.image_loader import load_image
from app.config import MODEL_PATH

router = APIRouter()

# 서비스 인스턴스
document_detector = DocumentDetector(model_path=MODEL_PATH)
image_detector = ImageDetector()
audio_detector = AudioDetector()
video_detector = VideoDetector()
masker = Masker()


def _get_original_image_bytes(image_data: str, image_format: str) -> bytes:
    """
    원본 이미지를 bytes로 반환하는 헬퍼 함수
    
    Args:
        image_data: 이미지 데이터 (base64 또는 파일 경로)
        image_format: "base64" 또는 "path"
        
    Returns:
        이미지 bytes
    """
    if image_format == "base64":
        # base64 디코딩
        if "," in image_data:
            base64_data = image_data.split(",")[1]
        else:
            base64_data = image_data
        return base64.b64decode(base64_data)
    else:
        # 파일 경로면 읽기
        with open(image_data, "rb") as f:
            return f.read()


def _get_original_image_base64(image_data: str, image_format: str) -> str:
    """
    원본 이미지를 base64 문자열로 반환하는 헬퍼 함수
    
    Args:
        image_data: 이미지 데이터 (base64 또는 파일 경로)
        image_format: "base64" 또는 "path"
        
    Returns:
        base64 인코딩된 이미지 문자열
    """
    if image_format == "base64":
        # 이미 base64 형식이면 그대로 반환
        if "," in image_data:
            return image_data.split(",")[1]
        else:
            return image_data
    else:
        # 파일 경로면 읽어서 base64로 변환
        img = load_image(image_data)
        _, encoded_img = cv2.imencode('.jpg', img)
        return base64.b64encode(encoded_img.tobytes()).decode()


@router.post("/document/detect", response_model=DocumentDetectionResponse)
async def detect_document_personal_info(request: DocumentDetectionRequest):
    """
    문서 개인정보 탐지 (BERT + NER)
    """
    try:
        logger.info(f"문서 탐지 요청: 타입={request.file_type}")
        
        detected_items = document_detector.detect(request.file_content)
        
        return DocumentDetectionResponse(
            detected_items=detected_items,
            is_masked=False  # Spring Boot에서 확인
        )
    
    except Exception as e:
        logger.error(f"문서 탐지 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/image/detect", response_model=ImageDetectionResponse)
async def detect_image_faces(request: ImageDetectionRequest):
    """
    이미지 얼굴 탐지 (Vision)
    """
    try:
        logger.info(f"이미지 탐지 요청: 포맷={request.image_format}")
        
        detected_faces = image_detector.detect_faces(request.image_data)
        
        return ImageDetectionResponse(detected_faces=detected_faces)
    
    except Exception as e:
        logger.error(f"이미지 탐지 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/image/detect-and-mask", response_model=MaskingResponse)
async def detect_and_mask_image(request: ImageDetectionRequest):
    """
    이미지 얼굴 탐지 및 마스킹 (한 번에 처리)
    이미지를 받아서 마스킹된 이미지를 Base64로 반환합니다.
    """
    try:
        logger.info(f"이미지 탐지 및 마스킹 요청: 포맷={request.image_format}")
        
        # 이미지 데이터
        image_data = request.image_data
        
        # 얼굴 탐지
        detected_faces = image_detector.detect_faces(image_data)
        
        # 얼굴이 탐지된 경우에만 마스킹
        if detected_faces:
            logger.info(f"{len(detected_faces)}개의 얼굴 탐지됨, 마스킹 처리 시작")
            masked_image_bytes = masker.mask_image(image_data, detected_faces)
            masked_image_base64 = base64.b64encode(masked_image_bytes).decode()
            
            return MaskingResponse(
                masked_file=masked_image_base64,
                file_type="image"
            )
        else:
            logger.info("얼굴이 탐지되지 않음, 원본 이미지 반환")
            # 얼굴이 없으면 원본 이미지를 그대로 반환
            masked_image_base64 = _get_original_image_base64(image_data, request.image_format)
            
            return MaskingResponse(
                masked_file=masked_image_base64,
                file_type="image"
            )
    
    except Exception as e:
        logger.error(f"이미지 탐지 및 마스킹 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/image/detect-and-mask/file")
async def detect_and_mask_image_file(request: ImageDetectionRequest):
    """
    이미지 얼굴 탐지 및 마스킹 (이미지 파일로 직접 반환)
    이미지를 받아서 마스킹된 이미지를 이미지 파일로 직접 반환합니다.
    Postman에서 바로 이미지를 확인할 수 있습니다.
    """
    try:
        logger.info(f"이미지 탐지 및 마스킹 요청 (파일 반환): 포맷={request.image_format}")
        
        # 이미지 데이터
        image_data = request.image_data
        
        # 얼굴 탐지
        detected_faces = image_detector.detect_faces(image_data)
        
        # 얼굴이 탐지된 경우에만 마스킹
        if detected_faces:
            logger.info(f"{len(detected_faces)}개의 얼굴 탐지됨, 마스킹 처리 시작")
            masked_image_bytes = masker.mask_image(image_data, detected_faces)
        else:
            logger.info("얼굴이 탐지되지 않음, 원본 이미지 반환")
            # 얼굴이 없으면 원본 이미지를 그대로 반환
            masked_image_bytes = _get_original_image_bytes(image_data, request.image_format)
        
        # 이미지 파일로 직접 반환
        return Response(
            content=masked_image_bytes,
            media_type="image/jpeg",
            headers={
                "Content-Disposition": "inline; filename=masked_image.jpg"
            }
        )
    
    except Exception as e:
        logger.error(f"이미지 탐지 및 마스킹 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/audio/detect", response_model=AudioDetectionResponse)
async def detect_audio_personal_info(request: AudioDetectionRequest):
    """
    음성 개인정보 탐지 (LLM)
    """
    try:
        logger.info(f"음성 탐지 요청: 포맷={request.audio_format}")
        
        detected_items = audio_detector.detect(request.audio_data)
        
        return AudioDetectionResponse(detected_items=detected_items)
    
    except Exception as e:
        logger.error(f"음성 탐지 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/video/detect", response_model=VideoDetectionResponse)
async def detect_video_personal_info(request: VideoDetectionRequest):
    """
    영상 개인정보 탐지 (Vision + LLM)
    """
    try:
        logger.info(f"영상 탐지 요청: 포맷={request.video_format}")
        
        result = video_detector.detect(request.video_data)
        
        # 탐지 결과 확인
        faces = result.get("faces", [])
        personal_info_in_audio = result.get("personal_info_in_audio", [])
        
        # 개인정보 탐지 여부 확인
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
            message=message
        )
    
    except Exception as e:
        logger.error(f"영상 탐지 오류: {str(e)}")
        return VideoDetectionResponse(
            success=False,
            status="error",
            message=f"영상 탐지 중 오류가 발생했습니다: {str(e)}"
        )


@router.post("/mask", response_model=MaskingResponse)
async def apply_masking(request: MaskingRequest):
    """
    마스킹 처리
    """
    try:
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
            # detected_items가 비어있으면 자동 탐지 수행
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
            # detected_items가 비어있으면 자동 탐지 수행
            if not request.detected_items:
                logger.info("비디오 detected_items가 비어있어 자동 탐지를 수행합니다.")
                detection_result = video_detector.detect(request.file_data)
                faces = detection_result.get("faces", [])
                audio_items = detection_result.get("personal_info_in_audio", [])
                logger.info(f"자동 탐지 완료: 얼굴 {len(faces)}개, 오디오 항목 {len(audio_items)}개 발견")
            else:
                # video는 faces와 audio_items 분리 필요
                faces = [item for item in request.detected_items if "x" in item]
                audio_items = [item for item in request.detected_items if "start_time" in item]
                logger.info(f"제공된 detected_items 사용: 얼굴 {len(faces)}개, 오디오 항목 {len(audio_items)}개")
            
            masked_bytes = masker.mask_video(request.file_data, faces, audio_items)
            masked_data = base64.b64encode(masked_bytes).decode()
        
        else:
            raise HTTPException(status_code=400, detail=f"지원하지 않는 파일 타입: {request.file_type}")
        
        return MaskingResponse(
            masked_file=masked_data,
            file_type=request.file_type
        )
    
    except Exception as e:
        logger.error(f"마스킹 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

