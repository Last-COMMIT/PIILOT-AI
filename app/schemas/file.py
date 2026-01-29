"""
File AI 요청/응답 스키마
"""
from pydantic import BaseModel
from typing import List, Dict, Optional


# ========== 요청 ==========

class DocumentDetectionRequest(BaseModel):
    """문서 개인정보 탐지 요청"""
    file_content: str
    file_type: Optional[str] = "txt"


class ImageDetectionRequest(BaseModel):
    """이미지 얼굴 탐지 요청"""
    image_data: str
    image_format: Optional[str] = "base64"


class AudioDetectionRequest(BaseModel):
    """음성 개인정보 탐지 요청"""
    audio_data: str
    audio_format: Optional[str] = "base64"


class VideoDetectionRequest(BaseModel):
    """영상 개인정보 탐지 요청"""
    video_data: str
    video_format: Optional[str] = "base64"


class MaskingRequest(BaseModel):
    """마스킹 처리 요청"""
    file_type: str
    file_data: str
    file_format: Optional[str] = "base64"  # "base64" 또는 "path"
    detected_items: Optional[List[Dict]] = []


# ========== 응답 ==========

class DetectedPersonalInfo(BaseModel):
    """탐지된 개인정보"""
    type: str
    value: Optional[str] = None
    start: Optional[int] = None
    end: Optional[int] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    confidence: float


class DetectedFace(BaseModel):
    """탐지된 얼굴"""
    x: int
    y: int
    width: int
    height: int
    confidence: float
    frame_number: Optional[int] = None


class DocumentDetectionResponse(BaseModel):
    """문서 탐지 응답"""
    detected_items: List[DetectedPersonalInfo]
    is_masked: bool = False


class ImageDetectionResponse(BaseModel):
    """이미지 탐지 응답"""
    detected_faces: List[DetectedFace]


class AudioDetectionResponse(BaseModel):
    """음성 탐지 응답"""
    detected_items: List[DetectedPersonalInfo]


class VideoDetectionResponse(BaseModel):
    """영상 탐지 응답"""
    success: bool
    status: str
    message: str


class MaskingResponse(BaseModel):
    """마스킹 처리 응답"""
    masked_file: str
    file_type: str
