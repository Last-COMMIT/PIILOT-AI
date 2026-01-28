"""
Base64 인코딩/디코딩 통합 유틸리티
"""
import base64
import cv2
import numpy as np
import tempfile
import os
from app.core.logging import logger


def decode_base64_data(data: str) -> bytes:
    """
    Base64 문자열을 디코딩하여 bytes로 반환

    data:image/..., data:video/... 등의 접두사를 자동 처리
    """
    if "," in data:
        base64_data = data.split(",")[1]
    else:
        base64_data = data
    return base64.b64decode(base64_data)


def encode_to_base64(data: bytes) -> str:
    """bytes를 Base64 문자열로 인코딩"""
    return base64.b64encode(data).decode()


def is_base64(data: str) -> bool:
    """문자열이 base64 인코딩된 데이터인지 판별"""
    return data.startswith("data:") or len(data) > 1000


def decode_base64_image(image_data: str) -> np.ndarray:
    """
    Base64 이미지를 OpenCV ndarray로 디코딩

    Args:
        image_data: base64 인코딩된 이미지 문자열

    Returns:
        OpenCV BGR 이미지
    """
    image_bytes = decode_base64_data(image_data)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Base64 이미지를 디코딩할 수 없습니다.")
    return img


def encode_image_to_base64(img: np.ndarray, ext: str = '.jpg') -> str:
    """
    OpenCV 이미지를 Base64 문자열로 인코딩

    Args:
        img: OpenCV BGR 이미지
        ext: 인코딩 확장자 (기본 '.jpg')

    Returns:
        base64 인코딩된 이미지 문자열
    """
    _, encoded_img = cv2.imencode(ext, img)
    return base64.b64encode(encoded_img.tobytes()).decode()


def decode_base64_to_temp_file(data: str, suffix: str = '.mp4') -> str:
    """
    Base64 데이터를 디코딩하여 임시 파일로 저장

    Args:
        data: base64 인코딩된 데이터
        suffix: 파일 확장자

    Returns:
        임시 파일 경로
    """
    decoded_bytes = decode_base64_data(data)
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp.write(decoded_bytes)
    temp.close()
    logger.info(f"Base64 데이터를 임시 파일로 저장: {temp.name}")
    return temp.name


def get_original_image_bytes(image_data: str, image_format: str) -> bytes:
    """
    원본 이미지를 bytes로 반환

    Args:
        image_data: 이미지 데이터 (base64 또는 파일 경로)
        image_format: "base64" 또는 "path"

    Returns:
        이미지 bytes
    """
    if image_format == "base64":
        return decode_base64_data(image_data)
    else:
        with open(image_data, "rb") as f:
            return f.read()


def get_original_image_base64(image_data: str, image_format: str) -> str:
    """
    원본 이미지를 base64 문자열로 반환

    Args:
        image_data: 이미지 데이터 (base64 또는 파일 경로)
        image_format: "base64" 또는 "path"

    Returns:
        base64 인코딩된 이미지 문자열
    """
    if image_format == "base64":
        if "," in image_data:
            return image_data.split(",")[1]
        return image_data
    else:
        from app.utils.image_utils import load_image
        img = load_image(image_data)
        return encode_image_to_base64(img)
