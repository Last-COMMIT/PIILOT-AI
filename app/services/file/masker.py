"""
마스킹 처리 (공통)
"""
from typing import List, Dict
import cv2
import numpy as np
from app.utils.logger import logger
from app.utils.image_loader import load_image
from app.services.file.audio_masking import audio_pii_service


class Masker:
    """파일 마스킹 처리"""
    
    def __init__(self):
        logger.info("Masker 초기화")
    
    def mask_document(self, text: str, detected_items: list) -> str:
        """
        문서 마스킹
        
        Args:
            text: 원본 텍스트
            detected_items: 탐지된 개인정보 리스트
            
        Returns:
            마스킹된 텍스트
        """
        # TODO: 구현 필요
        logger.info(f"문서 마스킹: {len(detected_items)}개 항목")
        pass
    
    def mask_image(self, image_path: str, faces: List[Dict]) -> bytes:
        """
        이미지 마스킹 (얼굴 모자이크)
        
        Args:
            image_path: 이미지 파일 경로 또는 base64 인코딩된 이미지
            faces: 얼굴 위치 리스트 [{"x": int, "y": int, "width": int, "height": int}, ...]
            
        Returns:
            마스킹된 이미지 (bytes)
        """
        logger.info(f"이미지 마스킹: {len(faces)}개 얼굴")
        
        # 이미지 로드 (공통 유틸리티 사용)
        img = load_image(image_path)
        
        # 얼굴 마스킹 처리
        for face in faces:
            x = face.get("x", 0)
            y = face.get("y", 0)
            width = face.get("width", 0)
            height = face.get("height", 0)
            
            # 좌표 유효성 검사
            if x < 0 or y < 0 or width <= 0 or height <= 0:
                continue
            
            # 이미지 범위 내인지 확인
            img_height, img_width = img.shape[:2]
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(img_width, x + width)
            y2 = min(img_height, y + height)
            
            if x2 > x1 and y2 > y1:
                # 얼굴 영역 추출 및 blur 처리
                face_region = img[y1:y2, x1:x2]
                if face_region.size > 0:
                    # 강한 blur 적용
                    blurred_face = cv2.blur(face_region, (100, 100))
                    img[y1:y2, x1:x2] = blurred_face
        
        # 이미지를 bytes로 변환
        _, encoded_img = cv2.imencode('.jpg', img)
        return encoded_img.tobytes()
    
    def mask_audio(self, audio_data: str, detected_items: list) -> bytes:
        """
        음성 마스킹 (개인정보 부분 음소거 또는 변조)
        
        Args:
            audio_data: 음성 파일 경로 또는 base64 인코딩된 오디오 데이터
            detected_items: 탐지된 개인정보 리스트
            
        Returns:
            마스킹된 오디오 (bytes)
        """
        try:
            logger.info(f"음성 마스킹 시작: {len(detected_items)}개 항목")
            
            # 오디오 마스킹 서비스 사용
            masked_bytes = audio_pii_service.mask_audio(audio_data, detected_items)
            
            logger.info("음성 마스킹 완료")
            return masked_bytes
            
        except Exception as e:
            logger.error(f"음성 마스킹 중 오류 발생: {str(e)}")
            raise e
    
    def mask_video(self, video_path: str, faces: list, audio_items: list) -> bytes:
        """
        영상 마스킹 (얼굴 모자이크 + 오디오 마스킹)
        
        Args:
            video_path: 영상 파일 경로
            faces: 얼굴 위치 리스트
            audio_items: 오디오 개인정보 리스트
            
        Returns:
            마스킹된 비디오 (bytes)
        """
        # TODO: 구현 필요
        logger.info(f"영상 마스킹: {len(faces)}개 얼굴, {len(audio_items)}개 오디오 항목")
        pass

