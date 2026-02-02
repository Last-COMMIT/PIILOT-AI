"""
파일 타입별 마스킹 디스패치 서비스
(기존 masker.py 리팩토링 - 통합 Masker 클래스)
"""
from typing import List, Dict
from app.core.logging import logger
from app.services.file.processors.document_masker import DocumentMasker
from app.services.file.processors.image_masker import ImageMasker
from app.services.file.processors.video_masker import VideoMasker
from app.services.file.processors.audio_masker import AudioMasker
from app.services.file.face_detector import YOLOFaceDetector

class Masker:
    """파일 마스킹 처리 (문서, 이미지, 오디오, 비디오 통합 디스패치)"""

    def __init__(self, mask_char='*'):
        # self.document_masker = DocumentMasker(mask_char=mask_char)
        self.image_masker = ImageMasker()
        self.face_detector = YOLOFaceDetector()
        
        # DocumentMasker에 face_detector와 image_masker 전달
        self.document_masker = DocumentMasker(
            mask_char=mask_char,
            face_detector=self.face_detector,
            image_masker=self.image_masker
        )

        self.video_masker = VideoMasker()
        self.audio_masker = AudioMasker()
        logger.info("Masker 초기화 (얼굴 인식 모델 로드 완료)")

    def mask_document(self, text: str, detected_items: List[Dict]) -> str:
        """문서 텍스트 마스킹"""
        entities = []
        for item in detected_items:
            if 'start' in item and 'end' in item:
                entities.append({
                    'start': item['start'],
                    'end': item['end'],
                    'text': item.get('value', item.get('text', '')),
                })
        return self.document_masker.mask_text(text, entities)

    def mask_text(self, text: str, entities: List[Dict]) -> str:
        """텍스트의 PII를 마스킹"""
        return self.document_masker.mask_text(text, entities)

    def mask_image(self, image_path: str, faces: List[Dict]) -> bytes:
        """이미지 마스킹"""
        return self.image_masker.mask_image(image_path, faces)

    def mask_audio(self, audio_data: str, detected_items: list) -> bytes:
        """음성 마스킹"""
        return self.audio_masker.mask_audio(audio_data, detected_items)

    def mask_video(self, video_path: str, faces: list, audio_items: list,
                   save_path: str = None, text_pii_regions: list = None) -> bytes:
        """영상 마스킹 (얼굴 + 화면 텍스트 PII + 오디오)"""
        return self.video_masker.mask_video(
            video_path, faces, audio_items, save_path,
            text_pii_regions=text_pii_regions or [],
        )

    # 하위 호환성: 기존 masker.py 인터페이스
    def mask_pdf(self, *args, **kwargs):
        return self.document_masker.mask_pdf(*args, **kwargs)

    def mask_docx(self, *args, **kwargs):
        return self.document_masker.mask_docx(*args, **kwargs)

    def mask_txt(self, *args, **kwargs):
        return self.document_masker.mask_txt(*args, **kwargs)
