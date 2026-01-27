
import cv2
import fitz
import numpy as np
from typing import List, Dict
from docx import Document
from app.utils.logger import logger
from app.utils.image_loader import load_image
from app.services.file.audio_masking import audio_pii_service

# Helper for image loading (simulating app.utils.image_loader)
def load_image(image_path: str):
    return cv2.imread(image_path)

class Masker:
    """파일 마스킹 처리 (문서, 이미지, 오디오, 비디오 통합)"""
    def __init__(self, mask_char='*'):
        self.mask_char = mask_char
        logger.info("Masker 초기화")

    def mask_text(self, text: str, entities: List[Dict]) -> str:
        """텍스트의 PII를 마스킹"""
        if not entities:
            return text

        entities_sorted = sorted(entities, key=lambda x: x['start'], reverse=True)

        masked_text = text
        for entity in entities_sorted:
            start = entity['start']
            end = entity['end']
            pii_text = entity['text']
            mask_str = self.mask_char * len(pii_text)
            masked_text = masked_text[:start] + mask_str + masked_text[end:]

        return masked_text

    def mask_image(self, image_path: str, faces: List[Dict]) -> bytes:
        """
        이미지 마스킹 (얼굴 모자이크) - 팀원 구현 부분 복원
        
        Args:
            image_path: 이미지 파일 경로 또는 base64 인코딩된 이미지
            faces: 얼굴 위치 리스트 [{"x": int, "y": int, "width": int, "height": int}, ...]
            
        Returns:
            마스킹된 이미지 (bytes)
        """
        logger.info(f"이미지 마스킹: {len(faces)}개 얼굴")
        
        # 이미지 로드 (공통 유틸리티 사용)
        img = load_image(image_path)
        if img is None:
            print(f"이미지 로드 실패: {image_path}")
            return b""
        
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
        """
        # TODO: 구현 필요
        logger.info(f"영상 마스킹: {len(faces)}개 얼굴, {len(audio_items)}개 오디오 항목")
        pass

    def mask_pdf(self, pdf_path: str, text_blocks: List[Dict],
                 entities_per_block: List[List[Dict]],
                 output_path: str):
        """PDF의 PII 영역을 정밀하게 마스킹"""
        doc = fitz.open(pdf_path)

        for block, entities in zip(text_blocks, entities_per_block):
            if not entities:
                continue

            page_num = block['page']
            page = doc[page_num]
            
            # block['bbox']는 2배 해상도 기준이므로, 원본 PDF 좌표로 변환 (나누기 2)
            scale_factor = 2.0
            x0, y0, x1, y1 = block['bbox'][0][0], block['bbox'][0][1], block['bbox'][2][0], block['bbox'][2][1]
            
            clip_rect = fitz.Rect(x0 / scale_factor, y0 / scale_factor, 
                                  x1 / scale_factor, y1 / scale_factor)
            
            # PII 엔티티별로 마스킹
            for entity in entities:
                target_text = entity['text']
                hit_rects = page.search_for(target_text, clip=clip_rect)
                
                if hit_rects:
                    for rect in hit_rects:
                        page.draw_rect(rect, color=(0, 0, 0), fill=(0, 0, 0))
                else:
                    try:
                        pts = block['bbox']
                        bx0 = pts[0][0] / scale_factor
                        by0 = pts[0][1] / scale_factor
                        bx1 = pts[2][0] / scale_factor
                        by1 = pts[2][1] / scale_factor
                        
                        fallback_rect = fitz.Rect(bx0, by0, bx1, by1)
                        page.draw_rect(fallback_rect, color=(0, 0, 0), fill=(0, 0, 0))
                    except Exception as e:
                        print(f"  [Warning] Fallback masking failed: {e}")

        doc.save(output_path)
        doc.close()
        print(f"PDF 저장: {output_path}")

    def mask_docx(self, docx_path: str, paragraphs: List[Dict],
                  entities_per_para: List[List[Dict]],
                  output_path: str):
        """DOCX의 PII를 마스킹 (문단 및 표 지원)"""
        doc = Document(docx_path)

        for para_info, entities in zip(paragraphs, entities_per_para):
            if not entities:
                continue

            target_para = None
            try:
                p_type = para_info.get('type', 'para')
                if p_type == 'para':
                    if 'idx' in para_info:
                        target_para = doc.paragraphs[para_info['idx']]
                    elif 'paragraph_idx' in para_info:
                        target_para = doc.paragraphs[para_info['paragraph_idx']]      
                elif p_type == 'table':
                    table = doc.tables[para_info['table_idx']]
                    row = table.rows[para_info['row_idx']]
                    cell = row.cells[para_info['col_idx']]
                    target_para = cell.paragraphs[para_info['para_idx']]
                
                if target_para:
                    masked_text = self.mask_text(target_para.text, entities)
                    target_para.text = masked_text
                    
            except IndexError:
                continue
            except Exception as e:
                print(f"마스킹 중 오류 발생: {e}")
                continue

        doc.save(output_path)
        print(f"DOCX 저장: {output_path}")

    def mask_txt(self, txt_path: str, lines: List[Dict],
                 entities_per_line: List[List[Dict]],
                 output_path: str):
        """TXT의 PII를 마스킹"""
        with open(txt_path, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()

        for line_info, entities in zip(lines, entities_per_line):
            if not entities:
                continue

            line_idx = line_info['line_idx']
            original_line = all_lines[line_idx]
            masked_text = self.mask_text(original_line.strip(), entities)
            all_lines[line_idx] = masked_text + '\n'

        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(all_lines)

        print(f"TXT 저장: {output_path}")