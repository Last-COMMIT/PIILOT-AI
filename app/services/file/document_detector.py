
import os
import re
from pathlib import Path
from typing import Dict, List
from .extractors.text_extractor import TextExtractor
from .detectors.hybrid_detector import HybridPIIDetector
from .masker import Masker
from app.config import PII_NAMES, OUTPUT_DIR

class DocumentDetector:
    """하이브리드 PII 탐지 및 비식별화 통합 파이프라인"""
    def __init__(self, model_path: str, use_gpu=True, mask_char='*',
                 confidence_thresholds: Dict[str, float] = None):
        self.extractor = TextExtractor(use_gpu=use_gpu)
        self.detector = HybridPIIDetector(model_path, confidence_thresholds)
        self.deidentifier = Masker(mask_char=mask_char)

    def process_file(self, input_path: str, output_path: str = None,
                     method: str = 'mask') -> Dict:
        """파일을 처리하여 PII 비식별화"""
        ext = Path(input_path).suffix.lower()

        if output_path is None:
            filename = Path(input_path).stem
            output_path = os.path.join(OUTPUT_DIR, f"{filename}_deidentified{ext}")

        if ext == '.pdf':
            return self._process_pdf(input_path, output_path, method)
        elif ext == '.docx':
            return self._process_docx(input_path, output_path)
        elif ext == '.txt':
            return self._process_txt(input_path, output_path)
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {ext}")

    def _process_image(self, input_path: str, output_path: str, method: str) -> Dict:
        """이미지 파일 처리 (사용 안 함)"""
        raise NotImplementedError("이미지 파일 처리는 비활성화되었습니다.")

    def _process_pdf(self, input_path: str, output_path: str, method: str) -> Dict:
        """PDF 파일 처리"""
        text_blocks = self.extractor.extract_from_pdf(input_path)

        entities_per_block = []
        total_pii = 0
        pii_stats = {}
        confidence_stats = []
        method_stats = {'koelectra': 0, 'regex': 0}

        global_known_names = set()

        # Context-based Name Detection Variables (Strict Korean Only)
        expecting_korean_name = False
        
        for block in text_blocks:
            text = block['text']
            # Remove spaces for checking
            clean_text = text.replace(" ", "")
            
            entities = self.detector.detect_pii(text, context_names=global_known_names)
            
            # 1. Context-based Forced Detection (Strict Korean Only)
            if expecting_korean_name:
                # 한글 2~4글자이고, 다른 문자가 섞이지 않은 경우만 이름으로 인정
                # 공백 제거 후 확인
                if re.fullmatch(r'[가-힣]{2,4}', clean_text):
                     # 예외: '생년월일', '성명', '주소' 등 라벨 자체가 이름으로 오인되는 경우 제외
                     exclude_keywords = ['성명', '생년월일', '주소', '연락처', '이메일', '학점', '지원분야', '경력여부']
                     if not any(k in clean_text for k in exclude_keywords):
                        # 이미 탐지된 엔티티와 중복 체크
                        already_detected = any(e['label'] == 'p_nm' for e in entities)
                        if not already_detected:
                            entities.append({
                                'text': text,
                                'label': 'p_nm',
                                'start': 0,
                                'end': len(text),
                                'confidence': 0.85, # Context score
                                'method': 'context_strict'
                            })
                expecting_korean_name = False # Reset flag

            # 2. Check for Korean Name Labels (Set flag for NEXT block)
            # '성명', '한글성명', '국문' 등이 포함된 경우
            # 여권의 'l구 성명' 같은 OCR 오류도 '성명'으로 커버됨
            # 단, 문장 속에 있는 '이름' 등이 트리거가 되지 않도록, 블록 길이가 짧은 경우(20자 미만)에만 라벨로 간주
            name_keywords = ['성명', '한글성명', '이름']
            if len(clean_text) < 20 and any(k in clean_text for k in name_keywords):
                expecting_korean_name = True
            
            entities_per_block.append(entities)
            
            for entity in entities:
                if entity['label'] == 'p_nm':
                    global_known_names.add(entity['text'])

            for entity in entities:
                total_pii += 1
                label = entity['label']
                pii_stats[label] = pii_stats.get(label, 0) + 1
                confidence_stats.append(entity['confidence'])
                method_key = entity.get('method', 'koelectra')
                method_stats[method_key] = method_stats.get(method_key, 0) + 1

        print(f"총 {total_pii}개 PII 탐지")
        for label, count in pii_stats.items():
            print(f"    - {PII_NAMES.get(label, label)}: {count}개")

        if confidence_stats:
            avg_confidence = sum(confidence_stats) / len(confidence_stats)
            print(f"평균 신뢰도: {avg_confidence:.3f}")

        print(f"비식별화 처리 (방법: {method})")
        self.deidentifier.mask_pdf(input_path, text_blocks,
                                   entities_per_block, output_path)

        return {
            'input_path': input_path,
            'output_path': output_path,
            'total_pii': total_pii,
            'pii_stats': pii_stats,
            'avg_confidence': avg_confidence if confidence_stats else 0.0,
            'method_stats': method_stats,
            'method': method
        }

    def _process_docx(self, input_path: str, output_path: str) -> Dict:
        """DOCX 파일 처리"""
        paragraphs = self.extractor.extract_from_docx(input_path)

        entities_per_para = []
        total_pii = 0
        pii_stats = {}
        confidence_stats = []
        
        global_known_names = set()

        for para_info in paragraphs:
            text = para_info['text']
            entities = self.detector.detect_pii(text, context_names=global_known_names)
            entities_per_para.append(entities)
            
            for entity in entities:
                if entity['label'] == 'p_nm':
                    global_known_names.add(entity['text'])
            
            for entity in entities:
                total_pii += 1
                label = entity['label']
                pii_stats[label] = pii_stats.get(label, 0) + 1
                confidence_stats.append(entity['confidence'])

        print(f"총 {total_pii}개 PII 탐지")
        for label, count in pii_stats.items():
            print(f"    - {PII_NAMES.get(label, label)}: {count}개")

        if confidence_stats:
            avg_confidence = sum(confidence_stats) / len(confidence_stats)
            print(f"평균 신뢰도: {avg_confidence:.3f}")

        self.deidentifier.mask_docx(input_path, paragraphs,
                                    entities_per_para, output_path)

        return {
            'input_path': input_path,
            'output_path': output_path,
            'total_pii': total_pii,
            'pii_stats': pii_stats,
            'avg_confidence': avg_confidence if confidence_stats else 0.0,
            'method': 'mask'
        }

    def _process_txt(self, input_path: str, output_path: str) -> Dict:
        """TXT 파일 처리"""
        lines = self.extractor.extract_from_txt(input_path)

        entities_per_line = []
        total_pii = 0
        pii_stats = {}
        confidence_stats = []
        
        global_known_names = set()

        for line_info in lines:
            text = line_info['text']
            entities = self.detector.detect_pii(text, context_names=global_known_names)
            entities_per_line.append(entities)
            
            for entity in entities:
                if entity['label'] == 'p_nm':
                    global_known_names.add(entity['text'])
            
            for entity in entities:
                total_pii += 1
                label = entity['label']
                pii_stats[label] = pii_stats.get(label, 0) + 1
                confidence_stats.append(entity['confidence'])

        print(f"총 {total_pii}개 PII 탐지")
        for label, count in pii_stats.items():
            print(f"    - {PII_NAMES.get(label, label)}: {count}개")

        if confidence_stats:
            avg_confidence = sum(confidence_stats) / len(confidence_stats)
            print(f"평균 신뢰도: {avg_confidence:.3f}")

        print(f"비식별화 처리 중...")
        self.deidentifier.mask_txt(input_path, lines,
                                   entities_per_line, output_path)

        return {
            'input_path': input_path,
            'output_path': output_path,
            'total_pii': total_pii,
            'pii_stats': pii_stats,
            'avg_confidence': avg_confidence if confidence_stats else 0.0,
            'method': 'mask'
        }