"""
오디오 마스킹 최종 통합 버전
- 8종 개인정보 탐지: 이름, 전화번호, 이메일, 주소, 주민등록번호, IP주소, 계좌번호, 여권번호
- 정규식 + KoELECTRA 하이브리드 탐지
- Faster-Whisper STT
"""
import sys
import subprocess
import os
from pathlib import Path

# ==================== 필수 라이브러리 자동 설치 ====================
def install_and_import(package, import_name=None):
    if import_name is None:
        import_name = package
    try:
        __import__(import_name)
    except ImportError:
        print(f"[알림] '{package}' 라이브러리가 없습니다. 설치를 진행합니다...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--break-system-packages"])
        print(f"'{package}' 설치 완료.")

required_packages = [
    ("faster-whisper", "faster_whisper"),
    ("pydub", "pydub"),
    ("librosa", "librosa"),
    ("soundfile", "soundfile"),
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("tqdm", "tqdm"),
]

for package, import_name in required_packages:
    install_and_import(package, import_name)

# ================================================================

import re
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from pydub import AudioSegment
from pydub.generators import Sine
import librosa
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForTokenClassification
from tqdm import tqdm
from typing import List, Dict, Set
from glob import glob

# ==================== 설정 ====================
PII_NAMES = {
    'p_nm': '이름',
    'p_ph': '전화번호',
    'p_em': '이메일',
    'p_add': '주소',
    'p_rrn': '주민등록번호',
    'p_ip': 'IP주소',
    'p_acct': '계좌번호',
    'p_passport': '여권번호',
}

CONFIDENCE_THRESHOLDS = {
    'p_nm': 0.70,
    'p_ph': 0.75,
    'p_em': 0.75,
    'p_add': 0.80,
    'p_rrn': 0.90,
    'p_ip': 0.75,
    'p_acct': 0.85,
    'p_passport': 0.90,
}

# ==================== 정규식 PII 탐지기 (완전 버전) ====================
class EnhancedRegexPIIDetector:
    """향상된 정규식 PII 탐지기 - 6종 지원 (이름, 전화, 이메일, 주소, 주민번호, IP)"""
    def __init__(self):
        print("✓ 정규식 PII 탐지기 초기화 (6종: 이름, 전화번호, 이메일, 주소, 주민번호, IP)")

    def detect_phones(self, text: str) -> List[Dict]:
        """전화번호 탐지"""
        entities = []
        seen = set()

        patterns = [
            (r'01[016789]-\d{3,4}-\d{4}', 'mobile'),
            (r'0(?:2|3[1-3]|4[1-4]|5[1-5]|6[1-4])-\d{3,4}-\d{4}', 'landline'),
            (r'01[016789]\d{7,8}', 'mobile-no-hyphen'),
        ]

        for pattern, phone_type in patterns:
            for match in re.finditer(pattern, text):
                phone = match.group()
                digits = re.sub(r'\D', '', phone)

                if 9 <= len(digits) <= 11 and digits not in seen:
                    seen.add(digits)
                    entities.append({
                        'text': phone,
                        'label': 'p_ph',
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': 1.0,
                        'method': f'regex-{phone_type}'
                    })

        return entities

    def detect_emails(self, text: str) -> List[Dict]:
        """이메일 탐지 - OCR 오류 보정 포함"""
        entities = []

        # 표준 이메일
        standard_pattern = r'[a-zA-Z0-9][a-zA-Z0-9._+-]*@[a-zA-Z0-9][a-zA-Z0-9.-]*\.[a-zA-Z]{2,}'

        for match in re.finditer(standard_pattern, text):
            entities.append({
                'text': match.group(),
                'label': 'p_em',
                'start': match.start(),
                'end': match.end(),
                'confidence': 1.0,
                'method': 'regex-standard'
            })

        # OCR 오류 패턴
        ocr_pattern = r'[a-zA-Z0-9][a-zA-Z0-9._+-]*@[a-zA-Z0-9]+(?:com|net|org|cokr|kr|jp|cn|edu|gov|info)\b'

        for match in re.finditer(ocr_pattern, text):
            if any(e['start'] == match.start() for e in entities):
                continue

            email_text = match.group()
            domain_part = email_text.split('@')[1]

            if '.' not in domain_part:
                known_suffixes = ['com', 'net', 'org', 'cokr', 'kr', 'jp', 'cn', 'edu', 'gov', 'info']

                for suffix in known_suffixes:
                    if domain_part.endswith(suffix):
                        domain_name = domain_part[:-len(suffix)]
                        if len(domain_name) >= 2:
                            local_part = email_text.split('@')[0]
                            
                            if suffix == 'cokr':
                                corrected = f"{local_part}@{domain_name}.co.kr"
                            else:
                                corrected = f"{local_part}@{domain_name}.{suffix}"
                                
                            entities.append({
                                'text': corrected,
                                'label': 'p_em',
                                'start': match.start(),
                                'end': match.end(),
                                'confidence': 0.95,
                                'method': 'regex-ocr-corrected'
                            })
                            break

        return entities

    def detect_addresses(self, text: str) -> List[Dict]:
        """주소 탐지"""
        entities = []

        patterns = [
            r'[가-힣]{2,}(?:특별시|광역시|도)\s+[가-힣]{2,}(?:시|군|구)\s+[가-힣]{2,}(?:로|길)\s+\d+[가-힣0-9\s-]*',
            r'[가-힣]{2,}\s+[가-힣]{2,}(?:시|군|구)\s+[가-힣]{2,}(?:로|길)?\s*\d*[가-힣0-9\s-]*',
            r'[가-힣]{2,}(?:시|군|구)\s+[가-힣]{2,}(?:로|길)\s+\d+[가-힣0-9\s-]*',
            r'[가-힣]{2,}구\s+[가-힣]{2,}(?:로|길)\s+\d+[가-힣0-9\s-]*',
            r'[가-힣]{2,}(?:시|구)\s+[가-힣]{2,}동(?:\s+\d+)?',
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text):
                address = match.group().strip()

                if self._is_valid_address_structure(address):
                    entities.append({
                        'text': address,
                        'label': 'p_add',
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': 1.0,
                        'method': 'regex'
                    })

        return entities

    def _is_valid_address_structure(self, address: str) -> bool:
        """주소 구조 검증"""
        if len(address) < 8:
            return False

        has_admin = any(kw in address for kw in ['도', '시', '군', '구'])
        has_location = any(kw in address for kw in ['로', '길', '동'])

        hangul_chars = sum(1 for c in address if '가' <= c <= '힣')
        total_chars = len(address.replace(' ', ''))
        hangul_ratio = hangul_chars / total_chars if total_chars > 0 else 0
        is_mostly_hangul = hangul_ratio >= 0.6

        valid_endings = ['동', '번길', '번지', '호', '층']
        ends_with_number = address[-1].isdigit()
        ends_with_valid_keyword = any(address.endswith(ending) for ending in valid_endings)

        ends_improperly = (
            (address.endswith('구') or address.endswith('시'))
            and not ends_with_valid_keyword
            and not ends_with_number
        )
        ends_properly = not ends_improperly

        return has_admin and has_location and is_mostly_hangul and ends_properly

    def detect_rrn(self, text: str) -> List[Dict]:
        """주민등록번호 탐지"""
        entities = []
        
        pattern1 = r'\d{6}-[1-4]\d{6}'
        pattern2 = r'(?<!\d)\d{13}(?!\d)'
        
        for pattern in [pattern1, pattern2]:
            for match in re.finditer(pattern, text):
                rrn = match.group()
                digits = re.sub(r'\D', '', rrn)
                
                if len(digits) == 13:
                    gender_code = digits[6]
                    if gender_code in '1234':
                        entities.append({
                            'text': rrn,
                            'label': 'p_rrn',
                            'start': match.start(),
                            'end': match.end(),
                            'confidence': 1.0,
                            'method': 'regex'
                        })
        
        return entities

    def detect_ip(self, text: str) -> List[Dict]:
        """IP 주소 탐지"""
        entities = []
        pattern = r'(?<!\d)\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?!\d)'

        for match in re.finditer(pattern, text):
            ip = match.group()
            try:
                octets = [int(x) for x in ip.split('.')]
                if all(0 <= octet <= 255 for octet in octets):
                    entities.append({
                        'text': ip,
                        'label': 'p_ip',
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': 1.0,
                        'method': 'regex'
                    })
            except:
                pass
        return entities

    def detect_names(self, text: str) -> List[Dict]:
        """이름 탐지 - 문맥 기반"""
        entities = []
        
        context_patterns = [
            r'(?:이름[은는:]\s*)([가-힣]{2,4})(?=\s|님|씨|입니다|$)',
            r'(?:성명[은는:]\s*)([가-힣]{2,4})(?=\s|님|씨|입니다|$)',
            r'([가-힣]{2,4})(?:\s+(?:고객|상담사|님|씨|장군|대표|부장|과장|대리|사원))',
            r'(?:고객\s+)([가-힣]{2,4})(?=\(|님|씨|$)',
        ]
        
        common_surnames = [
            '김', '이', '박', '최', '정', '강', '조', '윤', '장', '임',
            '한', '오', '서', '신', '권', '황', '안', '송', '류', '홍',
            '전', '고', '문', '손', '배', '백', '허', '유', '남', '심',
            '노', '하', '곽', '성', '차', '주', '우', '구', '방', '공'
        ]
        
        stopwords = {
            '이것', '그것', '저것', '무엇', '어디', '누구', '언제', '어떻게',
            '이제', '그제', '저제', '여기', '거기', '저기',
            '이름', '성명', '고객', '상담', '담당', '관리', '정보',
            '확인', '처리', '등록', '삭제', '수정', '조회',
            '전화', '연락', '전화번호', '주소', '이메일', '메일',
            '서버', '주민', '주민번호', '고객정보', '계좌',
            '하나', '하고', '하면', '합니다', '했습니다', '합니까',
            '입니다', '있습니다', '없습니다', '됩니다',
            '서울', '부산', '대구', '인천', '광주', '대전', '울산', '세종',
            '경기', '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주',
            '강남구', '서초구', '송파구', '강동구', '성남시', '고양시',
            '우동', '장군', '대표', '사장', '부장', '과장', '대리', '사원'
        }
        
        for pattern in context_patterns:
            for match in re.finditer(pattern, text):
                if match.groups():
                    name = match.group(1)
                    name_start = match.start(1)
                    name_end = match.end(1)
                else:
                    name = match.group()
                    name_start = match.start()
                    name_end = match.end()
                
                if not (2 <= len(name) <= 4):
                    continue
                
                if name[0] not in common_surnames:
                    continue
                
                if name in stopwords:
                    continue
                
                is_duplicate = any(
                    e['start'] == name_start and e['end'] == name_end and e['label'] == 'p_nm'
                    for e in entities
                )
                
                if not is_duplicate:
                    entities.append({
                        'text': name,
                        'label': 'p_nm',
                        'start': name_start,
                        'end': name_end,
                        'confidence': 0.90,
                        'method': 'regex-korean-name'
                    })
        
        return entities

    def detect_all(self, text: str) -> List[Dict]:
        """6종 PII 탐지 (정규식)"""
        all_entities = []
        
        # 주민등록번호를 먼저 탐지 (전화번호 오탐 방지)
        rrn_entities = self.detect_rrn(text)
        all_entities.extend(rrn_entities)
        
        rrn_ranges = [(e['start'], e['end']) for e in rrn_entities]
        
        # 전화번호 (주민번호 범위 제외)
        phone_entities = self.detect_phones(text)
        for phone in phone_entities:
            is_inside_rrn = any(
                rrn_start <= phone['start'] < rrn_end or 
                rrn_start < phone['end'] <= rrn_end
                for rrn_start, rrn_end in rrn_ranges
            )
            if not is_inside_rrn:
                all_entities.append(phone)
        
        all_entities.extend(self.detect_emails(text))
        all_entities.extend(self.detect_addresses(text))
        all_entities.extend(self.detect_ip(text))
        all_entities.extend(self.detect_names(text))

        # 중복 제거
        seen = set()
        unique = []

        for entity in all_entities:
            key = (entity['start'], entity['end'], entity['label'])
            if key not in seen:
                seen.add(key)
                unique.append(entity)

        unique.sort(key=lambda x: x['start'])
        return unique


# ==================== DL 기반 PII 탐지기 (KoELECTRA) ====================
class KoELECTRAPIIDetector:
    """KoELECTRA NER 모델 - 8종 PII 탐지 (특히 계좌번호, 여권번호)"""
    def __init__(self, model_path: str, confidence_thresholds: Dict[str, float] = None):
        print(f"  KoELECTRA 모델 로드 시도: {model_path}")
        
        # Hugging Face 모델 체크 (org/model 형식)
        is_hf_model = '/' in model_path and not os.path.exists(model_path)
        
        if not is_hf_model and not os.path.exists(model_path):
            print(f"  ⚠ 로컬 모델을 찾을 수 없습니다: {model_path}")
            print(f"  → 정규식 탐지만 사용됩니다 (계좌번호, 여권번호 탐지 불가)")
            self.model = None
            return

        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            if is_hf_model:
                print(f"  → Hugging Face에서 다운로드 중...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForTokenClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()

            self.id2label = self.model.config.id2label
            self.label2id = self.model.config.label2id

            self.confidence_thresholds = confidence_thresholds or CONFIDENCE_THRESHOLDS

            print(f"  ✓ KoELECTRA 모델 로드 완료 (디바이스: {self.device})")
            print(f"  → 8종 PII 탐지 가능 (계좌번호, 여권번호 포함)")
            print(f"  → 모델 레이블: {list(self.id2label.values())}")
            
        except Exception as e:
            print(f"  ⚠ 모델 로드 실패: {e}")
            print(f"  → 정규식 탐지만 사용됩니다 (계좌번호, 여권번호 탐지 불가)")
            self.model = None

    def detect_pii(self, text: str) -> List[Dict]:
        """텍스트에서 PII 탐지"""
        if self.model is None:
            return []
            
        if not text.strip():
            return []

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=-1)[0]
            predictions = torch.argmax(probabilities, dim=-1)

        predictions = predictions.cpu().numpy()
        offsets = inputs["offset_mapping"][0].cpu().numpy()

        entities = []
        current_entity = None

        for idx, (pred, offset) in enumerate(zip(predictions, offsets)):
            if offset[0] == 0 and offset[1] == 0:
                continue

            pred_label = self.id2label[pred]
            confidence = probabilities[idx][pred].item()

            if pred_label == 'O':
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
                continue

            if pred_label.startswith('B-'):
                if current_entity:
                    entities.append(current_entity)

                current_entity = {
                    'label': pred_label[2:],
                    'start': offset[0],
                    'end': offset[1],
                    'confidences': [confidence],
                }

            elif pred_label.startswith('I-'):
                if current_entity and (current_entity['label'] == pred_label[2:]):
                    current_entity['end'] = offset[1]
                    current_entity['confidences'].append(confidence)
                else:
                    if current_entity:
                        entities.append(current_entity)

                    current_entity = {
                        'label': pred_label[2:],
                        'start': offset[0],
                        'end': offset[1],
                        'confidences': [confidence],
                    }

        if current_entity:
            entities.append(current_entity)

        final_results = []
        for ent in entities:
            full_text = text[ent['start']:ent['end']]
            avg_conf = sum(ent['confidences']) / len(ent['confidences'])

            result = {
                'text': full_text,
                'label': ent['label'],
                'start': ent['start'],
                'end': ent['end'],
                'confidence': avg_conf
            }
            final_results.append(result)

        return self.apply_confidence_filter(final_results)

    def apply_confidence_filter(self, entities: List[Dict]) -> List[Dict]:
        """신뢰도 기반 필터링"""
        if self.model is None:
            return []
            
        filtered = []

        for entity in entities:
            label = entity['label']
            confidence = entity['confidence']
            threshold = self.confidence_thresholds.get(label, 0.75)

            if confidence >= threshold:
                filtered.append(entity)

        return filtered


# ==================== 하이브리드 PII 탐지기 ====================
class HybridPIIDetector:
    """KoELECTRA NER + 정규식 기반 하이브리드 탐지"""
    def __init__(self, model_path: str, confidence_thresholds: Dict[str, float] = None):
        self.ner_detector = KoELECTRAPIIDetector(model_path, confidence_thresholds)
        self.regex_detector = EnhancedRegexPIIDetector()

    def merge_entities(self, ner_entities: List[Dict], regex_entities: List[Dict]) -> List[Dict]:
        """DL 모델(KoELECTRA)과 정규식 결과 병합 (중복 제거)"""
        merged = []

        # 1. 정규식 결과는 무조건 포함 (위치나 내용이 같아도 regex가 우선이거나, 따로 처리)
        # 단, regex끼리 겹치는 경우는 앞서 regex_detector.detect_all에서 이미 처리되었다고 가정
        merged.extend(regex_entities)

        # 2. KoELECTRA 결과 추가 (위치 겹침 체크)
        for ner_entity in ner_entities:
            is_overlapping = False
            for existing in merged:
                if self._is_overlapping(ner_entity, existing):
                    is_overlapping = True
                    break

            if not is_overlapping:
                ner_entity['method'] = 'koelectra'
                merged.append(ner_entity)

        merged.sort(key=lambda x: x['start'])
        return merged

    def _is_overlapping(self, entity1: Dict, entity2: Dict) -> bool:
        """두 엔티티가 겹치는지 확인"""
        start1, end1 = entity1['start'], entity1['end']
        start2, end2 = entity2['start'], entity2['end']
        return not (end1 <= start2 or end2 <= start1)

    def _extend_address_entities(self, text: str, entities: List[Dict]) -> List[Dict]:
        """주소 엔티티 확장 (휴리스틱: '동', '호' 등 상세주소 포함으로 확장)"""
        extended_entities = []
        
        # 확장 패턴: (아파트명 | 동/호/층 | 숫자+동/호/층 | 숫자)
        extension_pattern = r'^[\s,]*((?:[가-힣a-zA-Z0-9]+(?:타운|빌라|맨션|아파트|오피스텔)|[가-힣0-9]+(?:동|호|층)|[\d-]+(?:동|호|층)?))'

        for entity in entities:
            if entity['label'] != 'p_add':
                extended_entities.append(entity)
                continue
            
            current_end = entity['end']
            
            # 반복적으로 뒤따르는 주소 요소 확인
            while True:
                remaining_text = text[current_end:]
                if not remaining_text:
                    break
                    
                match = re.match(extension_pattern, remaining_text)
                if match:
                    # 매칭된 부분만큼 확장
                    matched_str = match.group(0) 
                    
                    new_end = current_end + len(matched_str)
                    
                    # 엔티티 업데이트
                    entity['end'] = new_end
                    entity['text'] = text[entity['start']:new_end]
                    
                    current_end = new_end
                else:
                    break
            
            extended_entities.append(entity)
            
        return extended_entities

    def _extend_short_names(self, text: str, entities: List[Dict]) -> List[Dict]:
        """짧은 이름 확장"""
        STOP_CHARS = {'이', '가', '은', '는', '을', '를', '의', '에', '와', '과', '로', '도', '만', '씨', '님', '군', '양', '과', '장'}

        for entity in entities:
            if entity['label'] != 'p_nm':
                continue
                
            name_text = entity['text'].strip()
            # 2글자 이름인 경우
            if len(name_text) == 2:
                current_end = entity['end']
                if current_end < len(text):
                    next_char = text[current_end]
                    
                    if '가' <= next_char <= '힣' and next_char not in STOP_CHARS:
                        entity['end'] += 1
                        entity['text'] = text[entity['start']:entity['end']]
        
        return entities

    def _propagate_known_names(self, text: str, entities: List[Dict], context_names: Set[str] = None) -> List[Dict]:
        """문맥 전파: 확실한 이름에서 이름만 추출하여 검색"""
        known_names = set()
        if context_names:
            known_names.update(context_names)

        for entity in entities:
            if entity['label'] == 'p_nm':
                known_names.add(entity['text'])
        
        if not known_names:
            return entities
            
        search_terms = set()
        for name in known_names:
            clean_name = name.strip()
            if len(clean_name) >= 3:
                given_name = clean_name[-2:] 
                search_terms.add(given_name)
            elif len(clean_name) == 2:
                search_terms.add(clean_name)
                
        search_terms = {t for t in search_terms if len(t) >= 2}
        
        if not search_terms:
            return entities
            
        propagated_entities = []
        def is_overlapping(start, end, existing_entities):
            for e in existing_entities:
                if max(start, e['start']) < min(end, e['end']):
                    return True
            return False

        for term in search_terms:
            for match in re.finditer(re.escape(term), text):
                start, end = match.span()
                
                if not is_overlapping(start, end, entities) and not is_overlapping(start, end, propagated_entities):
                    propagated_entities.append({
                        'start': start,
                        'end': end,
                        'text': term,
                        'label': 'p_nm',
                        'confidence': 0.90
                    })

        return entities + propagated_entities

    def _refine_entities(self, entities: List[Dict]) -> List[Dict]:
        """엔티티 정제"""
        refined = []
        
        label_patterns = [
            r'^(?:주\s*소|거\s*주\s*지|Address|Addr)\s*[:.]?\s*',
            r'^(?:성\s*명|이\s*름|Name)\s*[:.]?\s*',
            r'^(?:연\s*락\s*처|Phone|Mobile|Tel)\s*[:.]?\s*',
            r'^(?:이\s*메\s*일|E-?mail)\s*[:.]?\s*',
            r'^(?:생\s*년\s*월\s*일|Birth)\s*[:.]?\s*',
            r'^(?:주\s*민\s*번\s*호|RRN)\s*[:.]?\s*'
        ]
        
        for entity in entities:
            text = entity['text']
            label = entity['label']
            
            # 1. 라벨 트리밍
            for pattern in label_patterns:
                match = re.match(pattern, text, re.IGNORECASE)
                if match:
                    trim_len = len(match.group(0))
                    entity['start'] += trim_len
                    entity['text'] = text[trim_len:]
                    break
            
            # 주소 후처리
            if label == 'p_add':
                suffix_match = re.search(r'(?<=[0-9가-힣])(에|에서|로|으로)(\s.*)?$', entity['text'])
                if suffix_match:
                    suffix = suffix_match.group(0)
                    should_trim = False
                    if suffix.startswith('에') or suffix.startswith('에서') or suffix.startswith('으로'):
                        should_trim = True
                    elif suffix.startswith('로'):
                        if re.search(r'(동|호|층|번지|[0-9])$', entity['text'][:suffix_match.start()]):
                            should_trim = True
                            
                    if should_trim:
                        trim_len = len(suffix)
                        entity['end'] -= trim_len
                        entity['text'] = entity['text'][:-trim_len]
            
            if not entity['text'].strip():
                continue
                
            # 2. 오탐 필터링
            if label == 'p_nm':
                clean_text = entity['text'].strip()
                if len(clean_text) <= 1:
                    continue
                if re.search(r'[0-9!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>/?]', clean_text):
                    continue
            
            refined.append(entity)
            
        return refined

    def detect_pii(self, text: str, context_names: Set[str] = None) -> List[Dict]:
        """하이브리드 PII 탐지"""
        ner_entities = self.ner_detector.detect_pii(text)
        regex_entities = self.regex_detector.detect_all(text)
        
        merged_entities = self.merge_entities(ner_entities, regex_entities)
        
        merged_entities = self._extend_short_names(text, merged_entities)
        merged_entities = self._extend_address_entities(text, merged_entities)
        merged_entities = self._propagate_known_names(text, merged_entities, context_names)
        merged_entities = self._refine_entities(merged_entities)

        if regex_entities:
            print(f"정규식 추가 탐지: {len(regex_entities)}개")
            for entity in regex_entities:
                print(f"    + {PII_NAMES.get(entity['label'], entity['label'])}: '{entity['text']}'")

        return merged_entities

# ==================== 오디오 처리 함수들 ====================
def initialize_whisper_model(model_size="large-v3", device="auto", num_workers=4):
    """Faster-Whisper 모델 초기화"""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cuda":
        compute_type = "float16"
    else:
        compute_type = "int8"
    
    print(f"    Faster-Whisper 모델: {model_size}")
    print(f"    장치: {device} | 연산 타입: {compute_type}")
    print(f"    워커 수: {num_workers}")
    
    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
        num_workers=num_workers,
        download_root=None
    )
    
    print("    ✓ 모델 로드 완료")
    return model


def transcribe_audio_with_words(audio_path, model, language='ko'):
    """오디오 → 텍스트 변환 (단어별 타임스탬프 포함)"""
    segments, info = model.transcribe(
        audio_path,
        language=language,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=500,
            speech_pad_ms=200
        )
    )
    
    all_words = []
    full_text_parts = []
    
    for segment in segments:
        if hasattr(segment, 'words') and segment.words:
            for word_info in segment.words:
                all_words.append({
                    'word': word_info.word.strip(),
                    'start': word_info.start,
                    'end': word_info.end,
                    'probability': word_info.probability
                })
                full_text_parts.append(word_info.word.strip())
    
    full_text = ' '.join(full_text_parts)
    
    return {
        'full_text': full_text,
        'words': all_words,
        'language': info.language,
        'language_probability': info.language_probability,
        'duration': info.duration
    }


def match_pii_timestamps(transcription, pii_entities):
    """PII 텍스트를 오디오 타임스탬프와 매칭"""
    full_text = transcription['full_text']
    words = transcription['words']
    
    if not words:
        return []
    
    def normalize(s):
        return re.sub(r'\s+', '', s.lower())
    
    normalized_full_text = normalize(full_text)
    
    pii_timestamps = []
    
    for pii in pii_entities:
        pii_text = pii['text']
        pii_normalized = normalize(pii_text)
        
        norm_start = normalized_full_text.find(pii_normalized)
        if norm_start == -1:
            continue
        
        norm_end = norm_start + len(pii_normalized)
        
        # 원본 텍스트 위치로 역매핑
        original_start = 0
        norm_pos = 0
        for i, char in enumerate(full_text):
            if not char.isspace():
                if norm_pos == norm_start:
                    original_start = i
                    break
                norm_pos += 1
        
        original_end = original_start
        norm_pos = norm_start
        for i in range(original_start, len(full_text)):
            if not full_text[i].isspace():
                norm_pos += 1
            original_end = i + 1
            if norm_pos >= norm_end:
                break
        
        # 단어 타임스탬프 매칭
        char_count = 0
        start_time = None
        end_time = None
        
        for word_info in words:
            word_len = len(word_info['word'])
            word_start = char_count
            word_end = char_count + word_len
            
            if word_start <= original_start < word_end or word_start < original_end <= word_end:
                if start_time is None:
                    start_time = word_info['start']
                end_time = word_info['end']
            
            char_count += word_len + 1
        
        if start_time is not None and end_time is not None:
            pii_timestamps.append({
                'start': max(0, start_time - 0.1),
                'end': end_time + 0.1,
                'type': PII_NAMES.get(pii['label'], pii['label']),
                'text': pii_text
            })
    
    return pii_timestamps


def mask_audio_segments(audio_path, pii_timestamps, tone_freq=1000, fade_duration=50):
    """오디오에서 PII 구간을 톤으로 마스킹"""
    audio = AudioSegment.from_file(audio_path)
    
    masked_segments = []
    current_time = 0
    
    for pii in pii_timestamps:
        start_ms = int(pii['start'] * 1000)
        end_ms = int(pii['end'] * 1000)
        
        start_ms = max(0, start_ms)
        end_ms = min(len(audio), end_ms)
        
        if start_ms >= end_ms:
            continue
        
        # 마스킹 전 구간
        if current_time < start_ms:
            pre_segment = audio[current_time:start_ms]
            if fade_duration > 0:
                pre_segment = pre_segment.fade_out(fade_duration)
            masked_segments.append(pre_segment)
        
        # 마스킹 구간
        duration_ms = end_ms - start_ms
        tone = Sine(tone_freq).to_audio_segment(
            duration=duration_ms,
            volume=-20
        )
        
        if fade_duration > 0:
            tone = tone.fade_in(fade_duration).fade_out(fade_duration)
        
        masked_segments.append(tone)
        current_time = end_ms
    
    # 마지막 구간
    if current_time < len(audio):
        final_segment = audio[current_time:]
        if fade_duration > 0:
            final_segment = final_segment.fade_in(fade_duration)
        masked_segments.append(final_segment)
    
    final_audio = sum(masked_segments) if masked_segments else audio
    return final_audio


def save_audio(audio, output_path, audio_format="wav"):
    """오디오 저장"""
    if not output_path.endswith(f'.{audio_format}'):
        output_path = f"{output_path}.{audio_format}"
    
    audio.export(output_path, format=audio_format)
    return output_path


# ==================== 메인 처리 함수 ====================
def process_single_audio(audio_path, output_path, whisper_model, pii_detector, verbose=True):
    """단일 오디오 파일 처리"""
    try:
        # 1. STT 변환
        if verbose:
            print("    STT 변환 중...")
        
        transcription = transcribe_audio_with_words(audio_path, whisper_model)
        
        if verbose:
            print(f"    ✓ 변환 완료")
            print(f"      텍스트: {transcription['full_text'][:100]}{'...' if len(transcription['full_text']) > 100 else ''}")
            print(f"      언어: {transcription['language']} (확률: {transcription['language_probability']:.2%})")
            print(f"      단어 수: {len(transcription['words'])}개")
        
        # 2. PII 탐지
        pii_items = pii_detector.detect_pii(transcription['full_text'])
        
        if not pii_items:
            if verbose:
                print(f"    ⚠ PII 미탐지 - 원본 복사")
            import shutil
            shutil.copy2(audio_path, output_path)
            return {
                'success': True,
                'pii_detected': False,
                'output_path': output_path,
                'transcription': transcription['full_text']
            }
        
        if verbose:
            print(f"    ✓ PII 탐지: {len(pii_items)}개")
            for pii in pii_items:
                method = pii.get('method', 'unknown')
                pii_type = PII_NAMES.get(pii['label'], pii['label'])
                conf = pii.get('confidence', 0)
                print(f"      - [{pii_type}] '{pii['text']}' (방법: {method}, 신뢰도: {conf:.2f})")
        
        # 3. 타임스탬프 매칭
        pii_timestamps = match_pii_timestamps(transcription, pii_items)
        
        if not pii_timestamps:
            if verbose:
                print(f"    ⚠ 타임스탬프 매칭 실패 - 원본 복사")
            import shutil
            shutil.copy2(audio_path, output_path)
            return {
                'success': True,
                'pii_detected': True,
                'timestamp_matched': False,
                'output_path': output_path,
                'transcription': transcription['full_text']
            }
        
        if verbose:
            print(f"    ✓ 타임스탬프 매칭 완료")
            for ts in pii_timestamps:
                print(f"      - {ts['type']}: {ts['start']:.2f}s ~ {ts['end']:.2f}s")
        
        # 4. 마스킹
        if verbose:
            print(f"    마스킹 처리 중...")
        
        masked = mask_audio_segments(audio_path, pii_timestamps)
        
        # 5. 저장
        final_path = save_audio(masked, output_path)
        
        if verbose:
            print(f"    ✓ 마스킹 완료: {len(pii_timestamps)}개 구간")
        
        return {
            'success': True,
            'pii_detected': True,
            'timestamp_matched': True,
            'pii_count': len(pii_timestamps),
            'output_path': final_path,
            'transcription': transcription['full_text']
        }
        
    except Exception as e:
        print(f"    ✗ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def process_audio_folder(input_folder, output_folder, whisper_model, pii_detector, audio_extensions=['.wav', '.mp3', '.m4a', '.flac']):
    """폴더 내 모든 오디오 파일 배치 처리"""
    print("\n" + "="*80)
    print(f"폴더 배치 처리: {input_folder}")
    print("="*80)
    
    os.makedirs(output_folder, exist_ok=True)
    
    # 오디오 파일 찾기
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(glob(os.path.join(input_folder, f"**/*{ext}"), recursive=True))
    
    if not audio_files:
        print(f"⚠ 오디오 파일을 찾을 수 없습니다: {input_folder}")
        return
    
    print(f"\n총 {len(audio_files)}개 파일 발견")
    
    # 통계
    stats = {
        'total': len(audio_files),
        'success': 0,
        'pii_detected': 0,
        'failed': 0
    }
    
    # 파일별 처리
    for idx, audio_path in enumerate(audio_files, 1):
        filename = os.path.basename(audio_path)
        print(f"\n{'='*80}")
        print(f"[{idx}/{len(audio_files)}] {filename}")
        print(f"{'='*80}")
        
        output_path = os.path.join(output_folder, f"masked_{filename}")
        
        result = process_single_audio(
            audio_path=audio_path,
            output_path=output_path,
            whisper_model=whisper_model,
            pii_detector=pii_detector,
            verbose=True
        )
        
        if result.get('success', False):
            stats['success'] += 1
            if result.get('pii_detected', False):
                stats['pii_detected'] += 1
        else:
            stats['failed'] += 1
    
    # 최종 통계
    print("\n" + "="*80)
    print("배치 처리 완료!")
    print("="*80)
    print(f"총 파일: {stats['total']}개")
    print(f"성공: {stats['success']}개")
    print(f"  - PII 탐지됨: {stats['pii_detected']}개")
    print(f"  - PII 없음: {stats['success'] - stats['pii_detected']}개")
    print(f"실패: {stats['failed']}개")
    print(f"\n출력 폴더: {os.path.abspath(output_folder)}")


# ==================== 실행 ====================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("🎯 PIILOT 오디오 마스킹 시스템")
    print("="*80)
    print("지원 개인정보:")
    print("  1. 이름 (정규식)")
    print("  2. 전화번호 (정규식)")
    print("  3. 이메일 (정규식 + OCR 보정)")
    print("  4. 주소 (정규식)")
    print("  5. 주민등록번호 (정규식)")
    print("  6. IP주소 (정규식)")
    print("  7. 계좌번호 (KoELECTRA 전용)")
    print("  8. 여권번호 (KoELECTRA 전용)")
    print("="*80 + "\n")
    
    # 설정
    INPUT_FOLDER = "./generated_audio_dataset"
    OUTPUT_FOLDER = "./masked_audio_output"
    
    # Whisper 설정
    WHISPER_MODEL_SIZE = "large-v3"
    DEVICE = "auto"
    NUM_WORKERS = 4
    
    # KoELECTRA 모델 경로 (Hugging Face에서 자동 다운로드)
    KOELECTRA_MODEL_PATH = "ParkJunSeong/PIILOT_NER_Model"
    
    # 입력 폴더 확인
    if not os.path.exists(INPUT_FOLDER):
        print(f"오류: 입력 폴더 '{INPUT_FOLDER}'이(가) 존재하지 않습니다.")
    elif not os.path.isdir(INPUT_FOLDER):
        print(f"오류: '{INPUT_FOLDER}'은(는) 폴더가 아닙니다.")
    else:
        # Whisper 초기화
        print("Faster-Whisper 모델 초기화")
        print("-" * 80)
        whisper_model = initialize_whisper_model(
            model_size=WHISPER_MODEL_SIZE,
            device=DEVICE,
            num_workers=NUM_WORKERS
        )
        
        # 하이브리드 PII 탐지기 초기화
        pii_detector = HybridPIIDetector(
            model_path=KOELECTRA_MODEL_PATH,  # Hugging Face에서 자동 다운로드
            confidence_thresholds=CONFIDENCE_THRESHOLDS
        )
        
        print("✓ 모든 모델 초기화 완료\n")
        
        # 배치 처리 실행
        process_audio_folder(
            input_folder=INPUT_FOLDER,
            output_folder=OUTPUT_FOLDER,
            whisper_model=whisper_model,
            pii_detector=pii_detector
        )
