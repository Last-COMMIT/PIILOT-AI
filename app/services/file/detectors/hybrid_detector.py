import re
from typing import List, Dict, Set
from .dl_detector import KoELECTRAPIIDetector
from .regex_detector import GeneralizedRegexPIIDetector
from app.config import PII_NAMES

class HybridPIIDetector:
    """KoELECTRA NER + 정규식 기반 하이브리드 탐지"""
    def __init__(self, model_path: str, confidence_thresholds: Dict[str, float] = None):
        self.ner_detector = KoELECTRAPIIDetector(model_path, confidence_thresholds)
        self.regex_detector = GeneralizedRegexPIIDetector()

    def merge_entities(self, ner_entities: List[Dict], regex_entities: List[Dict]) -> List[Dict]:
        """DL 모델(KoELECTRA)과 정규식 결과 병합 (중복 제거)"""
        merged = []
        merged.extend(regex_entities)

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

    def _detect_names_by_context(self, text: str) -> List[Dict]:
        """문맥(이름은 OO, 성명: OO)을 이용한 이름 정규식 탐지"""
        entities = []
        # '이름은 홍길동이고', '성명: 홍길동' 등의 패턴 탐지
        # 이름은 2~4글자 한글로 가정
        patterns = [
            r'(?:이름|성명|성함)[은는이가\s]*[:\s]\s*([가-힣]{2,4})(?:[이가은는을를]|\s|[.,]|$)',
            r'Name\s*[:]\s*([가-힣a-zA-Z]{2,20})'
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                name = match.group(1)
                entities.append({
                    'text': name,
                    'label': 'p_nm',
                    'start': match.start(1),
                    'end': match.end(1),
                    'confidence': 0.85,
                    'method': 'regex_context'
                })
        return entities

    def detect_pii(self, text: str, context_names: Set[str] = None) -> List[Dict]:
        """하이브리드 PII 탐지"""
        ner_entities = self.ner_detector.detect_pii(text)
        regex_entities = self.regex_detector.detect_all(text)
        
        # 문맥 기반 이름 탐지 추가
        context_name_entities = self._detect_names_by_context(text)
        
        # 모든 엔티티 병합 (Regex + Context Regex + NER)
        all_regex_entities = regex_entities + context_name_entities
        
        merged_entities = self.merge_entities(ner_entities, all_regex_entities)
        
        merged_entities = self._extend_short_names(text, merged_entities)
        merged_entities = self._extend_address_entities(text, merged_entities)
        merged_entities = self._propagate_known_names(text, merged_entities, context_names)
        merged_entities = self._refine_entities(merged_entities)

        if regex_entities:
            print(f"정규식 추가 탐지: {len(regex_entities)}개")
            for entity in regex_entities:
                print(f"    + {PII_NAMES.get(entity['label'], entity['label'])}: '{entity['text']}'")

        return merged_entities

    def update_confidence_threshold(self, entity_type: str, new_threshold: float):
        """KoELECTRA 모델의 신뢰도 임계값 업데이트"""
        self.ner_detector.update_confidence_threshold(entity_type, new_threshold)