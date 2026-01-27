from typing import List, Dict, Set, Optional
import re
from ....models.personal_info import PII_NAMES
from .regex import GeneralizedRegexPIIDetector
from .dl import KoELECTRAPIIDetector

class HybridPIIDetector:
    """KoELECTRA NER + 정규식 기반 하이브리드 탐지"""
    def __init__(self, model_path: str, confidence_thresholds: Dict[str, float] = None):
        self.ner_detector = KoELECTRAPIIDetector(model_path, confidence_thresholds)
        self.regex_detector = GeneralizedRegexPIIDetector()

    def merge_entities(self, ner_entities: List[Dict], regex_entities: List[Dict]) -> List[Dict]:
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
        start1, end1 = entity1['start'], entity1['end']
        start2, end2 = entity2['start'], entity2['end']
        return not (end1 <= start2 or end2 <= start1)

    def _extend_address_entities(self, text: str, entities: List[Dict]) -> List[Dict]:
        extended_entities = []
        extension_pattern = r'^[\s,]*((?:[가-힣a-zA-Z0-9]+(?:타운|빌라|맨션|아파트|오피스텔)|[가-힣0-9]+(?:동|호|층)|[\d-]+(?:동|호|층)?))'

        for entity in entities:
            if entity['label'] != 'p_add':
                extended_entities.append(entity)
                continue
            current_end = entity['end']
            while True:
                remaining_text = text[current_end:]
                match = re.match(extension_pattern, remaining_text)
                if match:
                    matched_str = match.group(0) 
                    new_end = current_end + len(matched_str)
                    entity['end'] = new_end
                    entity['text'] = text[entity['start']:new_end]
                    current_end = new_end
                else:
                    break
            extended_entities.append(entity)
        return extended_entities

    def _extend_short_names(self, text: str, entities: List[Dict]) -> List[Dict]:
        STOP_CHARS = {'이', '가', '은', '는', '을', '를', '의', '에', '와', '과', '로', '도', '만', '씨', '님', '군', '양', '과', '장'}
        for entity in entities:
            if entity['label'] != 'p_nm': continue
            name_text = entity['text'].strip()
            if len(name_text) == 2:
                current_end = entity['end']
                if current_end < len(text):
                    next_char = text[current_end]
                    if '가' <= next_char <= '힣' and next_char not in STOP_CHARS:
                        entity['end'] += 1
                        entity['text'] = text[entity['start']:entity['end']]
        return entities

    def _propagate_known_names(self, text: str, entities: List[Dict], context_names: Set[str] = None) -> List[Dict]:
        known_names = set()
        if context_names: known_names.update(context_names)
        for entity in entities:
            if entity['label'] == 'p_nm': known_names.add(entity['text'])
        
        if not known_names: return entities
        search_terms = {name.strip() for name in known_names if len(name.strip()) >= 2}
        if not search_terms: return entities
            
        propagated = []
        def is_overlapping(start, end, existing):
            for e in existing:
                if max(start, e['start']) < min(end, e['end']): return True
            return False

        for term in search_terms:
            for match in re.finditer(re.escape(term), text):
                start, end = match.span()
                if not is_overlapping(start, end, entities) and not is_overlapping(start, end, propagated):
                    propagated.append({
                        'start': start,
                        'end': end,
                        'text': term,
                        'label': 'p_nm',
                        'confidence': 0.90
                    })
        return entities + propagated

    def _refine_entities(self, entities: List[Dict]) -> List[Dict]:
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
            
            for pattern in label_patterns:
                match = re.match(pattern, text, re.IGNORECASE)
                if match:
                    trim_len = len(match.group(0))
                    entity['start'] += trim_len
                    entity['text'] = text[trim_len:]
                    break
            
            if label == 'p_add':
                pass
            
            if not entity['text'].strip(): continue
            if label == 'p_nm' and len(entity['text'].strip()) <= 1: continue
            
            refined.append(entity)
        return refined

    def detect_pii(self, text: str, context_names: Set[str] = None) -> List[Dict]:
        ner_entities = self.ner_detector.detect_pii(text)
        regex_entities = self.regex_detector.detect_all(text)
        merged = self.merge_entities(ner_entities, regex_entities)
        merged = self._extend_short_names(text, merged)
        merged = self._extend_address_entities(text, merged)
        merged = self._propagate_known_names(text, merged, context_names)
        merged = self._refine_entities(merged)
        
        if regex_entities:
            print(f"정규식 추가 탐지: {len(regex_entities)}개")
            
        return merged

    def update_confidence_threshold(self, entity_type: str, new_threshold: float):
        self.ner_detector.update_confidence_threshold(entity_type, new_threshold)