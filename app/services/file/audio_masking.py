"""
오디오 마스킹 서비스 모듈
"""
import sys
import subprocess
import os
import re
import numpy as np
from typing import List, Dict, Set
from glob import glob
import soundfile as sf
import torch
import torch.nn.functional as F
import base64
import tempfile
import uuid

# 필수 라이브러리 임포트 (설치 로직은 docker/deployment 단계에서 처리하는 것이 좋지만, 기존 유지)
try:
    from faster_whisper import WhisperModel
    from pydub import AudioSegment
    from pydub.generators import Sine
    from transformers import AutoTokenizer, AutoModelForTokenClassification
except ImportError:
    pass  # 실제 환경에서는 requirements.txt로 관리 권장

from app.utils.logger import logger

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

# ==================== 정규식 PII 탐지기 ====================
class EnhancedRegexPIIDetector:
    """향상된 정규식 PII 탐지기"""
    def __init__(self):
        pass

    def detect_phones(self, text: str) -> List[Dict]:
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
        entities = []
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
        return entities

    def detect_addresses(self, text: str) -> List[Dict]:
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
        if len(address) < 8: return False
        has_admin = any(kw in address for kw in ['도', '시', '군', '구'])
        has_location = any(kw in address for kw in ['로', '길', '동'])
        return has_admin and has_location

    def detect_rrn(self, text: str) -> List[Dict]:
        entities = []
        patterns = [r'\d{6}-[1-4]\d{6}', r'(?<!\d)\d{13}(?!\d)']
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                rrn = match.group()
                digits = re.sub(r'\D', '', rrn)
                if len(digits) == 13 and digits[6] in '1234':
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
        entities = []
        pattern = r'(?<!\d)\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?!\d)'
        for match in re.finditer(pattern, text):
            try:
                ip = match.group()
                if all(0 <= int(x) <= 255 for x in ip.split('.')):
                    entities.append({
                        'text': ip,
                        'label': 'p_ip',
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': 1.0,
                        'method': 'regex'
                    })
            except: pass
        return entities

    def detect_names(self, text: str) -> List[Dict]:
        # 간단한 이름 탐지 (실제로는 LLM/NER이 더 정확함)
        return []

    def detect_all(self, text: str) -> List[Dict]:
        all_entities = []
        all_entities.extend(self.detect_rrn(text))
        all_entities.extend(self.detect_phones(text))
        all_entities.extend(self.detect_emails(text))
        all_entities.extend(self.detect_addresses(text))
        all_entities.extend(self.detect_ip(text))
        
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
    def __init__(self, model_path: str, confidence_thresholds: Dict[str, float] = None):
        if not confidence_thresholds:
            confidence_thresholds = CONFIDENCE_THRESHOLDS
        self.confidence_thresholds = confidence_thresholds
        self.model = None
        
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForTokenClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            self.id2label = self.model.config.id2label
            logger.info(f"KoELECTRA 모델 로드 완료: {model_path}")
        except Exception as e:
            logger.error(f"KoELECTRA 모델 로드 실패: {e}")
            self.model = None

    def detect_pii(self, text: str) -> List[Dict]:
        if not self.model or not text.strip():
            return []

        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512, return_offsets_mapping=True
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = F.softmax(outputs.logits, dim=-1)[0]
            predictions = torch.argmax(probabilities, dim=-1)

        predictions = predictions.cpu().numpy()
        offsets = inputs["offset_mapping"][0].cpu().numpy()
        
        entities = []
        current_entity = None
        
        for idx, (pred, offset) in enumerate(zip(predictions, offsets)):
            if offset[0] == 0 and offset[1] == 0: continue
            
            pred_label = self.id2label[pred]
            confidence = probabilities[idx][pred].item()
            
            if pred_label == 'O':
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
                continue
                
            if pred_label.startswith('B-'):
                if current_entity: entities.append(current_entity)
                current_entity = {
                    'label': pred_label[2:],
                    'start': offset[0],
                    'end': offset[1],
                    'confidences': [confidence]
                }
            elif pred_label.startswith('I-') and current_entity and current_entity['label'] == pred_label[2:]:
                current_entity['end'] = offset[1]
                current_entity['confidences'].append(confidence)
        
        if current_entity: entities.append(current_entity)
        
        final_results = []
        for ent in entities:
            avg_conf = sum(ent['confidences']) / len(ent['confidences'])
            if avg_conf >= self.confidence_thresholds.get(ent['label'], 0.5):
                final_results.append({
                    'text': text[ent['start']:ent['end']],
                    'label': ent['label'],
                    'start': ent['start'],
                    'end': ent['end'],
                    'confidence': avg_conf
                })
        return final_results


# ==================== 통합 오디오 서비스 ====================
class AudioPIIService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AudioPIIService, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if self.initialized: return
        
        # 모델 설정
        self.whisper_model_size = "large-v3"
        self.koelectra_model_path = "ParkJunSeong/PIILOT_NER_Model"
        
        self.whisper_model = None
        self.pii_detector = None
        self.regex_detector = EnhancedRegexPIIDetector()
        
        self.initialized = True
        logger.info("AudioPIIService 초기화됨 (모델은 아직 로드되지 않음)")

    def initialize_models(self):
        """무거운 모델 지연 로딩"""
        if self.whisper_model is None:
            logger.info("Whisper 모델 로딩 시작...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            self.whisper_model = WhisperModel(
                self.whisper_model_size, device=device, compute_type=compute_type
            )
            logger.info("Whisper 모델 로딩 완료")
            
        if self.pii_detector is None:
            logger.info("KoELECTRA 모델 로딩 시작...")
            self.pii_detector = KoELECTRAPIIDetector(self.koelectra_model_path)
            logger.info("KoELECTRA 모델 로딩 완료")

    def transcribe_and_detect(self, audio_input) -> Dict:
        """오디오 → 텍스트 → PII 탐지"""
        self.initialize_models()
        
        # 임시 파일 처리 (bytes or path)
        temp_path = None
        if isinstance(audio_input, bytes):
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp.write(audio_input)
                temp_path = tmp.name
            audio_path = temp_path
        else:
            audio_path = audio_input

        try:
            # 1. STT
            segments, info = self.whisper_model.transcribe(
                audio_path, language='ko', word_timestamps=True, vad_filter=True
            )
            
            all_words = []
            full_text_parts = []
            
            for segment in segments:
                for word in segment.words:
                    all_words.append({
                        'word': word.word.strip(),
                        'start': word.start,
                        'end': word.end,
                        'probability': word.probability
                    })
                    full_text_parts.append(word.word.strip())
            
            full_text = ' '.join(full_text_parts)
            
            # 2. PII 탐지 (NER + Regex)
            ner_results = self.pii_detector.detect_pii(full_text)
            regex_results = self.regex_detector.detect_all(full_text)
            
            # 병합 로직 (간소화)
            merged = regex_results + ner_results
            # 중복 제거 (위치 기반)
            merged.sort(key=lambda x: x['start'])
            unique_pii = []
            for item in merged:
                if not any(self._is_overlapping(item, existing) for existing in unique_pii):
                    unique_pii.append(item)

            # 3. 타임스탬프 매칭
            pii_with_timestamps = self._match_timestamps(full_text, all_words, unique_pii)
            
            return {
                "text": full_text,
                "detected_items": pii_with_timestamps
            }
            
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    def mask_audio(self, audio_input, detected_items: List[Dict]) -> bytes:
        """오디오 마스킹"""
        # 임시 파일 처리
        temp_path = None
        is_temp = False
        
        if isinstance(audio_input, bytes):
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp.write(audio_input)
                temp_path = tmp.name
                is_temp = True
            audio_path = temp_path
        else:
            audio_path = audio_input

        try:
            audio = AudioSegment.from_file(audio_path)
            masked_segments = []
            current_time = 0
            
            # 시간순 정렬
            detected_items.sort(key=lambda x: x.get('start_time', 0))
            
            for item in detected_items:
                start_ms = int(item.get('start_time', 0) * 1000)
                end_ms = int(item.get('end_time', 0) * 1000)
                
                if start_ms >= end_ms: continue
                
                # 마스킹 전 구간
                if current_time < start_ms:
                    masked_segments.append(audio[current_time:start_ms])
                
                # 마스킹 구간 (삐- 소리)
                duration = end_ms - start_ms
                tone = Sine(1000).to_audio_segment(duration=duration, volume=-10)
                masked_segments.append(tone)
                
                current_time = end_ms
                
            if current_time < len(audio):
                masked_segments.append(audio[current_time:])
                
            final_audio = sum(masked_segments)
            
            # Bytes로 반환
            out_io = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            out_io.close()
            final_audio.export(out_io.name, format="mp3")
            
            with open(out_io.name, "rb") as f:
                masked_bytes = f.read()
            
            os.remove(out_io.name)
            return masked_bytes
            
        finally:
            if is_temp and temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    def _is_overlapping(self, entity1, entity2):
        return not (entity1['end'] <= entity2['start'] or entity2['end'] <= entity1['start'])

    def _match_timestamps(self, full_text, words, pii_entities):
        """텍스트 위치를 오디오 타임스탬프로 변환"""
        results = []
        # 공백 제거 텍스트 매핑 로직은 복잡하므로, 간소화된 버전 또는 기존 로직 차용 필요
        # 여기서는 단순화를 위해 문자 인덱스 비율로 추정하거나
        # 기존 audio_masking.py의 match_pii_timestamps 로직을 그대로 가져오는 것이 좋음
        
        # (기존 match_pii_timestamps 로직의 간소화 버전)
        for pii in pii_entities:
            start_char = pii['start']
            end_char = pii['end']
            
            # 해당 문자 범위에 걸치는 단어들의 시간 범위 찾기
            # (정확한 매칭을 위해서는 원본의 복잡한 로직이 필요함)
            
            # 간단한 근사치 (실제로는 더 정교해야 함)
            current_char = 0
            start_time = None
            end_time = None
            
            for word in words:
                word_len = len(word['word'])
                # 단어 찾기 (공백 등 고려 필요)
                word_start_char = full_text.find(word['word'], current_char)
                if word_start_char == -1: continue
                
                word_end_char = word_start_char + word_len
                current_char = word_end_char
                
                if word_start_char < end_char and word_end_char > start_char:
                    if start_time is None: start_time = word['start']
                    end_time = word['end']
            
            if start_time is not None and end_time is not None:
                results.append({
                    "type": pii['label'],
                    "value": pii['text'],
                    "start_time": start_time,
                    "end_time": end_time,
                    "confidence": pii['confidence']
                })
        return results

# 싱글톤 인스턴스 (Import용)
audio_pii_service = AudioPIIService()
