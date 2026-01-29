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
from app.core.config import AUDIO_OUTPUT_DIR

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

# ==================== 정규식/DL PII 탐지기 ====================
# 중복 클래스 제거됨 - app.ml.pii_detectors에서 통합된 클래스 사용
# - EnhancedRegexPIIDetector → GeneralizedRegexPIIDetector
# - KoELECTRAPIIDetector → app.ml.pii_detectors.dl_detector.KoELECTRAPIIDetector


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
        # 통합된 정규식 탐지기 사용
        from app.ml.pii_detectors.regex_detector import GeneralizedRegexPIIDetector
        self.regex_detector = GeneralizedRegexPIIDetector()
        
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
            # 통합된 KoELECTRA 탐지기 사용
            from app.ml.pii_detectors.dl_detector import KoELECTRAPIIDetector
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
            # 1. STT (타임스탬프 정확도를 위해 VAD 필터 비활성화)
            segments, info = self.whisper_model.transcribe(
                audio_path, 
                language='ko', 
                word_timestamps=True,
                vad_filter=True,  # 무음 구간 필터링
                no_speech_threshold=0.4,  # 기본값 0.6보다 낮게 (작은 소리에 더 민감)
                log_prob_threshold=-1.0,  # 기본값 -1.0보다 낮게 (낮은 확률도 인식)
                beam_size=5,  # 기본값 유지 (정확도와 속도 균형)
                condition_on_previous_text=True,  # 이전 텍스트 조건 사용
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
            audio_length_ms = len(audio)
            masked_segments = []
            current_time = 0
            
            # 시간순 정렬
            detected_items.sort(key=lambda x: x.get('start_time', 0))
            
            # 마스킹 설정
            MASK_VOLUME = -25  # 삐- 소리 볼륨 (기본 대화 수준보다 낮게)
            TIME_BUFFER_MS = 50  # 타임스탬프 보정 버퍼 (0.05초)
            
            for item in detected_items:
                # 원본 타임스탬프
                original_start_ms = int(item.get('start_time', 0) * 1000)
                original_end_ms = int(item.get('end_time', 0) * 1000)
                pii_text = item.get('value', '')
                
                if original_start_ms >= original_end_ms:
                    logger.warning(f"PII '{pii_text}'의 타임스탬프가 유효하지 않음: start={original_start_ms}ms, end={original_end_ms}ms")
                    continue
                
                # 타임스탬프는 _match_timestamps에서 계산되었지만, 
                # 반올림 오차와 경계 오차를 보정하기 위해 안전 버퍼 적용
                SAFE_BUFFER_MS = 25  # 25ms 안전 버퍼 (시작은 약간 앞서, 끝은 약간 늦게)
                start_ms = max(0, original_start_ms - SAFE_BUFFER_MS)
                end_ms = min(original_end_ms + SAFE_BUFFER_MS, audio_length_ms)
                
                # 최소 duration 보장
                if end_ms - start_ms < 50:  # 50ms 미만이면
                    start_ms = original_start_ms
                    end_ms = original_end_ms
                
                # 오디오 길이 제한 (범위를 벗어나지 않도록)
                start_ms = max(0, min(start_ms, audio_length_ms))
                end_ms = max(start_ms + 50, min(end_ms, audio_length_ms))  # 최소 50ms
                
                # 디버깅 로그
                original_duration = (original_end_ms - original_start_ms) / 1000.0
                final_duration = (end_ms - start_ms) / 1000.0
                logger.debug(
                    f"마스킹 구간: PII='{pii_text}', "
                    f"원본={original_start_ms/1000:.3f}s~{original_end_ms/1000:.3f}s (duration={original_duration:.3f}s), "
                    f"최종={start_ms/1000:.3f}s~{end_ms/1000:.3f}s (duration={final_duration:.3f}s)"
                )
                
                if start_ms >= end_ms:
                    logger.warning(f"PII '{pii_text}'의 최종 타임스탬프가 유효하지 않음: start={start_ms}ms, end={end_ms}ms")
                    continue
                
                # 마스킹 전 구간
                if current_time < start_ms:
                    masked_segments.append(audio[current_time:start_ms])
                
                # 마스킹 구간 (삐- 소리)
                duration = end_ms - start_ms
                tone = Sine(1000).to_audio_segment(duration=duration, volume=MASK_VOLUME)
                masked_segments.append(tone)
                
                current_time = end_ms
                
            if current_time < len(audio):
                masked_segments.append(audio[current_time:])
                
            final_audio = sum(masked_segments)
            
            # AUDIO_OUTPUT_DIR에 저장
            from pathlib import Path
            import datetime
            
            # 타임스탬프 기반 파일명 생성
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = Path(AUDIO_OUTPUT_DIR) / f"masked_audio_{timestamp}.mp3"
            
            # 파일로 저장
            final_audio.export(str(output_path), format="mp3")
            logger.info(f"마스킹된 오디오가 저장되었습니다: {output_path}")
            
            # Bytes로도 반환 (API 호환성 유지)
            with open(output_path, "rb") as f:
                masked_bytes = f.read()
            
            return masked_bytes
            
        finally:
            if is_temp and temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    def _is_overlapping(self, entity1, entity2):
        return not (entity1['end'] <= entity2['start'] or entity2['end'] <= entity1['start'])

    def _match_timestamps(self, full_text, words, pii_entities):
        """텍스트 위치를 오디오 타임스탬프로 변환 (텍스트 직접 매칭 방식)"""
        results = []
        
        for pii in pii_entities:
            pii_text = pii.get('text', '').strip()
            if not pii_text:
                continue
            
            # PII 텍스트를 단어 리스트에서 직접 찾기
            matched_words = []
            
            # 방법 1: PII 텍스트가 단어 전체와 정확히 일치하는 경우
            for word in words:
                word_text = word['word'].strip()
                if word_text == pii_text:
                    # 정확히 일치하는 단어 발견
                    matched_words.append({
                        'word': word_text,
                        'start': word['start'],
                        'end': word['end'],
                        'match_type': 'exact'
                    })
                    break
            
            # 방법 2: PII 텍스트가 단어에 포함되어 있는 경우 (부분 매칭)
            if not matched_words:
                for word in words:
                    word_text = word['word'].strip()
                    if pii_text in word_text:
                        # PII 텍스트가 단어에 포함되어 있음
                        matched_words.append({
                            'word': word_text,
                            'start': word['start'],
                            'end': word['end'],
                            'match_type': 'partial',
                            'pii_start_in_word': word_text.find(pii_text),
                            'pii_end_in_word': word_text.find(pii_text) + len(pii_text)
                        })
                        break
            
            # 방법 3: PII 텍스트가 여러 단어에 걸치는 경우
            if not matched_words:
                # PII 텍스트를 공백 제거하여 연속된 단어들에서 찾기
                pii_no_space = pii_text.replace(' ', '')
                consecutive_words = []
                combined_text = ""
                
                for i, word in enumerate(words):
                    word_text = word['word'].strip()
                    combined_text += word_text
                    
                    # PII 텍스트가 현재까지의 조합에 포함되는지 확인
                    if pii_no_space in combined_text.replace(' ', ''):
                        consecutive_words.append({
                            'word': word_text,
                            'start': word['start'],
                            'end': word['end'],
                            'index': i
                        })
                        
                        # PII 텍스트의 시작과 끝 위치 계산
                        combined_no_space = combined_text.replace(' ', '')
                        pii_start_in_combined = combined_no_space.find(pii_no_space)
                        pii_end_in_combined = pii_start_in_combined + len(pii_no_space)
                        
                        # 각 단어에서의 위치 계산
                        char_pos = 0
                        for cw in consecutive_words:
                            word_len = len(cw['word'])
                            word_start_in_combined = char_pos
                            word_end_in_combined = char_pos + word_len
                            
                            # PII가 이 단어에 걸치는지 확인
                            if word_start_in_combined < pii_end_in_combined and word_end_in_combined > pii_start_in_combined:
                                cw['pii_start_in_word'] = max(0, pii_start_in_combined - word_start_in_combined)
                                cw['pii_end_in_word'] = min(word_len, pii_end_in_combined - word_start_in_combined)
                            
                            char_pos += word_len
                        
                        matched_words = consecutive_words
                        break
                    
                    # 너무 많은 단어를 조합해도 안 되면 중단
                    if len(consecutive_words) > 10:
                        break
            
            if not matched_words:
                logger.warning(f"PII '{pii_text}'에 해당하는 단어를 찾을 수 없습니다.")
                continue
            
            # 타임스탬프 계산
            if len(matched_words) == 1:
                word = matched_words[0]
                if word.get('match_type') == 'exact':
                    # 정확히 일치: 단어 전체 시간 사용
                    start_time = word['start']
                    end_time = word['end']
                    logger.debug(f"PII '{pii_text}'가 단어 전체와 정확히 일치: {start_time:.3f}s~{end_time:.3f}s")
                else:
                    # 부분 매칭: 비율 계산
                    pii_start_in_word = word.get('pii_start_in_word', 0)
                    pii_end_in_word = word.get('pii_end_in_word', len(word['word']))
                    word_total_chars = len(word['word'])
                    word_duration = word['end'] - word['start']
                    
                    ratio_start = pii_start_in_word / word_total_chars if word_total_chars > 0 else 0
                    ratio_end = pii_end_in_word / word_total_chars if word_total_chars > 0 else 1.0
                    
                    start_time = word['start'] + (word_duration * ratio_start)
                    end_time = word['start'] + (word_duration * ratio_end)
                    logger.debug(f"PII '{pii_text}'가 단어 일부에 포함: {start_time:.3f}s~{end_time:.3f}s")
            else:
                # 여러 단어에 걸치는 경우
                first_word = matched_words[0]
                last_word = matched_words[-1]
                
                # 첫 번째 단어에서 시작 시간 계산
                if 'pii_start_in_word' in first_word:
                    pii_start_in_word = first_word['pii_start_in_word']
                    word_total_chars = len(first_word['word'])
                    word_duration = first_word['end'] - first_word['start']
                    ratio_start = pii_start_in_word / word_total_chars if word_total_chars > 0 else 0
                    start_time = first_word['start'] + (word_duration * ratio_start)
                else:
                    start_time = first_word['start']
                
                # 마지막 단어에서 끝 시간 계산
                if 'pii_end_in_word' in last_word:
                    pii_end_in_word = last_word['pii_end_in_word']
                    word_total_chars = len(last_word['word'])
                    word_duration = last_word['end'] - last_word['start']
                    ratio_end = pii_end_in_word / word_total_chars if word_total_chars > 0 else 1.0
                    end_time = last_word['start'] + (word_duration * ratio_end)
                else:
                    end_time = last_word['end']
                
                logger.debug(f"PII '{pii_text}'가 여러 단어에 걸침 ({len(matched_words)}개): {start_time:.3f}s~{end_time:.3f}s")
            
            # 디버깅 로그
            logger.debug(
                f"PII 타임스탬프 매칭: text='{pii_text}', "
                f"matched_words={len(matched_words)}, "
                f"start_time={start_time:.3f}s, end_time={end_time:.3f}s, "
                f"duration={end_time-start_time:.3f}s"
            )
            
            if start_time is not None and end_time is not None and start_time < end_time:
                results.append({
                    "type": pii['label'],
                    "value": pii_text,
                    "start_time": start_time,
                    "end_time": end_time,
                    "confidence": pii['confidence']
                })
            else:
                logger.warning(f"PII '{pii_text}'의 타임스탬프 계산 실패: start={start_time}, end={end_time}")
        
        return results

# 싱글톤 인스턴스 (Import용)
audio_pii_service = AudioPIIService()
