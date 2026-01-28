"""
오디오 마스킹 처리 (마스킹 로직만)
(기존 audio_masking.py에서 분리 - 탐지 로직은 ml/ 사용)
"""
import os
import tempfile
from typing import List, Dict
from app.core.logging import logger
from app.utils.overlap import is_overlapping
from app.ml.whisper_stt import WhisperSTT
from app.ml.pii_detectors.regex_detector import GeneralizedRegexPIIDetector
from app.ml.pii_detectors.dl_detector import KoELECTRAPIIDetector
from app.core.constants import CONFIDENCE_THRESHOLDS


class AudioMasker:
    """오디오 마스킹 처리"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.whisper = WhisperSTT()
        self.regex_detector = GeneralizedRegexPIIDetector()
        self.pii_detector = None
        self.koelectra_model_path = "ParkJunSeong/PIILOT_NER_Model"
        self._initialized = True
        logger.info("AudioMasker 초기화됨")

    def _ensure_pii_detector(self):
        if self.pii_detector is None:
            logger.info("KoELECTRA 모델 로딩 시작...")
            self.pii_detector = KoELECTRAPIIDetector(self.koelectra_model_path)
            logger.info("KoELECTRA 모델 로딩 완료")

    def transcribe_and_detect(self, audio_input) -> Dict:
        """오디오 → 텍스트 → PII 탐지"""
        self._ensure_pii_detector()

        temp_path = None
        if isinstance(audio_input, bytes):
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp.write(audio_input)
                temp_path = tmp.name
            audio_path = temp_path
        else:
            audio_path = audio_input

        try:
            segments, info = self.whisper.transcribe(audio_path)
            all_words = []
            full_text_parts = []
            for segment in segments:
                for word in segment.words:
                    all_words.append({
                        'word': word.word.strip(),
                        'start': word.start,
                        'end': word.end,
                        'probability': word.probability,
                    })
                    full_text_parts.append(word.word.strip())
            full_text = ' '.join(full_text_parts)

            ner_results = self.pii_detector.detect_pii(full_text)
            regex_results = self.regex_detector.detect_all(full_text)

            merged = regex_results + ner_results
            merged.sort(key=lambda x: x['start'])
            unique_pii = []
            for item in merged:
                if not any(is_overlapping(item, existing) for existing in unique_pii):
                    unique_pii.append(item)

            pii_with_timestamps = self._match_timestamps(full_text, all_words, unique_pii)

            return {
                "text": full_text,
                "detected_items": pii_with_timestamps,
            }
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    def mask_audio(self, audio_input, detected_items: List[Dict]) -> bytes:
        """오디오 마스킹"""
        from pydub import AudioSegment
        from pydub.generators import Sine

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

            detected_items.sort(key=lambda x: x.get('start_time', 0))
            MASK_VOLUME = -25
            SAFE_BUFFER_MS = 25

            for item in detected_items:
                original_start_ms = int(item.get('start_time', 0) * 1000)
                original_end_ms = int(item.get('end_time', 0) * 1000)
                pii_text = item.get('value', '')

                if original_start_ms >= original_end_ms:
                    continue

                start_ms = max(0, original_start_ms - SAFE_BUFFER_MS)
                end_ms = min(original_end_ms + SAFE_BUFFER_MS, audio_length_ms)
                if end_ms - start_ms < 50:
                    start_ms = original_start_ms
                    end_ms = original_end_ms
                start_ms = max(0, min(start_ms, audio_length_ms))
                end_ms = max(start_ms + 50, min(end_ms, audio_length_ms))

                if start_ms >= end_ms:
                    continue

                if current_time < start_ms:
                    masked_segments.append(audio[current_time:start_ms])

                duration = end_ms - start_ms
                tone = Sine(1000).to_audio_segment(duration=duration, volume=MASK_VOLUME)
                masked_segments.append(tone)
                current_time = end_ms

            if current_time < len(audio):
                masked_segments.append(audio[current_time:])

            final_audio = sum(masked_segments)

            from pathlib import Path
            import datetime
            project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
            test_audio_dir = project_root / "test_audio"
            test_audio_dir.mkdir(exist_ok=True)
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = test_audio_dir / f"masked_audio_{timestamp}.mp3"

            final_audio.export(str(output_path), format="mp3")
            logger.info(f"마스킹된 오디오가 저장되었습니다: {output_path}")

            with open(output_path, "rb") as f:
                masked_bytes = f.read()
            return masked_bytes

        finally:
            if is_temp and temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    def _match_timestamps(self, full_text, words, pii_entities):
        """텍스트 위치를 오디오 타임스탬프로 변환"""
        results = []
        for pii in pii_entities:
            pii_text = pii.get('text', '').strip()
            if not pii_text:
                continue

            matched_words = []
            # 방법 1: 정확히 일치
            for word in words:
                word_text = word['word'].strip()
                if word_text == pii_text:
                    matched_words.append({
                        'word': word_text, 'start': word['start'],
                        'end': word['end'], 'match_type': 'exact',
                    })
                    break

            # 방법 2: 부분 매칭
            if not matched_words:
                for word in words:
                    word_text = word['word'].strip()
                    if pii_text in word_text:
                        matched_words.append({
                            'word': word_text, 'start': word['start'],
                            'end': word['end'], 'match_type': 'partial',
                            'pii_start_in_word': word_text.find(pii_text),
                            'pii_end_in_word': word_text.find(pii_text) + len(pii_text),
                        })
                        break

            # 방법 3: 여러 단어
            if not matched_words:
                pii_no_space = pii_text.replace(' ', '')
                consecutive_words = []
                combined_text = ""
                for i, word in enumerate(words):
                    word_text = word['word'].strip()
                    combined_text += word_text
                    consecutive_words.append({
                        'word': word_text, 'start': word['start'],
                        'end': word['end'], 'index': i,
                    })
                    if pii_no_space in combined_text.replace(' ', ''):
                        matched_words = consecutive_words
                        break
                    if len(consecutive_words) > 10:
                        break

            if not matched_words:
                logger.warning(f"PII '{pii_text}'에 해당하는 단어를 찾을 수 없습니다.")
                continue

            # 타임스탬프 계산
            if len(matched_words) == 1:
                word = matched_words[0]
                if word.get('match_type') == 'exact':
                    start_time = word['start']
                    end_time = word['end']
                else:
                    pii_start_in_word = word.get('pii_start_in_word', 0)
                    pii_end_in_word = word.get('pii_end_in_word', len(word['word']))
                    word_total_chars = len(word['word'])
                    word_duration = word['end'] - word['start']
                    ratio_start = pii_start_in_word / word_total_chars if word_total_chars > 0 else 0
                    ratio_end = pii_end_in_word / word_total_chars if word_total_chars > 0 else 1.0
                    start_time = word['start'] + (word_duration * ratio_start)
                    end_time = word['start'] + (word_duration * ratio_end)
            else:
                start_time = matched_words[0]['start']
                end_time = matched_words[-1]['end']

            if start_time is not None and end_time is not None and start_time < end_time:
                results.append({
                    "type": pii['label'],
                    "value": pii_text,
                    "start_time": start_time,
                    "end_time": end_time,
                    "confidence": pii['confidence'],
                })

        return results
