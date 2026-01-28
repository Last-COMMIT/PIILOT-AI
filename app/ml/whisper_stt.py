"""
Whisper STT 모델 래퍼 (audio_masking.py에서 분리)
"""
import torch
from app.core.logging import logger


class WhisperSTT:
    """Whisper 모델을 이용한 음성→텍스트 변환"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.model = None
        self.model_size = "large-v3"
        self._initialized = True
        logger.info("WhisperSTT 초기화됨 (모델은 아직 로드되지 않음)")

    def _load_model(self):
        """무거운 모델 지연 로딩"""
        if self.model is not None:
            return

        from faster_whisper import WhisperModel

        logger.info("Whisper 모델 로딩 시작...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        self.model = WhisperModel(
            self.model_size, device=device, compute_type=compute_type
        )
        logger.info("Whisper 모델 로딩 완료")

    def transcribe(self, audio_path: str):
        """
        오디오 파일을 텍스트로 변환

        Returns:
            (segments, info) 튜플
        """
        self._load_model()
        segments, info = self.model.transcribe(
            audio_path,
            language='ko',
            word_timestamps=True,
            vad_filter=True,
            no_speech_threshold=0.4,
            log_prob_threshold=-1.0,
            beam_size=5,
            condition_on_previous_text=True,
        )
        return segments, info
