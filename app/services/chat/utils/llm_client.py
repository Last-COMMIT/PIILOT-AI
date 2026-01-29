"""
LLM 클라이언트 유틸리티 (KT 믿음 모델)
기존 regulation_search.py의 LLM 초기화 패턴 재사용
"""
from langchain_huggingface import ChatHuggingFace
from app.core.model_manager import ModelManager
from app.core.logging import logger
import torch
import time

# 싱글톤으로 LLM 인스턴스 관리 (한 번만 로드)
_llm_instance = None
_llm_config = {}


def get_llm(temperature: float = 0.001, max_new_tokens: int = 512) -> ChatHuggingFace:
    """
    KT 믿음 LLM 인스턴스 반환 (싱글톤)
    
    Args:
        temperature: 생성 온도 (기본값: 0.001)
        max_new_tokens: 최대 생성 토큰 수 (기본값: 512)
    
    Returns:
        ChatHuggingFace 인스턴스
    """
    global _llm_instance, _llm_config
    
    # 설정이 변경되면 새로 생성 (temperature나 max_new_tokens가 다를 수 있음)
    config_key = (temperature, max_new_tokens)
    
    if _llm_instance is None or _llm_config.get("config") != config_key:
        model_name = ModelManager.HUGGINGFACE_MODELS["llm"]["name"]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 캐시 디렉토리 가져오기 (기존 models 디렉토리 사용)
        ModelManager.setup_cache_dir()  # 캐시 디렉토리 초기화 보장
        cache_dir = ModelManager.get_cache_dir()
        
        logger.info(f"LLM 모델 로딩 시작: {model_name}, device={device}, cache_dir={cache_dir}, temperature={temperature}, max_new_tokens={max_new_tokens}")
        llm_start_time = time.time()
        
        try:
            _llm_instance = ChatHuggingFace.from_model_id(
                model_id=model_name,
                task="text-generation",
                device_map=device,
                model_kwargs={
                    "dtype": torch.bfloat16,
                    "trust_remote_code": True,
                    "cache_dir": cache_dir,  # 명시적으로 캐시 디렉토리 전달
                },
                tokenizer_kwargs={
                    "trust_remote_code": True,
                    "cache_dir": cache_dir,  # 명시적으로 캐시 디렉토리 전달
                },
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
            _llm_config["config"] = config_key
            llm_load_time = time.time() - llm_start_time
            logger.info(f"✓ LLM 모델 로딩 완료 (소요 시간: {llm_load_time:.2f}초)")
        except Exception as e:
            logger.error(f"LLM 모델 로딩 실패: {str(e)}", exc_info=True)
            raise
    
    return _llm_instance


def get_classification_llm() -> ChatHuggingFace:
    """분류용 LLM (낮은 temperature, 짧은 응답)"""
    return get_llm(temperature=0.001, max_new_tokens=100)


def get_answer_llm() -> ChatHuggingFace:
    """답변 생성용 LLM (기본 설정)"""
    return get_llm(temperature=0.3, max_new_tokens=512)


def get_evaluation_llm() -> ChatHuggingFace:
    """평가용 LLM (관련성, 환각 검증 - 낮은 temperature)"""
    return get_llm(temperature=0.001, max_new_tokens=200)
