"""
LLM 클라이언트 유틸리티 (KT 믿음 모델 또는 OpenAI GPT)
환경 변수 LANGGRAPH_LLM_TYPE으로 선택 가능 ("huggingface" 또는 "openai")
"""
from typing import Union
from langchain_huggingface import ChatHuggingFace
from langchain_openai import ChatOpenAI
from app.core.model_manager import ModelManager
from app.core.config import settings
from app.services.chat.config import LLM_TYPE, OPENAI_MODEL
from app.core.logging import logger
import torch
import time

# 싱글톤으로 LLM 인스턴스 관리 (한 번만 로드)
_llm_instance = None
_llm_config = {}


def get_llm(temperature: float = 0.001, max_new_tokens: int = 512) -> Union[ChatHuggingFace, ChatOpenAI]:
    """
    LLM 인스턴스 반환 (싱글톤)
    LLM_TYPE 설정에 따라 HuggingFace 또는 OpenAI GPT 사용
    
    Args:
        temperature: 생성 온도 (기본값: 0.001)
        max_new_tokens: 최대 생성 토큰 수 (기본값: 512, OpenAI의 경우 max_tokens로 변환)
    
    Returns:
        ChatHuggingFace 또는 ChatOpenAI 인스턴스
    """
    global _llm_instance, _llm_config
    
    # 설정이 변경되면 새로 생성 (LLM 타입, temperature, max_new_tokens가 다를 수 있음)
    config_key = (LLM_TYPE, temperature, max_new_tokens)
    
    if _llm_instance is None or _llm_config.get("config") != config_key:
        if LLM_TYPE == "openai":
            # OpenAI GPT 사용
            if not settings.OPENAI_API_KEY:
                logger.warning("OPENAI_API_KEY가 설정되지 않았습니다. HuggingFace 모델로 폴백합니다.")
                # HuggingFace로 폴백
                return _get_huggingface_llm(temperature, max_new_tokens)
            
            logger.info(f"OpenAI LLM 초기화 시작: {OPENAI_MODEL}, temperature={temperature}, max_tokens={max_new_tokens}")
            llm_start_time = time.time()
            
            try:
                _llm_instance = ChatOpenAI(
                    model=OPENAI_MODEL,
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                    api_key=settings.OPENAI_API_KEY,
                )
                _llm_config["config"] = config_key
                llm_load_time = time.time() - llm_start_time
                logger.info(f"✓ OpenAI LLM 초기화 완료 (소요 시간: {llm_load_time:.2f}초)")
            except Exception as e:
                logger.error(f"OpenAI LLM 초기화 실패: {str(e)}", exc_info=True)
                raise
        else:
            # HuggingFace 모델 사용 (기본값)
            _llm_instance = _get_huggingface_llm(temperature, max_new_tokens)
            _llm_config["config"] = config_key
    
    return _llm_instance


def _get_huggingface_llm(temperature: float = 0.001, max_new_tokens: int = 512) -> ChatHuggingFace:
    """
    HuggingFace LLM 인스턴스 생성 (내부 함수)
    
    Args:
        temperature: 생성 온도
        max_new_tokens: 최대 생성 토큰 수
    
    Returns:
        ChatHuggingFace 인스턴스
    """
    model_name = ModelManager.HUGGINGFACE_MODELS["llm"]["name"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 캐시 디렉토리 가져오기 (기존 models 디렉토리 사용)
    ModelManager.setup_cache_dir()  # 캐시 디렉토리 초기화 보장
    cache_dir = ModelManager.get_cache_dir()
    
    logger.info(f"HuggingFace LLM 모델 로딩 시작: {model_name}, device={device}, cache_dir={cache_dir}, temperature={temperature}, max_new_tokens={max_new_tokens}")
    llm_start_time = time.time()
    
    try:
        llm_instance = ChatHuggingFace.from_model_id(
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
        llm_load_time = time.time() - llm_start_time
        logger.info(f"✓ HuggingFace LLM 모델 로딩 완료 (소요 시간: {llm_load_time:.2f}초)")
        return llm_instance
    except Exception as e:
        logger.error(f"HuggingFace LLM 모델 로딩 실패: {str(e)}", exc_info=True)
        raise


def get_classification_llm() -> Union[ChatHuggingFace, ChatOpenAI]:
    """분류용 LLM (낮은 temperature, 짧은 응답)"""
    return get_llm(temperature=0.001, max_new_tokens=100)


def get_answer_llm() -> Union[ChatHuggingFace, ChatOpenAI]:
    """답변 생성용 LLM (기본 설정)"""
    return get_llm(temperature=0.3, max_new_tokens=512)


def get_evaluation_llm() -> Union[ChatHuggingFace, ChatOpenAI]:
    """평가용 LLM (관련성, 환각 검증 - 낮은 temperature)"""
    return get_llm(temperature=0.001, max_new_tokens=200)
