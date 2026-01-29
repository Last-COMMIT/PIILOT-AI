"""
LangGraph 라우팅 함수들 (Conditional Edges)
"""
from app.services.chat.state import ChatbotState
from app.services.chat.config import (
    RELEVANCE_THRESHOLD,
    GROUNDING_THRESHOLD,
    MAX_SEARCH_RETRIES,
    MAX_GENERATION_RETRIES
)
from app.core.logging import logger


def route_after_classify(state: ChatbotState) -> str:
    """
    classify 노드 이후 라우팅
    
    Returns:
        다음 노드 이름: "db_query" | "vector_search" | "both_query" | "generate_answer"
    """
    query_type = state.get("query_type", "general")
    logger.debug(f"route_after_classify: query_type={query_type}")
    
    # "general" 타입은 "generate_answer"로 매핑
    if query_type == "general":
        return "generate_answer"
    
    return query_type


def route_after_db(state: ChatbotState) -> str:
    """
    db_query 노드 이후 라우팅
    
    Returns:
        "generate_answer" | "vector_search" (DB 조회 실패 시 폴백)
    """
    db_result = state.get("db_result", "")
    query_type = state.get("query_type", "")
    
    # DB 조회 실패 시 vector_search로 폴백
    # 주의: query_type 변경은 db_query 노드에서 처리해야 함
    if query_type == "vector_search":
        logger.info("DB 조회 실패로 vector_search로 폴백")
        return "vector_search"
    
    logger.debug("route_after_db: DB 조회 성공, generate_answer로")
    return "generate_answer"


def route_after_vector(state: ChatbotState) -> str:
    """
    vector_search 노드 이후 라우팅
    
    Returns:
        항상 "check_relevance"
    """
    logger.debug("route_after_vector: 항상 check_relevance로")
    return "check_relevance"


def route_after_relevance(state: ChatbotState) -> str:
    """
    check_relevance 노드 이후 라우팅 (Self-RAG 재검색 루프)
    
    Returns:
        "rerank" | "vector_search" (재검색) | "generate_answer"
    """
    is_relevant = state.get("is_relevant", False)
    relevance_score = state.get("relevance_score", 0.0)
    retry_count = state.get("retry_count", 0)
    
    logger.debug(f"route_after_relevance: is_relevant={is_relevant}, score={relevance_score:.2f}, retry={retry_count}")
    
    # 관련성 높음 → rerank
    if is_relevant and relevance_score >= RELEVANCE_THRESHOLD:
        logger.info("관련성 충분, rerank로 이동")
        return "rerank"
    
    # 재시도 가능 → vector_search (재검색 루프)
    # 주의: state 수정은 vector_search 노드에서 처리해야 함
    elif retry_count < MAX_SEARCH_RETRIES:
        logger.info(f"관련성 부족, 재검색 시도 {retry_count + 1}/{MAX_SEARCH_RETRIES}")
        return "vector_search"
    
    # 재시도 초과 → generate_answer ("자료 없음" 답변)
    else:
        logger.warning("재검색 재시도 초과, generate_answer로 이동")
        return "generate_answer"


def route_after_hallucination(state: ChatbotState) -> str:
    """
    check_hallucination 노드 이후 라우팅 (Self-RAG 재생성 루프)
    
    주의: 라우팅 함수는 state를 수정할 수 없습니다. 카운터 증가는 노드에서 처리해야 합니다.
    
    Returns:
        "save_memory" | "generate_answer" (재생성)
    """
    is_grounded = state.get("is_grounded", False)
    hallucination_score = state.get("hallucination_score", 0.0)
    generation_retry_count = state.get("generation_retry_count", 0)
    
    logger.debug(f"route_after_hallucination: is_grounded={is_grounded}, score={hallucination_score:.2f}, retry={generation_retry_count}")
    
    # 근거 충분 → save_memory
    if is_grounded and hallucination_score >= GROUNDING_THRESHOLD:
        logger.info("근거 충분, save_memory로 이동")
        return "save_memory"
    
    # 재생성 가능 → generate_answer (재생성 루프)
    # 주의: 카운터 증가는 generate_answer 노드에서 처리
    elif generation_retry_count < MAX_GENERATION_RETRIES:
        logger.info(f"근거 부족, 재생성 시도 {generation_retry_count + 1}/{MAX_GENERATION_RETRIES}")
        return "generate_answer"
    
    # 재생성 초과 → save_memory (경고와 함께 저장)
    # 주의: 경고 메시지 추가는 save_memory 노드에서 처리
    else:
        logger.warning("재생성 재시도 초과, 경고와 함께 save_memory로 이동")
        return "save_memory"
