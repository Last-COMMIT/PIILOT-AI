"""
메모리 관리 노드 (load_memory, save_memory)

MemorySaver 체크포인터 대신 모듈 레벨 딕셔너리(_conversation_store)로
conversation_id별 대화 이력(messages)만 경량 관리.
→ 매 노드마다 full state 체크포인트가 누적되는 메모리 문제 해결.
"""
from app.services.chat.state import ChatbotState
from app.services.chat.config import MAX_MESSAGES, MAX_TOKENS
from app.core.logging import logger
from typing import Dict, List, Optional

from app.services.chat.utils.llm_client import get_llm

# ── 대화 이력 저장소 (MemorySaver 대체) ──
# conversation_id → List[Dict]  (최대 MAX_MESSAGES개)
_conversation_store: Dict[str, List[Dict]] = {}


def load_memory(state: ChatbotState) -> ChatbotState:
    """
    대화 이력 로드 및 초기화
    
    Args:
        state: 현재 상태
    
    Returns:
        업데이트된 상태
    """
    try:
        conversation_id = state.get("conversation_id", "default")
        logger.info(f"대화 이력 로드 시작: conversation_id={conversation_id}")
        
        # _conversation_store에서 이전 대화 이력 로드
        stored_messages = _conversation_store.get(conversation_id, [])
        messages = stored_messages[-MAX_MESSAGES:] if stored_messages else []
        
        if not messages:
            logger.debug("대화 이력 초기화(messages)")
        
        # conversation_summary 비활성화 (시연 안정성을 위해)
        state["conversation_summary"] = ""
        
        # 대화 이력 설정
        state["messages"] = messages
        
        # 재시도 카운터 초기화 (새로운 질문마다 초기화)
        state["retry_count"] = 0
        state["search_query_version"] = 0
        state["generation_retry_count"] = 0
        # 이전 답변 관련 필드 초기화 (재생성으로 잘못 인식되는 것 방지)
        state["final_answer"] = ""
        state["db_result"] = None
        state["vector_docs"] = []
        state["reranked_docs"] = []
        state["relevance_score"] = None
        state["is_relevant"] = None
        state["hallucination_score"] = None
        state["is_grounded"] = None
        
        logger.info(f"대화 이력 로드 완료: {len(messages)}개 메시지")
        return state
        
    except Exception as e:
        logger.error(f"대화 이력 로드 실패: {str(e)}", exc_info=True)
        # 기본값으로 초기화
        state["messages"] = []
        state["retry_count"] = 0
        state["search_query_version"] = 0
        state["generation_retry_count"] = 0
        return state


def save_memory(state: ChatbotState) -> ChatbotState:
    """
    대화 이력 저장
    
    Args:
        state: 현재 상태
    
    Returns:
        업데이트된 상태
    """
    try:
        conversation_id = state.get("conversation_id", "default")
        user_question = state.get("user_question", "")
        final_answer = state.get("final_answer", "")
        generation_retry_count = state.get("generation_retry_count", 0)
        
        logger.info(f"대화 이력 저장 시작: conversation_id={conversation_id}")
        
        # 재생성 재시도 초과인 경우에만 경고 메시지 추가
        from app.services.chat.config import MAX_GENERATION_RETRIES
        if generation_retry_count > MAX_GENERATION_RETRIES:
            if not final_answer.startswith("⚠️"):
                state["final_answer"] = f"⚠️ [답변 품질 경고] {final_answer}"
                logger.warning("재생성 재시도 초과로 경고 메시지 추가")
        
        # 현재 대화의 메시지 목록
        current_messages = list(state.get("messages") or [])
        
        # conversation_summary 비활성화 (시연 안정성을 위해)
        state["conversation_summary"] = ""
        
        # 새 메시지 추가
        new_messages = [
            {"role": "user", "content": user_question},
            {"role": "assistant", "content": final_answer},
        ]
        
        # 기존 메시지와 합치기
        updated_messages = current_messages + new_messages

        # FIFO: MAX_MESSAGES를 초과하면 가장 오래된 메시지 제거
        if len(updated_messages) > MAX_MESSAGES:
            overflow = len(updated_messages) - MAX_MESSAGES
            updated_messages = updated_messages[-MAX_MESSAGES:]
            logger.debug(f"대화 이력 제한: {overflow}개 메시지 제거 (요약 생성 없음)")
        
        # state 및 _conversation_store에 동시 저장
        state["messages"] = updated_messages
        _conversation_store[conversation_id] = updated_messages
        
        logger.info(f"대화 이력 저장 완료: 총 {len(updated_messages)}개 메시지")
        return state
        
    except Exception as e:
        logger.error(f"대화 이력 저장 실패: {str(e)}", exc_info=True)
        return state
