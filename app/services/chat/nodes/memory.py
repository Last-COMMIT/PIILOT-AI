"""
메모리 관리 노드 (load_memory, save_memory)
"""
from app.services.chat.state import ChatbotState
from app.services.chat.config import MAX_MESSAGES, MAX_TOKENS
from app.core.logging import logger
from typing import Dict, List


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
        
        # TODO: DB에서 대화 이력 불러오기 (현재는 메모리에서만 관리)
        # 현재는 state에 이미 messages가 있으면 그대로 사용, 없으면 초기화
        if "messages" not in state or not state["messages"]:
            state["messages"] = []
            logger.debug("대화 이력 초기화")
        
        # 최근 N개만 유지 (토큰 제한 고려)
        messages = state["messages"][-MAX_MESSAGES:]
        state["messages"] = messages
        
        # 재시도 카운터 초기화
        state["retry_count"] = 0
        state["search_query_version"] = 0
        state["generation_retry_count"] = 0
        
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
        
        # 재생성 재시도 초과인 경우 경고 메시지 추가
        from app.services.chat.config import MAX_GENERATION_RETRIES
        if generation_retry_count >= MAX_GENERATION_RETRIES:
            if not final_answer.startswith("⚠️"):
                state["final_answer"] = f"⚠️ [답변 품질 경고] {final_answer}"
                logger.warning("재생성 재시도 초과로 경고 메시지 추가")
        
        # 현재 질문과 답변을 messages에 추가
        if "messages" not in state:
            state["messages"] = []
        
        state["messages"].append({
            "role": "user",
            "content": user_question
        })
        state["messages"].append({
            "role": "assistant",
            "content": final_answer
        })
        
        # 최근 N개만 유지
        state["messages"] = state["messages"][-MAX_MESSAGES:]
        
        # TODO: DB에 영구 저장 (선택사항)
        logger.info(f"대화 이력 저장 완료: 총 {len(state['messages'])}개 메시지")
        return state
        
    except Exception as e:
        logger.error(f"대화 이력 저장 실패: {str(e)}", exc_info=True)
        return state
