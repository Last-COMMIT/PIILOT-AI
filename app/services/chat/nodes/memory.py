"""
메모리 관리 노드 (load_memory, save_memory)
"""
from app.services.chat.state import ChatbotState
from app.services.chat.config import MAX_MESSAGES, MAX_TOKENS
from app.core.logging import logger
from typing import Dict, List, Optional

from app.services.chat.utils.llm_client import get_llm


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
        # state에 messages/summary가 없으면 초기화 (체크포인터 state를 덮어쓰지 않도록 주의)
        if "messages" not in state:
            state["messages"] = []
            logger.debug("대화 이력 초기화(messages)")
        if "conversation_summary" not in state or state.get("conversation_summary") is None:
            state["conversation_summary"] = ""
            logger.debug("대화 요약 초기화(conversation_summary)")
        
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
        if "conversation_summary" not in state or state.get("conversation_summary") is None:
            state["conversation_summary"] = ""
        
        state["messages"].append({
            "role": "user",
            "content": user_question
        })
        state["messages"].append({
            "role": "assistant",
            "content": final_answer
        })

        # FIFO: MAX_MESSAGES를 초과하면, 초과분(가장 오래된 메시지)을 요약에 누적 후 제거
        overflow = len(state["messages"]) - MAX_MESSAGES
        if overflow > 0:
            evicted = state["messages"][:overflow]
            try:
                evicted_text = "\n".join(
                    f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in evicted
                ).strip()
                if evicted_text:
                    summary_prompt = (
                        "다음은 이전 대화 중 오래되어 제거될 메시지들입니다.\n"
                        "기존 요약이 있으면 그 의미를 유지하면서, 아래 내용을 3~6문장으로 한국어로 간결히 요약해 주세요.\n"
                        "사실/결정사항/요구사항/제약을 우선으로 남기고, 불필요한 수사는 제거하세요.\n\n"
                        f"[기존 요약]\n{state.get('conversation_summary','')}\n\n"
                        f"[제거될 대화]\n{evicted_text}\n\n"
                        "요약:"
                    )
                    llm = get_llm(temperature=0.001, max_new_tokens=256)
                    resp = llm.invoke(summary_prompt)
                    new_summary = getattr(resp, "content", None) or str(resp)
                    # 요약 길이 제한(간단한 문자 기준 컷)
                    merged = (new_summary or "").strip()
                    if merged:
                        state["conversation_summary"] = merged[-4000:]
                        logger.info("대화 요약 업데이트 완료 (len=%d)", len(state["conversation_summary"]))
            except Exception as e:
                logger.warning(f"대화 요약 생성 실패(무시): {e}", exc_info=True)

            # 최근 N개만 유지 (실제 제거)
            state["messages"] = state["messages"][-MAX_MESSAGES:]
        else:
            # 최근 N개만 유지
            state["messages"] = state["messages"][-MAX_MESSAGES:]
        
        # TODO: DB에 영구 저장 (선택사항)
        logger.info(f"대화 이력 저장 완료: 총 {len(state['messages'])}개 메시지")
        return state
        
    except Exception as e:
        logger.error(f"대화 이력 저장 실패: {str(e)}", exc_info=True)
        return state
