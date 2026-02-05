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
        if "messages" not in state or not state["messages"]:
            current_messages = []
            logger.debug("대화 이력 초기화(messages)")
        else:
            # 기존 메시지 가져오기 (리스트로 변환)
            current_messages = list(state["messages"]) if state["messages"] else []
        
        # conversation_summary는 reducer가 있으므로, 없을 때만 초기화
        # 이미 값이 있으면 덮어쓰지 않음 (체크포인터에서 복원된 값 유지)
        if "conversation_summary" not in state:
            state["conversation_summary"] = ""
            logger.debug("대화 요약 초기화(conversation_summary)")
        elif state.get("conversation_summary") is None:
            # None인 경우만 빈 문자열로 초기화
            state["conversation_summary"] = ""
            logger.debug("대화 요약 None -> 빈 문자열로 초기화")
        
        # 최근 N개만 유지 (토큰 제한 고려)
        messages = current_messages[-MAX_MESSAGES:] if current_messages else []
        state["messages"] = messages
        
        # 재시도 카운터 초기화 (새로운 질문마다 초기화)
        # 새로운 질문이 들어올 때마다 재시도 카운터를 리셋해야 함
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
        
        # 재생성 재시도 초과인 경우 경고 메시지 추가
        from app.services.chat.config import MAX_GENERATION_RETRIES
        if generation_retry_count >= MAX_GENERATION_RETRIES:
            if not final_answer.startswith("⚠️"):
                state["final_answer"] = f"⚠️ [답변 품질 경고] {final_answer}"
                logger.warning("재생성 재시도 초과로 경고 메시지 추가")
        
        # 현재 질문과 답변을 messages에 추가
        # LangGraph의 Annotated add_messages를 사용하므로 리스트로 반환
        if "messages" not in state:
            current_messages = []
        else:
            current_messages = list(state["messages"]) if state["messages"] else []
        
        # conversation_summary는 reducer가 있으므로, 없을 때만 초기화
        # 이미 값이 있으면 덮어쓰지 않음
        if "conversation_summary" not in state:
            state["conversation_summary"] = ""
        elif state.get("conversation_summary") is None:
            state["conversation_summary"] = ""
        
        # 새 메시지 추가
        new_messages = [
            {
                "role": "user",
                "content": user_question
            },
            {
                "role": "assistant",
                "content": final_answer
            }
        ]
        
        # 기존 메시지와 합치기
        updated_messages = current_messages + new_messages

        # FIFO: MAX_MESSAGES를 초과하면, 초과분(가장 오래된 메시지)을 요약에 누적 후 제거
        overflow = len(updated_messages) - MAX_MESSAGES
        if overflow > 0:
            evicted = updated_messages[:overflow]
            try:
                evicted_text = "\n".join(
                    f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in evicted
                ).strip()
                if evicted_text:
                    try:
                        # 입력 길이 제한 (API 에러 방지)
                        existing_summary = state.get('conversation_summary', '') or ''
                        summary_text = existing_summary[:1000]  # 기존 요약도 제한
                        evicted_text_limited = evicted_text[:1500]  # 제거될 대화도 제한
                        
                        summary_prompt = (
                            "다음은 이전 대화 중 오래되어 제거될 메시지들입니다.\n"
                            "기존 요약이 있으면 그 의미를 유지하면서, 아래 내용을 3~6문장으로 한국어로 간결히 요약해 주세요.\n"
                            "사실/결정사항/요구사항/제약을 우선으로 남기고, 불필요한 수사는 제거하세요.\n\n"
                            f"[기존 요약]\n{summary_text}\n\n"
                            f"[제거될 대화]\n{evicted_text_limited}\n\n"
                            "요약:"
                        )
                        
                        # 타임아웃 설정을 위해 LLM 호출을 간단하게
                        llm = get_llm(temperature=0.001, max_new_tokens=200)  # 토큰 수 줄임
                        
                        # 응답 파싱 (에러 처리 강화)
                        try:
                            resp = llm.invoke(summary_prompt)
                            
                            if hasattr(resp, "content"):
                                new_summary = resp.content.strip()
                            elif isinstance(resp, str):
                                new_summary = resp.strip()
                            else:
                                # dict 형태의 에러 응답 처리
                                if isinstance(resp, dict):
                                    if "error" in resp:
                                        logger.warning(f"대화 요약 생성 API 에러: {resp.get('error', 'unknown')}")
                                        new_summary = None
                                    else:
                                        # 다른 형태의 응답 시도
                                        new_summary = str(resp).strip()
                                else:
                                    new_summary = str(resp).strip()
                            
                            # 요약 길이 제한
                            if new_summary and len(new_summary) > 0:
                                merged = new_summary[:3000]  # 최대 3000자로 제한
                                state["conversation_summary"] = merged
                                logger.info("대화 요약 업데이트 완료 (len=%d)", len(merged))
                            else:
                                logger.warning("대화 요약 생성 결과가 비어있음, 기존 요약 유지")
                        except Exception as api_error:
                            # API 호출 자체가 실패한 경우
                            error_msg = str(api_error)
                            if "400" in error_msg or "Bad Request" in error_msg:
                                logger.warning(f"대화 요약 생성 API 400 에러 (무시): {error_msg[:200]}")
                            else:
                                logger.warning(f"대화 요약 생성 API 에러 (무시): {error_msg[:200]}")
                            # 에러 발생 시 기존 요약 유지 (변경하지 않음)
                    except Exception as e:
                        logger.warning(f"대화 요약 생성 중 예외 발생(무시): {e}", exc_info=True)
                        # 에러 발생 시 기존 요약 유지
            except Exception as e:
                logger.warning(f"대화 요약 생성 중 예외 발생(무시): {e}", exc_info=True)

            # 최근 N개만 유지 (실제 제거)
            updated_messages = updated_messages[-MAX_MESSAGES:]
        
        # State에 메시지 설정 (Annotated add_messages를 위해 리스트로 반환)
        state["messages"] = updated_messages
        
        # TODO: DB에 영구 저장 (선택사항)
        logger.info(f"대화 이력 저장 완료: 총 {len(state['messages'])}개 메시지")
        return state
        
    except Exception as e:
        logger.error(f"대화 이력 저장 실패: {str(e)}", exc_info=True)
        return state
