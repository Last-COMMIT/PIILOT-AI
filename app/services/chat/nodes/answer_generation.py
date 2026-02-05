"""
답변 생성 노드 (generate_answer)
"""
from app.services.chat.state import ChatbotState
from app.services.chat.utils.llm_client import get_answer_llm
from app.services.chat.config import MAX_GENERATION_RETRIES
from app.core.logging import logger
from langchain_core.prompts import PromptTemplate


ANSWER_PROMPT = """다음 정보를 참고하여 사용자 질문에 답변하세요.

대화 이력:
{conversation_history}

DB 조회 결과:
{db_result}

참고 문서:
{context}

사용자 질문: {question}

지시사항:
- 제공된 자료만 사용하여 답변하세요
- DB 조회 결과가 "DB 조회를 수행할 수 없습니다" 또는 "DB 조회 결과 없음"이고 참고 문서도 "참고 문서 없음"인 경우, 자료가 없어 답변할 수 없다고 명확히 알려주세요
- 근거 없으면 추측하지 마세요. 일반적인 지식으로 답변하지 마세요
- 참고한 문서의 출처를 명시하세요
- 법령을 나열하지 말고, 적절히 요약하여 답변하세요

[SQL 결과 해석 가이드]
- "가장 많은", "최대", "어떤 것이 가장" 같은 비교 질문의 경우:
  * DB 조회 결과에서 COUNT, 개수, 수치가 포함된 행을 확인하세요
  * 모든 행의 개수가 같으면 "모든 항목에 동일하게 X개씩 있습니다"라고 명확히 답변하세요
  * 개수가 다르면 가장 큰 값을 가진 항목을 "가장 많은" 항목으로 답변하세요
  * 결과가 없으면 "해당하는 데이터가 없습니다"라고 답변하세요
- 숫자 데이터는 정확히 그대로 사용하세요 (추측하거나 변경하지 마세요)
- DB 조회 결과의 모든 행을 확인하여 패턴을 파악한 후 답변하세요

답변:"""


def generate_answer(state: ChatbotState) -> ChatbotState:
    """
    최종 답변 생성
    
    Args:
        state: 현재 상태
    
    Returns:
        업데이트된 상태 (final_answer 설정)
    """
    try:
        user_question = state.get("user_question", "")
        messages = state.get("messages", [])
        conversation_summary = (state.get("conversation_summary") or "").strip()
        db_result = state.get("db_result")
        reranked_docs = state.get("reranked_docs", [])
        generation_retry_count = state.get("generation_retry_count", 0)
        
        logger.info(f"답변 생성 시작 (재생성 횟수: {generation_retry_count})")
        
        # 재생성 시도인 경우 카운터 증가 (라우팅 함수는 state를 수정할 수 없으므로 여기서 처리)
        # final_answer가 이미 있으면 재생성으로 간주하여 카운터 증가
        if state.get("final_answer") and generation_retry_count < MAX_GENERATION_RETRIES:
            state["generation_retry_count"] = generation_retry_count + 1
            logger.debug(f"재생성 시도로 카운터 증가: {generation_retry_count} -> {generation_retry_count + 1}")
        
        # 대화 이력 구성 (요약 + 최근 5개)
        recent_history = "\n".join([
            f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
            for msg in messages[-5:]
        ]) if messages else "대화 이력 없음"
        if conversation_summary:
            conversation_history = f"이전 대화 요약: {conversation_summary}\n\n최근 대화:\n{recent_history}"
        else:
            conversation_history = recent_history
        
        # DB 결과 구성
        db_result_text = db_result if db_result else "DB 조회 결과 없음"
        
        # 참고 문서 구성 (reranked_docs 우선, 없으면 vector_docs)
        if reranked_docs:
            context_docs = reranked_docs
        else:
            context_docs = state.get("vector_docs", [])
        
        context_text = "\n\n---\n\n".join([
            f"문서 {i+1} ({doc.get('metadata', {}).get('law_name', '')} {doc.get('metadata', {}).get('article', '')}):\n{doc['content'][:500]}"
            for i, doc in enumerate(context_docs[:5])  # 최대 5개만
        ]) if context_docs else "참고 문서 없음"
        
        # 프롬프트 생성
        prompt = PromptTemplate(
            template=ANSWER_PROMPT,
            input_variables=["conversation_history", "db_result", "context", "question"]
        )
        
        # LLM 호출
        llm = get_answer_llm()
        formatted_prompt = prompt.format(
            conversation_history=conversation_history,
            db_result=db_result_text,
            context=context_text,
            question=user_question
        )
        response = llm.invoke(formatted_prompt)
        
        # LLM 응답 파싱 (시스템 프롬프트 제거)
        answer_text = response.content.strip()
        
        # 시스템 프롬프트 토큰 제거 (HuggingFace 모델의 경우 원시 출력에 포함될 수 있음)
        # <|start_header_id|>assistant<|end_header_id|> 이후의 내용만 추출
        if "<|start_header_id|>assistant<|end_header_id|>" in answer_text:
            parts = answer_text.split("<|start_header_id|>assistant<|end_header_id|>")
            if len(parts) > 1:
                answer_text = parts[-1].strip()
        
        # <|eot_id|> 토큰 제거
        answer_text = answer_text.replace("<|eot_id|>", "").strip()
        
        # 답변 저장
        state["final_answer"] = answer_text
        
        logger.info(f"답변 생성 완료: {len(state['final_answer'])}자")
        return state
        
    except Exception as e:
        logger.error(f"답변 생성 실패: {str(e)}", exc_info=True)
        state["final_answer"] = "죄송합니다. 답변 생성 중 오류가 발생했습니다."
        return state
