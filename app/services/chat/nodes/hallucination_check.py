"""
환각 검증 노드 (check_hallucination) - Self-RAG ②
"""
from app.services.chat.state import ChatbotState
from app.services.chat.utils.llm_client import get_evaluation_llm
from app.services.chat.config import GROUNDING_THRESHOLD
from app.core.logging import logger
from langchain_core.prompts import PromptTemplate


HALLUCINATION_PROMPT = """다음 생성된 답변이 제공된 문서에 근거하는지 평가하세요.

생성된 답변:
{answer}

제공된 문서:
{context}

DB 조회 결과:
{db_result}

평가 기준:
- 답변의 각 주장이 제공된 문서에 근거하는가?
- 추측이나 외부 지식을 사용했는가?
- 문서에 없는 정보를 포함했는가?

중요: 다음 형식으로만 응답하세요. 다른 설명은 포함하지 마세요.
근거 점수: 0.0 ~ 1.0 사이의 숫자
근거 여부: yes 또는 no

예시:
근거 점수: 0.9
근거 여부: yes"""


def check_hallucination(state: ChatbotState) -> ChatbotState:
    """
    생성된 답변의 환각(hallucination) 검증
    
    Args:
        state: 현재 상태
    
    Returns:
        업데이트된 상태 (hallucination_score, is_grounded 설정)
    """
    try:
        final_answer = state.get("final_answer", "")
        db_result = state.get("db_result")
        reranked_docs = state.get("reranked_docs", [])
        
        logger.info("환각 검증 시작")
        
        # DB 조회만 한 경우: 환각 체크 스킵 (is_grounded=True)
        if db_result and not reranked_docs and not state.get("vector_docs"):
            logger.info("DB 조회만 수행했으므로 환각 체크 스킵")
            state["hallucination_score"] = 1.0
            state["is_grounded"] = True
            return state
        
        # 검색 결과 없음: 참고 문서가 없으면 환각 검증 스킵
        vector_docs = state.get("vector_docs", [])
        if not reranked_docs and not vector_docs:
            # "자료 없음" 답변이면 OK
            if "자료 없음" in final_answer or "찾을 수 없" in final_answer or "정보 없" in final_answer or "보안상의 이유" in final_answer:
                logger.info("자료 없음 또는 보안 거부 답변이므로 OK")
                state["hallucination_score"] = 1.0
                state["is_grounded"] = True
                return state
            # 참고 문서가 없고 일반 질문인 경우: 재생성 루프 방지를 위해 통과 처리
            # (general 타입 질문은 참고 문서가 없어도 정상)
            query_type = state.get("query_type", "")
            if query_type == "general":
                logger.info("일반 질문이고 참고 문서 없음, 통과 처리")
                state["hallucination_score"] = 1.0
                state["is_grounded"] = True
                return state
            # 참고 문서가 없으면 기본 점수 부여 (재생성 루프 방지)
            else:
                logger.warning("참고 문서가 없어 환각 검증 스킵, 기본 점수 부여")
                state["hallucination_score"] = 0.5  # 중간 점수로 재생성 루프 방지
                state["is_grounded"] = False
                return state
        
        # 참고 문서 구성
        context_docs = reranked_docs if reranked_docs else vector_docs
        context_text = "\n\n---\n\n".join([
            f"문서 {i+1}:\n{doc['content'][:500]}"
            for i, doc in enumerate(context_docs[:5])  # 최대 5개만
        ]) if context_docs else "참고 문서 없음"
        
        db_result_text = db_result if db_result else "DB 조회 결과 없음"
        
        # 프롬프트 생성
        prompt = PromptTemplate(
            template=HALLUCINATION_PROMPT,
            input_variables=["answer", "context", "db_result"]
        )
        
        # LLM 호출
        llm = get_evaluation_llm()
        formatted_prompt = prompt.format(
            answer=final_answer,
            context=context_text,
            db_result=db_result_text
        )
        response = llm.invoke(formatted_prompt)
        
        # 응답 파싱
        import re
        response_text = response.content.strip()
        logger.debug(f"환각 검증 LLM 원본 응답: {response_text[:200]}...")
        
        # 점수 추출
        hallucination_score = 0.0
        is_grounded = False
        
        try:
            # "근거 점수: 0.9" 형식 파싱 (정규식 강화)
            # 숫자 패턴: 0.0 ~ 1.0 사이의 소수점 숫자
            score_pattern = r'(\d+\.?\d*)'
            for line in response_text.split("\n"):
                line_lower = line.lower()
                if "근거 점수" in line_lower or ("점수" in line_lower and "근거" in line_lower):
                    match = re.search(score_pattern, line)
                    if match:
                        score = float(match.group(1))
                        # 0.0~1.0 범위로 제한
                        hallucination_score = max(0.0, min(1.0, score))
                        logger.debug(f"근거 점수 추출: {hallucination_score}")
                        break
                
                if "근거 여부" in line_lower or ("여부" in line_lower and "근거" in line_lower):
                    if "yes" in line_lower or "예" in line_lower or "있" in line_lower or "충분" in line_lower:
                        is_grounded = True
                        logger.debug("근거 여부: yes 추출")
                        break
            
            # 점수 기반으로도 판단
            if hallucination_score >= GROUNDING_THRESHOLD:
                is_grounded = True
                
        except Exception as parse_error:
            logger.warning(f"응답 파싱 실패: {parse_error}, 기본값 사용")
            # 파싱 실패 시 기본값 (보수적으로 처리)
            hallucination_score = 0.5
            is_grounded = hallucination_score >= GROUNDING_THRESHOLD
        
        state["hallucination_score"] = hallucination_score
        state["is_grounded"] = is_grounded
        
        logger.info(f"환각 검증 완료: score={hallucination_score:.2f}, is_grounded={is_grounded}")
        return state
        
    except Exception as e:
        logger.error(f"환각 검증 실패: {str(e)}", exc_info=True)
        # 기본값 설정 (보수적으로 처리)
        state["hallucination_score"] = 0.5
        state["is_grounded"] = False
        return state
