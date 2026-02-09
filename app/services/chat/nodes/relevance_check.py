"""
관련성 평가 노드 (check_relevance) - Self-RAG ①
"""
from app.services.chat.state import ChatbotState
from app.services.chat.utils.llm_client import get_evaluation_llm
from app.services.chat.config import RELEVANCE_THRESHOLD, MAX_SEARCH_RETRIES
from app.core.logging import logger
from langchain_core.prompts import PromptTemplate


RELEVANCE_PROMPT = """다음 검색된 문서가 사용자 질문과 관련이 있는지 평가하세요.

질문: {question}

검색된 문서:
{document}

평가 기준:
- 문서가 질문에 답할 수 있는 정보를 포함하는가?
- 주제가 일치하는가?

다음 형식으로 응답하세요:
관련성 점수: 0.0 ~ 1.0 사이의 숫자
관련성 여부: yes 또는 no

예시:
관련성 점수: 0.8
관련성 여부: yes"""


def check_relevance(state: ChatbotState) -> ChatbotState:
    """
    검색 결과의 관련성 평가
    
    Args:
        state: 현재 상태
    
    Returns:
        업데이트된 상태 (relevance_score, is_relevant 설정)
    """
    try:
        user_question = state.get("user_question", "")
        vector_docs = state.get("vector_docs", [])
        retry_count = state.get("retry_count", 0)
        
        logger.info(f"관련성 평가 시작: {len(vector_docs)}개 문서, retry_count={retry_count}")
        
        # 문서 수가 비정상적으로 많으면 경고 (누적 버그 감지)
        if len(vector_docs) > 1000:
            logger.warning(f"⚠️ 문서 수가 비정상적으로 많습니다: {len(vector_docs)}개 (누적 버그 가능성)")
        
        # 재검색 루프인 경우 카운터 증가 (라우팅 함수에서 증가하지 않으므로 여기서 처리)
        # 단, 이미 증가했다면 중복 증가 방지
        if retry_count < MAX_SEARCH_RETRIES:
            # 재검색 시도인 경우에만 증가 (vector_search에서 이미 증가했을 수도 있음)
            # 안전하게: 현재 retry_count가 0이 아니면 이미 증가한 것으로 간주
            pass  # vector_search에서 이미 처리됨
        
        if not vector_docs:
            logger.warning("평가할 문서가 없음")
            state["relevance_score"] = 0.0
            state["is_relevant"] = False
            return state
        
        # 상위 3개 문서만 평가 (비용 절감)
        top_docs = vector_docs[:3]
        document_text = "\n\n---\n\n".join([
            f"문서 {i+1}:\n{doc['content'][:500]}..."
            for i, doc in enumerate(top_docs)
        ])
        
        # 프롬프트 생성
        prompt = PromptTemplate(
            template=RELEVANCE_PROMPT,
            input_variables=["question", "document"]
        )
        
        # LLM 호출
        llm = get_evaluation_llm()
        formatted_prompt = prompt.format(question=user_question, document=document_text)
        response = llm.invoke(formatted_prompt)
        
        # 응답 파싱
        response_text = response.content.strip()
        
        # 점수 추출
        relevance_score = 0.0
        is_relevant = False
        
        try:
            # "관련성 점수: 0.8" 형식 파싱
            for line in response_text.split("\n"):
                if "관련성 점수" in line or "점수" in line:
                    import re
                    match = re.search(r'(\d+\.?\d*)', line)
                    if match:
                        relevance_score = float(match.group(1))
                        relevance_score = max(0.0, min(1.0, relevance_score))  # 0.0~1.0 범위로 제한
                        break
                
                if "관련성 여부" in line or "여부" in line:
                    if "yes" in line.lower() or "예" in line or "있" in line:
                        is_relevant = True
                        break
            
            # 점수 기반으로도 판단
            if relevance_score >= RELEVANCE_THRESHOLD:
                is_relevant = True
                
        except Exception as parse_error:
            logger.warning(f"응답 파싱 실패: {parse_error}, 기본값 사용")
            # 점수 기반으로만 판단
            is_relevant = relevance_score >= RELEVANCE_THRESHOLD
        
        state["relevance_score"] = relevance_score
        state["is_relevant"] = is_relevant
        
        logger.info(f"관련성 평가 완료: score={relevance_score:.2f}, is_relevant={is_relevant}")
        return state
        
    except Exception as e:
        logger.error(f"관련성 평가 실패: {str(e)}", exc_info=True)
        # 기본값 설정
        state["relevance_score"] = 0.0
        state["is_relevant"] = False
        return state
