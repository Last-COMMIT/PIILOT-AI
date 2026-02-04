"""
질문 분류 노드 (classify)
"""
from app.services.chat.state import ChatbotState
from app.services.chat.utils.llm_client import get_classification_llm
from app.core.logging import logger
from langchain_core.prompts import PromptTemplate


CLASSIFICATION_PROMPT = """다음 질문을 분석하여 질문 유형을 분류하세요.

질문: {question}

대화 맥락:
{context}

다음 중 하나로 분류하세요:
- "db_query": 구조화된 데이터베이스 테이블 조회/집계가 필요한 질문 (예: "고객 수는?", "매출 통계는?", "이슈 몇 건?", "스캔 실패 몇 개?")
- "vector_search": 법령/규정/문서 검색이 필요한 질문 (예: "개인정보보호법은?", "암호화 규정은?", "법령 조항", "규정 내용")
- "both": DB 조회와 법령 검색이 모두 필요한 질문
- "general": 일반 대화 또는 위 유형에 해당하지 않는 질문

중요 규칙:
1. "법", "규정", "조항", "법령", "개인정보보호법" 같은 법령/규정 질의면 "vector_search"
2. "몇 개/개수/건수/통계/집계/카운트/count" 처럼 수량을 묻고, 대상이 시스템/서비스/서버/파일/스캔/이슈/오류/연결/작업/로그 등 운영 데이터이면 "db_query"
3. 질문이 특정 DB 테이블/컬럼을 직접 언급하지 않아도, 운영/관제 지표(이슈 수, 실패 수, 처리 건수 등)면 "db_query"로 분류하세요.
3. 분류 결과만 한 단어로 출력하세요. 다른 설명이나 텍스트는 포함하지 마세요.

예시: vector_search"""


def classify(state: ChatbotState) -> ChatbotState:
    """
    질문 유형 분류
    
    Args:
        state: 현재 상태
    
    Returns:
        업데이트된 상태 (query_type 설정)
    """
    try:
        user_question = state.get("user_question", "")
        messages = state.get("messages", [])
        conversation_summary = (state.get("conversation_summary") or "").strip()
        
        logger.info(f"질문 분류 시작: {user_question[:50]}...")
        
        # 대화 맥락 구성 (요약 + 최근 3개 메시지)
        context_messages = messages[-3:] if len(messages) > 0 else []
        recent_context = "\n".join([
            f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
            for msg in context_messages
        ]) if context_messages else "대화 맥락 없음"
        if conversation_summary:
            context = f"이전 대화 요약: {conversation_summary}\n\n최근 대화:\n{recent_context}"
        else:
            context = recent_context
        
        # 프롬프트 생성
        prompt = PromptTemplate(
            template=CLASSIFICATION_PROMPT,
            input_variables=["question", "context"]
        )
        
        # LLM 호출
        llm = get_classification_llm()
        formatted_prompt = prompt.format(question=user_question, context=context)
        response = llm.invoke(formatted_prompt)
        
        # 응답 파싱 (정규식으로 유효한 타입만 추출)
        import re
        response_text = response.content.strip()
        logger.debug(f"분류 LLM 원본 응답: {response_text[:200]}...")
        
        # 정규식으로 유효한 타입 추출
        valid_types = ["db_query", "vector_search", "both", "general"]
        query_type = None
        
        # 1. 정규식으로 직접 매칭 시도
        pattern = r'\b(' + '|'.join(valid_types) + r')\b'
        match = re.search(pattern, response_text.lower())
        if match:
            query_type = match.group(1)
        else:
            # 2. 각 타입을 직접 검색
            for valid_type in valid_types:
                if valid_type in response_text.lower():
                    query_type = valid_type
                    break
        
        # 3. 파싱 실패 시 질문 내용 기반 휴리스틱 (법령 키워드 우선)
        if query_type is None:
            logger.warning(f"분류 파싱 실패, 휴리스틱 적용: {response_text[:100]}...")
            question_lower = user_question.lower()
            
            # 법령/규정 키워드 우선 검사 (더 많은 키워드 추가)
            law_keywords = [
                "법", "규정", "조항", "법령", "개인정보보호법", "정보통신망법",
                "개인정보", "보호", "규칙", "지침", "가이드라인", "표준"
            ]
            # DB 조회 키워드
            db_keywords = [
                "고객", "매출", "통계", "데이터", "조회", "테이블", "집계",
                "count", "sum", "avg", "통계", "수량", "건수"
            ]
            
            if any(keyword in question_lower for keyword in law_keywords):
                query_type = "vector_search"
                logger.info(f"휴리스틱 분류 결과: {query_type} (법령 키워드 감지)")
            elif any(keyword in question_lower for keyword in db_keywords):
                query_type = "db_query"
                logger.info(f"휴리스틱 분류 결과: {query_type} (DB 키워드 감지)")
            else:
                query_type = "general"
                logger.info(f"휴리스틱 분류 결과: {query_type} (기본값)")
        
        state["query_type"] = query_type
        logger.info(f"질문 분류 완료: {query_type}")
        return state
        
    except Exception as e:
        logger.error(f"질문 분류 실패: {str(e)}", exc_info=True)
        # 기본값으로 설정
        state["query_type"] = "general"
        return state
