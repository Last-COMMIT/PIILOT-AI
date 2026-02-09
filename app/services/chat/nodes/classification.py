"""
질문 분류 노드 (classify)
"""
from app.services.chat.state import ChatbotState
from app.services.chat.utils.llm_client import get_classification_llm
from app.core.logging import logger
from langchain_core.prompts import PromptTemplate
import re


CLASSIFICATION_PROMPT = """당신은 질문 분류 전문가입니다. 사용자의 질문을 분석하여 적절한 처리 방식을 결정하세요.

## 역할 및 목적
- 질문의 의도와 필요한 정보 소스를 정확히 파악
- DB 조회가 필요한지, 법령 검색이 필요한지, 둘 다 필요한지 판단
- 단계별로 분석하여 정확한 분류 수행

## 분석 단계 (Chain-of-Thought)
다음 순서로 분석하세요:

1단계: 질문의 핵심 키워드 파악
- 질문에서 주요 키워드를 추출하세요
- 예: "법", "규정", "개수", "이슈", "파일" 등

2단계: 정보 소스 판단
- 질문이 요구하는 정보가 어디에 있는지 판단하세요
- DB 테이블에 저장된 운영 데이터인가?
- 법령/규정 문서에 있는 정보인가?
- 둘 다 필요한가?

3단계: 분류 결정
- 1-2단계 분석 결과를 바탕으로 최종 분류 결정

## 분류 유형

### "db_query"
- 구조화된 데이터베이스 테이블 조회/집계가 필요한 질문
- 특징: 수량, 통계, 집계, 개수, 상태 조회
- 예시:
  * "DB 서버별 이슈 개수는?"
  * "파일별 PII 유형별 개수 조회"
  * "최근 일주일간 스캔 내역"
  * "암호화 안 된 파일 몇 개?"

### "vector_search"
- 법령/규정/문서 검색이 필요한 질문
- 특징: 법령명, 조항, 규정 내용, 가이드라인
- 예시:
  * "개인정보보호법 제15조는?"
  * "65조 내용 알려줘" (조항 번호만 있어도 vector_search)
  * "제15조는 뭐야?" (조항 번호 패턴)
  * "암호화 규정은 어떻게 되어있나요?"
  * "개인정보 처리 원칙은?"
  * "법령에서 개인정보 보관 기간은?"

### "both"
- DB 조회와 법령 검색이 모두 필요한 질문
- 예시:
  * "우리 DB에 있는 개인정보가 법령에 위반되는지 확인해줘"
  * "현재 암호화 상태가 법령 기준에 맞는지 조회"

### "general"
- 일반 대화 또는 위 유형에 해당하지 않는 질문
- 예시:
  * "안녕하세요"
  * "도움말"
  * "시스템 사용법"

## Few-shot Examples

질문: "우리 DB서버 이슈 뭐있어"
분석:
1단계: 키워드 = "DB서버", "이슈"
2단계: DB 테이블에서 이슈 정보 조회 필요
3단계: db_query
답변: db_query

질문: "개인정보보호법 제15조 내용 알려줘"
분석:
1단계: 키워드 = "개인정보보호법", "제15조"
2단계: 법령 문서 검색 필요
3단계: vector_search
답변: vector_search

질문: "65조 내용좀 알려줘"
분석:
1단계: 키워드 = "65조" (조항 번호 패턴)
2단계: 법령 문서 검색 필요
3단계: vector_search
답변: vector_search

질문: "파일 서버별로 이슈 개수 정리해줘"
분석:
1단계: 키워드 = "파일 서버", "이슈 개수"
2단계: DB 테이블에서 집계 조회 필요
3단계: db_query
답변: db_query

## 중요 규칙 (우선순위 순)
1. **부정 표현 + 명확한 키워드 우선 처리**:
   - "DB 말고 파일서버", "파일 말고 DB" 같은 부정 표현이 있으면, 부정된 것이 아닌 키워드로 분류
   - 예: "DB 말고 파일서버 이슈" → 파일서버 키워드 우선 → db_query
2. **법령 질문 패턴 (매우 중요!)**:
   - "어떤 법령", "어떤 규정", "어떤 조항", "어떤 법", "어떤 법률" 같은 표현이 있으면 반드시 "vector_search" 또는 "both"로 분류
   - "위반할 수 있는", "위반하는", "위반 가능" 같은 표현과 함께 "법령", "규정", "조항"이 있으면 "vector_search" 또는 "both"
   - 예: "어떤 법령을 위반할 수 있는건데?" → vector_search 또는 both
   - 예: "어떤 규정에 위배되는거야?" → vector_search 또는 both
3. **조항 번호 패턴(예: "65조", "제15조", "15조")이 있으면 반드시 "vector_search"로 분류**
4. **운영 데이터 키워드 우선**:
   - "파일서버", "파일 서버", "DB서버", "DB 서버", "파일", "이슈", "스캔" 등이 있으면 "db_query" 우선 고려
   - 단, "법령", "법", "규정", "조항"과 함께 나오면 "both" 고려
5. 법령/규정 키워드("법", "규정", "조항", "법령", "개인정보보호법")만 있으면 "vector_search"
6. 수량/통계 키워드("몇 개", "개수", "건수", "통계", "집계") + 운영 데이터 키워드면 "db_query"
7. DB 테이블/컬럼 직접 언급 없어도 운영 지표 질문이면 "db_query"
8. 분류 결과만 한 단어로 출력 (설명 없이)

## 현재 질문 분석

질문: {question}

대화 맥락:
{context}

위 분석 단계를 따라 분류하세요. 답변은 분류 유형만 한 단어로 출력하세요:"""


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
        
        # 대화 맥락 구성 (최소화: 최근 1개만 - 시연 안정성을 위해)
        context_messages = messages[-1:] if len(messages) > 0 else []  # 최근 1개만
        recent_context = "\n".join([
            f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
            for msg in context_messages
        ]) if context_messages else "대화 맥락 없음"
        context = recent_context  # conversation_summary 사용 안 함
        
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
            
            # 1. 부정 표현 + 명확한 키워드 우선 처리
            negation_patterns = [
                (r"db\s*말고|데이터베이스\s*말고", ["파일서버", "파일 서버", "파일", "file"]),
                (r"파일\s*말고|파일서버\s*말고", ["db", "데이터베이스", "db서버", "db 서버"]),
            ]
            for neg_pattern, positive_keywords in negation_patterns:
                if re.search(neg_pattern, question_lower):
                    if any(kw in question_lower for kw in positive_keywords):
                        query_type = "db_query"
                        logger.info(f"휴리스틱 분류 결과: {query_type} (부정 표현 + 명확한 키워드: {positive_keywords})")
                        break
            
                # 2. 법령 질문 패턴 체크 - "어떤 법령", "어떤 규정", "위반할 수 있는" 등
                if query_type is None:
                    law_question_patterns = [
                        r"어떤\s*법령",
                        r"어떤\s*규정",
                        r"어떤\s*조항",
                        r"어떤\s*법",
                        r"어떤\s*법률",
                        r"위반할\s*수\s*있는",
                        r"위반하는",
                        r"위반\s*가능",
                        r"위배되는",
                        r"위배할"
                    ]
                    if any(re.search(pattern, user_question, re.IGNORECASE) for pattern in law_question_patterns):
                        # 이전 대화에서 파일/DB 관련 언급이 있으면 both, 없으면 vector_search
                        if any(keyword in question_lower for keyword in ["파일", "db", "데이터베이스", "서버", "이슈"]):
                            query_type = "both"
                            logger.info(f"휴리스틱 분류 결과: {query_type} (법령 질문 패턴 + 운영 데이터 키워드)")
                        else:
                            query_type = "vector_search"
                            logger.info(f"휴리스틱 분류 결과: {query_type} (법령 질문 패턴 감지)")
                
                # 3. 조항 번호 패턴 체크 - "65조", "제15조", "15조" 등
                if query_type is None:
                    article_pattern = r'\d+\s*조|제\s*\d+\s*조'
                    if re.search(article_pattern, user_question):
                        query_type = "vector_search"
                        logger.info(f"휴리스틱 분류 결과: {query_type} (조항 번호 패턴 감지)")
                
                # 4. 운영 데이터 키워드 우선 (파일서버, DB서버, 이슈 등)
            if query_type is None:
                if any(keyword in question_lower for keyword in [
                    "파일서버", "파일 서버", "파일", "db서버", "db 서버", "이슈", "스캔",
                    "file", "server", "issue"
                ]):
                    # 법령 키워드도 함께 있으면 both, 아니면 db_query
                    if any(keyword in question_lower for keyword in [
                        "법령", "법", "규정", "조항", "개인정보보호법"
                    ]):
                        query_type = "both"
                        logger.info(f"휴리스틱 분류 결과: {query_type} (운영 데이터 + 법령 키워드)")
                    else:
                        query_type = "db_query"
                        logger.info(f"휴리스틱 분류 결과: {query_type} (운영 데이터 키워드 감지)")
            
            # 5. 법령/규정 키워드 (운영 데이터 키워드가 없을 때만)
            if query_type is None:
                if any(keyword in question_lower for keyword in [
                    "법", "규정", "조항", "법령", "개인정보보호법", "정보통신망법",
                    "개인정보", "보호", "규칙", "지침", "가이드라인", "표준", "내용", "알려줘"
                ]):
                    query_type = "vector_search"
                    logger.info(f"휴리스틱 분류 결과: {query_type} (법령 키워드 감지)")
            
            # 6. DB 조회 키워드
            if query_type is None:
                if any(keyword in question_lower for keyword in [
                    "고객", "매출", "통계", "데이터", "조회", "테이블", "집계",
                    "count", "sum", "avg", "통계", "수량", "건수"
                ]):
                    query_type = "db_query"
                    logger.info(f"휴리스틱 분류 결과: {query_type} (DB 키워드 감지)")
            
            # 7. 기본값
            if query_type is None:
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
