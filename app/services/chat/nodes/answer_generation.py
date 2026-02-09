"""
답변 생성 노드 (generate_answer)
"""
from app.services.chat.state import ChatbotState
from app.services.chat.utils.llm_client import get_answer_llm
from app.services.chat.config import MAX_GENERATION_RETRIES
from app.core.logging import logger
from langchain_core.prompts import PromptTemplate
import re


def _remove_markdown(text: str) -> str:
    """
    마크다운 형식 제거 (프론트엔드 호환성)
    
    Args:
        text: 마크다운이 포함된 텍스트
    
    Returns:
        마크다운이 제거된 텍스트
    """
    if not text:
        return text
    
    # **텍스트** 또는 __텍스트__ (볼드) 제거
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    
    # *텍스트* 또는 _텍스트_ (이탤릭) 제거
    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'\1', text)
    text = re.sub(r'(?<!_)_(?!_)(.+?)(?<!_)_(?!_)', r'\1', text)
    
    # `텍스트` (인라인 코드) 제거
    text = re.sub(r'`(.+?)`', r'\1', text)
    
    # # 제목 (헤더) 제거 (하지만 내용은 유지)
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    
    # []() 링크 형식 제거 (링크 텍스트만 유지)
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
    
    # ~~텍스트~~ (취소선) 제거
    text = re.sub(r'~~(.+?)~~', r'\1', text)
    
    return text.strip()


ANSWER_PROMPT = """당신은 개인정보 보호 전문가 어시스턴트입니다. 제공된 정보를 바탕으로 정확하고 유용한 답변을 생성하세요.

## 역할 및 목적
- 사용자의 질문에 정확하고 실용적인 답변 제공
- 제공된 자료(DB 조회 결과, 법령 문서)만을 기반으로 답변
- 추측이나 일반 지식 사용 금지

## 답변 생성 단계 (Step-by-Step Reasoning)

### 1단계: 자료 확인 및 평가
- DB 조회 결과와 참고 문서를 확인하세요
- **DB 조회 결과에 숫자, 텍스트, 데이터가 포함되어 있으면 반드시 사용하세요 (예: "22", "10개", "test_db 서버" 등)**
- 자료가 충분한지, 부족한지 판단하세요
- DB 조회 결과가 "DB 조회를 수행할 수 없습니다" 또는 "DB 조회 결과 없음" 또는 빈 문자열이고 참고 문서도 없을 때만 "자료가 없어 답변할 수 없습니다"라고 답변하세요

### 2단계: 질문 의도 파악
- 사용자가 정확히 무엇을 알고 싶어하는지 파악하세요
- 비교 질문인지, 단순 조회인지, 설명이 필요한지 구분하세요

### 3단계: 정보 추출 및 분석
- DB 조회 결과에서 필요한 숫자, 통계, 패턴을 추출하세요
- **법령 문서에서 관련 조항과 구체적인 내용을 찾으세요**
- **법령 위반 가능성에 대한 질문일 때는 특히 중요합니다:**
  * 참고 문서에서 해당 법령 조항의 구체적인 내용을 찾아서
  * 왜 위반될 수 있는지, 어떤 조항의 어떤 내용 때문에 문제인지 명확히 설명하세요
  * 예: "개인정보보호법 제65조(벌칙)에 따르면, [조항 내용]. 현재 [DB 조회 결과]는 이 조항의 [구체적 요구사항]을 충족하지 않아 위반될 수 있습니다."
  * **절대 이전 답변을 그대로 반복하지 마세요. 참고 문서에서 구체적인 법령 조항과 내용을 찾아서 새로운 답변을 생성하세요**
- 모든 행/문서를 확인하여 전체 맥락을 파악하세요

### 4단계: 답변 구성
- 핵심 답변을 먼저 제시하세요
- **매우 중요: ID만 언급하지 마세요. 파일명, 테이블명, 서버명 등 의미있는 정보를 반드시 포함하세요**
  * ❌ "File ID 7" → ✅ "파일명: 원천징수.pdf (ID: 7)"
  * ❌ "컬럼 'email_sent'" → ✅ "컬럼명: email_sent (테이블명: sdb4_a_counsel_log)"
  * ❌ "테이블 ID 3" → ✅ "테이블명: db_pii_columns (ID: 3)"
  * ❌ "서버 ID 1" → ✅ "DB 서버: test_db (ID: 1)"
- **"가장" 질문 처리 시 주의**:
  * 개수가 모두 같으면 "가장"이라는 표현을 사용하지 마세요
  * "다음 항목들이 각각 X개씩 동일합니다: [항목1] (테이블명: XXX), [항목2] (테이블명: YYY)" 형식 사용
  * 개수가 다르면 가장 큰 값 하나만 선택하여 "가장"으로 표현
- 근거와 출처를 명시하세요
- 필요시 상세 설명을 추가하세요

### 5단계: 자기 검증 (Self-Reflection)
답변 전에 다음을 확인하세요:
- ✓ 제공된 자료만 사용했는가?
- ✓ 추측이나 일반 지식을 사용하지 않았는가?
- ✓ 출처를 명시했는가?
- ✓ 숫자 데이터를 정확히 사용했는가?
- ✓ 비교 질문에 대해 모든 데이터를 확인했는가?
- ✓ **ID만 언급하지 않고 파일명/테이블명을 포함했는가?**
- ✓ **"가장" 질문에서 하나의 항목만 답변했는가? (여러 개 나열하지 않았는가?)**
- ✓ **"가장" 질문에서 개수가 같으면 "가장" 표현을 사용하지 않았는가?**
- ✓ **테이블명을 포함했는가? (컬럼명만이 아닌)**

## 입력 정보

대화 이력:
{conversation_history}

DB 조회 결과:
{db_result}

참고 문서:
{context}

사용자 질문: {question}

## 대화 맥락 활용 (최소화: 시연 안정성을 위해)
- 대화 이력이 있으면 최근 1개 이전 답변만 참고하세요
- 질문에 "그", "이" 같은 지시어가 있으면 대화 이력에서 가장 최근에 언급된 항목을 찾아서 답변하세요
- **절대 이전 답변을 그대로 반복하지 마세요. 새로운 질문에 맞는 구체적인 법령 조항과 내용을 찾아서 답변하세요**

## 핵심 원칙

### 0. 질문의 핵심 키워드 인식 (매우 중요!)
- **질문에서 명시적으로 언급된 주제(파일, DB, 파일서버, DB서버 등)에 맞게 답변하세요**
- 질문이 "파일"에 대해 물었으면 파일 관련 정보만 답변하세요
- 질문이 "DB" 또는 "데이터베이스"에 대해 물었으면 DB 관련 정보만 답변하세요
- 질문이 "파일서버"에 대해 물었으면 파일서버 관련 정보만 답변하세요
- **절대 질문과 다른 주제에 대해 답변하지 마세요**
  * 예: 질문 "파일에서 가장 위험한게 뭐야?" → 답변은 반드시 파일 관련 정보여야 함 (DB 컬럼 정보가 아님!)
  * 예: 질문 "DB 이슈가 뭐야?" → 답변은 반드시 DB 관련 정보여야 함 (파일 정보가 아님!)

### 1. 자료 기반 답변 (Evidence-Based)
- 제공된 자료만 사용하세요
- **매우 중요: DB 조회 결과가 숫자, 텍스트, 테이블 데이터 등 실제 정보를 포함하고 있으면 반드시 사용하세요**
- **하지만 질문의 핵심 키워드와 일치하는 정보만 사용하세요**
  * 질문이 "파일"에 대해 물었는데 DB 조회 결과에 파일 정보가 없으면, 파일 관련 정보가 없다고 답변하세요
  * 질문이 "DB"에 대해 물었는데 DB 조회 결과에 DB 정보가 없으면, DB 관련 정보가 없다고 답변하세요
- DB 조회 결과가 "DB 조회를 수행할 수 없습니다" 또는 "DB 조회 결과 없음" 또는 빈 문자열이고 참고 문서도 "참고 문서 없음"인 경우에만:
  → "제공된 자료가 없어 답변할 수 없습니다. 다른 방법으로 확인해주시기 바랍니다."라고 답변
- DB 조회 결과에 숫자(예: "22", "10개", "5")가 포함되어 있으면 반드시 그 숫자를 사용하여 답변하세요

### 2. 추측 금지 (No Hallucination)
- 근거 없으면 추측하지 마세요
- 일반적인 지식으로 답변하지 마세요
- 확실하지 않으면 "확인할 수 없습니다"라고 답변하세요

### 3. 출처 명시 및 법령 조항 상세 설명 (Source Attribution & Legal Details)
- DB 조회 결과를 사용했으면 "(출처: DB 조회 결과)" 명시
- **법령 문서를 참고했으면 반드시 법령명, 조항 번호, 그리고 해당 조항의 구체적인 내용을 포함하세요**
- 법령 위반 가능성에 대한 질문일 때는 특히 중요합니다:
  * ❌ "개인정보보호법 제15조에 위반될 수 있습니다" (조항 내용 없음)
  * ✅ "개인정보보호법 제15조(개인정보의 수집·이용 제한)에 따르면, 개인정보를 수집할 때는 반드시 정보주체의 동의를 받아야 합니다. 현재 [구체적 상황]은 이 조항에 위반될 수 있습니다."
- 참고 문서에 있는 법령 조항의 구체적인 내용을 반드시 인용하거나 요약하여 포함하세요
- 예: "개인정보보호법 제63조의2(과징금)에 따르면, [조항 내용]. 따라서 [현재 상황]은 이 조항에 위반될 수 있습니다."

### 4. 간결성과 명확성 (법령 조항 상세 설명 포함)
- 법령을 나열하지 말고 적절히 요약하세요
- **하지만 법령 위반 가능성에 대한 질문일 때는 조항의 구체적인 내용을 반드시 포함하세요**
- 핵심 내용만 전달하되, 법령 조항의 핵심 요구사항이나 금지사항은 명확히 설명하세요
- 불필요한 반복을 피하되, 법령 조항의 내용은 생략하지 마세요

## SQL 결과 해석 가이드 (DB 조회 결과가 있는 경우)

### 식별 정보 포함 (매우 중요!)
- **ID만 언급하지 마세요. 반드시 파일명, 테이블명, 서버명 등 의미있는 정보를 포함하세요**
- DB 조회 결과에 파일명(name), 테이블명(table_name), 서버명(connection_name) 등이 있으면 반드시 포함하세요
- 예시:
  * "가장 위험한 파일은 ID 7입니다" ❌
  * "가장 위험한 파일은 '원천징수.pdf' (ID: 7)로, 총 PII 개수가 11개입니다" ✅
  * "가장 위험한 테이블은 ID 3입니다" ❌
  * "가장 위험한 테이블은 'db_pii_columns' (ID: 3)로, 위험 수준이 HIGH입니다" ✅

### 비교 질문 처리 ("가장 많은", "최대", "어떤 것이 가장")
⚠️ 매우 중요: "가장"이라는 표현은 반드시 하나의 항목만을 의미합니다.

**핵심 원칙:**
- 질문에 "가장", "최대", "최고" 같은 표현이 있으면 반드시 하나의 항목만 답변하세요
- 여러 항목을 나열하지 마세요
- 개수가 같아도 하나만 선택하거나, "다음 항목들이 동일합니다"라고 명시하세요

1. DB 조회 결과의 모든 행을 확인하세요
2. COUNT, 개수, 수치가 포함된 컬럼을 찾으세요 (예: total_piis_count, total_records_count)
3. **파일명, 테이블명, 서버명 등 식별 정보를 확인하세요**
4. 모든 행의 개수를 비교하세요:
   - **개수가 다르면**: 가장 큰 값을 가진 **하나의** 항목만을 "가장" 항목으로 답변
     → "가장 위험한 파일은 '[파일명]'으로, 총 PII 개수가 X개입니다" 형식
     → "가장 위험한 컬럼은 '[컬럼명]' (테이블명: [테이블명])로, X개의 레코드가 있습니다" 형식
   - **모든 행의 개수가 같으면**: "가장"이라는 표현을 사용하지 마세요!
     → "다음 항목들이 각각 X개씩 동일하게 있습니다: [항목1 이름], [항목2 이름]" 형식으로 답변
     → 또는 "가장 위험한 항목은 [항목1 이름]으로, X개입니다 (다른 항목들도 동일한 수치)" 형식
   - **결과가 없으면**: "해당하는 데이터가 없습니다"라고 답변

⚠️ 절대 하지 말 것:
- "가장 위험한 파일은 A, B, C로..." ❌ (가장은 하나만! 여러 개 나열 금지!)
- "가장 위험한 컬럼은 A와 B로..." ❌ (가장은 하나만!)
- "File ID 7" ❌ (파일명 포함 필수!)
- "컬럼 'email_sent'" ❌ (테이블명 포함 필수!)
- "총 10개의 파일이 확인되었습니다" ❌ (가장 질문에는 하나만 답변!)

### 숫자 데이터 처리
- 숫자 데이터는 정확히 그대로 사용하세요
- 반올림하거나 변경하지 마세요
- 추측하지 마세요

### 패턴 분석
- DB 조회 결과의 모든 행을 확인하여 패턴을 파악하세요
- 일부만 보고 판단하지 마세요
- 전체 데이터를 종합적으로 분석하세요
- **각 항목의 이름(파일명, 테이블명 등)을 확인하여 답변에 포함하세요**

## 출력 형식

답변은 다음 형식으로 작성하세요:
1. 핵심 답변 (1-2문장)
2. 상세 정보 (필요시)
3. 출처 명시

⚠️ 마크다운 형식은 절대 사용하지 마세요. 일반 텍스트로만 작성하세요.
- ❌ "**텍스트**" (볼드 표시 금지)
- ❌ "*텍스트*" (이탤릭 표시 금지)
- ❌ "# 제목" (헤더 표시 금지)
- ❌ "`코드`" (코드 블록 금지)
- ✅ "텍스트" (일반 텍스트만 사용)
- ✅ "1. 항목" (번호 목록은 사용 가능)

## 답변 생성

위 단계를 따라 답변을 생성하세요:

## 예시

### 예시 1: DB 조회 결과가 있는 경우
- 질문: "DB서버 이슈랑 파일서버 이슈 합쳐서 몇개야"
- DB 조회 결과: "[DB 조회 성공] 22"
- 올바른 답변: "DB서버 이슈와 파일서버 이슈를 합쳐서 총 22개입니다. (출처: DB 조회 결과)"
- 잘못된 답변: "제공된 자료가 없어 답변할 수 없습니다" ❌

### 예시 2: DB 조회 결과가 없는 경우
- 질문: "어떤 테이블이 있나요?"
- DB 조회 결과: "DB 조회 결과 없음"
- 참고 문서: "참고 문서 없음"
- 올바른 답변: "제공된 자료가 없어 답변할 수 없습니다. 다른 방법으로 확인해주시기 바랍니다."

**중요: "[DB 조회 성공]"이라는 표시가 있으면 반드시 그 결과를 사용하여 답변하세요!**"""


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
        vector_docs = state.get("vector_docs", [])
        generation_retry_count = state.get("generation_retry_count", 0)
        
        # 문서 수가 비정상적으로 많으면 경고 (누적 버그 감지)
        if len(reranked_docs) > 1000:
            logger.warning(f"⚠️ Reranked 문서 수가 비정상적으로 많습니다: {len(reranked_docs)}개 (누적 버그 가능성)")
        if len(vector_docs) > 1000:
            logger.warning(f"⚠️ Vector 문서 수가 비정상적으로 많습니다: {len(vector_docs)}개 (누적 버그 가능성)")
        
        logger.info(f"답변 생성 시작 (재생성 횟수: {generation_retry_count})")
        
        # 재생성 시도인 경우 카운터 증가
        # 주의: 재생성은 check_hallucination에서 근거 부족으로 판단되어 재생성 루프로 돌아온 경우에만 발생
        # check_hallucination에서 재생성 루프로 돌아올 때는 이미 final_answer가 설정되어 있고 generation_retry_count가 증가함
        # 첫 번째 답변 생성 시에는 generation_retry_count가 0이고 final_answer가 빈 문자열이어야 함
        
        existing_answer = state.get("final_answer", "")
        
        # 재생성 판단: generation_retry_count가 0보다 크면 이미 재생성 루프에 진입한 것
        # 또는 final_answer가 있고 generation_retry_count가 0이면 재생성으로 간주
        # 단, load_memory에서 초기화된 경우(빈 문자열)는 제외
        is_regeneration = (
            generation_retry_count > 0 or  # 이미 재생성 루프에 진입한 경우
            (existing_answer and 
             existing_answer.strip() and 
             generation_retry_count == 0 and
             generation_retry_count < MAX_GENERATION_RETRIES and
             len(existing_answer.strip()) > 0)
        )
        
        if is_regeneration and generation_retry_count < MAX_GENERATION_RETRIES:
            new_count = generation_retry_count + 1
            state["generation_retry_count"] = new_count
            logger.info(f"재생성 시도로 카운터 증가: {generation_retry_count} -> {new_count}")
        elif generation_retry_count >= MAX_GENERATION_RETRIES:
            logger.warning(f"재생성 재시도 초과 ({generation_retry_count}/{MAX_GENERATION_RETRIES}), 더 이상 재생성하지 않음")
        else:
            # 첫 번째 답변 생성인 경우 (재생성이 아님)
            # generation_retry_count가 0이고 final_answer가 비어있음
            logger.debug(f"첫 번째 답변 생성 (재생성 아님, retry_count={generation_retry_count})")
        
        # 대화 이력 구성 (최소화: 최근 2개만 - 시연 안정성을 위해)
        recent_history = "\n".join([
            f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
            for msg in messages[-2:]  # 최근 2개만 (현재 질문 + 이전 답변)
        ]) if messages else "대화 이력 없음"
        conversation_history = recent_history  # conversation_summary 사용 안 함
        
        # DB 결과 구성
        # db_result가 실제 데이터를 포함하는지 확인
        if db_result and db_result.strip():
            # 실제 데이터인 경우 (에러 메시지가 아닌 경우)
            error_phrases = ["DB 조회를 수행할 수 없습니다", "DB 조회 결과 없음", "DB 조회 중 오류가 발생했습니다"]
            if not any(phrase in db_result for phrase in error_phrases):
                db_result_text = f"[DB 조회 성공] {db_result}"
            else:
                db_result_text = "DB 조회 결과 없음"
        else:
            db_result_text = "DB 조회 결과 없음"
        
        # 참고 문서 구성 (reranked_docs 우선, 없으면 vector_docs)
        if reranked_docs:
            context_docs = reranked_docs
        else:
            context_docs = state.get("vector_docs", [])
        
        # 참고 문서 구성: 법령 조항의 구체적인 내용을 더 많이 포함하도록 길이 제한 증가
        context_text = "\n\n---\n\n".join([
            f"문서 {i+1} ({doc.get('metadata', {}).get('law_name', '')} {doc.get('metadata', {}).get('article', '')}):\n{doc['content'][:800]}"
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
        
        # 마크다운 제거 (프론트엔드 호환성)
        answer_text = _remove_markdown(answer_text)
        
        # 답변 저장
        state["final_answer"] = answer_text
        
        logger.info(f"답변 생성 완료: {len(state['final_answer'])}자")
        return state
        
    except Exception as e:
        logger.error(f"답변 생성 실패: {str(e)}", exc_info=True)
        state["final_answer"] = "죄송합니다. 답변 생성 중 오류가 발생했습니다."
        return state
