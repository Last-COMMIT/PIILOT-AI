"""
Structured Memory (Entity Memory) 노드
엔티티와 사실을 추출하여 구조화된 메모리로 저장
"""
from app.services.chat.state import ChatbotState
from app.services.chat.utils.llm_client import get_evaluation_llm
from app.core.logging import logger
from langchain_core.prompts import PromptTemplate
import json
import re


ENTITY_EXTRACTION_PROMPT = """다음 대화에서 중요한 엔티티와 사실을 추출하세요.

사용자 질문: {question}
생성된 답변: {answer}

엔티티 타입:
- DB_SERVER: 데이터베이스 서버명 (예: "test2", "ttest_db")
- FILE: 파일명 (예: "원천징수.pdf", "test_video5.mp4")
- TABLE: 테이블명 (예: "db_pii_columns", "file_pii")
- COLUMN: 컬럼명 (예: "email_sent", "call_phone")
- LAW: 법령명 (예: "개인정보보호법")
- NUMBER: 중요한 숫자 (예: "17개", "11개")

사실: 질문-답변 쌍에서 확인된 사실 (예: "test2 서버에 17개 이슈가 있음")

응답 형식 (JSON):
{{
    "entities": {{
        "DB_SERVER": ["서버1", "서버2"],
        "FILE": ["파일1", "파일2"],
        "TABLE": ["테이블1"],
        "COLUMN": ["컬럼1"],
        "LAW": ["법령1"],
        "NUMBER": ["17", "11"]
    }},
    "facts": [
        "사실1",
        "사실2"
    ]
}}

JSON만 출력하세요 (설명 없이):"""


def extract_entities(state: ChatbotState) -> ChatbotState:
    """
    엔티티와 사실 추출하여 구조화된 메모리로 저장
    
    Args:
        state: 현재 상태
    
    Returns:
        업데이트된 상태 (entities, facts 설정)
    """
    try:
        user_question = state.get("user_question", "")
        final_answer = state.get("final_answer", "")
        
        if not user_question or not final_answer:
            logger.debug("질문 또는 답변이 없어 엔티티 추출 스킵")
            return state
        
        logger.info("엔티티 및 사실 추출 시작")
        
        # 프롬프트 생성
        prompt = PromptTemplate(
            template=ENTITY_EXTRACTION_PROMPT,
            input_variables=["question", "answer"]
        )
        
        # LLM 호출
        llm = get_evaluation_llm()
        formatted_prompt = prompt.format(
            question=user_question,
            answer=final_answer
        )
        response = llm.invoke(formatted_prompt)
        
        # 응답 파싱
        response_text = response.content.strip()
        
        # 시스템 프롬프트 토큰 제거
        if "<|start_header_id|>assistant<|end_header_id|>" in response_text:
            parts = response_text.split("<|start_header_id|>assistant<|end_header_id|>")
            if len(parts) > 1:
                response_text = parts[-1].strip()
        
        response_text = response_text.replace("<|eot_id|>", "").strip()
        
        # JSON 추출 (코드 블록 제거)
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            json_str = json_match.group(0)
        else:
            json_str = response_text
        
        # JSON 파싱
        try:
            extracted_data = json.loads(json_str)
            entities = extracted_data.get("entities", {})
            facts = extracted_data.get("facts", [])
            
            # 기존 엔티티와 병합
            existing_entities = state.get("entities", {})
            if existing_entities:
                for entity_type, values in entities.items():
                    if entity_type in existing_entities:
                        # 중복 제거하면서 병합
                        existing_entities[entity_type] = list(set(existing_entities[entity_type] + values))
                    else:
                        existing_entities[entity_type] = values
                entities = existing_entities
            
            # 기존 사실과 병합
            existing_facts = state.get("facts", [])
            if existing_facts:
                facts = list(set(existing_facts + facts))
            
            state["entities"] = entities
            state["facts"] = facts
            
            logger.info(f"엔티티 추출 완료: {len(entities)}개 타입, {len(facts)}개 사실")
            
        except json.JSONDecodeError as e:
            logger.warning(f"엔티티 추출 JSON 파싱 실패: {e}, 기본값 사용")
            state["entities"] = state.get("entities", {})
            state["facts"] = state.get("facts", [])
        
        return state
        
    except Exception as e:
        logger.error(f"엔티티 추출 실패: {str(e)}", exc_info=True)
        # 기본값 유지
        state["entities"] = state.get("entities", {})
        state["facts"] = state.get("facts", [])
        return state
