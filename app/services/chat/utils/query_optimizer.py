"""
검색 쿼리 최적화 유틸리티 (Self-RAG 재시도용)
LLM 기반 지능형 쿼리 확장
"""
from app.core.logging import logger
from app.services.chat.utils.llm_client import get_evaluation_llm
from langchain_core.prompts import PromptTemplate


QUERY_EXPANSION_PROMPT = """당신은 검색 쿼리 최적화 전문가입니다. 사용자의 질문을 더 효과적인 검색 쿼리로 확장하세요.

원본 질문: {query}
재시도 버전: {version}

지시사항:
1. 법령 검색에 유용한 동의어, 관련 용어, 상위/하위 개념을 추가하세요
2. 법령 용어, 조항 번호, 관련 키워드를 포함하세요
3. 원본 질문의 의미를 유지하면서 검색 범위를 확장하세요
4. 불필요한 단어는 추가하지 마세요

예시:
- "개인정보" → "개인정보 개인 데이터 PII 민감정보"
- "암호화 규정" → "암호화 규정 암호화 의무 개인정보 암호화"
- "보관 기간" → "보관 기간 보관 기한 보존 기간"

확장된 쿼리만 출력하세요 (설명 없이):"""


def optimize_query(query: str, version: int) -> str:
    """
    LLM 기반 지능형 쿼리 확장 (Self-RAG 재시도용)
    
    Args:
        query: 원본 쿼리
        version: 쿼리 버전 (0=원본, 1+=확장)
    
    Returns:
        개선된 쿼리
    """
    if version == 0:
        return query
    
    try:
        # LLM 기반 쿼리 확장
        prompt = PromptTemplate(
            template=QUERY_EXPANSION_PROMPT,
            input_variables=["query", "version"]
        )
        
        llm = get_evaluation_llm()
        formatted_prompt = prompt.format(query=query, version=version)
        response = llm.invoke(formatted_prompt)
        
        expanded_query = response.content.strip()
        
        # 시스템 프롬프트 토큰 제거
        if "<|start_header_id|>assistant<|end_header_id|>" in expanded_query:
            parts = expanded_query.split("<|start_header_id|>assistant<|end_header_id|>")
            if len(parts) > 1:
                expanded_query = parts[-1].strip()
        
        expanded_query = expanded_query.replace("<|eot_id|>", "").strip()
        
        # 결과가 비어있거나 원본과 같으면 fallback
        if not expanded_query or expanded_query == query:
            logger.debug(f"LLM 쿼리 확장 실패, 기본 전략 사용")
            return _fallback_expand_query(query, version)
        
        logger.debug(f"쿼리 확장 (v{version}): '{query}' -> '{expanded_query}'")
        return expanded_query
        
    except Exception as e:
        logger.warning(f"LLM 쿼리 확장 실패: {e}, 기본 전략 사용")
        return _fallback_expand_query(query, version)


def _fallback_expand_query(query: str, version: int) -> str:
    """
    기본 쿼리 확장 전략 (LLM 실패 시 fallback)
    """
    if version == 1:
        optimized = f"{query} 관련 법령 조항"
    elif version == 2:
        optimized = f"{query} 법적 근거 및 규정"
    else:
        optimized = f"{query} 법률 규정 법령"
    
    logger.debug(f"기본 쿼리 확장 (v{version}): '{query}' -> '{optimized}'")
    return optimized
