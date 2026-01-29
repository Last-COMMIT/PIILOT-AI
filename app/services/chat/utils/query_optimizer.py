"""
검색 쿼리 최적화 유틸리티 (Self-RAG 재시도용)
"""
from app.core.logging import logger


def optimize_query(query: str, version: int) -> str:
    """
    검색 쿼리를 개선하여 재시도 시 다른 결과를 얻도록 함
    
    Args:
        query: 원본 쿼리
        version: 쿼리 버전 (1, 2, 3...)
    
    Returns:
        개선된 쿼리
    """
    if version == 0:
        return query
    
    # 버전에 따라 다른 개선 전략 적용
    if version == 1:
        # 동의어 추가, 더 구체적인 표현
        optimized = f"{query} 관련 법령 조항"
        logger.debug(f"쿼리 개선 (v1): '{query}' -> '{optimized}'")
        return optimized
    elif version == 2:
        # 상위 개념 추가
        optimized = f"{query} 법적 근거 및 규정"
        logger.debug(f"쿼리 개선 (v2): '{query}' -> '{optimized}'")
        return optimized
    else:
        # 관련 키워드 확장
        optimized = f"{query} 법률 규정 법령"
        logger.debug(f"쿼리 개선 (v3+): '{query}' -> '{optimized}'")
        return optimized
