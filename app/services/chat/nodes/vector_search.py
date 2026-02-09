"""
벡터 검색 노드 (vector_search)
"""
from app.services.chat.state import ChatbotState
from app.services.chat.retriever import CustomLawRetriever
from app.services.chat.vector_db import VectorDB
from app.services.chat.utils.query_optimizer import optimize_query
from app.services.chat.config import VECTOR_SEARCH_K, MAX_SEARCH_RETRIES
from app.core.logging import logger

# ── VectorDB 싱글톤 캐싱 (매번 인스턴스 재생성 방지) ──
_cached_vector_db = None


def _get_vector_db() -> VectorDB:
    """VectorDB 싱글톤 반환 (최초 1회만 생성)"""
    global _cached_vector_db
    if _cached_vector_db is None:
        _cached_vector_db = VectorDB()
        logger.info("✓ VectorDB 싱글톤 생성 완료")
    return _cached_vector_db


def vector_search(state: ChatbotState) -> ChatbotState:
    """
    pgvector로 법령/내규 검색
    
    Args:
        state: 현재 상태
    
    Returns:
        업데이트된 상태 (vector_docs 설정)
    """
    try:
        user_question = state.get("user_question", "")
        search_query_version = state.get("search_query_version", 0)
        retry_count = state.get("retry_count", 0)
        
        # 재검색 루프인 경우 카운터 및 버전 증가
        if retry_count > 0:
            search_query_version = retry_count
            logger.info(f"벡터 재검색 시작: {user_question[:50]}..., version={search_query_version}, retry={retry_count}")
        else:
            logger.info(f"벡터 검색 시작: {user_question[:50]}..., version={search_query_version}")
        
        # 재시도 시 쿼리 개선 및 카운터 증가
        query = user_question
        
        # 재검색 루프인 경우: check_relevance에서 재검색이 필요하다고 판단된 경우
        # vector_docs가 이미 있고 비어있지 않으면 재검색으로 간주하여 카운터 증가
        existing_docs = state.get("vector_docs", [])
        if existing_docs and len(existing_docs) > 0 and retry_count < MAX_SEARCH_RETRIES:
            # 재검색 시도: 카운터 증가 및 쿼리 개선
            retry_count = retry_count + 1
            state["retry_count"] = retry_count
            state["search_query_version"] = retry_count
            query = optimize_query(user_question, retry_count)
            logger.debug(f"재검색 쿼리 개선 적용: '{user_question}' -> '{query}' (retry={retry_count})")
        elif search_query_version > 0:
            # 초기 검색이지만 버전이 있는 경우 (이전 재시도에서 설정됨)
            query = optimize_query(user_question, search_query_version)
            logger.debug(f"쿼리 개선 적용: '{user_question}' -> '{query}'")
        
        # VectorDB 싱글톤 및 CustomLawRetriever 사용 (Hybrid Search 활성화)
        vector_db = _get_vector_db()
        retriever = CustomLawRetriever(vector_db, n_results=VECTOR_SEARCH_K)
        
        # Hybrid Search 사용 (기본값: True)
        use_hybrid = True
        
        # 검색 실행 (Hybrid Search)
        docs = retriever._get_relevant_documents(query, use_hybrid=use_hybrid)
        
        # 결과를 state에 저장 (reduce_replace_list reducer가 완전 교체 보장)
        state["vector_docs"] = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": doc.metadata.get("similarity", 0.0)
            }
            for doc in docs
        ]
        
        logger.info(f"벡터 검색 완료: {len(state['vector_docs'])}개 문서 발견")
        return state
        
    except Exception as e:
        logger.error(f"벡터 검색 실패: {str(e)}", exc_info=True)
        state["vector_docs"] = []
        return state
