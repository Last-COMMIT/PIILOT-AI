"""
Reranker로 재순위화 노드 (rerank)
"""
from app.services.chat.state import ChatbotState
from app.services.chat.retriever import CustomLawRetriever
from app.services.chat.vector_db import VectorDB
from langchain_community.document_compressors import FlashrankRerank
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from app.services.chat.config import RERANK_TOP_N, VECTOR_SEARCH_K
from app.core.logging import logger
from app.core.model_manager import ModelManager

# Flashrank Ranker import
try:
    from flashrank import Ranker
except ImportError:
    Ranker = None

# ── Flashrank 싱글톤 캐싱 (메모리 절약: 매번 ~120MB 재할당 방지) ──
_cached_ranker = None
_cached_compressor = None


def _get_compressor() -> FlashrankRerank:
    """Flashrank compressor 싱글톤 반환 (최초 1회만 생성)"""
    global _cached_ranker, _cached_compressor
    if _cached_compressor is not None:
        logger.debug("캐싱된 FlashrankRerank 재사용")
        return _cached_compressor

    ModelManager.setup_cache_dir()
    flashrank_cache_dir = ModelManager.get_flashrank_cache_dir()
    logger.debug(f"Flashrank 캐시 디렉토리: {flashrank_cache_dir}")

    try:
        if Ranker is not None:
            logger.info("Flashrank Ranker 최초 생성 중... (model=ms-marco-MiniLM-L-12-v2)")
            _cached_ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir=flashrank_cache_dir)
            _cached_compressor = FlashrankRerank(client=_cached_ranker, top_n=RERANK_TOP_N)
            logger.info("✓ FlashrankRerank 싱글톤 생성 완료 (Ranker 사용)")
        else:
            logger.info("Ranker 클래스 없음, 기본 방식으로 FlashrankRerank 생성 중...")
            _cached_compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=RERANK_TOP_N)
            logger.info("✓ FlashrankRerank 싱글톤 생성 완료 (기본 방식)")
    except Exception as e:
        logger.warning(f"Flashrank Ranker 생성 실패, 기본 방식 사용: {e}", exc_info=True)
        _cached_compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=RERANK_TOP_N)
        logger.info("✓ FlashrankRerank 싱글톤 생성 완료 (fallback)")

    return _cached_compressor


def rerank(state: ChatbotState) -> ChatbotState:
    """
    Reranker로 검색 결과 재순위화
    
    Args:
        state: 현재 상태
    
    Returns:
        업데이트된 상태 (reranked_docs 설정)
    """
    # 노드 진입 즉시 로그 (디버깅용)
    logger.info("=== Rerank 노드 진입 ===")
    try:
        user_question = state.get("user_question", "")
        vector_docs = state.get("vector_docs", [])
        
        logger.info(f"Rerank 시작: {len(vector_docs)}개 문서, 질문: {user_question[:50]}...")
        
        if not vector_docs:
            logger.warning("Rerank할 문서가 없음")
            state["reranked_docs"] = []
            return state
        
        # 이미 검색된 vector_docs를 Document 객체로 변환
        from langchain_core.documents import Document
        documents = [
            Document(
                page_content=doc.get("content", ""),
                metadata=doc.get("metadata", {})
            )
            for doc in vector_docs
        ]
        
        # 타임아웃 방지를 위해 문서 수가 너무 많으면 제한 (VECTOR_SEARCH_K의 2배까지만 허용)
        max_docs_for_rerank = min(len(documents), VECTOR_SEARCH_K * 2)
        if len(documents) > max_docs_for_rerank:
            logger.warning(f"문서 수가 많아 rerank 대상 제한: {len(documents)}개 -> {max_docs_for_rerank}개")
            documents = documents[:max_docs_for_rerank]
        
        # 싱글톤 compressor 가져오기 (최초 1회만 모델 로드)
        compressor = _get_compressor()
        
        logger.debug(f"Rerank 실행 중: {len(documents)}개 문서 -> 최대 {RERANK_TOP_N}개 선별")
        try:
            # FlashrankRerank의 compress_documents 메서드를 직접 사용하여 이미 검색된 문서를 rerank
            logger.debug(f"compress_documents 호출 시작 (문서 수: {len(documents)})")
            reranked_docs = compressor.compress_documents(documents, user_question)
            logger.debug(f"compress_documents 호출 완료 (결과 수: {len(reranked_docs)})")
        except Exception as rerank_error:
            error_msg = str(rerank_error)
            logger.error(f"Rerank 실행 중 오류 발생: {error_msg[:200]}", exc_info=True)
            # Rerank 실패 시 상위 문서만 반환
            logger.warning(f"Rerank 실패, 상위 {RERANK_TOP_N}개 문서만 반환")
            reranked_docs = documents[:RERANK_TOP_N]

        # Flashrank의 relevance_score를 float로 변환 (Flashrank 모델 실행을 위해 필요)
        for doc in reranked_docs:
            if 'relevance_score' in doc.metadata:
                doc.metadata['relevance_score'] = float(doc.metadata['relevance_score'])
        
        # 결과를 state에 저장 (reduce_replace_list reducer가 완전 교체 보장)
        state["reranked_docs"] = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": doc.metadata.get("similarity", 0.0)
            }
            for doc in reranked_docs
        ]
        
        logger.info(f"Rerank 완료: {len(state['reranked_docs'])}개 문서 선별")
        logger.debug("=== Rerank 노드 종료 ===")
        return state
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Rerank 실패: {error_msg}", exc_info=True)
        logger.debug("=== Rerank 노드 예외 종료 ===")
        # 기본값으로 vector_docs 사용
        state["reranked_docs"] = state.get("vector_docs", [])[:RERANK_TOP_N]
        return state
