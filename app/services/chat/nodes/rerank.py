"""
Reranker로 재순위화 노드 (rerank)
"""
from app.services.chat.state import ChatbotState
from app.services.chat.retriever import CustomLawRetriever
from app.services.chat.vector_db import VectorDB
from langchain_community.document_compressors import FlashrankRerank
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from app.services.chat.config import RERANK_TOP_N
from app.core.logging import logger


def rerank(state: ChatbotState) -> ChatbotState:
    """
    Reranker로 검색 결과 재순위화
    
    Args:
        state: 현재 상태
    
    Returns:
        업데이트된 상태 (reranked_docs 설정)
    """
    try:
        user_question = state.get("user_question", "")
        vector_docs = state.get("vector_docs", [])
        
        logger.info(f"Rerank 시작: {len(vector_docs)}개 문서")
        
        if not vector_docs:
            logger.warning("Rerank할 문서가 없음")
            state["reranked_docs"] = []
            return state
        
        # VectorDB 및 CustomLawRetriever 사용
        vector_db = VectorDB()
        retriever = CustomLawRetriever(vector_db, n_results=len(vector_docs))
        
        # FlashrankRerank 사용 (regulation_search.py와 동일한 패턴)
        compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=RERANK_TOP_N)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=retriever
        )
        
        # 재순위화된 문서 가져오기
        reranked_docs = compression_retriever.get_relevant_documents(user_question)
        
        # 결과를 state에 저장
        state["reranked_docs"] = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": doc.metadata.get("similarity", 0.0)
            }
            for doc in reranked_docs
        ]
        
        logger.info(f"Rerank 완료: {len(state['reranked_docs'])}개 문서 선별")
        return state
        
    except Exception as e:
        logger.error(f"Rerank 실패: {str(e)}", exc_info=True)
        # 기본값으로 vector_docs 사용
        state["reranked_docs"] = state.get("vector_docs", [])[:RERANK_TOP_N]
        return state
