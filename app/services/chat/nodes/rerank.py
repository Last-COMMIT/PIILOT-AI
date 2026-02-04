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
from app.core.model_manager import ModelManager

# Flashrank Ranker import
try:
    from flashrank import Ranker
except ImportError:
    Ranker = None


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
        
        # Flashrank 캐시 디렉토리 설정
        ModelManager.setup_cache_dir()
        flashrank_cache_dir = ModelManager.get_flashrank_cache_dir()
        
        # FlashrankRerank 사용 (cache_dir 지정)
        try:
            if Ranker is not None:
                ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir=flashrank_cache_dir)
                compressor = FlashrankRerank(client=ranker, top_n=RERANK_TOP_N)
            else:
                compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=RERANK_TOP_N)
        except Exception as e:
            # Ranker 생성 실패 시 기본 방식으로 fallback
            logger.warning(f"Flashrank Ranker 생성 실패, 기본 방식 사용: {e}")
            compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=RERANK_TOP_N)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=retriever
        )
        
        # 재순위화된 문서 가져오기 (LangChain 0.1.0+ 버전에서는 get_relevant_documents() 메서드가 invoke()로 변경)
        reranked_docs = compression_retriever.invoke(user_question)

        # Flashrank의 relevance_score를 float로 변환 (Flashrank 모델 실행을 위해 필요)
        for doc in reranked_docs:
            if 'relevance_score' in doc.metadata:
                doc.metadata['relevance_score'] = float(doc.metadata['relevance_score'])
        
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
