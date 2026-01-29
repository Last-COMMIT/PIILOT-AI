from app.services.chat.vector_db import VectorDB
from app.services.chat.utils.llm_client import get_llm
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_classic.chains import RetrievalQA
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from app.utils.logger import logger
from typing import List, Dict
from pydantic import Field
import time

# FlashrankRerank의 Pydantic 순환 참조 문제 해결을 위해 Ranker를 먼저 import
try:
    from flashrank import Ranker
    # FlashrankRerank 모델 재빌드 (Pydantic v2 순환 참조 해결)
    FlashrankRerank.model_rebuild()
except ImportError:
    # flashrank가 없어도 FlashrankRerank는 사용 가능하므로 경고만
    logger.warning("flashrank 패키지가 설치되지 않았습니다. FlashrankRerank 사용 시 문제가 발생할 수 있습니다.")


class CustomLawRetriever(BaseRetriever):
    """법령 데이터베이스에서 검색하는 Custom Retriever"""
    
    vector_db: VectorDB = Field(exclude=True)  # Pydantic 필드로 선언 (직렬화 제외)
    n_results: int = 10
    
    def __init__(self, vector_db: VectorDB, n_results: int = 10):
        super().__init__(vector_db=vector_db, n_results=n_results)
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """질문에 대한 유사한 법령 청크 검색"""
        logger.debug(f"CustomLawRetriever: Vector DB 검색 시작 (n_results={self.n_results})")
        results = self.vector_db.search(query, n_results=self.n_results)
        logger.debug(f"CustomLawRetriever: Vector DB 검색 완료 ({len(results)}개 결과)")
        
        # 검색 결과를 Document 객체 리스트로 변환
        logger.debug("검색 결과를 Document 객체로 변환 중...")
        docs = []
        for result in results:
            # result: {"id": str, "text": str, "metadata": Dict, "distance": float}
            metadata = result.get("metadata", {})
            docs.append(Document(
                page_content=result.get("text", ""),
                metadata={
                    'id': result.get("id"),
                    'law_name': metadata.get("law_name"),
                    'article': metadata.get("article"),
                    'page': metadata.get("page"),
                    'similarity': 1.0 - result.get("distance", 0.0)  # distance를 similarity로 변환
                }
            ))
        logger.debug(f"✓ Document 객체 변환 완료 ({len(docs)}개)")
        return docs
    
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """비동기 버전 (필수 구현)"""
        return self._get_relevant_documents(query)


class RegulationSearch:
    """법령 검색 응답 생성"""
    
    # Custom Prompt Template
    PROMPT_TEMPLATE = """다음 법령 조항을 참고하여 질문에 답변하세요.

    {context}

    질문: {question}

    법령을 나열하지 말고, 적절히 요약하여 답변하세요.
    참고 문항은 하단에 dictionary 형식으로 제공하세요.

    답변:
    """
    
    def __init__(self):
        init_start_time = time.time()
        logger.info("RegulationSearch 초기화 시작")
        
        # VectorDB 초기화
        logger.info("VectorDB 초기화 중...")
        self.vector_db = VectorDB()
        logger.info("✓ VectorDB 초기화 완료")
        
        # LLM 초기화 (공통 llm_client 사용 → KT 믿음 / GPT API 설정에 따라 자동 선택)
        llm_start_time = time.time()
        self.llm = get_llm(temperature=0.001, max_new_tokens=512)
        llm_load_time = time.time() - llm_start_time
        logger.info(f"✓ LLM 초기화 완료 (소요 시간: {llm_load_time:.2f}초)")
        
        # Prompt Template 생성
        logger.debug("Prompt Template 생성 중...")
        self.prompt = PromptTemplate(
            template=self.PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )
        logger.debug("✓ Prompt Template 생성 완료")
        
        # Retriever는 respond()에서 동적으로 생성
        
        total_init_time = time.time() - init_start_time
        logger.info(f"✓ RegulationSearch 초기화 완료 (총 소요 시간: {total_init_time:.2f}초)")
    
    def respond(self, query: str, n_results: int = 10, top_n: int = 5) -> Dict:
        """
        법령 검색 기반 답변 생성
        
        Args:
            query: 사용자 질문
            n_results: 검색할 결과 수 (기본값: 10)
            top_n: Reranking 후 상위 N개 문서 선택 (기본값: 5)
        
        Returns:
            {
                "answer": str,  # 생성된 답변
                "sources": List[Dict]  # 참고 문서 리스트
            }
        """
        try:
            total_start_time = time.time()
            logger.info(f"[법령 검색 시작] 질의: {query}, n_results={n_results}, top_n={top_n}")
            
            # Retriever 생성 (n_results에 따라 동적으로 생성)
            logger.debug("CustomLawRetriever 생성 중...")
            retriever = CustomLawRetriever(self.vector_db, n_results=n_results)
            logger.debug("✓ CustomLawRetriever 생성 완료")
            
            # Compressor 및 Compression Retriever 생성 (top_n에 따라 동적으로 생성)
            logger.debug(f"FlashrankRerank Compressor 생성 중 (top_n={top_n})...")
            compressor_start_time = time.time()
            compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=top_n)
            compressor_time = time.time() - compressor_start_time
            logger.debug(f"✓ FlashrankRerank Compressor 생성 완료 (소요 시간: {compressor_time:.2f}초)")
            
            logger.debug("ContextualCompressionRetriever 생성 중...")
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=retriever
            )
            logger.debug("✓ ContextualCompressionRetriever 생성 완료")
            
            # RetrievalQA Chain 생성 (매번 새로 생성)
            logger.debug("RetrievalQA Chain 생성 중...")
            chain_start_time = time.time()
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=compression_retriever,
                chain_type_kwargs={"prompt": self.prompt},
                return_source_documents=True
            )
            chain_time = time.time() - chain_start_time
            logger.debug(f"✓ RetrievalQA Chain 생성 완료 (소요 시간: {chain_time:.2f}초)")
            
            # RetrievalQA Chain 실행
            logger.info("RetrievalQA Chain 실행 중... (이 단계에서 시간이 오래 걸릴 수 있습니다)")
            qa_start_time = time.time()
            result = qa_chain({"query": query})
            qa_time = time.time() - qa_start_time
            logger.info(f"✓ RetrievalQA Chain 실행 완료 (소요 시간: {qa_time:.2f}초)")
            
            # 소스 문서 정렬 (유사도 기준)
            logger.debug("소스 문서 정렬 중...")
            sorted_sources = sorted(
                result['source_documents'],
                key=lambda x: x.metadata.get('similarity', 0.0),
                reverse=True
            )
            logger.debug(f"✓ 소스 문서 정렬 완료 ({len(sorted_sources)}개 문서)")
            
            # 소스 문서 포맷팅 (RegulationSearchResult 형식에 맞게)
            logger.debug("소스 문서 포맷팅 중...")
            sources = []
            for doc in sorted_sources:
                sources.append({
                    "document_title": doc.metadata.get('law_name', ''),
                    "content": doc.page_content,
                    "article": doc.metadata.get('article', ''),
                    "page": str(doc.metadata.get('page', '')),
                    "similarity": doc.metadata.get('similarity', 0.0)
                })
            logger.debug(f"✓ 소스 문서 포맷팅 완료 ({len(sources)}개)")
            
            total_time = time.time() - total_start_time
            logger.info(f"[법령 검색 완료] 총 소요 시간: {total_time:.2f}초")
            
            return {
                "answer": result['result'],
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"법령 검색 응답 생성 오류: {str(e)}", exc_info=True)
            return {
                "answer": "죄송합니다. 답변 생성 중 오류가 발생했습니다.",
                "sources": []
            }
