"""
법령 데이터베이스 Custom Retriever
regulation_search.py의 CustomLawRetriever를 별도 파일로 분리하여 재사용
"""
from app.services.chat.vector_db import VectorDB
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pydantic import Field
from typing import List
from app.core.logging import logger


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
