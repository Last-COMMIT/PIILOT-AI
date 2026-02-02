from app.services.chat.vector_db import VectorDB
from app.services.chat.utils.llm_client import get_llm
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_classic.chains import RetrievalQA
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from app.utils.logger import logger
from app.core.model_manager import ModelManager
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
    PROMPT_TEMPLATE = """
    # 역할 및 목적
    당신은 기업의 개인정보 관리 실무자를 지원하는 법률 정보 어시스턴트입니다.
    개인정보보호법, 정보통신망법 등 관련 법령과 규정에 대한 질문에 답변합니다.

    # 핵심 원칙
    1. 검색된 문서 기반 답변: 반드시 제공된 검색 결과(Vector DB에서 조회된 내용)만을 기반으로 답변하세요.
    2. 출처 명시: 법령, 조항, 가이드라인 등의 출처를 명확히 밝히세요.
    3. 정보 부족 시 솔직하게: 검색 결과에 관련 정보가 없으면 "제공된 자료에서 관련 내용을 찾을 수 없습니다"라고 답하세요.
    4. 추측 금지: 확실하지 않은 내용은 절대 추측하거나 일반적인 지식으로 보완하지 마세요.

    답변 구조
    다음 번호 체계로 답변하세요:

    1. 핵심 답변
    질문에 대한 직접적인 답 (1-2문장)

    2. 관련 법령 및 조항
    법령명, 조항, 조문 내용

    3. 상세 설명
    법령의 의미와 적용 방법

    4. 실무 적용 시 주의사항
    고려해야 할 점

    5. 출처
    참조한 법령

    1. 핵심 답변
    - 질문에 대한 직접적인 답을 1-2문장으로 요약

    2. 관련 법령 및 조항
    - 해당하는 법령명과 조항 번호 명시
    - 핵심 조문 내용 인용 (원문 그대로 또는 요약)
    - 예시: "개인정보보호법 제15조 제1항에 따르면..."

    3. 상세 설명
    - 법령의 의미와 적용 방법을 알기 쉽게 설명
    - 법률 용어가 있다면 쉬운 말로 풀어서 설명

    4. 실무 적용 시 주의사항
    - 실제 업무에 적용할 때 고려해야 할 점
    - 흔히 발생하는 오해나 실수 (검색 결과에 있는 경우)

     5. 출처
    - 참조한 법령, 가이드라인, 판례 등을 명시

    답변 방식 가이드
    - 명확성: 전문적이되 이해하기 쉬운 언어를 사용하세요
    - 간결성: 불필요한 반복을 피하고 핵심만 전달하세요
    - 객관성: 법령과 검색된 자료의 내용을 객관적으로 전달하세요
    - 친절성: 질문자가 실무자임을 고려해 실용적인 정보를 제공하세요

    제약사항 및 면책
    답변 마지막에 다음 중 적절한 면책 문구를 포함하세요:

    - 법률 해석이 복잡하거나 여러 해석이 가능한 경우:
    "이 답변은 참고용 정보이며, 구체적인 사안에 대해서는 법무팀 또는 전문 변호사와 상담하시기 바랍니다."

    - 최신성이 중요한 경우:
    "법령은 수시로 개정될 수 있으므로, 최신 내용은 국가법령정보센터(www.law.go.kr)에서 확인하시기 바랍니다."

    - 검색 결과가 불충분한 경우:
    "제공된 자료에서 관련 내용을 찾을 수 없습니다. 더 구체적인 질문이나 추가 키워드로 다시 질문해주시거나, 개인정보보호위원회(www.pipc.go.kr)에 문의하시기 바랍니다."

    금지사항
    - 검색 결과에 없는 내용을 임의로 생성하거나 추측하지 마세요
    - 개별 사안에 대한 법적 판단이나 자문을 제공하지 마세요
    - 불확실한 정보를 확정적으로 제시하지 마세요
    - 검색 결과를 왜곡하거나 과장하지 마세요
    - 마크다운 기호 사용 금지
    - 번호와 제목만 사용

    특수 상황 처리
    1. 상충하는 정보: 검색 결과에 서로 다른 정보가 있으면 모두 제시하고 출처를 구분하세요
    2. 개정 정보: 법령 개정 내용이 있으면 시행일과 함께 명시하세요
    3. 판례: 판례 정보가 있으면 사건번호와 판결 요지를 포함하세요

    이제 사용자의 질문에 답변할 준비가 되었습니다.
    
    다음 법령 조항을 참고하여 질문에 답변하세요.

    {context}

    질문: {question}

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
            
            # Flashrank 캐시 디렉토리 설정
            ModelManager.setup_cache_dir()
            flashrank_cache_dir = ModelManager.get_flashrank_cache_dir()
            logger.debug(f"Flashrank 캐시 디렉토리: {flashrank_cache_dir}")
            
            # flashrank.Ranker를 직접 생성하여 cache_dir 지정
            try:
                ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir=flashrank_cache_dir)
                compressor = FlashrankRerank(client=ranker, top_n=top_n)
            except Exception as e:
                # Ranker 생성 실패 시 기본 방식으로 fallback
                logger.warning(f"Flashrank Ranker 생성 실패, 기본 방식 사용: {e}")
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
