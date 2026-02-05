"""
AI 어시스턴트 API
"""
from fastapi import APIRouter, Depends
from app.schemas.chat import (
    RegulationSearchRequest,
    RegulationSearchResponse,
    RegulationSearchResult,
    LangGraphChatRequest,
    LangGraphChatResponse,
    RegulationUploadResponse,
    RegulationUploadRequest,
)
from app.api.deps import get_regulation_search, get_langgraph_chatbot, get_regulation_upload
from app.core.logging import logger
from app.core.async_utils import run_in_thread
import time

router = APIRouter()


@router.post("/search-regulations", response_model=RegulationSearchResponse)
async def search_regulations(
    request: RegulationSearchRequest,
    regulation_search=Depends(get_regulation_search),
):
    """법령 검색"""
    logger.info(f"법령 검색: {request.query}")

    results = await run_in_thread(
        regulation_search.respond,
        query=request.query,
        n_results=request.n_results,
        top_n=request.top_n,
    )

    sources = [
        RegulationSearchResult(**source)
        for source in results.get("sources", [])
    ]

    return RegulationSearchResponse(
        answer=results.get("answer", ""),
        sources=sources,
    )


@router.post("/chatbot", response_model=LangGraphChatResponse)
async def chatbot(
    request: LangGraphChatRequest,
    chatbot=Depends(get_langgraph_chatbot),
):
    """LangGraph Self-RAG 챗봇"""
    total_start_time = time.time()
    logger.info(f"[LangGraph 챗봇 시작] 질의: {request.question[:50]}..., conversation_id={request.conversation_id}")

    try:
        # 초기 상태 구성
        config = {
            "configurable": {
                "thread_id": request.conversation_id
            }
        }

        initial_state = {
            "user_question": request.question,
            "conversation_id": request.conversation_id,
            "query_type": "",
            "db_result": None,
            "vector_docs": [],
            "reranked_docs": [],
            "final_answer": "",
            "relevance_score": None,
            "is_relevant": None,
            "hallucination_score": None,
            "is_grounded": None,
            "retry_count": 0,
            "search_query_version": 0,
            "generation_retry_count": 0,
        }

        # LangGraph 실행
        graph_start_time = time.time()
        result = await run_in_thread(
            chatbot.invoke,
            initial_state,
            config
        )
        graph_elapsed = time.time() - graph_start_time
        logger.debug(f"LangGraph 실행 완료 (소요 시간: {graph_elapsed:.2f}초)")

        # 응답 구성
        sources = []
        if result.get("reranked_docs"):
            for doc in result["reranked_docs"]:
                metadata = doc.get("metadata", {})
                law_name = metadata.get("law_name", "")
                article = metadata.get("article", "")
                if law_name or article:
                    sources.append(f"{law_name} {article}".strip())
        elif result.get("vector_docs"):
            for doc in result["vector_docs"]:
                metadata = doc.get("metadata", {})
                law_name = metadata.get("law_name", "")
                article = metadata.get("article", "")
                if law_name or article:
                    sources.append(f"{law_name} {article}".strip())

        total_elapsed = time.time() - total_start_time
        logger.info(f"[LangGraph 챗봇 완료] 총 소요 시간: {total_elapsed:.2f}초, query_type={result.get('query_type', 'general')}, answer_length={len(result.get('final_answer', ''))}자")

        return LangGraphChatResponse(
            answer=result.get("final_answer", ""),
            sources=sources,
        )

    except Exception as e:
        total_elapsed = time.time() - total_start_time
        error_msg = str(e)
        logger.error(f"[LangGraph 챗봇 실패] 총 소요 시간: {total_elapsed:.2f}초, 오류: {error_msg}", exc_info=True)
        
        # 사용자에게는 친화적인 메시지만 표시 (기술적 오류 내용은 숨김)
        user_friendly_message = "죄송합니다. 일시적인 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."
        
        # 특정 오류 타입에 따른 처리
        if "INVALID_CONCURRENT_GRAPH_UPDATE" in error_msg or "messages" in error_msg.lower():
            logger.error("State 충돌 오류 발생 - LangGraph State 업데이트 문제")
        elif "timeout" in error_msg.lower() or "time limit" in error_msg.lower():
            user_friendly_message = "죄송합니다. 처리 시간이 초과되었습니다. 질문을 간단히 다시 작성해 주세요."
        elif "connection" in error_msg.lower() or "network" in error_msg.lower():
            user_friendly_message = "죄송합니다. 네트워크 연결 문제가 발생했습니다. 잠시 후 다시 시도해 주세요."
        
        return LangGraphChatResponse(
            answer=user_friendly_message,
            sources=[],
        )


@router.post("/upload-regulations", response_model=RegulationUploadResponse)
async def upload_pdf(request: RegulationUploadRequest):
    """PDF 파일을 벡터 DB에 저장 (로컬 파일 또는 S3 URL 지원)"""
    logger.info(f"법규 PDF 임베딩 요청: {request.file_path}")

    try:
        processor = get_regulation_upload()
        await processor(request.file_path)

        logger.info(f"법규 PDF 임베딩 완료: {request.file_path}")
        return RegulationUploadResponse(
            success=True,
            message="임베딩 완료"
        )

    except Exception as e:
        logger.error(f"법규 PDF 임베딩 실패: {str(e)}", exc_info=True)
        return RegulationUploadResponse(
            success=False,
            message=f"임베딩 실패: {str(e)}"
        )