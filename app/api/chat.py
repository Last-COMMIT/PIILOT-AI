"""
AI 어시스턴트 API
"""
from fastapi import APIRouter, Depends
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    RegulationSearchRequest,
    RegulationSearchResponse,
    RegulationSearchResult,
    LangGraphChatRequest,
    LangGraphChatResponse,
    RegulationUploadResponse,
    RegulationUploadRequest,
)
from app.api.deps import get_assistant, get_regulation_search, get_langgraph_chatbot, get_regulation_upload
from app.core.logging import logger
import time

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    assistant=Depends(get_assistant),
):
    """자연어 질의응답"""
    logger.info(f"AI 어시스턴트 질의: {request.query}")

    result = await assistant.chat(
        query=request.query,
        context=request.context,
    )

    return ChatResponse(
        answer=result.get("answer", ""),
        sources=result.get("sources", []),
    )


@router.post("/search-regulations", response_model=RegulationSearchResponse)
async def search_regulations(
    request: RegulationSearchRequest,
    regulation_search=Depends(get_regulation_search),
):
    """법령 검색"""
    logger.info(f"법령 검색: {request.query}")

    results = regulation_search.respond(
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


@router.post("/chat/langgraph", response_model=LangGraphChatResponse)
async def langgraph_chat(
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
            "messages": [],
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
        result = chatbot.invoke(initial_state, config)
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
            query_type=result.get("query_type", "general"),
            relevance_score=result.get("relevance_score"),
            hallucination_score=result.get("hallucination_score"),
        )

    except Exception as e:
        total_elapsed = time.time() - total_start_time
        logger.error(f"[LangGraph 챗봇 실패] 총 소요 시간: {total_elapsed:.2f}초, 오류: {str(e)}", exc_info=True)
        return LangGraphChatResponse(
            answer=f"죄송합니다. 챗봇 실행 중 오류가 발생했습니다: {str(e)}",
            sources=[],
            query_type="general",
            relevance_score=None,
            hallucination_score=None,
        )


@router.post("/upload-regulations", response_model=RegulationUploadResponse)
async def upload_pdf(request: RegulationUploadRequest):
    """PDF 파일을 벡터 DB에 저장"""
    logger.info(f"PDF 처리 요청: {request.file_path}")
    
    try:
        processor = get_regulation_upload()
        await processor(request.file_path)
        
        logger.info(f"PDF 처리 완료: {request.file_path}")
        return RegulationUploadResponse(
            status="PDF 파일이 성공적으로 벡터 DB에 저장되었습니다."
        )
    
    except Exception as e:
        logger.error(f"PDF 처리 중 오류 발생: {str(e)}", exc_info=True)
        return RegulationUploadResponse(
            status=f"PDF 처리 중 오류가 발생했습니다: {str(e)}"
        )