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
)
from app.api.deps import get_assistant, get_regulation_search
from app.core.logging import logger

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
