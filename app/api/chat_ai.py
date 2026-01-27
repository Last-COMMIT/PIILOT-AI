"""
AI 어시스턴트 API
"""
from fastapi import APIRouter, HTTPException
from app.models.request import ChatRequest, RegulationSearchRequest
from app.models.response import ChatResponse, RegulationSearchResponse
from app.services.chat.assistant import AIAssistant
from app.services.chat.regulation_search import RegulationSearch
from app.utils.logger import logger

router = APIRouter()

# 서비스 인스턴스
assistant = AIAssistant()
regulation_search = RegulationSearch()

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    자연어 질의응답
    
    Spring Boot에서 질의와 컨텍스트를 전달받아
    LLM + LangChain으로 응답 생성
    """
    try:
        logger.info(f"AI 어시스턴트 질의: {request.query}")
        
        result = await assistant.chat(
            query=request.query,
            context=request.context
        )
        
        return ChatResponse(
            answer=result.get("answer", ""),
            sources=result.get("sources", [])
        )
    
    except Exception as e:
        logger.error(f"AI 어시스턴트 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search-regulations", response_model=RegulationSearchResponse)
async def search_regulations(request: RegulationSearchRequest):
    """
    법령 검색
    
    Spring Boot에서 검색 쿼리를 전달받아
    Vector DB에서 관련 법령 검색
    """
    try:
        logger.info(f"법령 검색: {request.query}")
        
        results = regulation_search.respond(
            query=request.query,
            n_results=request.n_results,
            top_n=request.top_n
        )
        
        # RegulationSearchResult 리스트로 변환
        from app.models.response import RegulationSearchResult
        sources = [
            RegulationSearchResult(**source) 
            for source in results.get("sources", [])
        ]
        
        return RegulationSearchResponse(
            answer=results.get("answer", ""),
            sources=sources
        )
    
    except Exception as e:
        logger.error(f"법령 검색 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

