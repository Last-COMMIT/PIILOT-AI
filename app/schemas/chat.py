"""
Chat AI 요청/응답 스키마
"""
from pydantic import BaseModel
from typing import List, Dict, Optional


# ========== 요청 ==========

class ChatRequest(BaseModel):
    """AI 어시스턴트 질의 요청"""
    query: str
    context: Optional[Dict] = None


class RegulationSearchRequest(BaseModel):
    """법령 검색 요청"""
    query: str
    n_results: Optional[int] = 10
    top_n: Optional[int] = 5


# ========== 응답 ==========

class ChatResponse(BaseModel):
    """AI 어시스턴트 응답"""
    answer: str
    sources: List[str] = []


class RegulationSearchResult(BaseModel):
    """법령 검색 결과"""
    document_title: str
    content: str
    article: str
    page: str
    similarity: float


class RegulationSearchResponse(BaseModel):
    """법령 검색 응답"""
    answer: str
    sources: List[RegulationSearchResult]


class LangGraphChatRequest(BaseModel):
    """LangGraph 챗봇 질의 요청"""
    question: str
    conversation_id: str = "default"


class LangGraphChatResponse(BaseModel):
    """LangGraph 챗봇 응답"""
    answer: str
    sources: List[str] = []
    query_type: str = "general"
    relevance_score: Optional[float] = None
    hallucination_score: Optional[float] = None
