"""
LangGraph 챗봇 State 정의
"""
from typing import TypedDict, List, Dict, Optional


class ChatbotState(TypedDict):
    """LangGraph 챗봇 State"""
    # 기본
    messages: List[Dict]  # 대화 이력 [{"role": "user"|"assistant", "content": str}]
    conversation_summary: Optional[str]  # 오래된 대화 요약(누적)
    user_question: str  # 현재 사용자 질문
    conversation_id: str  # 대화 세션 ID
    
    # 분류
    query_type: str  # "db_query" | "vector_search" | "both" | "general"
    
    # DB
    db_result: Optional[str]  # DB 조회 결과
    
    # Vector
    vector_docs: List[Dict]  # [{content, metadata, similarity_score}]
    
    # Rerank
    reranked_docs: List[Dict]  # [{content, metadata, score}]
    
    # 답변
    final_answer: str  # 최종 생성된 답변
    
    # Self-RAG 평가
    relevance_score: Optional[float]  # 관련성 점수 (0.0 ~ 1.0)
    is_relevant: Optional[bool]  # 관련성 여부
    hallucination_score: Optional[float]  # 근거 점수 (0.0 ~ 1.0)
    is_grounded: Optional[bool]  # 근거 충분 여부
    
    # 재시도 카운터
    retry_count: int  # 검색 재시도 횟수
    search_query_version: int  # 검색 쿼리 버전 (재시도 시 개선)
    generation_retry_count: int  # 답변 재생성 횟수
