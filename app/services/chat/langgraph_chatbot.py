"""
LangGraph Self-RAG 챗봇 그래프 구성
"""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from app.services.chat.state import ChatbotState
from app.services.chat.nodes.memory import load_memory, save_memory
from app.services.chat.nodes.classification import classify
from app.services.chat.nodes.db_query import db_query
from app.services.chat.nodes.vector_search import vector_search
from app.services.chat.nodes.both_query import both_query
from app.services.chat.nodes.relevance_check import check_relevance
from app.services.chat.nodes.rerank import rerank
from app.services.chat.nodes.answer_generation import generate_answer
from app.services.chat.nodes.hallucination_check import check_hallucination
from app.services.chat.routers.routing import (
    route_after_classify,
    route_after_db,
    route_after_vector,
    route_after_relevance,
    route_after_hallucination
)
from app.core.logging import logger


def create_chatbot_app():
    """
    LangGraph 챗봇 앱 생성 및 컴파일
    
    Returns:
        컴파일된 LangGraph 앱
    """
    try:
        logger.info("LangGraph 챗봇 그래프 구성 시작")
        
        # StateGraph 생성
        workflow = StateGraph(ChatbotState)
        
        # 노드 추가
        workflow.add_node("load_memory", load_memory)
        workflow.add_node("classify", classify)
        workflow.add_node("db_query", db_query)
        workflow.add_node("vector_search", vector_search)
        workflow.add_node("both_query", both_query)
        workflow.add_node("check_relevance", check_relevance)
        workflow.add_node("rerank", rerank)
        workflow.add_node("generate_answer", generate_answer)
        workflow.add_node("check_hallucination", check_hallucination)
        workflow.add_node("save_memory", save_memory)
        
        # 시작점 설정
        workflow.set_entry_point("load_memory")
        
        # 고정 엣지
        workflow.add_edge("load_memory", "classify")
        workflow.add_edge("both_query", "check_relevance")
        workflow.add_edge("rerank", "generate_answer")
        workflow.add_edge("generate_answer", "check_hallucination")
        workflow.add_edge("save_memory", END)
        
        # 조건부 엣지
        workflow.add_conditional_edges(
            "classify",
            route_after_classify,
            {
                "db_query": "db_query",
                "vector_search": "vector_search",
                "both_query": "both_query",
                "generate_answer": "generate_answer"  # "general" 타입도 "generate_answer"로 매핑됨
            }
        )
        
        workflow.add_conditional_edges(
            "db_query",
            route_after_db,
            {
                "generate_answer": "generate_answer",
                "vector_search": "vector_search"  # DB 조회 실패 시 폴백
            }
        )
        
        workflow.add_conditional_edges(
            "vector_search",
            route_after_vector,
            {"check_relevance": "check_relevance"}
        )
        
        workflow.add_conditional_edges(
            "check_relevance",
            route_after_relevance,
            {
                "rerank": "rerank",
                "vector_search": "vector_search",  # 재검색 루프
                "generate_answer": "generate_answer"
            }
        )
        
        workflow.add_conditional_edges(
            "check_hallucination",
            route_after_hallucination,
            {
                "save_memory": "save_memory",
                "generate_answer": "generate_answer"  # 재생성 루프
            }
        )
        
        # 컴파일 (MemorySaver로 세션별 대화 유지)
        memory = MemorySaver()
        app = workflow.compile(checkpointer=memory)
        
        logger.info("✓ LangGraph 챗봇 그래프 구성 완료")
        return app
        
    except Exception as e:
        logger.error(f"LangGraph 챗봇 그래프 구성 실패: {str(e)}", exc_info=True)
        raise
