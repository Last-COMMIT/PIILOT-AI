"""
DB와 Vector 검색 순차 실행 노드 (both_query)
LangGraph는 순차 실행을 전제로 하므로 병렬 실행 시 State 충돌 발생
"""
from app.services.chat.state import ChatbotState
from app.services.chat.nodes.db_query import db_query
from app.services.chat.nodes.vector_search import vector_search
from app.core.logging import logger


def both_query(state: ChatbotState) -> ChatbotState:
    """
    DB와 Vector 검색 순차 실행
    
    주의: LangGraph는 State를 순차적으로 업데이트하므로 병렬 실행 시 충돌 발생
    성능 최적화는 각 노드 내부에서 비동기 처리로 해결
    
    Args:
        state: 현재 상태
    
    Returns:
        업데이트된 상태 (db_result, vector_docs 설정)
    """
    try:
        logger.info("DB와 Vector 검색 순차 실행 시작")
        
        # DB 조회 실행
        state = db_query(state)
        
        # Vector 검색 실행
        state = vector_search(state)
        
        logger.info("DB와 Vector 검색 순차 실행 완료")
        return state
        
    except Exception as e:
        logger.error(f"DB와 Vector 검색 실행 실패: {str(e)}", exc_info=True)
        # 부분 실패 시에도 기본값 설정
        if "db_result" not in state:
            state["db_result"] = "DB 조회 실패"
        if "vector_docs" not in state:
            state["vector_docs"] = []
        return state
