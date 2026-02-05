"""
DB 조회 노드 (db_query)
"""
from app.services.chat.state import ChatbotState
from app.services.chat.utils.text_to_sql import generate_sql_query
from app.core.logging import logger


def db_query(state: ChatbotState) -> ChatbotState:
    """
    PostgreSQL DB 조회
    
    Args:
        state: 현재 상태
    
    Returns:
        업데이트된 상태 (db_result 설정)
    """
    try:
        user_question = state.get("user_question", "")
        messages = state.get("messages", [])
        
        logger.info(f"DB 조회 시작: {user_question[:50]}...")
        
        # 대화 맥락 구성
        context = {
            "previous_messages": messages[-3:] if len(messages) > 0 else []
        }
        
        # Text-to-SQL: SQL Agent 사용 (SQL 생성 + 실행 자동 처리)
        # SQL Agent는 SQL 생성과 실행을 함께 처리하므로 execute_sql_query 호출 불필요
        db_result = generate_sql_query(user_question, context)
        
        if db_result and db_result not in ["DB 조회를 수행할 수 없습니다.", "DB 조회 결과 없음"]:
            state["db_result"] = db_result
        else:
            logger.warning("SQL Agent 실행 실패")
            state["db_result"] = "DB 조회를 수행할 수 없습니다."
            # DB 질문인 경우 vector_search로 폴백하지 않음
            # (DB 관련 질문은 법령 검색으로 폴백해도 의미가 없으므로)
            # query_type은 그대로 유지하여 상위 라우터에서 처리하도록 함
        
        logger.info(f"DB 조회 완료: {state['db_result'][:100]}...")
        return state
        
    except Exception as e:
        logger.error(f"DB 조회 실패: {str(e)}", exc_info=True)
        state["db_result"] = f"DB 조회 중 오류가 발생했습니다: {str(e)}"
        return state
