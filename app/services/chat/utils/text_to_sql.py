"""
Text-to-SQL 변환 유틸리티 (SQL Agent 사용)
"""
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from app.crud.db_connect import get_sql_database
from app.core.config import settings
from app.core.logging import logger
from app.services.chat.utils.llm_client import get_answer_llm
from urllib.parse import urlparse
from typing import Optional, Dict
import re


def get_db_schema() -> Optional[SQLDatabase]:
    """
    SQLDatabase 인스턴스 반환
    
    Returns:
        SQLDatabase 인스턴스 또는 None
    """
    try:
        parsed_url = urlparse(settings.DATABASE_URL.replace('postgresql+psycopg://', 'postgresql://'))
        db = get_sql_database(
            user=parsed_url.username,
            password=parsed_url.password,
            host=parsed_url.hostname,
            port=parsed_url.port or 5432,
            database=parsed_url.path.lstrip('/')
        )
        return db
    except Exception as e:
        logger.error(f"DB 스키마 가져오기 실패: {str(e)}", exc_info=True)
        return None


def _validate_sql_query(sql_query: str) -> bool:
    """
    SQL 쿼리 보안 검증 (읽기 전용 쿼리만 허용)
    
    Args:
        sql_query: 검증할 SQL 쿼리
    
    Returns:
        True: 안전한 쿼리, False: 위험한 쿼리
    """
    # 위험한 키워드 검사 (대소문자 무시)
    dangerous_keywords = [
        'drop', 'delete', 'truncate', 'alter', 'create', 
        'insert', 'update', 'grant', 'revoke', 'exec', 'execute'
    ]
    
    sql_lower = sql_query.lower().strip()
    
    # 주석 제거 후 검사
    sql_clean = re.sub(r'--.*?$', '', sql_lower, flags=re.MULTILINE)
    sql_clean = re.sub(r'/\*.*?\*/', '', sql_clean, flags=re.DOTALL)
    
    for keyword in dangerous_keywords:
        # 단어 경계로 검사 (예: "drop"은 "dropped"와 구분)
        pattern = r'\b' + keyword + r'\b'
        if re.search(pattern, sql_clean):
            logger.warning(f"위험한 SQL 키워드 감지: {keyword}")
            return False
    
    # SELECT로 시작하는지 확인 (읽기 전용)
    if not sql_clean.strip().startswith('select'):
        logger.warning("SELECT가 아닌 쿼리는 허용되지 않습니다.")
        return False
    
    return True


def generate_sql_query_and_execute(question: str, context: Optional[Dict] = None) -> Optional[str]:
    """
    자연어 질문을 SQL로 변환하고 실행하여 결과 반환 (SQL Agent 사용)
    
    Args:
        question: 사용자 질문
        context: 추가 컨텍스트 (이전 대화 등)
    
    Returns:
        쿼리 결과 문자열 또는 None
    """
    try:
        db = get_db_schema()
        if db is None:
            logger.warning("DB 스키마를 가져올 수 없어 SQL 생성 불가")
            return None
        
        logger.info(f"SQL Agent 실행 요청: {question}")
        
        # 대화 맥락 구성
        context_text = ""
        if context and context.get("previous_messages"):
            context_text = "\n이전 대화 맥락:\n"
            for msg in context["previous_messages"][-3:]:  # 최근 3개만
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                context_text += f"{role}: {content}\n"
        
        # 질문에 맥락 추가
        full_question = question
        if context_text:
            full_question = f"{context_text}\n질문: {question}"
        
        # LLM 가져오기
        llm = get_answer_llm()
        
        # SQL Agent 생성
        # HuggingFace 모델은 tool-calling을 완전히 지원하지 않으므로 기본 ReAct 방식 사용
        agent = create_sql_agent(
            llm=llm,
            db=db,
            verbose=False,  # 로그가 너무 많을 수 있으므로 False
            # agent_type 제거: HuggingFace 모델과의 호환성을 위해 기본 ReAct 방식 사용
            handle_parsing_errors=True,  # 파싱 에러 처리
            max_iterations=5,  # 최대 반복 횟수 제한
        )
        
        # Agent 실행 (SQL 생성 + 실행 + 결과 반환)
        logger.debug(f"SQL Agent 실행 시작: {question[:50]}...")
        result = agent.invoke({"input": full_question})
        
        # 결과 추출
        result_text = str(result.get("output", ""))
        
        if result_text:
            logger.info(f"SQL Agent 실행 완료: {result_text[:100]}...")
            return result_text
        else:
            logger.warning("SQL Agent 실행 결과가 비어있음")
            return None
        
    except Exception as e:
        logger.error(f"SQL Agent 실행 실패: {str(e)}", exc_info=True)
        return None


def generate_sql_query(question: str, context: Optional[Dict] = None) -> Optional[str]:
    """
    자연어 질문을 SQL 쿼리로 변환 (SQL Agent 사용)
    
    주의: SQL Agent는 SQL 생성과 실행을 함께 처리하므로,
    이 함수는 generate_sql_query_and_execute를 호출하여 결과를 반환합니다.
    
    Args:
        question: 사용자 질문
        context: 추가 컨텍스트 (이전 대화 등)
    
    Returns:
        SQL 쿼리 결과 문자열 또는 None
    """
    # SQL Agent는 SQL 생성과 실행을 함께 처리하므로
    # 결과를 바로 반환 (SQL 문자열이 아닌 실행 결과)
    return generate_sql_query_and_execute(question, context)


def execute_sql_query(sql_query: str) -> Optional[str]:
    """
    SQL 쿼리 실행 및 결과 반환
    
    Args:
        sql_query: 실행할 SQL 쿼리
    
    Returns:
        쿼리 결과 문자열 또는 None
    """
    try:
        db = get_db_schema()
        if db is None:
            logger.warning("DB 스키마를 가져올 수 없어 SQL 실행 불가")
            return None
        
        logger.info(f"SQL 쿼리 실행: {sql_query}")
        result = db.run(sql_query)
        logger.info(f"SQL 쿼리 실행 완료: {result[:100]}...")
        return result
        
    except Exception as e:
        logger.error(f"SQL 쿼리 실행 실패: {str(e)}", exc_info=True)
        return None
