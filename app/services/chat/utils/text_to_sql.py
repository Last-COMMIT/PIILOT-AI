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
from datetime import date, timedelta
import re

_AGENT_STOP_PHRASE = "Agent stopped due to iteration limit or time limit"


def _build_table_schema_guide() -> str:
    """
    간소화된 테이블 스키마 가이드
    """
    return """
[테이블 구조 요약]
- db_server_connections: id, connection_name, host, port, db_name, username, status
- db_pii_columns: id, table_id, name, total_records_count, enc_records_count, risk_level
- db_pii_issues: id, column_id, db_server_connection_id, issue_status (ACTIVE/RESOLVED)
- file_pii: id, file_id, total_piis_count, masked_piis_count
- file_pii_issues: id, file_id, file_server_connection_id, issue_status
- files: id, name, path, file_type, size, created_at, updated_at

[테이블 관계 및 JOIN 가이드 - 매우 중요!]
⚠️ 상세한 정보를 얻으려면 반드시 JOIN을 사용하세요:

1. DB 관련:
   - db_pii_columns (컬럼 정보) ←→ db_pii_issues (이슈 정보)
     JOIN: db_pii_issues.column_id = db_pii_columns.id
   - db_pii_issues ←→ db_server_connections (서버 정보)
     JOIN: db_pii_issues.db_server_connection_id = db_server_connections.id
   - 예: "어떤 컬럼에 문제가 있나요?" → db_pii_columns와 db_pii_issues를 JOIN하여 컬럼명, 위험도, 이슈 상태를 함께 조회

2. 파일 관련:
   - files (파일 정보) ←→ file_pii (PII 통계)
     JOIN: file_pii.file_id = files.id
   - files ←→ file_pii_issues (이슈 정보)
     JOIN: file_pii_issues.file_id = files.id
   - file_pii ←→ file_pii_issues
     JOIN: file_pii_issues.file_id = file_pii.file_id
   - 예: "어떤 파일에 문제가 있나요?" → files, file_pii, file_pii_issues를 JOIN하여 파일명, PII 개수, 이슈 상태를 함께 조회

[중요]
- 개수 질문: db_pii_columns.total_records_count, file_pii.total_piis_count 사용
- 상세 정보 질문: 반드시 JOIN을 사용하여 관련 테이블의 정보를 함께 조회하세요
- 이슈가 있는 항목 조회: 이슈 테이블과 메인 테이블을 JOIN하여 상세 정보를 함께 보여주세요
"""


def _build_date_context() -> str:
    """
    간소화된 날짜 컨텍스트
    """
    today = date.today()
    iso_weekday = today.isoweekday()
    monday_this = today - timedelta(days=iso_weekday - 1)
    monday_last = monday_this - timedelta(days=7)
    sunday_this = monday_this + timedelta(days=6)
    sunday_last = monday_last + timedelta(days=6)
    return (
        f"[날짜] 오늘: {today.isoformat()}, 이번 주: {monday_this.isoformat()}~{sunday_this.isoformat()}\n"
        "[사용법] 테이블 스키마 확인 후 SQL 작성. 개수 질문은 total_records_count, total_piis_count 사용."
    )


def _build_query_examples() -> str:
    """
    핵심 쿼리 샘플만 제공 (간소화)
    """
    return """
[쿼리 샘플 - 유사한 질문이면 이 패턴 사용]

1. DB 서버별 이슈 개수: "DB 서버별 이슈 개수" 
   → SELECT dsc.connection_name, COUNT(dpi.id) FROM db_pii_issues dpi JOIN db_server_connections dsc ON dpi.db_server_connection_id = dsc.id WHERE dpi.issue_status = 'ACTIVE' GROUP BY dsc.connection_name;

2. DB 컬럼별 상세 이슈 정보 (JOIN 필수!): "어떤 컬럼에 문제가 있나요?"
   → SELECT dpc.name AS column_name, dpc.risk_level, dpc.total_records_count, dpi.issue_status, dsc.connection_name 
     FROM db_pii_issues dpi 
     JOIN db_pii_columns dpc ON dpi.column_id = dpc.id 
     JOIN db_server_connections dsc ON dpi.db_server_connection_id = dsc.id 
     WHERE dpi.issue_status = 'ACTIVE';

3. 파일 서버별 이슈 개수: "파일 서버별 이슈 개수" 
   → SELECT fsc.connection_name, COUNT(fpi.id) FROM file_pii_issues fpi JOIN file_server_connections fsc ON fpi.file_server_connection_id = fsc.id WHERE fpi.issue_status = 'ACTIVE' GROUP BY fsc.connection_name;

4. 파일별 상세 PII 및 이슈 정보 (JOIN 필수!): "어떤 파일에 문제가 있나요?"
   → SELECT f.name AS file_name, f.file_type, fp.total_piis_count, fp.masked_piis_count, fpi.issue_status 
     FROM file_pii_issues fpi 
     JOIN files f ON fpi.file_id = f.id 
     JOIN file_pii fp ON fpi.file_id = fp.file_id 
     WHERE fpi.issue_status = 'ACTIVE';

5. 파일별 PII 개수: "파일별 PII 개수" 
   → SELECT f.name, fp.total_piis_count FROM file_pii fp JOIN files f ON fp.file_id = f.id;

[핵심 원칙]
- 이슈나 문제를 조회할 때는 반드시 이슈 테이블과 메인 테이블을 JOIN하여 상세 정보를 함께 조회하세요
- 파일 관련 질문: files 테이블과 JOIN하여 파일명을 반드시 포함하세요
- DB 컬럼 관련 질문: db_pii_columns와 JOIN하여 컬럼명, 위험도 등을 함께 조회하세요
"""


def _is_agent_stopped_message(text: str) -> bool:
    t = (text or "").strip()
    return bool(t) and _AGENT_STOP_PHRASE.lower() in t.lower()


def _extract_data_from_agent_output(output: str) -> Optional[str]:
    """
    SQL Agent 출력에서 실제 데이터 추출
    파싱 에러가 발생해도 마크다운이나 텍스트 형식의 결과에서 데이터 추출 시도
    """
    if not output:
        return None
    
    import re
    
    # 파싱 에러 메시지가 포함된 경우, 실제 결과 부분만 추출
    if "could not parse" in output.lower() or "parsing error" in output.lower():
        lines = output.split('\n')
        data_lines = []
        skip_patterns = [
            'error', 'parsing', 'could not', 'output parsing', 
            'troubleshooting', 'visit:', 'http', 'langchain'
        ]
        
        for line in lines:
            line_lower = line.lower()
            # 에러 메시지나 링크 제외
            if any(pattern in line_lower for pattern in skip_patterns):
                continue
            
            # 데이터가 포함된 라인만 추출 (ID, 숫자, 날짜, 상태 등)
            if re.search(r'\d+|ID|Status|Detected|Issue|Count|Name|Type', line, re.IGNORECASE):
                # 마크다운 포맷 제거
                cleaned_line = re.sub(r'^\*\*|\*\*$|^#+\s*|^-+\s*', '', line).strip()
                if cleaned_line and len(cleaned_line) > 3:
                    data_lines.append(cleaned_line)
        
        if data_lines:
            result = '\n'.join(data_lines)
            logger.info(f"파싱 에러에서 {len(data_lines)}개 라인 추출 성공")
            return result
    
    return output


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


def _add_sql_result_explanation(result_text: str, question: str) -> str:
    """
    SQL 결과에 간단한 설명 추가 (고도화)
    
    Args:
        result_text: SQL 쿼리 결과 텍스트
        question: 사용자 질문
    
    Returns:
        설명이 추가된 결과 텍스트
    """
    if not result_text or len(result_text.strip()) < 10:
        return result_text
    
    # 간단한 구조 분석
    lines = result_text.strip().split('\n')
    num_lines = len([l for l in lines if l.strip()])
    
    # 숫자 패턴 감지 (개수, 통계 등)
    import re
    numbers = re.findall(r'\d+', result_text)
    
    # 설명 추가 (간단하게)
    explanation_parts = []
    
    if num_lines == 0:
        explanation_parts.append("[결과 없음]")
    elif num_lines == 1:
        explanation_parts.append(f"[단일 결과: {num_lines}개 행]")
    else:
        explanation_parts.append(f"[조회 결과: {num_lines}개 행]")
    
    if numbers:
        explanation_parts.append(f"[수치 데이터 포함: {len(numbers)}개 숫자]")
    
    if explanation_parts:
        explanation = " ".join(explanation_parts)
        return f"{result_text}\n\n{explanation}"
    
    return result_text


def _filter_sensitive_data(result_text: str) -> str:
    """
    SQL 쿼리 결과에서 민감한 정보(비밀번호, 패스워드 등) 및 users 테이블 정보 제거
    
    Args:
        result_text: SQL 쿼리 결과 텍스트
    
    Returns:
        민감한 정보가 제거된 텍스트
    """
    if not result_text:
        return result_text
    
    import re
    
    # users 테이블 관련 정보가 포함되어 있으면 전체 결과 거부
    if re.search(r'\busers\b', result_text, re.IGNORECASE):
        logger.warning("결과에 users 테이블 정보가 포함되어 있어 차단")
        return "보안상의 이유로 해당 정보는 제공할 수 없습니다."
    
    # 민감한 컬럼명 패턴 (대소문자 무시)
    sensitive_patterns = [
        r'password\s*[:=]\s*[^\n]+',
        r'pwd\s*[:=]\s*[^\n]+',
        r'passwd\s*[:=]\s*[^\n]+',
        r'secret\s*[:=]\s*[^\n]+',
        r'credential\s*[:=]\s*[^\n]+',
        r'token\s*[:=]\s*[^\n]+',
        r'api[_-]?key\s*[:=]\s*[^\n]+',
        r'encrypted[_-]?password\s*[:=]\s*[^\n]+',
    ]
    
    filtered_text = result_text
    for pattern in sensitive_patterns:
        filtered_text = re.sub(pattern, '[민감한 정보는 보호됩니다]', filtered_text, flags=re.IGNORECASE)
    
    # 비밀번호 같은 단어가 포함된 라인 전체 제거
    lines = filtered_text.split('\n')
    safe_lines = []
    for line in lines:
        line_lower = line.lower()
        # users 테이블 관련 라인 제거
        if 'users' in line_lower or 'user' in line_lower:
            # "user_id" 같은 외래키는 허용하되, users 테이블 자체 정보는 차단
            if re.search(r'\busers\b', line_lower) and not re.search(r'user_id|user_name', line_lower):
                continue
        
        # 민감한 키워드가 포함된 라인 제거
        if any(keyword in line_lower for keyword in ['password', 'pwd', 'passwd', 'secret', 'credential', 'token', 'api_key', 'encrypted_password']):
            # 단, 컬럼명만 언급하는 경우는 허용 (예: "password 컬럼이 있습니다")
            if not re.search(r'(password|pwd|passwd|secret|credential|token|api[_-]?key|encrypted[_-]?password)\s*[:=]', line_lower):
                continue  # 값이 없는 경우만 제거
        safe_lines.append(line)
    
    return '\n'.join(safe_lines)


def _validate_sql_query(sql_query: str) -> bool:
    """
    SQL 쿼리 보안 검증 (읽기 전용 쿼리만 허용, 민감한 정보 차단)
    
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
    
    # 민감한 컬럼명 (SELECT 절에서 차단)
    sensitive_columns = [
        'password', 'pwd', 'passwd', 'secret', 'credential', 
        'token', 'api_key', 'encrypted_password'
    ]
    
    # 접근 금지 테이블
    forbidden_tables = ['users']
    
    sql_lower = sql_query.lower().strip()
    
    # 주석 제거 후 검사
    sql_clean = re.sub(r'--.*?$', '', sql_lower, flags=re.MULTILINE)
    sql_clean = re.sub(r'/\*.*?\*/', '', sql_clean, flags=re.DOTALL)
    
    # 위험한 키워드 검사
    for keyword in dangerous_keywords:
        pattern = r'\b' + keyword + r'\b'
        if re.search(pattern, sql_clean):
            logger.warning(f"위험한 SQL 키워드 감지: {keyword}")
            return False
    
    # SELECT로 시작하는지 확인 (읽기 전용)
    if not sql_clean.strip().startswith('select'):
        logger.warning("SELECT가 아닌 쿼리는 허용되지 않습니다.")
        return False
    
    # 접근 금지 테이블 차단 (FROM, JOIN 절에서)
    for table in forbidden_tables:
        # FROM users, JOIN users, users. 등의 패턴 감지
        patterns = [
            r'\bfrom\s+' + table + r'\b',
            r'\bjoin\s+' + table + r'\b',
            r'\b' + table + r'\.',
        ]
        for pattern in patterns:
            if re.search(pattern, sql_clean, re.IGNORECASE):
                logger.warning(f"접근 금지 테이블 감지: {table}")
                return False
    
    # 민감한 컬럼 SELECT 차단
    for col in sensitive_columns:
        # SELECT 절에서 민감한 컬럼 선택 시도 감지
        # 예: SELECT password, SELECT encrypted_password 등
        pattern = r'select\s+.*?\b' + col + r'\b'
        if re.search(pattern, sql_clean, re.IGNORECASE):
            logger.warning(f"민감한 컬럼 선택 시도 감지: {col}")
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
        
        # 에이전트용 컨텍스트: 오늘 날짜·주차 + 테이블 스키마 가이드 + DB 스키마 활용 지시 + 쿼리 샘플
        date_and_schema_context = _build_date_context()
        table_schema_guide = _build_table_schema_guide()
        query_examples = _build_query_examples()
        
        # 대화 맥락 구성
        context_text = ""
        if context and context.get("previous_messages"):
            context_text = "\n이전 대화 맥락:\n"
            for msg in context["previous_messages"][-3:]:  # 최근 3개만
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                context_text += f"{role}: {content}\n"
        
        # 보안 지시 추가
        security_instruction = """
[보안 필수 지시 - 절대 위반 금지]
⚠️ SQL 쿼리 작성 시:
- users 테이블은 절대 접근하지 마세요 (FROM users, JOIN users 등 모두 금지)
- password, pwd, passwd, encrypted_password, secret, credential, token, api_key 컬럼은 절대 SELECT하지 마세요
- 이런 테이블/컬럼을 사용하는 쿼리는 생성하지 마세요

⚠️ 결과 반환 시:
- users 테이블의 정보는 절대 반환하지 마세요
- 비밀번호, 패스워드 등 민감한 정보는 절대 반환하지 마세요
- 이런 정보를 요청받으면 "보안상의 이유로 해당 정보는 제공할 수 없습니다"라고만 답변하세요
"""
        
        output_format_instruction = "\n[출력 형식] 마크다운 없이 간단한 텍스트만 반환하세요.\n"
        
        # JOIN 사용 강조 지시사항
        join_instruction = """
[매우 중요: JOIN 사용 원칙]
⚠️ 상세한 정보를 조회할 때는 반드시 JOIN을 사용하세요:

1. 파일 관련 질문:
   - "어떤 파일에 문제가 있나요?" → files, file_pii, file_pii_issues를 JOIN하여 파일명, PII 개수, 이슈 상태를 함께 조회
   - "파일별 이슈" → file_pii_issues와 files를 JOIN하여 파일명을 반드시 포함

2. DB 컬럼 관련 질문:
   - "어떤 컬럼에 문제가 있나요?" → db_pii_columns와 db_pii_issues를 JOIN하여 컬럼명, 위험도, 이슈 상태를 함께 조회
   - "컬럼별 이슈" → db_pii_issues와 db_pii_columns를 JOIN하여 컬럼명을 반드시 포함

3. 서버 관련 질문:
   - 서버명을 보여줄 때는 db_server_connections 또는 file_server_connections와 JOIN

⚠️ 단순 개수만 묻는 경우가 아니라면, 항상 관련 테이블을 JOIN하여 상세 정보를 함께 조회하세요!
"""
        
        # 간소화된 프롬프트 구조
        full_question = f"{table_schema_guide}\n{date_and_schema_context}\n{security_instruction}\n{join_instruction}\n[질문] {question}\n{query_examples}\n{output_format_instruction}"
        
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
            max_iterations=15,  # 반복 제한 상향 (스키마 탐색/집계 질문 대응)
        )
        
        # Agent 실행 (SQL 생성 + 실행 + 결과 반환)
        logger.debug(f"SQL Agent 실행 시작: {question[:50]}...")
        result = agent.invoke({"input": full_question})
        
        # 결과 추출
        result_text = str(result.get("output", ""))
        
        if result_text:
            # iteration/time limit으로 중단된 경우는 실패로 간주 (상위에서 폴백/재시도 처리)
            if _is_agent_stopped_message(result_text):
                logger.warning(f"SQL Agent가 제한에 의해 중단됨: {result_text[:120]}...")
                return None
            
            # 민감한 정보 요청인지 먼저 확인
            question_lower = question.lower()
            sensitive_keywords = ['비밀번호', 'password', 'pwd', 'passwd', 'secret', 'credential', 'token', 'api_key', 'encrypted_password']
            forbidden_table_keywords = ['users 테이블', 'users table', 'user 정보', '사용자 정보', '사용자 목록', 'user list']
            
            is_sensitive_request = any(keyword in question_lower for keyword in sensitive_keywords)
            is_forbidden_table_request = any(keyword in question_lower for keyword in forbidden_table_keywords)
            
            if is_sensitive_request or is_forbidden_table_request:
                logger.warning(f"민감한 정보 요청 감지 (sensitive: {is_sensitive_request}, forbidden_table: {is_forbidden_table_request}), 즉시 거부")
                return "보안상의 이유로 해당 정보는 제공할 수 없습니다."
            
            # 파싱 에러가 발생했지만 결과가 있는 경우 추출 시도
            if "parsing error" in result_text.lower() or "could not parse" in result_text.lower():
                logger.warning("SQL Agent 파싱 에러 발생, 결과 추출 시도")
                extracted_result = _extract_data_from_agent_output(result_text)
                if extracted_result:
                    # 추출된 결과도 필터링 및 설명 추가
                    filtered_extracted = _filter_sensitive_data(extracted_result)
                    explained_result = _add_sql_result_explanation(filtered_extracted, question)
                    logger.info(f"파싱 에러에서 결과 추출 성공: {explained_result[:100]}...")
                    return explained_result
                else:
                    logger.warning("파싱 에러에서 결과 추출 실패")
                    return None
            
            # 민감한 정보 필터링
            filtered_result = _filter_sensitive_data(result_text)
            
            # SQL 결과 설명 추가 (고도화)
            explained_result = _add_sql_result_explanation(filtered_result, question)
            
            logger.info(f"SQL Agent 실행 완료: {explained_result[:100]}...")
            return explained_result
        else:
            logger.warning("SQL Agent 실행 결과가 비어있음")
            return None
        
    except Exception as e:
        error_msg = str(e)
        
        # 파싱 에러인 경우, 에러 메시지에서 결과 추출 시도
        if "parsing error" in error_msg.lower() or "could not parse" in error_msg.lower():
            logger.warning(f"SQL Agent 파싱 에러 발생: {error_msg[:200]}...")
            # 에러 메시지에서 실제 결과 부분 추출 시도
            extracted_result = _extract_data_from_agent_output(error_msg)
            if extracted_result:
                logger.info(f"예외에서 결과 추출 성공: {extracted_result[:100]}...")
                return extracted_result
        
        logger.error(f"SQL Agent 실행 실패: {error_msg}", exc_info=True)
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
