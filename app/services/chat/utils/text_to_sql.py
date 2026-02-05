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
    ERD 기반 테이블 스키마 가이드 제공
    모든 테이블의 관계, 컬럼 의미, ENUM 값 등을 명시적으로 제공
    """
    return """
[데이터베이스 테이블 구조 및 관계 가이드]

=== 사용자 및 공지사항 ===
- users: 사용자 정보
  * id (PK), email, name, role (ENUM: ADMIN, USER)
  * 관계: db_server_connections.user_id, file_server_connections.user_id

- notices: 공지사항
  * id (PK), user_id (FK → users.id), title, content

=== DB 서버 관련 ===
- dbms_types: DBMS 유형 정의
  * id (PK), name (예: PostgreSQL, MySQL, Oracle), jdbc_url_prefix, default_port, driver_class_name, test_query
  * 관계: db_server_connections.dbms_type_id

- db_server_connections: DB 서버 연결 정보
  * id (PK), user_id (FK → users.id), dbms_type_id (FK → dbms_types.id)
  * connection_name, host, port, db_name, username, status (ENUM: CONNECTED, DISCONNECTED)
  * 관계: db_scan_history.db_server_connection_id, db_pii_issues.db_server_connection_id

- db_scan_history: DB 스캔 히스토리
  * id (PK), db_server_connection_id (FK → db_server_connections.id)
  * status (ENUM: IN_PROGRESS, COMPLETED), scan_start_time, scan_end_time
  * total_tables_count, total_columns_count, scanned_columns_count

- db_tables: DB 테이블 메타데이터
  * id (PK), db_server_connection_id (FK → db_server_connections.id)
  * name (테이블명), total_columns, last_scanned_at
  * 관계: db_pii_columns.table_id

=== PII 관련 ===
- pii_types: 개인정보 유형 정의
  * id (PK), type (ENUM: NM, RRN, ADDR, PPH, ACN, PREM, FACE)
  * risk_weight (위험도 가중치)
  * 관계: db_pii_columns.pii_type_id, file_pii.pii_type_id

- db_pii_columns: DB 개인정보 컬럼 상세 정보 (★ 실제 개수 데이터 포함)
  * id (PK), table_id (FK → db_tables.id), pii_type_id (FK → pii_types.id)
  * name (컬럼명), total_records_count (전체 레코드 수), enc_records_count (암호화된 레코드 수)
  * risk_level (ENUM: HIGH, MEDIUM, LOW), is_issue_open (BOOLEAN)
  * 관계: db_pii_issues.column_id
  * 중요: "개수", "가장 많은" 질문 시 total_records_count, enc_records_count, unenc_records_count 사용

- db_pii_issues: DB 개인정보 이슈 (★ 존재 여부만 표시)
  * id (PK), column_id (FK → db_pii_columns.id), db_server_connection_id (FK → db_server_connections.id)
  * user_status (ENUM: ISSUE_RUNNING_DONE), issue_status (ENUM: ACTIVE, RESOLVED)
  * detected_at, resolved_at
  * 주의: 이 테이블은 이슈 존재 여부만 표시. 실제 개수는 db_pii_columns를 봐야 함

=== 파일 서버 관련 ===
- file_server_types: 파일 서버 유형
  * id (PK), name
  * 관계: file_server_connections.server_type_id

- file_server_connections: 파일 서버 연결 정보
  * id (PK), server_type_id (FK → file_server_types.id), user_id (FK → users.id)
  * connection_name, host, port, default_path, status (ENUM: CONNECTED, DISCONNECTED)
  * 관계: file_scan_history.file_server_connection_id, file_pii_issues.file_server_connection_id, masking_logs.file_server_connection_id

- file_scan_history: 파일 서버 스캔 히스토리
  * id (PK), file_server_connection_id (FK → file_server_connections.id)
  * status (ENUM: IN_PROGRESS, COMPLETED), scan_start_time, scan_end_time
  * total_files_count, scanned_files_count

- file_type: 파일 유형
  * id (PK), type (ENUM: DOCUMENT, PHOTO, AUDIO, VIDEO), extension
  * 관계: files.file_type_id

- files: 파일 메타데이터
  * id (PK), connection_id (FK → file_server_connections.id), file_type_id (FK → file_type.id)
  * name (파일명), file_path, is_encrypted (BOOLEAN), has_personal_info (BOOLEAN)
  * risk_level (ENUM: HIGH, MEDIUM, LOW), last_modified_time, last_scanned_at
  * 관계: file_pii.file_id, file_pii_issues.file_id, masking_logs.file_id

- file_pii: 파일별 개인정보 상세 (★ 실제 개수 데이터 포함)
  * id (PK), file_id (FK → files.id), pii_type_id (FK → pii_types.id)
  * total_piis_count (총 PII 개수), masked_piis_count (마스킹된 PII 개수)
  * 중요: "파일별 PII 개수" 질문 시 이 테이블의 total_piis_count 사용

- file_pii_issues: 파일 개인정보 이슈 (★ 존재 여부만 표시)
  * id (PK), file_id (FK → files.id), file_server_connection_id (FK → file_server_connections.id)
  * user_status (ENUM: ISSUE_RUNNING_DONE), issue_status (ENUM: ACTIVE, RESOLVED)
  * detected_at, resolved_at
  * 주의: 이 테이블은 이슈 존재 여부만 표시. 실제 개수는 file_pii를 봐야 함

- masking_logs: 마스킹 작업 로그
  * id (PK), file_id (FK → files.id), file_server_connection_id (FK → file_server_connections.id)
  * original_file_path, masked_file_path, performed_at

=== 문서 및 법령 ===
- document: 문서 정보
  * id (PK), user_id (FK → users.id), title, type (ENUM: LAWS, INTERNAL_REGULATIONS, DB_INFO), url

- law_data: 법령 데이터 (벡터 DB)
  * id (PK), chunk_text (TEXT), embedding (VECTOR(1024))
  * document_title, law_name, article, page (INTEGER), effective_date (DATE)

[중요 원칙]
1. "개수", "가장 많은", "문제 개수" 질문:
   - db_pii_columns의 total_records_count, unenc_records_count 사용 (COUNT(*) 아님)
   - file_pii의 total_piis_count 사용 (COUNT(*) 아님)
   - db_pii_issues, file_pii_issues는 존재 여부만 표시하므로 개수 질문에 부적합

2. JOIN 관계:
   - db_pii_issues.column_id = db_pii_columns.id
   - db_pii_columns.table_id = db_tables.id
   - db_tables.db_server_connection_id = db_server_connections.id
   - file_pii.file_id = files.id
   - files.connection_id = file_server_connections.id

3. ENUM 값:
   - status: CONNECTED, DISCONNECTED
   - issue_status: ACTIVE, RESOLVED
   - risk_level: HIGH, MEDIUM, LOW
   - pii_type.type: NM, RRN, ADDR, PPH, ACN, PREM, FACE

4. 날짜/시간 컬럼:
   - detected_at, resolved_at, performed_at: TIMESTAMP
   - scan_start_time, scan_end_time: TIMESTAMP
   - effective_date: DATE
"""


def _build_date_context() -> str:
    """
    SQL Agent에 넘길 기준일·주차 정보 문자열 생성.
    오늘, 이번 주, 저번 주 구간을 명시해 '이번 주/저번 주 비교' 등 질문에 활용되도록 함.
    """
    today = date.today()
    # ISO 주: 월요일 시작
    iso_weekday = today.isoweekday()  # 1=월 .. 7=일
    monday_this = today - timedelta(days=iso_weekday - 1)
    monday_last = monday_this - timedelta(days=7)
    sunday_this = monday_this + timedelta(days=6)
    sunday_last = monday_last + timedelta(days=6)
    return (
        f"[기준일 정보] 오늘: {today.isoformat()} (YYYY-MM-DD). "
        f"이번 주: {monday_this.isoformat()} ~ {sunday_this.isoformat()}. "
        f"저번 주: {monday_last.isoformat()} ~ {sunday_last.isoformat()}. "
        "날짜/기간 관련 질문(오늘, 이번 주, 저번 주 등)은 위 구간을 사용하세요.\n"
        "[DB 사용] 반드시 먼저 테이블 목록을 확인한 뒤, 필요한 테이블의 스키마(컬럼·타입)를 조회하고 그에 맞는 SQL만 작성하세요.\n"
        "[중요: 테이블 관계 및 의미 파악]\n"
        "- '이슈 개수', '문제 개수', '레코드 개수', '가장 많은' 같은 질문은 실제 개수 데이터가 있는 테이블과 컬럼을 확인하세요\n"
        "- 예시 1: db_pii_issues는 이슈 존재 여부(id, status)만 표시합니다. 실제 레코드 개수는 db_pii_columns의 total_records_count, unenc_records_count 등을 봐야 합니다\n"
        "- 예시 2: '어떤 컬럼에 문제가 가장 많아?' → db_pii_columns의 unenc_records_count (암호화 안 된 레코드 수)를 ORDER BY로 정렬\n"
        "- 예시 3: '파일별 PII 개수' → file_pii 테이블의 total_piis_count 컬럼 사용\n"
        "- 원칙: COUNT(*)보다는 실제 개수 컬럼(total_*, count, num 등)을 우선 사용하세요\n"
        "- JOIN이 필요한 경우: db_pii_issues.column_id = db_pii_columns.id 같은 관계를 확인하고 적절히 JOIN하세요\n"
        "- 테이블 스키마를 확인할 때 각 컬럼의 의미(개수인지, 존재 여부인지, 상태인지)를 파악하세요\n"
        "[비교 질문 처리] '가장 많은', '최대', '최소', '어떤 것이 가장' 같은 비교 질문의 경우:\n"
        "- COUNT, SUM, MAX, MIN 등 집계 함수를 사용하여 비교 기준을 명확히 하세요\n"
        "- 단, 실제 개수 데이터가 있는 컬럼(total_records_count, unenc_records_count 등)이 있으면 그것을 우선 사용하세요\n"
        "- ORDER BY로 정렬하여 상위 결과를 반환하세요\n"
        "- 결과가 모두 같으면 그 사실을 명확히 표시하세요 (예: '모든 컬럼에 이슈가 1개씩 있습니다')\n"
        "- 결과가 없으면 '해당하는 데이터가 없습니다'라고 표시하세요"
    )


def _build_query_examples() -> str:
    """
    쿼리 샘플 제공 (Few-shot learning)
    ERD 기반 실제 사용 가능한 쿼리 패턴 예시
    """
    return """
[쿼리 샘플 - 참고용]

다음은 데이터베이스에서 자주 사용되는 쿼리 패턴입니다. 유사한 질문이 들어오면 이 패턴을 참고하세요.

1. 연결된 DB 서버 목록 조회
   질문: "연결된 데이터베이스 서버 목록 보여줘"
   SQL: SELECT id, connection_name, host, db_name, status FROM db_server_connections WHERE status = 'CONNECTED';

2. 사용자별 DB 연결 수 조회
   질문: "각 사용자가 만든 DB 연결이 몇 개인지 보여줘"
   SQL: SELECT u.name, COUNT(dsc.id) as connection_count 
        FROM users u 
        LEFT JOIN db_server_connections dsc ON u.id = dsc.user_id 
        GROUP BY u.id, u.name 
        ORDER BY connection_count DESC;

3. 최근 DB 스캔 히스토리 조회
   질문: "최근 일주일간 DB 스캔 내역 보여줘"
   SQL: SELECT dsh.id, dsc.connection_name, dsh.status, dsh.scan_start_time, dsh.scan_end_time, dsh.total_tables_count
        FROM db_scan_history dsh
        JOIN db_server_connections dsc ON dsh.db_server_connection_id = dsc.id
        WHERE dsh.scan_start_time >= CURRENT_DATE - INTERVAL '7 days'
        ORDER BY dsh.scan_start_time DESC;

4. HIGH 위험 PII 컬럼 조회
   질문: "위험도가 높은 개인정보 컬럼 목록 보여줘"
   SQL: SELECT dt.name as table_name, dpc.name as column_name, pt.type as pii_type, dpc.risk_level, dpc.total_records_count
        FROM db_pii_columns dpc
        JOIN db_tables dt ON dpc.table_id = dt.id
        JOIN pii_types pt ON dpc.pii_type_id = pt.id
        WHERE dpc.risk_level = 'HIGH'
        ORDER BY dpc.total_records_count DESC;

5. 활성화된 PII 이슈 조회
   질문: "현재 해결되지 않은 DB 개인정보 이슈 보여줘"
   SQL: SELECT dpi.id, dt.name as table_name, dpc.name as column_name, dpi.detected_at, dpi.issue_status
        FROM db_pii_issues dpi
        JOIN db_pii_columns dpc ON dpi.column_id = dpc.id
        JOIN db_tables dt ON dpc.table_id = dt.id
        WHERE dpi.issue_status = 'ACTIVE'
        ORDER BY dpi.detected_at DESC;

6. 파일 서버 연결 상태 조회
   질문: "연결된 파일 서버 목록 보여줘"
   SQL: SELECT fsc.id, fsc.connection_name, fst.name as server_type, fsc.host, fsc.status
        FROM file_server_connections fsc
        JOIN file_server_types fst ON fsc.server_type_id = fst.id
        WHERE fsc.status = 'CONNECTED';

7. 개인정보가 있는 파일 중 암호화되지 않은 파일 조회
   질문: "개인정보가 있는데 암호화 안 된 파일 목록 보여줘"
   SQL: SELECT f.id, f.name, f.file_path, f.risk_level, f.last_scanned_at
        FROM files f
        WHERE f.has_personal_info = TRUE AND f.is_encrypted = FALSE
        ORDER BY f.risk_level DESC, f.last_scanned_at DESC;

8. 파일별 PII 유형별 개수 조회
   질문: "각 파일에 어떤 개인정보가 몇 개씩 있는지 보여줘"
   SQL: SELECT f.name, pt.type as pii_type, fp.total_piis_count, fp.masked_piis_count
        FROM file_pii fp
        JOIN files f ON fp.file_id = f.id
        JOIN pii_types pt ON fp.pii_type_id = pt.id
        ORDER BY f.name, fp.total_piis_count DESC;

9. 최근 마스킹 작업 로그 조회
   질문: "최근 일주일간 마스킹 작업 내역 보여줘"
   SQL: SELECT ml.id, f.name as file_name, ml.original_file_path, ml.masked_file_path, ml.performed_at
        FROM masking_logs ml
        JOIN files f ON ml.file_id = f.id
        WHERE ml.performed_at >= CURRENT_DATE - INTERVAL '7 days'
        ORDER BY ml.performed_at DESC;

10. 사용자별 생성한 이슈 수 조회
    질문: "각 사용자가 만든 개인정보 이슈가 몇 개인지 보여줘"
    SQL: SELECT u.name, COUNT(dpi.id) as issue_count
         FROM users u
         LEFT JOIN db_server_connections dsc ON u.id = dsc.user_id
         LEFT JOIN db_pii_issues dpi ON dsc.id = dpi.db_server_connection_id
         GROUP BY u.id, u.name
         ORDER BY issue_count DESC;

11. DB 서버별 활성 이슈 개수 조회
    질문: "우리 DB서버 이슈 뭐있어" 또는 "DB 서버별 이슈 개수"
    SQL: SELECT dsc.connection_name, COUNT(dpi.id) as issue_count, dpi.issue_status
         FROM db_pii_issues dpi
         JOIN db_server_connections dsc ON dpi.db_server_connection_id = dsc.id
         WHERE dpi.issue_status = 'ACTIVE'
         GROUP BY dsc.connection_name, dpi.issue_status
         ORDER BY issue_count DESC;

12. 파일 서버별 이슈 개수 조회
    질문: "파일 서버별로 이슈 개수 정리해줘"
    SQL: SELECT fsc.connection_name, COUNT(fpi.id) as issue_count, fpi.issue_status
         FROM file_pii_issues fpi
         JOIN file_server_connections fsc ON fpi.file_server_connection_id = fsc.id
         WHERE fpi.issue_status = 'ACTIVE'
         GROUP BY fsc.connection_name, fpi.issue_status
         ORDER BY issue_count DESC;

[주의사항]
- 위 예시는 참고용이며, 실제 질문에 맞게 테이블명과 컬럼명을 확인한 후 쿼리를 작성하세요.
- JOIN 시 외래키 관계를 정확히 파악하세요.
- ENUM 값은 정확히 사용하세요 (예: 'CONNECTED', 'ACTIVE', 'HIGH' 등).
- 날짜 비교 시 CURRENT_DATE, INTERVAL 등을 활용하세요.
- 결과를 반환할 때는 간단한 텍스트 형식으로만 반환하세요. 마크다운, HTML, 특수 형식은 사용하지 마세요.
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
        
        # 질문에 맥락 추가: 테이블 스키마 가이드 → 기준일/스키마 지시 → 쿼리 샘플 → (선택) 이전 대화 → 질문
        # 중요: 결과를 간단한 텍스트 형식으로만 반환하도록 지시 추가
        output_format_instruction = "\n[중요] 결과를 반환할 때는 마크다운 형식(**굵게**, # 헤더, - 리스트 등)을 사용하지 말고, 간단한 텍스트 형식으로만 반환하세요. 숫자와 데이터만 명확하게 표시하세요.\n"
        
        full_question = f"{table_schema_guide}\n{date_and_schema_context}\n{query_examples}\n{output_format_instruction}"
        if context_text:
            full_question += f"{context_text}\n"
        full_question += f"질문: {question}"
        
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
            
            # 파싱 에러가 발생했지만 결과가 있는 경우 추출 시도
            if "parsing error" in result_text.lower() or "could not parse" in result_text.lower():
                logger.warning("SQL Agent 파싱 에러 발생, 결과 추출 시도")
                extracted_result = _extract_data_from_agent_output(result_text)
                if extracted_result:
                    logger.info(f"파싱 에러에서 결과 추출 성공: {extracted_result[:100]}...")
                    return extracted_result
                else:
                    logger.warning("파싱 에러에서 결과 추출 실패")
                    return None
            
            logger.info(f"SQL Agent 실행 완료: {result_text[:100]}...")
            return result_text
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
