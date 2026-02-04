"""
암호화 여부 판단 (분류 모델)
"""
from typing import Dict, Optional, Any, List, Tuple
from pathlib import Path
from urllib.parse import urlparse
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from app.utils.logger import logger
from app.config import settings
from app.utils.db_connect import create_db_engine, create_db_engine_for_connection
from app.utils.aes_decrypt import decrypt_db_password
from app.services.db.detectors.classifier import classify_batch


class EncryptionClassifier:
    """암호화 여부 판단 분류 모델"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Args:
            model_path: 학습된 모델 경로 (없으면 기본 모델 사용)
        """
        self.model_path = model_path
        # 메인 DB 연결 (연결 정보 조회용)
        # DATABASE_URL 파싱: postgresql+psycopg://user:password@host:port/dbname
        if hasattr(settings, 'DATABASE_URL') and settings.DATABASE_URL:
            parsed_url = urlparse(settings.DATABASE_URL.replace('postgresql+psycopg://', 'postgresql://'))
            self.main_engine = create_db_engine(
                user=parsed_url.username,
                password=parsed_url.password,
                host=parsed_url.hostname,
                port=parsed_url.port or 5432,
                database=parsed_url.path.lstrip('/')
            )
        else:
            self.main_engine = None
            
        # 연결 정보 캐시
        self._connection_cache: Dict[str, Dict] = {}
        logger.info(f"EncryptionClassifier 초기화: {model_path}")
    
    def _get_connection_info(self, connection_id: str) -> Dict:
        """
        메인 DB에서 connection_id로 연결 정보 조회
        """
        if not self.main_engine:
             raise RuntimeError("Main DB Engine이 초기화되지 않았습니다.")

        # 캐시 확인
        if connection_id in self._connection_cache:
            return self._connection_cache[connection_id]
        
        # 메인 DB에서 조회 (dbms_types JOIN으로 DBMS 종류 포함). JOIN 실패 시 단일 테이블로 폴백
        try:
            query = text("""
                SELECT dsc.host, dsc.port, dsc.db_name, dsc.username, dsc.encrypted_password,
                       dt.name AS dbms_type_name
                FROM db_server_connections dsc
                LEFT JOIN dbms_types dt ON dsc.dbms_type_id = dt.id
                WHERE dsc.id = :connection_id
            """)
            with self.main_engine.connect() as conn:
                result = conn.execute(query, {"connection_id": connection_id})
                row = result.fetchone()
        except Exception as e:
            logger.debug("dbms_types JOIN 조회 실패, 단일 테이블로 폴백: %s", e)
            query = text("""
                SELECT host, port, db_name, username, encrypted_password
                FROM db_server_connections
                WHERE id = :connection_id
            """)
            with self.main_engine.connect() as conn:
                result = conn.execute(query, {"connection_id": connection_id})
                row = result.fetchone()
        
        if not row:
            raise ValueError(f"연결 정보를 찾을 수 없습니다: connection_id={connection_id}")
        
        connection_info = {
            "host": row.host,
            "port": row.port or 5432,
            "db_name": row.db_name,
            "username": row.username,
            "encrypted_password": row.encrypted_password,
            "dbms_type_name": getattr(row, "dbms_type_name", None) or "PostgreSQL",
        }
        
        # 비밀번호 복호화: ENCRYPTION_AES_KEY가 있으면 AES-256-GCM 복호화 필수, 없으면 평문으로 사용
        encrypted_password = connection_info["encrypted_password"]
        if encrypted_password is None:
            encrypted_password = ""
        raw_str = str(encrypted_password).strip()
        aes_key = getattr(settings, "ENCRYPTION_AES_KEY", None) or ""
        if aes_key and len(aes_key.encode("utf-8")) == 32:
            # 키가 설정된 경우 복호화 실패 시 평문 폴백 없이 예외 전파 (접속 실패 → API 오류 반환)
            connection_info["password"] = decrypt_db_password(raw_str, aes_key)
            logger.info("연결 정보 조회 connection_id=%s, 비밀번호 복호화 완료", connection_id)
        else:
            connection_info["password"] = raw_str
            logger.info("연결 정보 조회 connection_id=%s, 비밀번호 평문 사용(ENCRYPTION_AES_KEY 미설정)", connection_id)
        
        # 캐시에 저장
        self._connection_cache[connection_id] = connection_info
        
        return connection_info
    
    def _parse_table_name(self, table_name: str) -> Tuple[str, str]:
        """table_name을 schema, table로 분리. 'schema.table' 또는 'table' 형식."""
        if "." in table_name:
            schema, table = table_name.split(".", 1)
            return schema.strip(), table.strip()
        return "public", table_name.strip()
    
    def _dialect(self, dbms_type_name: Optional[str] = None) -> str:
        """dbms_type_name -> dialect ('postgresql' | 'mysql' | 'oracle')."""
        raw = (dbms_type_name or "").strip().lower()
        if "mysql" in raw or raw == "mysql":
            return "mysql"
        if "oracle" in raw or raw == "oracle":
            return "oracle"
        return "postgresql"
    
    def _quote_identifier(self, name: str, dialect: str) -> str:
        """DBMS별 식별자 따옴표. PostgreSQL/Oracle \", MySQL `."""
        if dialect == "mysql":
            return f"`{name}`"
        return f'"{name}"'
    
    def _get_primary_key_columns(
        self,
        engine: Engine,
        table_name: str,
        dialect: str,
        default_schema: str = "public",
    ) -> List[str]:
        """
        해당 테이블의 PK 컬럼 목록 조회.
        PostgreSQL/MySQL: information_schema 사용.
        Oracle: ALL_CONSTRAINTS / ALL_CONS_COLUMNS 사용.
        """
        schema, table = self._parse_table_name(table_name)
        if dialect == "mysql" and schema == "public":
            schema = default_schema
        if dialect == "oracle" and schema == "public":
            schema = default_schema
        try:
            with engine.connect() as conn:
                if dialect == "oracle":
                    query = text("""
                        SELECT acc.COLUMN_NAME
                        FROM ALL_CONSTRAINTS ac
                        JOIN ALL_CONS_COLUMNS acc
                            ON ac.OWNER = acc.OWNER
                            AND ac.CONSTRAINT_NAME = acc.CONSTRAINT_NAME
                            AND ac.TABLE_NAME = acc.TABLE_NAME
                        WHERE ac.CONSTRAINT_TYPE = 'P'
                          AND UPPER(ac.TABLE_NAME) = UPPER(:table_name)
                          AND UPPER(ac.OWNER) = UPPER(:table_schema)
                        ORDER BY acc.POSITION
                    """)
                    result = conn.execute(query, {"table_schema": schema, "table_name": table})
                else:
                    query = text("""
                        SELECT kcu.column_name
                        FROM information_schema.table_constraints tc
                        JOIN information_schema.key_column_usage kcu
                            ON tc.constraint_name = kcu.constraint_name
                            AND tc.table_schema = kcu.table_schema
                        WHERE tc.constraint_type = 'PRIMARY KEY'
                            AND tc.table_schema = :table_schema
                            AND tc.table_name = :table_name
                        ORDER BY kcu.ordinal_position
                    """)
                    result = conn.execute(query, {"table_schema": schema, "table_name": table})
                return [row[0] for row in result]
        except Exception as e:
            logger.warning(f"PK 조회 실패 table={table_name}: {e}")
            return []

    def _get_first_column(
        self,
        engine: Engine,
        table_name: str,
        dialect: str,
        default_schema: str = "public",
    ) -> Optional[str]:
        """
        PK가 없을 때 사용할 테이블의 첫 번째 컬럼명 조회.
        information_schema.columns (또는 Oracle 동등) 사용.
        """
        schema, table = self._parse_table_name(table_name)
        if dialect == "mysql" and schema == "public":
            schema = default_schema
        if dialect == "oracle" and schema == "public":
            schema = default_schema
        try:
            with engine.connect() as conn:
                if dialect == "oracle":
                    query = text("""
                        SELECT COLUMN_NAME
                        FROM ALL_TAB_COLUMNS
                        WHERE OWNER = :table_schema AND TABLE_NAME = :table_name
                        ORDER BY COLUMN_ID
                        FETCH FIRST 1 ROW ONLY
                    """)
                    result = conn.execute(query, {"table_schema": schema, "table_name": table})
                else:
                    query = text("""
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_schema = :table_schema AND table_name = :table_name
                        ORDER BY ordinal_position
                        LIMIT 1
                    """)
                    result = conn.execute(query, {"table_schema": schema, "table_name": table})
                row = result.fetchone()
                return row[0] if row else None
        except Exception as e:
            logger.warning(f"첫 번째 컬럼 조회 실패 table={table_name}: {e}")
            return None

    def check_encryption(self, values: list) -> str:
        """
        값 리스트를 받아 암호화 여부("되어있음"/"안되어있음")를 반환
        
        로직:
            - PII(개인정보)가 식별되면 -> 평문이므로 "안되어있음"
            - PII가 식별되지 않으면(NONE) -> 암호화된 것으로 간주하여 "되어있음"
        """
        if not values:
            return "판단불가"

        # PII 분류 (배치 처리)
        pii_results = classify_batch(values, model_dir=self.model_path)
        
        # PII가 하나라도 발견되면 "안되어있음"으로 판단
        # (단, 오탐 방지를 위해 임계값을 둘 수 있음, 여기서는 1개라도 나오면 안된 것으로 간주)
        pii_detected_count = sum(1 for res in pii_results if res != "NONE")
        
        if pii_detected_count > 0:
            return "안되어있음" # PII 식별됨 -> 암호화 안됨
        else:
            return "되어있음" # PII 식별 안됨 -> 암호화 됨
    
    def classify(
        self,
        connection_id: str,
        table_name: str,
        column_name: str,
        key_column: Optional[str] = None,
    ) -> Dict[str, Any]:
        """전체 행 기준 암호화 여부 판단. key_column이 None이면 테이블 PK를 자동 조회하여 사용."""
        try:
            connection_info = self._get_connection_info(connection_id)
            target_engine = create_db_engine_for_connection(connection_info)
            dialect = self._dialect(connection_info.get("dbms_type_name"))
            if dialect == "mysql":
                default_schema = connection_info.get("db_name", "public")
            elif dialect == "oracle":
                default_schema = connection_info.get("username", "public")  # Oracle owner = username
            else:
                default_schema = "public"
            schema, table = self._parse_table_name(table_name)
            if dialect == "mysql" and schema == "public":
                schema = default_schema
            if dialect == "oracle" and schema == "public":
                schema = default_schema
            if key_column is None:
                pk_cols = self._get_primary_key_columns(
                    target_engine, table_name, dialect, default_schema
                )
                if pk_cols:
                    key_column = pk_cols[0]
                    logger.info(f"자동 PK 사용: table={table_name} -> key_column={key_column}")
                else:
                    key_column = self._get_first_column(
                        target_engine, table_name, dialect, default_schema
                    ) or "id"
                    logger.info(f"PK 없음, 첫 번째 컬럼 사용: table={table_name} -> key_column={key_column}")
            logger.info(
                f"암호화 여부 판단 시작: connection_id={connection_id}, dbms={dialect}, "
                f"table={table_name}, column={column_name}, key_column={key_column}"
            )
            # 전체 행 조회 (키 + 값 컬럼). DBMS별 식별자 따옴표
            qs, qt = self._quote_identifier(schema, dialect), self._quote_identifier(table, dialect)
            qk, qc = self._quote_identifier(key_column, dialect), self._quote_identifier(column_name, dialect)
            from_clause = f"{qs}.{qt}"
            query = text(f"SELECT {qk}, {qc} FROM {from_clause}")
            with target_engine.connect() as conn:
                df = pd.read_sql(query, conn)
            total_records = len(df)
            if total_records == 0:
                return {
                    "table_name": table_name,
                    "column_name": column_name,
                    "key_column": key_column,
                    "encryption_status": "판단불가",
                    "total_records": 0,
                    "encrypted_records": 0,
                    "unenc_record_keys": [],
                    "reason": "데이터 없음",
                }
            # null/빈 값은 ""로 통일해 행 순서 유지 (키와 1:1 매칭)
            values = [
                str(val).strip() if pd.notna(val) and str(val).strip() else ""
                for val in df[column_name]
            ]
            pii_results = classify_batch(values, model_dir=self.model_path)
            encrypted_count = sum(1 for r in pii_results if r == "NONE")
            # 미암호화(PII 감지) 행의 키 수집 (JSON 직렬화 가능한 타입으로)
            unenc_keys: List[Any] = []
            for i in range(len(pii_results)):
                if pii_results[i] != "NONE":
                    key_val = df.iloc[i][key_column]
                    if pd.isna(key_val):
                        continue
                    if isinstance(key_val, (np.integer, np.int64, np.int32)):
                        unenc_keys.append(int(key_val))
                    elif isinstance(key_val, (int, float)) and not isinstance(key_val, bool):
                        unenc_keys.append(int(key_val) if key_val == int(key_val) else key_val)
                    else:
                        unenc_keys.append(str(key_val))
            # 반환 시 오름차순 정렬 후 최대 10개만 (전체 검사는 유지)
            sort_key = lambda x: (0 if isinstance(x, (int, float)) else 1, x if isinstance(x, (int, float)) else str(x))
            unenc_record_keys = sorted(unenc_keys, key=sort_key)[:10]
            status = "안되어있음" if unenc_keys else "되어있음"
            return {
                "table_name": table_name,
                "column_name": column_name,
                "key_column": key_column,
                "encryption_status": status,
                "total_records": total_records,
                "encrypted_records": encrypted_count,
                "unenc_record_keys": unenc_record_keys,
            }
        except Exception as e:
            logger.error(f"암호화 여부 판단 오류: {str(e)}", exc_info=True)
            return {
                "table_name": table_name,
                "column_name": column_name,
                "key_column": key_column if 'key_column' in dir() else None,
                "encryption_status": "에러",
                "error_message": str(e),
                "total_records": 0,
                "encrypted_records": 0,
                "unenc_record_keys": [],
            }