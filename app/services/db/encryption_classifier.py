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
from app.utils.db_connect import create_db_engine
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
        
        # 메인 DB에서 조회
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
                "encrypted_password": row.encrypted_password
            }
        
        # 비밀번호 복호화 (TODO: 나중에 구현)
        decrypted_password = connection_info["encrypted_password"]
        connection_info["password"] = decrypted_password
        
        # 캐시에 저장
        self._connection_cache[connection_id] = connection_info
        
        return connection_info
    
    def _parse_table_name(self, table_name: str) -> Tuple[str, str]:
        """table_name을 schema, table로 분리. 'schema.table' 또는 'table' 형식."""
        if "." in table_name:
            schema, table = table_name.split(".", 1)
            return schema.strip(), table.strip()
        return "public", table_name.strip()
    
    def _get_primary_key_columns(self, engine: Engine, table_name: str) -> List[str]:
        """
        information_schema로 해당 테이블의 PK 컬럼 목록 조회 (ordinal_position 순).
        PostgreSQL/표준 information_schema 사용.
        """
        schema, table = self._parse_table_name(table_name)
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
        try:
            with engine.connect() as conn:
                result = conn.execute(query, {"table_schema": schema, "table_name": table})
                return [row.column_name for row in result]
        except Exception as e:
            logger.warning(f"PK 조회 실패 table={table_name}: {e}")
            return []
    
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
            target_engine = create_db_engine(
                user=connection_info["username"],
                password=connection_info["password"],
                host=connection_info["host"],
                port=connection_info["port"],
                database=connection_info["db_name"],
            )
            schema, table = self._parse_table_name(table_name)
            if key_column is None:
                pk_cols = self._get_primary_key_columns(target_engine, table_name)
                key_column = pk_cols[0] if pk_cols else "id"
                logger.info(f"자동 PK 사용: table={table_name} -> key_column={key_column}")
            logger.info(
                f"암호화 여부 판단 시작: connection_id={connection_id}, "
                f"table={table_name}, column={column_name}, key_column={key_column}"
            )
            # 전체 행 조회 (키 + 값 컬럼). schema.table 식별자 사용
            from_clause = f'"{schema}"."{table}"'
            query = text(
                f'SELECT "{key_column}", "{column_name}" FROM {from_clause}'
            )
            with target_engine.connect() as conn:
                df = pd.read_sql(query, conn)
            total_records = len(df)
            if total_records == 0:
                return {
                    "table_name": table_name,
                    "column_name": column_name,
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
                "encryption_status": "에러",
                "error_message": str(e),
                "total_records": 0,
                "encrypted_records": 0,
                "unenc_record_keys": [],
            }