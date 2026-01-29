"""
암호화 여부 판단 (분류 모델)
"""
from typing import Dict, Optional, Any
from pathlib import Path
from urllib.parse import urlparse
import pandas as pd
from sqlalchemy import create_engine, text
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
    
    def classify(self, connection_id: str, table_name: str, column_name: str) -> Dict[str, Any]:
        """데이터 샘플의 암호화 여부 판단 (DB 접속 포함)."""
        try:
            logger.info(f"암호화 여부 판단 시작: connection_id={connection_id}, table={table_name}, column={column_name}")
            
            # 1. 연결 정보 조회
            connection_info = self._get_connection_info(connection_id)
            
            # 2. 실제 DB 연결
            target_engine = create_db_engine(
                user=connection_info["username"],
                password=connection_info["password"],
                host=connection_info["host"],
                port=connection_info["port"],
                database=connection_info["db_name"]
            )
            
            # 3. 데이터 조회 (샘플링)
            query = text(f'SELECT "{column_name}" FROM "{table_name}" LIMIT 100')
            with target_engine.connect() as conn:
                df = pd.read_sql(query, conn)
            
            # 빈 값 제거 및 문자열 변환
            values = [str(val).strip() for val in df[column_name] if pd.notna(val) and str(val).strip()]
            
            total_records = len(values)
            
            if total_records == 0:
                return {
                    "table_name": table_name,
                    "column_name": column_name,
                    "encryption_status": "판단불가",
                    "total_records": 0,
                    "encrypted_records": 0,
                    "reason": "데이터 없음",
                }
            
            # 4. PII 분류 (샘플별 결과 사용)
            pii_results = classify_batch(values, model_dir=self.model_path)
            encrypted_count = sum(1 for r in pii_results if r == "NONE")
            pii_detected_count = sum(1 for r in pii_results if r != "NONE")
            status = "안되어있음" if pii_detected_count > 0 else "되어있음"
            
            return {
                "table_name": table_name,
                "column_name": column_name,
                "encryption_status": status,
                "total_records": total_records,
                "encrypted_records": encrypted_count,
            }
        
        except Exception as e:
            logger.error(f"암호화 여부 판단 오류: {str(e)}", exc_info=True)
            return {
                "table_name": table_name,
                "column_name": column_name,
                "encryption_status": "에러",
                "error_message": str(e),
            }