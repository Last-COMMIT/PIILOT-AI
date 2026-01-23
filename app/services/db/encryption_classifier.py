"""
암호화 여부 판단 (분류 모델)
"""
from typing import Dict, Optional
from pathlib import Path
from urllib.parse import urlparse
import pandas as pd
from sqlalchemy import create_engine, text
from app.utils.logger import logger
from app.config import settings
from app.utils.db_connect import create_db_engine
from app.services.db.pii_classifier import classify_batch


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
        parsed_url = urlparse(settings.DATABASE_URL.replace('postgresql+psycopg://', 'postgresql://'))
        self.main_engine = create_db_engine(
            user=parsed_url.username,
            password=parsed_url.password,
            host=parsed_url.hostname,
            port=parsed_url.port or 5432,
            database=parsed_url.path.lstrip('/')
        )
        # 연결 정보 캐시
        self._connection_cache: Dict[str, Dict] = {}
        logger.info(f"EncryptionClassifier 초기화: {model_path}")
    
    def _get_connection_info(self, connection_id: str) -> Dict:
        """
        메인 DB에서 connection_id로 연결 정보 조회
        
        Args:
            connection_id: 연결 ID
        
        Returns:
            연결 정보 딕셔너리 (host, port, db_name, username, password)
        """
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
        # TODO: encrypted_password 복호화 로직 추가
        decrypted_password = connection_info["encrypted_password"]  # 임시로 암호화된 비밀번호 그대로 사용
        
        connection_info["password"] = decrypted_password
        
        # 캐시에 저장
        self._connection_cache[connection_id] = connection_info
        
        return connection_info
    
    def classify(self, connection_id: str, table_name: str, column_name: str) -> Dict[str, int]:
        """
        데이터 샘플의 암호화 여부 판단
        
        Args:
            connection_id: DB 연결 ID
            table_name: 테이블 이름
            column_name: 컬럼 이름
            
        Returns:
            {"total_records": int, "encrypted_records": int} 형태
        """
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
            
            # 3. 데이터 조회
            query = text(f'SELECT "{column_name}" FROM "{table_name}"')
            with target_engine.connect() as conn:
                df = pd.read_sql(query, conn)
            
            # 빈 값 제거 및 문자열 변환
            values = [str(val).strip() for val in df[column_name] if pd.notna(val) and str(val).strip()]
            
            total_records = len(values)
            
            if total_records == 0:
                logger.warning(f"조회된 데이터가 없습니다: table={table_name}, column={column_name}")
                return {"total_records": 0, "encrypted_records": 0}
            
            logger.debug(f"조회된 데이터 개수: {total_records}개")
            
            # 4. PII 분류 (배치 처리)
            pii_results = classify_batch(values, model_dir=self.model_path)
            
            # 5. 암호화된 레코드 수 계산 (NONE으로 분류된 개수)
            encrypted_records = sum(1 for result in pii_results if result == "NONE")
            
            logger.info(f"분류 완료: 총 {total_records}개 중 암호화 {encrypted_records}개")
            
            return {
                "total_records": total_records,
                "encrypted_records": encrypted_records
            }
        
        except Exception as e:
            logger.error(f"암호화 여부 판단 오류: {str(e)}", exc_info=True)
            # 에러 발생 시 0 반환
            return {"total_records": 0, "encrypted_records": 0}
    
    def is_encrypted(self, connection_id: str, table_name: str, column_name: str, threshold: float = 0.5) -> bool:
        """
        암호화 여부를 boolean으로 반환
        
        Args:
            connection_id: DB 연결 ID
            table_name: 테이블 이름
            column_name: 컬럼 이름
            threshold: 판단 임계값 (기본 0.5, 암호화 비율)
        
        Returns:
            True: 암호화됨, False: 평문
        """
        result = self.classify(connection_id, table_name, column_name)
        total = result.get("total_records", 0)
        encrypted = result.get("encrypted_records", 0)
        
        if total == 0:
            return False
        
        encryption_rate = encrypted / total
        return encryption_rate > threshold