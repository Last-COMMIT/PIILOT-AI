"""
DB 커넥션 정보 조회 (encryption_classifier에서 분리)
"""
from typing import Dict
from sqlalchemy import text
from app.core.logging import logger


class ConnectionRepository:
    """메인 DB에서 연결 정보를 조회하는 리포지토리"""

    def __init__(self, main_engine):
        self.main_engine = main_engine
        self._cache: Dict[str, Dict] = {}

    def get_connection_info(self, connection_id: str) -> Dict:
        """
        메인 DB에서 connection_id로 연결 정보 조회
        """
        if not self.main_engine:
            raise RuntimeError("Main DB Engine이 초기화되지 않았습니다.")

        if connection_id in self._cache:
            return self._cache[connection_id]

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
            }

        # 비밀번호 복호화 (TODO: 나중에 구현)
        connection_info["password"] = connection_info["encrypted_password"]

        self._cache[connection_id] = connection_info

        return connection_info
