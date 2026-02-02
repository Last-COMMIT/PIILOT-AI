"""
파일 서버 연결 정보 조회 (메인 DB file_server_connections)
"""
from typing import Dict, Any, Optional
from urllib.parse import urlparse
from sqlalchemy import create_engine, text

from app.core.config import settings
from app.utils.aes_decrypt import decrypt_db_password
from app.utils.db_connect import create_db_engine
from app.core.logging import logger

_engine = None
_connection_cache: Dict[int, Dict[str, Any]] = {}


def _get_main_engine():
    """메인 DB 엔진 (DATABASE_URL 기반, 캐시)"""
    global _engine
    if _engine is not None:
        return _engine
    if not getattr(settings, "DATABASE_URL", None):
        raise RuntimeError("DATABASE_URL이 설정되지 않았습니다.")
    parsed = urlparse(settings.DATABASE_URL.replace("postgresql+psycopg://", "postgresql://"))
    _engine = create_db_engine(
        user=parsed.username,
        password=parsed.password,
        host=parsed.hostname,
        port=parsed.port or 5432,
        database=parsed.path.lstrip("/"),
    )
    return _engine


def get_file_server_connection(connection_id: int) -> Dict[str, Any]:
    """
    메인 DB에서 파일 서버 연결 정보 조회 (비밀번호 복호화 포함)

    Returns:
        host, port, default_path, username, password, server_type_name
    """
    if connection_id in _connection_cache:
        return _connection_cache[connection_id]

    engine = _get_main_engine()
    try:
        q = text("""
            SELECT fsc.host, fsc.port, fsc.default_path, fsc.username, fsc.encrypted_password,
                   fst.name AS server_type_name
            FROM file_server_connections fsc
            LEFT JOIN file_server_types fst ON fsc.server_type_id = fst.id
            WHERE fsc.id = :connection_id
        """)
        with engine.connect() as conn:
            row = conn.execute(q, {"connection_id": connection_id}).fetchone()
    except Exception as e:
        logger.debug("file_server_types JOIN 조회 실패, 단일 테이블로 폴백: %s", e)
        q = text("""
            SELECT host, port, default_path, username, encrypted_password
            FROM file_server_connections
            WHERE id = :connection_id
        """)
        with engine.connect() as conn:
            row = conn.execute(q, {"connection_id": connection_id}).fetchone()

    if not row:
        raise ValueError(f"파일 서버 연결 정보를 찾을 수 없습니다: connection_id={connection_id}")

    info = {
        "host": row.host,
        "port": int(row.port or 9000),
        "default_path": (row.default_path or "").strip().rstrip("/"),
        "username": row.username,
        "encrypted_password": row.encrypted_password,
        "server_type_name": getattr(row, "server_type_name", None) or "WebDAV",
    }

    raw = str(info["encrypted_password"] or "").strip()
    aes_key = getattr(settings, "ENCRYPTION_AES_KEY", None) or ""
    if aes_key and len(aes_key.encode("utf-8")) == 32:
        info["password"] = decrypt_db_password(raw, aes_key)
    else:
        info["password"] = raw
    del info["encrypted_password"]

    _connection_cache[connection_id] = info
    return info
