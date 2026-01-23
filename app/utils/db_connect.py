import sys
from pathlib import Path

# 직접 실행 시 프로젝트 루트를 sys.path에 추가
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import psycopg
from sqlalchemy import create_engine
from typing import Optional, Dict
from langchain_community.utilities import SQLDatabase

# 전역 Engine 저장소 (연결 정보별로 관리)
_engines: Dict[str, any] = {}
_db_instances: Dict[str, SQLDatabase] = {}

def create_db_engine(
    user: str,
    password: str,
    host: str,
    port: int = 5432,
    database: str = None,
    dbname: str = None
) -> any:
    """
    DB 연결 파라미터를 받아서 SQLAlchemy Engine을 생성하고 캐시합니다.
    한번 생성한 Engine은 계속 재사용됩니다 (연결 풀 사용).
    
    Args:
        user: DB 사용자명
        password: DB 비밀번호
        host: DB 호스트
        port: DB 포트 (기본값: 5432)
        database: DB 이름 (database 또는 dbname 중 하나 필수)
        dbname: DB 이름 (database와 동일, 호환성을 위해)
    
    Returns:
        SQLAlchemy Engine 인스턴스
    """
    # dbname과 database 중 하나 사용
    db_name = database or dbname
    if not db_name:
        raise ValueError("database 또는 dbname 중 하나는 필수입니다")
    
    # 연결 정보를 키로 사용 (동일한 연결 정보면 같은 Engine 재사용)
    connection_key = f"{user}@{host}:{port}/{db_name}"
    
    if connection_key not in _engines:
        # SQLAlchemy connection string 생성
        connection_string = f"postgresql+psycopg://{user}:{password}@{host}:{port}/{db_name}"
        _engines[connection_key] = create_engine(
            connection_string,
            pool_pre_ping=True,  # 연결이 끊어졌는지 확인하고 재연결
            pool_recycle=3600,   # 1시간마다 연결 재생성
        )
    
    return _engines[connection_key]

def get_sql_database(
    user: str,
    password: str,
    host: str,
    port: int = 5432,
    database: str = None,
    dbname: str = None
) -> SQLDatabase:
    """
    DB 연결 파라미터를 받아서 SQLDatabase 인스턴스를 생성하고 캐시합니다.
    
    Returns:
        SQLDatabase 인스턴스
    """
    db_name = database or dbname
    connection_key = f"{user}@{host}:{port}/{db_name}"
    
    if connection_key not in _db_instances:
        engine = create_db_engine(user, password, host, port, database, dbname)
        _db_instances[connection_key] = SQLDatabase(engine)
    
    return _db_instances[connection_key]

def get_connection(
    user: Optional[str] = None,
    password: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    database: Optional[str] = None,
    dbname: Optional[str] = None
):
    """
    psycopg Connection을 반환합니다.
    파라미터가 없으면 기본값 사용 (하위 호환성)
    
    Args:
        user, password, host, port, database/dbname: DB 연결 정보
    """
    if all([user, password, host]):
        port = port or 5432
        db_name = database or dbname
        if not db_name:
            raise ValueError("database 또는 dbname 중 하나는 필수입니다")
        return psycopg.connect(
            f"dbname={db_name} user={user} password={password} host={host} port={port}"
        )
    else:
        # 기본값 사용 (하위 호환성)
        return psycopg.connect(
            "dbname=finance_db user=finance_user password=1q2w3e4r host=localhost"
        )
