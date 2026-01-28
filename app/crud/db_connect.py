"""
DB 연결 관리 (SQLAlchemy Engine 풀링 + psycopg 직접 연결)
"""
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
    dbname: str = None,
) -> any:
    """
    DB 연결 파라미터를 받아서 SQLAlchemy Engine을 생성하고 캐시합니다.
    """
    db_name = database or dbname
    if not db_name:
        raise ValueError("database 또는 dbname 중 하나는 필수입니다")

    connection_key = f"{user}@{host}:{port}/{db_name}"

    if connection_key not in _engines:
        connection_string = f"postgresql+psycopg://{user}:{password}@{host}:{port}/{db_name}"
        _engines[connection_key] = create_engine(
            connection_string,
            pool_pre_ping=True,
            pool_recycle=3600,
        )

    return _engines[connection_key]


def get_sql_database(
    user: str,
    password: str,
    host: str,
    port: int = 5432,
    database: str = None,
    dbname: str = None,
) -> SQLDatabase:
    """
    DB 연결 파라미터를 받아서 SQLDatabase 인스턴스를 생성하고 캐시합니다.
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
    dbname: Optional[str] = None,
):
    """
    psycopg Connection을 반환합니다.
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
        return psycopg.connect(
            "dbname=finance_db user=finance_user password=1q2w3e4r host=localhost"
        )
