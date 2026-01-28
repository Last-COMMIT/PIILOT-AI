"""
대상 DB 데이터 샘플링 (encryption_classifier에서 분리)
"""
import pandas as pd
from sqlalchemy import text
from typing import List
from app.core.logging import logger


def sample_column_values(engine, table_name: str, column_name: str, limit: int = 100) -> List[str]:
    """
    대상 DB에서 특정 컬럼의 데이터를 샘플링

    Args:
        engine: SQLAlchemy Engine
        table_name: 테이블 이름
        column_name: 컬럼 이름
        limit: 샘플 개수

    Returns:
        빈 값 제거된 문자열 리스트
    """
    query = text(f'SELECT "{column_name}" FROM "{table_name}" LIMIT {limit}')
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    values = [str(val).strip() for val in df[column_name] if pd.notna(val) and str(val).strip()]
    logger.info(f"데이터 샘플링 완료: {table_name}.{column_name} -> {len(values)}건")
    return values
