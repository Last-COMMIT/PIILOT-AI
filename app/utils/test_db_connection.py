import pandas as pd
import psycopg

def get_psycopg_connection():
    """psycopg Connection을 반환합니다 (기존 코드 호환성)"""
    return psycopg.connect(
        # "dbname=kt_db user=kt_user password=1q2w3e4r host=localhost"
        "dbname=pgvector_test user=postgres password=1234 host=localhost"
    )

def get_data(query):
    with get_psycopg_connection() as conn:
        return pd.read_sql(query, conn)