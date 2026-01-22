"""
법령 Vector DB (읽기 전용)
"""
from typing import List, Dict, Optional

from app.config import settings
from app.utils.logger import logger

# NOTE:
# - 기존 ChromaDB 구현을 제거하고 PostgreSQL + pgvector 기반으로 전환합니다.
# - 실제 DB 연결/테이블 생성/임베딩 생성/검색 구현은 프로젝트 요구사항에 맞게 TODO로 남깁니다.


class VectorDB:
    """법령 데이터 Vector DB 관리 (읽기 전용) - PostgreSQL + pgvector"""
    
    def __init__(
        self,
        database_url: Optional[str] = None,
        table_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
        embedding_dim: Optional[int] = None,
    ):
        """
        Args:
            database_url: PostgreSQL 접속 URL (미지정 시 settings.PGVECTOR_DATABASE_URL 또는 settings.DATABASE_URL 사용)
            table_name: 벡터 테이블명 (미지정 시 settings.PGVECTOR_TABLE_NAME 사용)
            embedding_model: 임베딩 모델명 (미지정 시 settings.PGVECTOR_EMBEDDING_MODEL 사용)
            embedding_dim: 임베딩 차원 (미지정 시 settings.PGVECTOR_EMBEDDING_DIM 사용)
        """
        self.database_url = database_url or settings.PGVECTOR_DATABASE_URL or settings.DATABASE_URL
        self.table_name = table_name or settings.PGVECTOR_TABLE_NAME
        self.embedding_model = embedding_model or settings.PGVECTOR_EMBEDDING_MODEL
        self.embedding_dim = embedding_dim or settings.PGVECTOR_EMBEDDING_DIM

        # TODO(pgvector):
        # - SQLAlchemy 엔진/세션 생성 또는 psycopg 커넥션 풀 구성
        # - pgvector extension 활성화 (CREATE EXTENSION IF NOT EXISTS vector;)
        # - regulations 테이블 스키마 정의/마이그레이션
        #   예시 컬럼: id (uuid/bigserial), text (text), metadata (jsonb), embedding (vector(embedding_dim))
        # - 인덱스(HNSW/IVFFLAT) 생성 전략 결정
        self._engine = None

        logger.info(
            "VectorDB 초기화(pgvector): db=%s, table=%s, embedding_model=%s, dim=%s",
            self.database_url,
            self.table_name,
            self.embedding_model,
            self.embedding_dim,
        )
    
    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        법령 데이터 검색 (읽기 전용)
        
        Args:
            query: 검색 쿼리
            n_results: 반환할 결과 수
            
        Returns:
            [
                {
                    "id": str,
                    "text": str,
                    "metadata": Dict,
                    "distance": float
                },
                ...
            ]
        """
        # TODO(pgvector):
        # - query를 embedding_model로 임베딩(embedding_dim) 생성
        # - pgvector 코사인/유클리드 거리로 TOP-K 검색
        #   예: ORDER BY embedding <=> :query_embedding LIMIT :n_results
        # - 결과를 위 스펙(id/text/metadata/distance)으로 변환해 반환
        logger.debug(f"법령 검색: {query}, 결과 수: {n_results}")
        return []

    # TODO(pgvector):
    # def upsert_regulations(self, items: List[Dict]) -> None:
    #     """법령 데이터 적재/갱신 (setup_vector_db.py에서 사용)."""
    #     ...

