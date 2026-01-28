"""
법령 Vector DB (읽기 전용)
"""
from typing import List, Dict, Optional
from app.crud.db_connect import create_db_engine, get_connection
from urllib.parse import urlparse
from app.core.config import settings
from app.core.logging import logger
from app.ml.embedding_model import EmbeddingModel


class VectorDB:
    """법령 데이터 Vector DB 관리 (읽기 전용) - PostgreSQL + pgvector"""

    def __init__(self, table_name: Optional[str] = None):
        parsed_url = urlparse(settings.DATABASE_URL.replace('postgresql+psycopg://', 'postgresql://'))
        self.db_user = parsed_url.username
        self.db_password = parsed_url.password
        self.db_host = parsed_url.hostname
        self.db_port = parsed_url.port or 5432
        self.db_name = parsed_url.path.lstrip('/')
        self.table_name = table_name or "law_data"

        self.main_engine = create_db_engine(
            user=self.db_user,
            password=self.db_password,
            host=self.db_host,
            port=self.db_port,
            database=self.db_name,
        )

        self._embedding_model = EmbeddingModel()
        logger.info(f"VectorDB 초기화: table={self.table_name}")

    def search(self, query: str, n_results: int = 5, law_name_filter: Optional[str] = None) -> List[Dict]:
        """법령 데이터 검색 (읽기 전용)"""
        try:
            import time
            search_start_time = time.time()
            logger.debug(f"법령 검색 시작: query='{query[:50]}...', n_results={n_results}")

            query_embedding = self._embedding_model.generate_query_embedding(query)
            query_vector_str = self._embedding_model.vector_to_str(query_embedding)

            conn = get_connection(
                user=self.db_user,
                password=self.db_password,
                host=self.db_host,
                port=self.db_port,
                database=self.db_name,
            )

            try:
                with conn.cursor() as cursor:
                    if law_name_filter:
                        cursor.execute(f"""
                            SELECT
                                id, law_name, article, chunk_text, page,
                                1 - (embedding <=> %s::vector) as similarity
                            FROM {self.table_name}
                            WHERE law_name = %s
                            ORDER BY embedding <=> %s::vector
                            LIMIT %s
                        """, (query_vector_str, law_name_filter, query_vector_str, n_results))
                    else:
                        cursor.execute(f"""
                            SELECT
                                id, law_name, article, chunk_text, page,
                                1 - (embedding <=> %s::vector) as similarity
                            FROM {self.table_name}
                            ORDER BY embedding <=> %s::vector
                            LIMIT %s
                        """, (query_vector_str, query_vector_str, n_results))

                    results = cursor.fetchall()

                    formatted_results = []
                    for row in results:
                        formatted_results.append({
                            "id": str(row[0]),
                            "text": row[3],
                            "metadata": {
                                "law_name": row[1],
                                "article": row[2],
                                "page": row[4],
                            },
                            "distance": 1.0 - float(row[5]),
                        })

                    total_search_time = time.time() - search_start_time
                    logger.info(f"법령 검색 완료: {len(formatted_results)}개 결과 반환 (총 소요 시간: {total_search_time:.2f}초)")
                    return formatted_results

            finally:
                conn.close()

        except Exception as e:
            logger.error(f"법령 검색 오류: {str(e)}", exc_info=True)
            return []
