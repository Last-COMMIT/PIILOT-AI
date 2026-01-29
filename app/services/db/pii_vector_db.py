"""
PII 표준단어 Vector DB (읽기 전용)
- pii_standard_words 테이블에서 유사도 검색
"""
from typing import List, Dict, Optional
from torch import Tensor
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from urllib.parse import urlparse

from app.core.config import settings
from app.core.logging import logger
from app.crud.db_connect import get_connection
from app.core.model_manager import ModelManager


class PIIVectorDB:
    """PII 표준단어 검색용 VectorDB"""

    # 클래스 변수로 임베딩 모델 캐싱 (싱글톤)
    _tokenizer: Optional[AutoTokenizer] = None
    _model: Optional[AutoModel] = None

    def __init__(self, table_name: str = "pii_standard_words"):
        """
        Args:
            table_name: PII 표준단어 테이블명
        """
        parsed_url = urlparse(settings.DATABASE_URL.replace('postgresql+psycopg://', 'postgresql://'))
        self.db_user = parsed_url.username
        self.db_password = parsed_url.password
        self.db_host = parsed_url.hostname
        self.db_port = parsed_url.port or 5432
        self.db_name = parsed_url.path.lstrip('/')
        self.table_name = table_name

        logger.info(f"PIIVectorDB 초기화: table={self.table_name}")

    def _load_embedding_model(self):
        """E5 임베딩 모델 로드 (한 번만)"""
        if PIIVectorDB._tokenizer is None or PIIVectorDB._model is None:
            model_name = 'intfloat/multilingual-e5-large-instruct'
            cache_dir = ModelManager.get_cache_dir()

            PIIVectorDB._tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            PIIVectorDB._model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
            PIIVectorDB._model.eval()

            logger.info("PIIVectorDB: 임베딩 모델 로드 완료")

        return PIIVectorDB._tokenizer, PIIVectorDB._model

    @staticmethod
    def _average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """평균 풀링으로 임베딩 생성"""
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    @staticmethod
    def _get_detailed_instruct(task_description: str, query: str) -> str:
        """임베딩용 instruction 포맷"""
        return f'Instruct: {task_description}\nQuery: {query}'

    def _generate_batch_embeddings(self, column_names: List[str], batch_size: int = 32) -> Tensor:
        """
        여러 컬럼명을 배치 단위로 임베딩

        Args:
            column_names: 컬럼명 리스트
            batch_size: 한 번에 처리할 컬럼 수

        Returns:
            Tensor: [num_columns, embedding_dim]
        """
        tokenizer, model = self._load_embedding_model()

        task = "데이터베이스 컬럼명을 분석하여 개인정보 포함 여부와 유형을 식별합니다"
        formatted_texts = [self._get_detailed_instruct(task, col) for col in column_names]

        all_embeddings = []

        # batch_size씩 나눠서 처리
        for i in range(0, len(formatted_texts), batch_size):
            batch_texts = formatted_texts[i:i + batch_size]

            batch_dict = tokenizer(
                batch_texts,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )

            with torch.no_grad():
                outputs = model(**batch_dict)
                embeddings = self._average_pool(
                    outputs.last_hidden_state,
                    batch_dict['attention_mask']
                )
                embeddings = F.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings)

        # 모든 배치 결과 합치기
        return torch.cat(all_embeddings, dim=0)

    @staticmethod
    def _vector_to_str(vector: Tensor) -> str:
        """벡터를 PostgreSQL pgvector 형식 문자열로 변환"""
        return '[' + ','.join(map(str, vector.tolist())) + ']'

    def batch_search(
        self,
        column_names: List[str],
        top_k: int = 1
    ) -> Dict[str, List[Dict]]:
        """
        여러 컬럼명을 한 번에 검색 (배치 최적화)

        Args:
            column_names: 컬럼명 리스트
            top_k: 각 컬럼당 검색할 상위 결과 수

        Returns:
            {
                "user_nm": [{"abbr": "NM", "chunk_text": "...", "similarity": 0.95}],
                "email": [{"abbr": "EM", "chunk_text": "...", "similarity": 0.88}],
                ...
            }
        """
        if not column_names:
            return {}

        # 1. 배치 임베딩 생성
        embeddings = self._generate_batch_embeddings(column_names)

        # 2. DB 연결
        conn = get_connection(
            user=self.db_user,
            password=self.db_password,
            host=self.db_host,
            port=self.db_port,
            database=self.db_name
        )

        results = {}

        try:
            with conn.cursor() as cursor:
                # 3. 각 컬럼별로 검색
                for idx, column_name in enumerate(column_names):
                    query_vector_str = self._vector_to_str(embeddings[idx])

                    cursor.execute(f"""
                        SELECT
                            abbr,
                            chunk_text,
                            1 - (embedding <=> %s::vector) as similarity
                        FROM {self.table_name}
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                    """, (query_vector_str, query_vector_str, top_k))

                    rows = cursor.fetchall()

                    results[column_name] = [
                        {
                            "abbr": row[0],
                            "chunk_text": row[1],
                            "similarity": float(row[2])
                        }
                        for row in rows
                    ]
        finally:
            conn.close()

        return results
