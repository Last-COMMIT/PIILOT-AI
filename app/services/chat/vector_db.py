"""
법령 Vector DB (읽기 전용)
"""
from typing import List, Dict, Optional
from app.utils.db_connect import create_db_engine, get_connection
from urllib.parse import urlparse
from app.config import settings
from app.utils.logger import logger
from transformers import AutoTokenizer, AutoModel
from torch import Tensor
import torch
import torch.nn.functional as F

class VectorDB:
    """법령 데이터 Vector DB 관리 (읽기 전용) - PostgreSQL + pgvector"""
    
    # 클래스 변수로 모델 캐싱 (모든 인스턴스가 공유)
    _tokenizer: Optional[AutoTokenizer] = None
    _model: Optional[AutoModel] = None
    
    def __init__(self, table_name: Optional[str] = None):
        """
        Args:
            table_name: 법령 데이터 테이블명 (기본값: "law_data")
        """
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
            database=self.db_name
        )
        logger.info(f"VectorDB 초기화: table={self.table_name}")

    @staticmethod
    def _average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """평균 풀링으로 임베딩 생성"""
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    @staticmethod
    def _get_detailed_instruct(task_description: str, query: str) -> str:
        """임베딩용 instruction 포맷"""
        return f'Instruct: {task_description}\nQuery: {query}'

    def _load_embedding_model(self):
        """임베딩 모델 로드 (한 번만 로드)"""
        if VectorDB._tokenizer is None or VectorDB._model is None:
            import time
            model_start_time = time.time()
            logger.info("임베딩 모델 로드 시작: intfloat/multilingual-e5-large-instruct")
            logger.info("(첫 로드 시 다운로드 및 로딩에 시간이 걸릴 수 있습니다)")
            model_name = 'intfloat/multilingual-e5-large-instruct'
            
            try:
                logger.debug("Tokenizer 로딩 중...")
                VectorDB._tokenizer = AutoTokenizer.from_pretrained(model_name)
                logger.debug("✓ Tokenizer 로딩 완료")
                
                logger.debug("Model 로딩 중... (시간이 오래 걸릴 수 있습니다)")
                VectorDB._model = AutoModel.from_pretrained(model_name)
                VectorDB._model.eval()
                logger.debug("✓ Model 로딩 완료")
                
                model_load_time = time.time() - model_start_time
                logger.info(f"✓ 임베딩 모델 로드 완료! (소요 시간: {model_load_time:.2f}초)")
            except Exception as e:
                logger.error(f"임베딩 모델 로드 실패: {str(e)}", exc_info=True)
                raise
        
        return VectorDB._tokenizer, VectorDB._model

    def _generate_query_embedding(self, question: str) -> Tensor:
        """질문을 임베딩 벡터로 변환"""
        import time
        embedding_start_time = time.time()
        logger.debug("임베딩 모델 로드 확인 중...")
        tokenizer, model = self._load_embedding_model()
        
        task = '법률 질문에 대해 관련된 법령 조항을 검색합니다'
        formatted_text = self._get_detailed_instruct(task, question)
        
        # 토크나이징
        logger.debug("질문 토크나이징 중...")
        batch_dict = tokenizer(
            [formatted_text],
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        logger.debug("✓ 토크나이징 완료")
        
        # 임베딩 생성
        logger.debug("임베딩 벡터 생성 중...")
        with torch.no_grad():
            outputs = model(**batch_dict)
            embedding = self._average_pool(
                outputs.last_hidden_state,
                batch_dict['attention_mask']
            )
        
        # 정규화
        embedding = F.normalize(embedding, p=2, dim=1)
        
        embedding_time = time.time() - embedding_start_time
        logger.debug(f"✓ 임베딩 벡터 생성 완료 (소요 시간: {embedding_time:.2f}초)")
        
        return embedding[0]  # 첫 번째 (유일한) 임베딩 반환

    @staticmethod
    def _vector_to_str(vector: Tensor) -> str:
        """벡터를 PostgreSQL pgvector 형식 문자열로 변환"""
        return '[' + ','.join(map(str, vector.tolist())) + ']'
    
    def search(self, query: str, n_results: int = 5, law_name_filter: Optional[str] = None) -> List[Dict]:
        """
        법령 데이터 검색 (읽기 전용)
        
        Args:
            query: 검색 쿼리
            n_results: 반환할 결과 수
            law_name_filter: 법령명 필터 (선택사항)
            
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
        try:
            import time
            search_start_time = time.time()
            logger.debug(f"법령 검색 시작: query='{query[:50]}...', n_results={n_results}")
            
            # 1. 질문을 임베딩 벡터로 변환
            logger.debug("질문 임베딩 생성 중...")
            query_embedding = self._generate_query_embedding(query)
            query_vector_str = self._vector_to_str(query_embedding)
            logger.debug("✓ 질문 임베딩 생성 완료")
            
            # 2. 데이터베이스에서 유사한 청크 검색
            logger.debug(f"DB 연결 중... (host={self.db_host}, db={self.db_name})")
            db_connect_start = time.time()
            conn = get_connection(
                user=self.db_user,
                password=self.db_password,
                host=self.db_host,
                port=self.db_port,
                database=self.db_name
            )
            db_connect_time = time.time() - db_connect_start
            logger.debug(f"✓ DB 연결 완료 (소요 시간: {db_connect_time:.2f}초)")
            
            try:
                logger.debug(f"SQL 쿼리 실행 중... (table={self.table_name})")
                query_start_time = time.time()
                with conn.cursor() as cursor:
                    if law_name_filter:
                        # 법령명 필터링 포함
                        cursor.execute(f"""
                            SELECT 
                                id,
                                law_name,
                                article,
                                chunk_text,
                                page,
                                1 - (embedding <=> %s::vector) as similarity
                            FROM {self.table_name}
                            WHERE law_name = %s
                            ORDER BY embedding <=> %s::vector
                            LIMIT %s
                        """, (query_vector_str, law_name_filter, query_vector_str, n_results))
                    else:
                        # 전체 검색
                        cursor.execute(f"""
                            SELECT 
                                id,
                                law_name,
                                article,
                                chunk_text,
                                page,
                                1 - (embedding <=> %s::vector) as similarity
                            FROM {self.table_name}
                            ORDER BY embedding <=> %s::vector
                            LIMIT %s
                        """, (query_vector_str, query_vector_str, n_results))
                    
                    results = cursor.fetchall()
                    query_time = time.time() - query_start_time
                    logger.debug(f"✓ SQL 쿼리 실행 완료 (소요 시간: {query_time:.2f}초, 결과: {len(results)}개)")
                    
                    # 3. 결과를 스펙에 맞게 변환
                    logger.debug("검색 결과 포맷팅 중...")
                    formatted_results = []
                    for row in results:
                        formatted_results.append({
                            "id": str(row[0]),
                            "text": row[3],  # chunk_text
                            "metadata": {
                                "law_name": row[1],
                                "article": row[2],
                                "page": row[4]
                            },
                            "distance": 1.0 - float(row[5])  # similarity를 distance로 변환
                        })
                    
                    total_search_time = time.time() - search_start_time
                    logger.info(f"법령 검색 완료: {len(formatted_results)}개 결과 반환 (총 소요 시간: {total_search_time:.2f}초)")
                    return formatted_results
                    
            finally:
                conn.close()
                
        except Exception as e:
            logger.error(f"법령 검색 오류: {str(e)}", exc_info=True)
            return []
