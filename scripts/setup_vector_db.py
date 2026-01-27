"""
Vector DB 초기화 스크립트
"""
import asyncio
from app.services.chat.vector_db import VectorDB

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModel
from torch import Tensor
from scripts.embeddings import load_chunks_from_json
from scripts.embeddings import generate_embeddings_batch
from app.utils.logger import logger
# 테스트용 connection
from app.utils.test_db_connection import get_psycopg_connection

# # 프로젝트 루트를 sys.path에 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent
# sys.path.insert(0, str(project_root))

# 모델 경로 설정 (임베딩 모델용)
# model_cache_dir = os.path.join(project_root, "models", "multilingual-e5-large-instruct")
# os.makedirs(model_cache_dir, exist_ok=True)

# 환경변수로도 설정
# os.environ["HF_HOME"] = model_cache_dir
# os.environ["TRANSFORMERS_CACHE"] = model_cache_dir

def insert_to_database(chunks: List[Dict[str, Any]], embeddings: Tensor, batch_size: int = 100):
    """데이터베이스에 청크와 임베딩 삽입"""
    # conn = get_connection()
    conn = get_psycopg_connection()
    insert_query = """
        INSERT INTO test2_law_data (
            chunk_text, embedding,
            document_title, law_name, article, page, effective_date
        ) VALUES (
            %s, %s::vector, %s, %s, %s, %s, %s
        )
    """
    
    try:
        with conn.cursor() as cursor:
            inserted_count = 0
            
            for batch_start in range(0, len(chunks), batch_size):
                batch_data = []
                for chunk, embedding in zip(
                    chunks[batch_start:batch_start + batch_size],
                    embeddings[batch_start:batch_start + batch_size]
                ):
                    metadata = chunk.get('metadata', {})
                    
                    # 날짜 처리
                    effective_date = None
                    if metadata.get('effective_date'):
                        try:
                            effective_date = datetime.strptime(
                                str(metadata['effective_date']), '%Y-%m-%d'
                            ).date()
                        except:
                            pass
                    
                    batch_data.append((
                        chunk['chunk_text'],
                        embedding.tolist(),
                        metadata.get('document_title'),
                        metadata.get('law_name'),
                        metadata.get('article'),
                        metadata.get('page'),
                        effective_date
                    ))
                
                cursor.executemany(insert_query, batch_data)
                conn.commit()
                inserted_count += len(batch_data)
    finally:
        conn.close()

async def main():
    """법령 데이터를 Vector DB에 로드"""
    json_path = project_root / "test_pdf" / "chunks_output.json"
    if not os.path.exists(json_path):
        logger.info(f"파일 없음: {json_path}")
        return
    
    # JSON 파일 로드
    chunks = load_chunks_from_json(json_path)

    # 모델 로드
    model_name = 'intfloat/multilingual-e5-large-instruct'
    # tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModel.from_pretrained(model_name, cache_dir=model_cache_dir)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    # 임베딩 생성
    embeddings = generate_embeddings_batch(
        [chunk['chunk_text'] for chunk in chunks],
        tokenizer,
        model,
        batch_size=32
    )

    # 데이터베이스 삽입
    insert_to_database(chunks, embeddings)
    
    vector_db = VectorDB()

    # TODO(pgvector):
    # - PostgreSQL에 pgvector extension 설치/활성화
    # - regulations 테이블 생성/마이그레이션
    # - 법령 텍스트 로드(파일/DB/크롤링 등)
    # - 임베딩 생성 후 upsert_regulations(items)로 적재
    
    logger.info(f"Vector DB 초기화 완료")


if __name__ == "__main__":
    asyncio.run(main())