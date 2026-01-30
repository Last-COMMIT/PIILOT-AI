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
from scripts.law_pdf_preprocess import build_chunks_with_metadata, extract_and_fix_pages
from scripts.embeddings import generate_embeddings_batch
from app.utils.logger import logger
# DB 연결 (기존 방식 사용)
from app.crud.db_connect import get_connection
from urllib.parse import urlparse
from app.core.config import settings

# # 프로젝트 루트를 sys.path에 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent

def insert_to_database(chunks: List[Dict[str, Any]], embeddings: Tensor, batch_size: int = 100):
    """데이터베이스에 청크와 임베딩 삽입"""
    # config.py의 DATABASE_URL 파싱하여 연결
    parsed_url = urlparse(settings.DATABASE_URL.replace('postgresql+psycopg://', 'postgresql://'))
    conn = get_connection(
        user=parsed_url.username,
        password=parsed_url.password,
        host=parsed_url.hostname,
        port=parsed_url.port or 5432,
        database=parsed_url.path.lstrip('/')
    )
    insert_query = """
        INSERT INTO law_data (
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

async def law_pdf_to_vector(file_path: str):
    """법령 데이터를 Vector DB에 로드"""

    # TODO(pgvector):
    # - PostgreSQL에 pgvector extension 설치/활성화
    # - regulations 테이블 생성/마이그레이션
    # - 법령 텍스트 로드(파일/DB/크롤링 등)
    # - 임베딩 생성 후 upsert_regulations(items)로 적재
    
    if not os.path.exists(file_path):
        logger.info(f"파일 없음: {file_path}")
        return

    fixed_pages = extract_and_fix_pages(file_path)

    if not fixed_pages:
        logger.error("fixed_pages가 비어있음")
        return

    # 청크 생성
    chunks = build_chunks_with_metadata(
        fixed_pages=fixed_pages,
        file_path=file_path
    )

    logger.info(f"chunks 생성 완료: {len(chunks) if chunks else 0}개")
    if not chunks:
        logger.error("chunks가 비어있음")
        return

    # 모델 로드
    model_name = 'intfloat/multilingual-e5-large-instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
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

    vector_db = VectorDB
    
    logger.info(f"Vector DB 저장 완료: {file_path}")
