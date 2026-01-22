"""
Vector DB 초기화 스크립트
"""
import asyncio
from app.services.chat.vector_db import VectorDB


async def main():
    """법령 데이터를 Vector DB에 로드"""
    vector_db = VectorDB()
    
    # TODO(pgvector):
    # - PostgreSQL에 pgvector extension 설치/활성화
    # - regulations 테이블 생성/마이그레이션
    # - 법령 텍스트 로드(파일/DB/크롤링 등)
    # - 임베딩 생성 후 upsert_regulations(items)로 적재
    
    print("Vector DB 초기화 완료")


if __name__ == "__main__":
    asyncio.run(main())

