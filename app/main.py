"""
FastAPI 메인 애플리케이션
AI 처리 전용 마이크로서비스
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import logger
from app.core.exceptions import global_exception_handler, validation_exception_handler
from fastapi.exceptions import RequestValidationError
from app.core.model_manager import ModelManager
from app.api import db, file, chat

app = FastAPI(
    title="PIILOT",
    description="AI 기반 개인정보 보호 및 유출 관제 플랫폼 - AI 처리 전용 서비스",
    version="0.1.0",
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 글로벌 예외 핸들러 등록
app.add_exception_handler(Exception, global_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)

# API 라우터 등록
app.include_router(
    db.router,
    prefix="/api/ai/db",
    tags=["DB AI"],
)
app.include_router(
    file.router,
    prefix="/api/ai/file",
    tags=["File AI"],
)
app.include_router(
    chat.router,
    prefix="/api/ai/chat",
    tags=["Chat AI"],
)


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "PIILOT API",
        "version": "0.1.0",
        "description": "AI 처리 전용 마이크로서비스",
    }


@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {"status": "healthy"}


@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 실행"""
    logger.info("PIILOT 서비스 시작")
    
    # 모든 모델 다운로드 확인 및 다운로드
    try:
        ModelManager.ensure_all_models()
    except Exception as e:
        logger.error(f"모델 준비 중 오류 발생: {e}", exc_info=True)
        # 서비스는 계속 실행 (모델이 없으면 해당 기능만 비활성화)


@app.on_event("shutdown")
async def shutdown_event():
    """애플리케이션 종료 시 실행"""
    logger.info("PIILOT 서비스 종료")
