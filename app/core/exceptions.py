"""
커스텀 예외 및 글로벌 에러 핸들러
"""
from fastapi import Request
from fastapi.responses import JSONResponse
from app.core.logging import logger


class PIILOTException(Exception):
    """기본 예외 클래스"""
    pass


class ServerConnectionError(PIILOTException):
    """서버 연결 오류"""
    pass


class ScanError(PIILOTException):
    """스캔 오류"""
    pass


class ModelLoadError(PIILOTException):
    """모델 로드 오류"""
    pass


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """글로벌 예외 핸들러 - API 전체에 적용"""
    logger.error(f"처리되지 않은 예외: {request.method} {request.url.path} - {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)}
    )
