"""
커스텀 예외 및 글로벌 에러 핸들러
"""
import traceback
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
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
    """
    글로벌 예외 핸들러 - API 전체에 적용
    모든 예외를 안전하게 처리하여 서버가 멈추지 않도록 함
    """
    try:
        # 상세한 에러 정보 로깅
        error_msg = f"처리되지 않은 예외: {request.method} {request.url.path}"
        logger.error(error_msg, exc_info=True)
        
        # 스택 트레이스도 로깅
        traceback_str = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        logger.error(f"스택 트레이스:\n{traceback_str}")
        
        # 사용자에게는 간단한 메시지만 반환 (보안상 상세 정보 노출 방지)
        error_detail = str(exc) if str(exc) else "알 수 없는 오류가 발생했습니다."
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "detail": error_detail,
                "path": str(request.url.path)
            }
        )
    except Exception as handler_error:
        # 예외 핸들러 자체에서 오류가 발생한 경우 (극단적인 상황)
        logger.critical(f"예외 핸들러에서 오류 발생: {handler_error}", exc_info=True)
        # 최소한의 응답 반환
        return JSONResponse(
            status_code=500,
            content={"error": "Critical Error", "detail": "서버 오류가 발생했습니다."}
        )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """요청 검증 오류 핸들러"""
    try:
        logger.warning(f"요청 검증 오류: {request.method} {request.url.path} - {exc.errors()}")
        return JSONResponse(
            status_code=422,
            content={
                "error": "Validation Error",
                "detail": exc.errors(),
                "path": str(request.url.path)
            }
        )
    except Exception as e:
        logger.error(f"검증 오류 핸들러에서 오류 발생: {e}", exc_info=True)
        return JSONResponse(
            status_code=422,
            content={"error": "Validation Error", "detail": "요청 형식이 올바르지 않습니다."}
        )
