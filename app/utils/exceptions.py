"""
커스텀 예외
"""


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

