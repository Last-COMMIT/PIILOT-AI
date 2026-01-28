"""
임시 파일 관리 유틸리티
"""
import os
import tempfile
from app.core.logging import logger


class TempFileManager:
    """임시 파일을 자동으로 정리하는 컨텍스트 매니저"""

    def __init__(self):
        self._files: list[str] = []

    def create(self, suffix: str = '', prefix: str = 'piilot_') -> str:
        """임시 파일을 생성하고 경로를 반환"""
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix=prefix)
        temp.close()
        self._files.append(temp.name)
        return temp.name

    def add(self, path: str):
        """기존 파일을 관리 목록에 추가"""
        if path:
            self._files.append(path)

    def cleanup(self):
        """등록된 모든 임시 파일 삭제"""
        for path in self._files:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                    logger.debug(f"임시 파일 삭제: {path}")
                except Exception as e:
                    logger.warning(f"임시 파일 삭제 실패: {path} - {e}")
        self._files.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False
