"""
임시 파일 관리 유틸리티
"""
import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse, unquote

import requests

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


def download_file_from_url(url: str, timeout: int = 120) -> Path:
    """
    URL에서 파일을 다운로드하여 임시 파일로 저장

    Args:
        url: 다운로드할 파일 URL (S3 등)
        timeout: 요청 타임아웃 (초)

    Returns:
        Path: 다운로드된 임시 파일 경로

    Raises:
        requests.RequestException: 다운로드 실패 시
    """
    logger.info(f"URL에서 파일 다운로드 시작: {url}")

    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    # URL에서 파일 확장자 추출
    parsed_url = urlparse(url)
    url_path = unquote(parsed_url.path)
    suffix = Path(url_path).suffix or ".pdf"

    # 임시 파일에 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="piilot_download_") as tmp:
        tmp.write(response.content)
        tmp_path = Path(tmp.name)

    logger.info(f"파일 다운로드 완료: {tmp_path} ({len(response.content):,} bytes)")
    return tmp_path


def is_url(path: str) -> bool:
    """경로가 URL인지 확인"""
    return path.startswith("http://") or path.startswith("https://")
