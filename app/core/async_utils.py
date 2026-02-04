"""
비동기 처리 유틸리티
ThreadPoolExecutor를 사용하여 동기 함수를 비동기로 실행
"""
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Any
from app.core.logging import logger

# 전역 ThreadPoolExecutor 인스턴스
_executor: ThreadPoolExecutor = None


def get_executor() -> ThreadPoolExecutor:
    """ThreadPoolExecutor 인스턴스 반환 (싱글톤)"""
    global _executor
    if _executor is None:
        # 워커 수: CPU 코어 수 + 4 (I/O 대기 작업 고려)
        max_workers = min(32, (os.cpu_count() or 1) + 4)
        _executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="api_worker")
        logger.info(f"ThreadPoolExecutor 초기화 완료 (max_workers={max_workers})")
    return _executor


def shutdown_executor():
    """ThreadPoolExecutor 종료"""
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=True)
        _executor = None
        logger.info("ThreadPoolExecutor 종료 완료")


async def run_in_thread(func: Callable, *args, **kwargs) -> Any:
    """
    동기 함수를 ThreadPoolExecutor에서 비동기로 실행
    
    Args:
        func: 실행할 동기 함수
        *args, **kwargs: 함수에 전달할 인자
    
    Returns:
        함수 실행 결과
    """
    executor = get_executor()
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, lambda: func(*args, **kwargs))
