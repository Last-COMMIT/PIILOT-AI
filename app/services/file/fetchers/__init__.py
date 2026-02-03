"""
파일 서버 타입별 파일 다운로드 (WebDAV, S3, SFTP 등)
"""
from typing import Dict, Any

from app.core.logging import logger


def fetch_file_bytes(connection_info: Dict[str, Any], path: str) -> bytes:
    """
    연결 정보와 경로로 파일 바이트 반환.

    Args:
        connection_info: host, port, default_path, username, password, server_type_name
        path: 파일 경로 (예: documents/test_file.txt)

    Returns:
        파일 내용 bytes

    Raises:
        NotImplementedError: 해당 server_type 미구현
        Exception: 네트워크/파일 없음 등
    """
    server_type = (connection_info.get("server_type_name") or "WebDAV").strip()
    if not server_type:
        server_type = "WebDAV"

    server_type_lower = server_type.lower()
    if "webdav" in server_type_lower or "web" in server_type_lower or "http" in server_type_lower:
        return _fetch_webdav(connection_info, path)
    if "s3" in server_type_lower:
        return _fetch_s3(connection_info, path)
    if "sftp" in server_type_lower or "ftp" in server_type_lower:
        return _fetch_sftp(connection_info, path)

    logger.warning("알 수 없는 파일 서버 유형 %s, WebDAV로 시도", server_type)
    return _fetch_webdav(connection_info, path)


def _fetch_webdav(connection_info: Dict[str, Any], path: str) -> bytes:
    """WebDAV: HTTP(S) GET + Basic Auth"""
    import httpx

    # host = connection_info.get("host", "")
    host = (connection_info.get("host") or "").strip()
    port = int(connection_info.get("port") or 9000)
    default_path = (connection_info.get("default_path") or "").strip().rstrip("/")
    username = connection_info.get("username") or ""
    password = connection_info.get("password") or ""

    path_part = path.strip().lstrip("/")
    if default_path:
        path_part = f"{default_path.rstrip('/')}/{path_part}" if path_part else default_path
    path_part = path_part or ""

    # host에 이미 스킴(http://, https://)이 있으면 그대로 사용, 없으면 스킴+호스트 조합
    # host_lower = host.lower()
    # if "ngrok" in host_lower:
    #     scheme = "https"
    #     base = f"{scheme}://{host}"
    # else:
    #     scheme = "https" if port == 443 else "http"
    #     base = f"{scheme}://{host}" if port in (80, 443) else f"{scheme}://{host}:{port}"
    host_lower = host.lower()
    if host_lower.startswith("http://") or host_lower.startswith("https://"):
        base = host.rstrip("/")
    elif "ngrok" in host_lower:
        base = f"https://{host}"
    else:
        scheme = "https" if port == 443 else "http"
        base = f"{scheme}://{host}" if port in (80, 443) else f"{scheme}://{host}:{port}"
    url = f"{base}/{path_part}" if path_part else base.rstrip("/")

    logger.info(f"파일 서버 GET 요청 url={url} (Basic Auth={bool(username or password)})")
    auth = (username, password) if username or password else None
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(url, auth=auth)
        resp.raise_for_status()
    return resp.content


def _fetch_s3(connection_info: Dict[str, Any], path: str) -> bytes:
    """S3: 미구현 시 스텁"""
    raise NotImplementedError("S3 파일 서버는 아직 지원하지 않습니다.")


def _fetch_sftp(connection_info: Dict[str, Any], path: str) -> bytes:
    """SFTP: 미구현 시 스텁"""
    raise NotImplementedError("SFTP 파일 서버는 아직 지원하지 않습니다.")
