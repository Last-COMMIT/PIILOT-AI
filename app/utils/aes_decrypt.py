"""
AES-256-GCM 복호화 (백엔드 Java AesEncryptor와 동일 포맷)
- IV 12바이트 + ciphertext+tag, Base64 인코딩
"""
import base64
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


GCM_IV_LENGTH = 12


def decrypt_db_password(encrypted_base64: str, aes_key_utf8: str) -> str:
    """
    백엔드(Java AesEncryptor)에서 암호화한 DB 비밀번호를 복호화.

    Args:
        encrypted_base64: Base64( IV(12) + ciphertext+tag ) 문자열
        aes_key_utf8: UTF-8 기준 32바이트 키 (encryption.aes-key와 동일)

    Returns:
        복호화된 평문 비밀번호

    Raises:
        ValueError: 키 길이 오류
        Exception: Base64/복호화 실패
    """
    key_bytes = aes_key_utf8.encode("utf-8")
    if len(key_bytes) != 32:
        raise ValueError("AES key must be 32 bytes (UTF-8) for AES-256")

    combined = base64.b64decode(encrypted_base64)
    if len(combined) < GCM_IV_LENGTH + 16:
        raise ValueError("Invalid ciphertext length (need IV + ciphertext + tag)")

    nonce = combined[:GCM_IV_LENGTH]
    ciphertext_and_tag = combined[GCM_IV_LENGTH:]

    aesgcm = AESGCM(key_bytes)
    decrypted = aesgcm.decrypt(nonce, ciphertext_and_tag, None)
    return decrypted.decode("utf-8")
