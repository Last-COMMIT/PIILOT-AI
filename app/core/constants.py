"""
개인정보 타입 및 상수 정의 (단일 소스)
"""
PII_CATEGORIES = ["p_nm", "p_rrn", "p_add", "p_ip", "p_ph", "p_acn", "p_pp", "p_em"]

PII_NAMES = {
    "p_nm": "이름",
    "p_rrn": "주민번호",
    "p_add": "주소",
    "p_ip": "IP주소",
    "p_ph": "전화번호",
    "p_acn": "계좌번호",
    "p_pp": "여권번호",
    "p_em": "이메일",
    # audio_masking 호환 키
    "p_acct": "계좌번호",
    "p_passport": "여권번호",
}

CONFIDENCE_THRESHOLDS = {
    "p_nm": 0.70,    # 이름
    "p_rrn": 0.95,   # 주민번호
    "p_add": 0.70,   # 주소
    "p_ip": 0.80,    # IP주소
    "p_ph": 0.70,    # 전화번호
    "p_acn": 0.90,   # 계좌번호
    "p_pp": 0.75,    # 여권번호
    "p_em": 0.70,    # 이메일
    # audio_masking 호환 키
    "p_acct": 0.85,
    "p_passport": 0.90,
}
