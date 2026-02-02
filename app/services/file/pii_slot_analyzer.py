"""
문서 PII 슬롯 분석 (방법 A): 필드명(레이블) 뒤 값을 "PII 자리"로 보고,
탐지 안 되고 비어있거나 마스킹 패턴이면 maskedCount로 집계.
"""
import re
from typing import List, Dict, Tuple, Any

from app.core.logging import logger

# 내부 label (p_nm 등) -> 응답 piiType 코드
PII_LABEL_TO_CODE = {
    "p_nm": "NM", "p_ph": "PH", "p_add": "ADD", "p_rrn": "RRN",
    "p_em": "EM", "p_ip": "IP", "p_acn": "ACN", "p_pp": "PP",
    "p_acct": "ACN", "p_passport": "PP",
}

# 응답 piiType 코드 (한글 타입명 -> NM, PH 등)
PII_TYPE_NAME_TO_CODE = {
    "이름": "NM",
    "전화번호": "PH",
    "주소": "ADD",
    "주민번호": "RRN",
    "이메일": "EM",
    "IP주소": "IP",
    "계좌번호": "ACN",
    "여권번호": "PP",
}

# 필드 키워드: piiType 코드 -> 레이블 키워드 목록 (자기소개서/양식 등)
FIELD_KEYWORDS: Dict[str, List[str]] = {
    "NM": ["이름", "성명", "이름(한글)", "한글성명"],
    "PH": ["전화번호", "휴대폰", "연락처", "핸드폰"],
    "ADD": ["주소", "현주소", "주소지"],
    "RRN": ["주민번호", "주민등록번호"],
    "EM": ["이메일", "이메일주소", "e-mail"],
    "BTH": ["생년월일", "생일"],
}

# 마스킹 패턴: 값이 이 패턴에 맞으면 "마스킹된 자리"로 간주
MASKING_PATTERNS = [
    re.compile(r"^[가-힣]\*+[가-힣]$"),           # 이*욱
    re.compile(r"^[가-힣]\*+[가-힣]*$"),          # 홍*동
    re.compile(r"^010[-*]*\d{4}[-*]*\d{4}$"),     # 010-****-5678
    re.compile(r"^\d{2,3}[-*]+\d{3,4}[-*]+\d{4}$"),
    re.compile(r"^[*]+$"),                        # ***
]


def _is_masking_pattern(value: str) -> bool:
    """값이 마스킹 패턴에 해당하는지"""
    v = (value or "").strip()
    if not v:
        return False
    for pat in MASKING_PATTERNS:
        if pat.match(v):
            return True
    return False


def parse_field_slots(text: str) -> List[Tuple[str, str]]:
    """
    문서 텍스트에서 "레이블: 값" 형태의 PII 슬롯 추출.

    Returns:
        [(piiType_code, value), ...]
    """
    slots: List[Tuple[str, str]] = []
    if not (text or text.strip()):
        return slots

    lines = text.split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue
        matched = False
        for pii_code, keywords in FIELD_KEYWORDS.items():
            if matched:
                break
            for kw in keywords:
                # "키워드:" 또는 "키워드 :" 뒤 값 (해당 줄 끝까지)
                if kw + ":" in line or kw + " :" in line:
                    idx = line.find(kw)
                    rest = line[idx + len(kw):].lstrip(" :\t")
                    value = rest.strip()
                    slots.append((pii_code, value))
                    matched = True
                    break

    return slots


def analyze_slots_for_masked(
    text: str,
    document_detector: Any,
) -> Dict[str, int]:
    """
    필드 슬롯을 파싱한 뒤, 값이 비어있거나 마스킹 패턴인데 탐지가 안 되면 maskedCount로 집계.

    Returns:
        { piiType_code: maskedCount, ... }
    """
    masked: Dict[str, int] = {}
    slots = parse_field_slots(text)
    for pii_code, value in slots:
        value_stripped = (value or "").strip()
        if not value_stripped:
            masked[pii_code] = masked.get(pii_code, 0) + 1
            continue
        if not _is_masking_pattern(value_stripped):
            continue
        try:
            detected = document_detector.detect(value_stripped)
        except Exception as e:
            logger.debug("슬롯 값 탐지 예외: %s", e)
            detected = []
        if not detected:
            masked[pii_code] = masked.get(pii_code, 0) + 1
    return masked


def aggregate_document_pii(
    detected_items: List[Dict],
    masked_counts: Dict[str, int],
) -> Dict[str, Dict[str, int]]:
    """
    전체 문서 탐지 결과(detected_items)와 슬롯 maskedCount를 합쳐 piiType별 totalCount, maskedCount 반환.

    detected_items: document_detector.detect() 반환값 [{"type": "이름", ...}, ...]
    masked_counts: analyze_slots_for_masked() 반환값 { "NM": 1, ... }

    Returns:
        { "NM": {"totalCount": n, "maskedCount": m}, ... }
    """
    total_by_code: Dict[str, int] = {}
    for item in detected_items:
        t = item.get("type") or ""
        code = PII_TYPE_NAME_TO_CODE.get(t) or t
        total_by_code[code] = total_by_code.get(code, 0) + 1

    all_codes = set(total_by_code) | set(masked_counts)
    result: Dict[str, Dict[str, int]] = {}
    for code in all_codes:
        result[code] = {
            "totalCount": total_by_code.get(code, 0),
            "maskedCount": masked_counts.get(code, 0),
        }
    return result
