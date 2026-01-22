"""
개인정보 타입 상수
"""
from enum import Enum


class PersonalInfoType(str, Enum):
    """개인정보 8종류"""
    NAME = "name"  # 이름
    RRN = "rrn"  # 주민번호
    ADDRESS = "address"  # 집주소
    IP_ADDRESS = "ip_address"  # IP주소
    PHONE = "phone"  # 전화번호
    ACCOUNT = "account"  # 계좌번호
    PASSPORT = "passport"  # 여권번호
    EMAIL = "email"  # 이메일


# 개인정보 종류별 가중치
PERSONAL_INFO_WEIGHTS = {
    PersonalInfoType.RRN: 10,
    PersonalInfoType.PASSPORT: 9,
    PersonalInfoType.ACCOUNT: 8,
    PersonalInfoType.NAME: 5,
    PersonalInfoType.ADDRESS: 5,
    PersonalInfoType.PHONE: 4,
    PersonalInfoType.EMAIL: 3,
    PersonalInfoType.IP_ADDRESS: 2,
}

