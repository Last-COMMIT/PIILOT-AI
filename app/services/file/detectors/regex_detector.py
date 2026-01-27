import re
from typing import List, Dict, Set

class GeneralizedRegexPIIDetector:
    """일반화된 정규식 PII 탐지기"""
    def __init__(self):
        print("일반화된 정규식 탐지기 준비 완료")

    def detect_phones(self, text: str) -> List[Dict]:
        """전화번호 탐지"""
        entities = []
        seen = set()

        # 한국 전화번호: 0으로 시작, 10-11자리
        patterns = [
            r'01[016789]-\d{3,4}-\d{4}',     # 휴대폰 (하이픈)
            r'0\d{1,2}-\d{3,4}-\d{4}',       # 일반전화 (하이픈)
            r'01[016789]\d{7,8}',             # 휴대폰 (하이픈 없음)
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text):
                phone = match.group()
                digits = re.sub(r'\D', '', phone)

                if 10 <= len(digits) <= 11 and digits not in seen:
                    seen.add(digits)
                    entities.append({
                        'text': phone,
                        'label': 'p_ph',
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': 1.0,
                        'method': 'regex'
                    })

        return entities

    def detect_emails(self, text: str) -> List[Dict]:
        entities = []

        # 1단계: 표준 이메일
        standard_pattern = r'[a-zA-Z0-9][a-zA-Z0-9._+-]*@[a-zA-Z0-9][a-zA-Z0-9.-]*\.[a-zA-Z]{2,}'

        for match in re.finditer(standard_pattern, text):
            entities.append({
                'text': match.group(),
                'label': 'p_em',
                'start': match.start(),
                'end': match.end(),
                'confidence': 1.0,
                'method': 'regex'
            })

        # 2단계: OCR 오류 패턴 (@있고 도메인 접미사 있지만 점 없음)
        ocr_pattern = r'[a-zA-Z0-9][a-zA-Z0-9._+-]*@[a-zA-Z0-9]+(?:com|net|org|co\.?kr|kr|jp|cn|edu|gov)'

        for match in re.finditer(ocr_pattern, text):
            # 이미 표준 패턴으로 잡혔으면 스킵
            if any(e['start'] == match.start() for e in entities):
                continue

            email_text = match.group()
            domain_part = email_text.split('@')[1]

            # 점이 없거나 1개 이하인 경우만 보정
            if domain_part.count('.') <= 0:
                known_suffixes = [
                    'com', 'net', 'org', 'co.kr', 'kr',
                    'jp', 'cn', 'edu', 'gov', 'info'
                ]

                for suffix in known_suffixes:
                    if domain_part.endswith(suffix):
                        domain_name = domain_part[:-len(suffix)]
                        if len(domain_name) >= 2:
                            local_part = email_text.split('@')[0]
                            corrected = f"{local_part}@{domain_name}.{suffix}"
                            entities.append({
                                'text': corrected,
                                'label': 'p_em',
                                'start': match.start(),
                                'end': match.end(),
                                'confidence': 0.95,
                                'method': 'regex'
                            })
                            break

        return entities

    def detect_addresses(self, text: str) -> List[Dict]:
        entities = []

        # 주소 패턴들
        patterns = [
            # 완전한 주소: 도/시/구 + 로/길 + 번호
            r'[가-힣]{2,}(?:특별시|광역시|도)\s+[가-힣]{2,}(?:시|군|구)\s+[가-힣]{2,}(?:로|길)\s+\d+[가-힣0-9\s-]*',
            # 도 축약형: "경기 성남시"
            r'[가-힣]{2,}\s+[가-힣]{2,}(?:시|군|구)\s+[가-힣]{2,}(?:로|길)?\s*\d*[가-힣0-9\s-]*',
            # 시/군/구 + 도로명 + 번호
            r'[가-힣]{2,}(?:시|군|구)\s+[가-힣]{2,}(?:로|길)\s+\d+[가-힣0-9\s-]*',
            # 구 + 도로명 + 번호 (서울 등)
            r'[가-힣]{2,}구\s+[가-힣]{2,}(?:로|길)\s+\d+[가-힣0-9\s-]*',
            # 시/구 + 동 (번지 주소)
            r'[가-힣]{2,}(?:시|구)\s+[가-힣]{2,}동',
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text):
                address = match.group().strip()

                # 구조적 검증
                if self._is_valid_address_structure(address):
                    entities.append({
                        'text': address,
                        'label': 'p_add',
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': 1.0,
                        'method': 'regex'
                    })

        return entities

    def _is_valid_address_structure(self, address: str) -> bool:
        """주소 구조 검증"""
        if len(address) < 8: return False

        has_admin = any(kw in address for kw in ['도', '시', '군', '구'])
        has_location = any(kw in address for kw in ['로', '길', '동'])

        hangul_chars = sum(1 for c in address if '가' <= c <= '힣')
        total_chars = len(address.replace(' ', ''))
        hangul_ratio = hangul_chars / total_chars if total_chars > 0 else 0
        is_mostly_hangul = hangul_ratio >= 0.6

        valid_endings = ['동', '번길', '번지', '호', '층']
        ends_with_number = address[-1].isdigit()
        ends_with_valid_keyword = any(address.endswith(ending) for ending in valid_endings)

        ends_improperly = (
            (address.endswith('구') or address.endswith('시'))
            and not ends_with_valid_keyword
            and not ends_with_number
        )
        ends_properly = not ends_improperly

        return has_admin and has_location and is_mostly_hangul and ends_properly

    def detect_rrn(self, text: str) -> List[Dict]:
        """주민등록번호 탐지"""
        entities = []
        patterns = [r'\d{6}-[1-4]\d{6}', r'\d{6}[1-4]\d{6}']

        for pattern in patterns:
            for match in re.finditer(pattern, text):
                entities.append({
                    'text': match.group(),
                    'label': 'p_rrn',
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 1.0,
                    'method': 'regex'
                })
        return entities

    def detect_ip(self, text: str) -> List[Dict]:
        """IP 주소 탐지"""
        entities = []
        pattern = r'(?<!\d)\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?!\d)'

        for match in re.finditer(pattern, text):
            ip = match.group()
            try:
                octets = [int(x) for x in ip.split('.')]
                if all(0 <= octet <= 255 for octet in octets):
                    entities.append({
                        'text': ip,
                        'label': 'p_ip',
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': 1.0,
                        'method': 'regex'
                    })
            except:
                pass
        return entities

    def detect_passports(self, text: str) -> List[Dict]:
        """여권번호 탐지 (신여권/구여권 포괄)"""
        entities = []
        # 한국 여권: 영문 1자리 + 숫자 8자리 (구여권) 또는 영문 1자리 + 숫자 3자리 + 영문 1자리 + 숫자 4자리 (신여권)
        # 구여권: M12345678
        # 신여권: M123A4567 (중간 4번째(인덱스 4) 자리가 영문일 수 있음)
        # 통합 Regex: 영문 1개 + 숫자 3개 + (숫자 또는 영문) 1개 + 숫자 4개
        pattern = r'[a-zA-Z][0-9]{3}[a-zA-Z0-9][0-9]{4}'
        
        for match in re.finditer(pattern, text):
            entities.append({
                'text': match.group(),
                'label': 'p_pp', # Passport Number
                'start': match.start(),
                'end': match.end(),
                'confidence': 1.0,
                'method': 'regex'
            })
        return entities

    def detect_all(self, text: str) -> List[Dict]:
        """모든 PII 탐지 및 중복 제거"""
        all_entities = []
        all_entities.extend(self.detect_phones(text))
        all_entities.extend(self.detect_emails(text))
        all_entities.extend(self.detect_addresses(text))
        all_entities.extend(self.detect_rrn(text))
        all_entities.extend(self.detect_ip(text))
        all_entities.extend(self.detect_passports(text))

        # 중복 제거: 같은 위치 + 같은 레이블
        seen = set()
        unique = []

        for entity in all_entities:
            key = (entity['start'], entity['end'], entity['label'])
            if key not in seen:
                seen.add(key)
                unique.append(entity)

        return unique