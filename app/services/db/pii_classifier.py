"""
개인정보 분류 모델 - 독립 실행 가능한 파일
값을 넣으면 개인정보 타입을 반환

사용법:
    from app.services.db.pii_classifier import classify, classify_batch
    
    # 단일 값 분류
    result = classify("010-1234-5678")  # 'p_ph' 반환
    
    # 여러 값 분류
    results = classify_batch(["홍길동", "010-1234-5678"])  # ['p_nm', 'p_ph'] 반환
"""
import re
import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
from collections import Counter
import warnings

# XGBoost 경고 메시지 무시
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')


def _extract_features(value: str, text_feature_names: Optional[List[str]] = None) -> Dict[str, float]:
    """
    텍스트 값을 특징 벡터로 변환하는 함수
    
    모델이 학습 시 사용한 특징과 동일한 방식으로 추출해야 함.
    모델 파일에는 특징 이름만 저장되고, 실제 추출 로직은 이 함수에 구현되어 있음.
    
    Args:
        value: 분류할 텍스트 값
        text_feature_names: 모델에서 사용하는 텍스트 특징 이름 리스트 (n-gram 특징)
    
    Returns:
        특징 딕셔너리 (키: 특징 이름, 값: 특징 값)
    """
    # None이나 NaN 값 처리
    if pd.isna(value) or value is None:
        value = ""
    value = str(value).strip()
    
    # ===== 기본 통계 특징 =====
    # 텍스트의 기본적인 통계 정보 (길이, 숫자/문자 개수 등)
    features = {
        # 길이 및 개수 특징
        'length': len(value),  # 전체 길이
        'num_digits': len(re.findall(r'\d', value)),  # 숫자 개수
        'num_letters': len(re.findall(r'[a-zA-Z가-힣]', value)),  # 문자 개수 (영문+한글)
        'num_special': len(re.findall(r'[^a-zA-Z0-9가-힣\s]', value)),  # 특수문자 개수
        'num_spaces': len(re.findall(r'\s', value)),  # 공백 개수
        'num_hyphens': len(re.findall(r'-', value)),  # 하이픈 개수
        'num_dots': len(re.findall(r'\.', value)),  # 점 개수
        'num_at': len(re.findall(r'@', value)),  # @ 기호 개수
        'num_colons': len(re.findall(r':', value)),  # 콜론 개수
        
        # 비율 특징 (0~1 사이 값)
        'digit_ratio': len(re.findall(r'\d', value)) / max(len(value), 1),  # 숫자 비율
        'letter_ratio': len(re.findall(r'[a-zA-Z가-힣]', value)) / max(len(value), 1),  # 문자 비율
        'special_ratio': len(re.findall(r'[^a-zA-Z0-9가-힣\s]', value)) / max(len(value), 1),  # 특수문자 비율
        
        # ===== 마스킹 감지 특징 =====
        # 마스킹된 데이터는 NONE으로 분류되어야 함
        'has_asterisk': 1.0 if '*' in value else 0.0,  # 별표 존재 여부
        'num_asterisks': float(value.count('*')),  # 별표 개수
        'has_masking_pattern': 1.0 if '*' in value and value.count('*') >= 2 else 0.0,  # 마스킹 패턴 (별표 2개 이상)
        'has_consecutive_asterisks': 1.0 if '**' in value else 0.0,  # 연속된 별표
        # ===== 패턴 매칭 특징 =====
        # 각 개인정보 유형의 패턴을 정규식으로 감지 (마스킹이 있으면 패턴 매칭 실패)
        'has_rrn_pattern': 1.0 if '*' not in value and re.match(r'^\d{6}-?\d{7}$', value.replace('-', '')) else 0.0,  # 주민등록번호 패턴 (6자리-7자리)
        'has_phone_pattern': 1.0 if '*' not in value and re.match(r'^0\d{1,2}-?\d{3,4}-?\d{4}$', value.replace('-', '')) else 0.0,  # 전화번호 패턴 (010-1234-5678)
        'has_email_pattern': 1.0 if '*' not in value and bool(re.search(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', value)) else 0.0,  # 이메일 패턴
        'has_acn_pattern': 1.0 if '*' not in value and (
            (value[:3].isdigit() and value.count('-') >= 2 and 10 <= len(value.replace('-', '')) <= 16) or  # 계좌번호: 하이픈 2개 이상
            (value[:3].isdigit() and value.count('-') == 1 and 10 <= len(value.replace('-', '')) <= 16) or  # 계좌번호: 하이픈 1개
            (value[:3].isdigit() and value.count('-') == 0 and 10 <= len(value) <= 16)  # 계좌번호: 하이픈 없음
        ) else 0.0,
        'has_pp_pattern': 1.0 if '*' not in value and re.match(r'^[A-Z]\d{8}$', value) else 0.0,  # 여권번호 패턴 (영문 1자리 + 숫자 8자리)
        'has_ip_pattern': 1.0 if '*' not in value and re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', value) else 0.0,  # IP주소 패턴
        'has_date_pattern': 1.0 if '*' not in value and re.match(r'^\d{4}[-/]?\d{2}[-/]?\d{2}$', value) else 0.0,  # 날짜 패턴 (YYYY-MM-DD)
        # ===== 이름 특화 특징 =====
        'has_korean_name_pattern': 1.0 if bool(re.match(r'^[가-힣]{2,4}$', value)) and not bool(re.search(r'[0-9]', value)) else 0.0,  # 한국 이름 패턴 (한글 2-4자, 숫자 없음)
        'is_korean_only': 1.0 if bool(re.match(r'^[가-힣]+$', value)) else 0.0,  # 한글만 포함
        'name_like_length': 1.0 if 2 <= len(value) <= 4 and bool(re.match(r'^[가-힣]+$', value)) else 0.0,  # 이름 같은 길이 (2-4자 한글)
        
        # ===== 이메일 특화 특징 =====
        'has_at_symbol': 1.0 if '@' in value else 0.0,  # @ 기호 존재
        'has_dot_after_at': 1.0 if '@' in value and '.' in value.split('@')[1] else 0.0,  # @ 뒤에 점 존재
        'email_like_length': 1.0 if 5 <= len(value) <= 50 and '@' in value else 0.0,  # 이메일 같은 길이
        
        # ===== 주소 특화 특징 =====
        'has_korean': 1.0 if '*' not in value and bool(re.search(r'[가-힣]', value)) else 0.0,  # 한글 포함 여부
        'has_address_keywords': 1.0 if '*' not in value and bool(re.search(r'(시|구|동|군|면|리|번지|로|길|대로)', value)) else 0.0,  # 주소 키워드 포함
        'address_like_length': 1.0 if '*' not in value and len(value) >= 10 and bool(re.search(r'[가-힣]', value)) else 0.0,  # 주소 같은 길이 (10자 이상 + 한글)
        'has_space_in_middle': 1.0 if '*' not in value and ' ' in value and 0 < value.find(' ') < len(value) - 1 else 0.0,  # 중간에 공백 존재
        
        # ===== 날짜 특화 특징 =====
        'has_date_keywords': 1.0 if bool(re.search(r'(년|월|일)', value)) else 0.0,  # 날짜 키워드 포함 (년/월/일)
        'is_korean_date': 1.0 if bool(re.search(r'\d{4}년\s*\d{1,2}월\s*\d{1,2}일', value)) or bool(re.search(r'\d{4}년\s*\d{1,2}월', value)) else 0.0,  # 한글 날짜 패턴
        # ===== 길이 기반 특징 =====
        'is_length_13_14': 1.0 if 13 <= len(value.replace('-', '')) <= 14 else 0.0,  # 주민등록번호 길이
        'is_length_10_11': 1.0 if 10 <= len(value.replace('-', '')) <= 11 else 0.0,  # 전화번호 길이
        'is_length_9': 1.0 if len(value.replace('-', '')) == 9 else 0.0,  # 여권번호 길이
        'is_length_12_13': 1.0 if 12 <= len(value.replace('-', '')) <= 13 else 0.0,  # 계좌번호 길이 (기존)
        'is_length_10_16': 1.0 if 10 <= len(value.replace('-', '')) <= 16 else 0.0,  # 계좌번호 길이 (확장 범위)
        'is_length_2_4_korean': 1.0 if 2 <= len(value) <= 4 and bool(re.match(r'^[가-힣]+$', value)) else 0.0,  # 이름 길이 (한글 2-4자)
        
        # ===== 시작/끝 문자 특징 =====
        'starts_with_digit': 1.0 if value and value[0].isdigit() else 0.0,  # 숫자로 시작
        'ends_with_digit': 1.0 if value and value[-1].isdigit() else 0.0,  # 숫자로 끝남
        'starts_with_letter': 1.0 if value and value[0].isalpha() else 0.0,  # 문자로 시작
        
        # ===== 하이픈 관련 특징 =====
        'has_hyphen_middle': 1.0 if '-' in value and 0 < value.find('-') < len(value) - 1 else 0.0,  # 중간에 하이픈 존재
        'hyphen_position_ratio': value.find('-') / max(len(value), 1) if '-' in value else 0.0,  # 하이픈 위치 비율
        'num_hyphens': float(value.count('-')),  # 하이픈 개수
        'has_multiple_hyphens': 1.0 if value.count('-') >= 2 else 0.0,  # 여러 하이픈 존재
        # ===== 계좌번호 특화 특징 =====
        # 계좌번호는 다양한 형식이 있어서 여러 특징으로 감지
        'starts_with_bank_code': 1.0 if '*' not in value and value and len(value) >= 3 and value[:3].isdigit() and value[:3] in ['004', '088', '020', '081', '003', '011', '090', '089', '023', '027'] else 0.0,  # 유효한 은행 코드로 시작
        'has_hyphen_after_3digits': 1.0 if '*' not in value and len(value) > 3 and value[:3].isdigit() and value[3] == '-' else 0.0,  # 처음 3자리 뒤에 하이픈
        'acn_like_structure': 1.0 if '*' not in value and (value[:3].isdigit() if len(value) >= 3 else False) and value.count('-') >= 2 and len(value.replace('-', '')) >= 10 else 0.0,  # 계좌번호 같은 구조
        'no_letters_in_acn': 1.0 if '*' not in value and bool(re.match(r'^[\d-]+$', value)) and value[:3].isdigit() and len(value.replace('-', '')) >= 10 else 0.0,  # 계좌번호는 영문자 없음
        'acn_starts_with_valid_bank_and_has_hyphens': 1.0 if '*' not in value and value and len(value) >= 3 and value[:3].isdigit() and value[:3] in ['004', '088', '020', '081', '003', '011', '090', '089', '023', '027'] and value.count('-') >= 1 and 10 <= len(value.replace('-', '')) <= 16 else 0.0,  # 유효 은행코드 + 하이픈 + 길이 체크
        'acn_has_multiple_hyphens_and_valid_length': 1.0 if '*' not in value and value.count('-') >= 2 and value[:3].isdigit() and 10 <= len(value.replace('-', '')) <= 16 else 0.0,  # 하이픈 2개 이상 + 유효 길이
        'acn_no_hyphen_but_valid_bank_code': 1.0 if '*' not in value and value and len(value) >= 10 and value[:3].isdigit() and value[:3] in ['004', '088', '020', '081', '003', '011', '090', '089', '023', '027'] and '-' not in value and 10 <= len(value) <= 16 else 0.0,  # 하이픈 없지만 유효 은행코드 + 길이
        
        # ===== 연속 문자 특징 =====
        'max_consecutive_digits': max([len(match) for match in re.findall(r'\d+', value)] or [0]),  # 최대 연속 숫자 길이
        'max_consecutive_letters': max([len(match) for match in re.findall(r'[a-zA-Z가-힣]+', value)] or [0]),  # 최대 연속 문자 길이
    }
    
    # ===== 텍스트 특징 (n-gram) =====
    # 문자 2-gram, 3-gram 빈도를 특징으로 사용 (모델 학습 시 선택된 n-gram만 사용)
    if text_feature_names:
        # 2-gram과 3-gram 추출
        bigrams = [value[i:i+2] for i in range(len(value) - 1)] if len(value) >= 2 else []
        trigrams = [value[i:i+3] for i in range(len(value) - 2)] if len(value) >= 3 else []
        bigram_counts = Counter(bigrams)  # 2-gram 빈도 계산
        trigram_counts = Counter(trigrams)  # 3-gram 빈도 계산
        
        # 모델에서 사용하는 n-gram 특징만 추가
        for feat_name in text_feature_names:
            if feat_name.startswith('bigram_'):
                ngram = feat_name.replace('bigram_', '')
                features[feat_name] = bigram_counts.get(ngram, 0.0)  # 해당 2-gram의 빈도
            elif feat_name.startswith('trigram_'):
                ngram = feat_name.replace('trigram_', '')
                features[feat_name] = trigram_counts.get(ngram, 0.0)  # 해당 3-gram의 빈도
    
    return features


# ===== 전역 모델 인스턴스 =====
# 모델은 최초 1회만 로드하고 메모리에 캐싱하여 재사용
_model = None  # 메인 다중 클래스 분류 모델
_acn_model = None  # 계좌번호 이진 분류 모델 (2단계 분류 사용 시)
_feature_names = None  # 모델이 사용하는 특징 이름 리스트
_pii_types = None  # PII 타입 매핑 딕셔너리
_use_two_stage = False  # 2단계 분류 사용 여부
_reverse_mapping = None  # 레이블 재매핑 딕셔너리 (2단계 분류 시 필요)
_text_feature_names = None  # 텍스트 특징 이름 리스트 (n-gram)


def classify(value: str, model_dir: Optional[str] = None) -> str:
    """
    단일 값을 분류하여 개인정보 타입을 반환
    
    Args:
        value: 분류할 텍스트 값
        model_dir: 모델 파일이 있는 디렉토리 경로 (None이면 기본 경로 사용)
    
    Returns:
        str: 개인정보 타입 ('p_nm', 'p_rrn', 'p_add', 'p_ip', 'p_ph', 'p_acn', 'p_pp', 'p_em', 'NONE')
    
    예시:
        >>> classify("010-1234-5678")
        'p_ph'
        >>> classify("홍길동")
        'p_nm'
    """
    global _model, _acn_model, _feature_names, _pii_types, _use_two_stage, _reverse_mapping, _text_feature_names
    
    # ===== 모델 로드 (최초 1회만) =====
    if _model is None:
        # 모델 디렉토리 경로 설정
        if model_dir is None:
            # 기본 경로: 프로젝트 루트의 models/encryption_classifier
            project_root = Path(__file__).parent.parent.parent.parent
            model_dir = project_root / "models" / "encryption_classifier"
        else:
            model_dir = Path(model_dir)
        
        # 메인 모델 로드 (다중 클래스 분류기)
        model_path = model_dir / "pii_xgboost_model.pkl"
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        _model = data['model']  # XGBoost 모델 객체
        _feature_names = data['feature_names']  # 모델이 사용하는 특징 이름 리스트
        _pii_types = data['pii_types']  # PII 타입 매핑
        _use_two_stage = data.get('use_two_stage', False)  # 2단계 분류 사용 여부
        _reverse_mapping = data.get('reverse_other_class_mapping', None)  # 레이블 재매핑 (2단계 분류 시 필요)
        _text_feature_names = data.get('text_feature_names', []) if data.get('use_text_features', True) else None  # 텍스트 특징 이름
        
        # 계좌번호 이진 분류기 로드 (2단계 분류 사용 시)
        if _use_two_stage:
            acn_path = model_dir / "pii_acn_binary_model.pkl"
            if acn_path.exists():
                with open(acn_path, 'rb') as f:
                    acn_data = pickle.load(f)
                _acn_model = acn_data['model']  # 계좌번호 전용 이진 분류기
    
    # ===== 특징 추출 =====
    # 텍스트 값을 모델이 학습 시 사용한 특징 벡터로 변환
    features = _extract_features(value, _text_feature_names)
    features_df = pd.DataFrame([features])  # DataFrame으로 변환
    
    # ===== 특징 순서 및 누락 특징 처리 =====
    # 모델이 기대하는 특징 순서에 맞추고, 누락된 특징은 0으로 채움
    for name in _feature_names:
        if name not in features_df.columns:
            features_df[name] = 0.0
    features_df = features_df[_feature_names]  # 특징 순서 맞추기
    
    # ===== 2단계 분류: 계좌번호 이진 분류 =====
    # 먼저 계좌번호인지 확인 (False Negative 최소화를 위해 낮은 임계값 0.3 사용)
    if _use_two_stage and _acn_model is not None:
        acn_prob = _acn_model.predict_proba(features_df)[0]  # 계좌번호일 확률
        acn_prob_positive = acn_prob[1] if len(acn_prob) > 1 else acn_prob[0]  # 양성 클래스 확률
        if acn_prob_positive >= 0.3:  # 임계값 이상이면 계좌번호로 판단
            return 'p_acn'
    
    # ===== 메인 모델 예측 =====
    # 계좌번호가 아니면 나머지 클래스 분류
    pred_mapped = _model.predict(features_df)[0]  # 예측 결과 (재매핑된 레이블)
    
    # 2단계 분류 사용 시: 재매핑된 레이블을 원본 레이블로 변환
    if _use_two_stage and _reverse_mapping is not None:
        pred = _reverse_mapping[pred_mapped]
    else:
        pred = pred_mapped
    
    # 숫자 레이블을 문자열 타입으로 변환
    reverse_pii_types = {v: k for k, v in _pii_types.items()}
    return reverse_pii_types[pred]


def classify_batch(values: List[str], model_dir: Optional[str] = None) -> List[str]:
    """
    여러 값을 일괄 분류하여 개인정보 타입 리스트를 반환
    
    Args:
        values: 분류할 텍스트 값들의 리스트
        model_dir: 모델 파일이 있는 디렉토리 경로 (None이면 기본 경로 사용)
    
    Returns:
        List[str]: 개인정보 타입 리스트
    
    예시:
        >>> classify_batch(["홍길동", "010-1234-5678"])
        ['p_nm', 'p_ph']
    """
    global _model, _acn_model, _feature_names, _pii_types, _use_two_stage, _reverse_mapping, _text_feature_names
    
    # ===== 모델 로드 (최초 1회만) =====
    # classify() 함수와 동일한 로직
    if _model is None:
        if model_dir is None:
            project_root = Path(__file__).parent.parent.parent.parent
            model_dir = project_root / "models" / "encryption_classifier"
        else:
            model_dir = Path(model_dir)
        
        model_path = model_dir / "pii_xgboost_model.pkl"
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        _model = data['model']
        _feature_names = data['feature_names']
        _pii_types = data['pii_types']
        _use_two_stage = data.get('use_two_stage', False)
        _reverse_mapping = data.get('reverse_other_class_mapping', None)
        _text_feature_names = data.get('text_feature_names', []) if data.get('use_text_features', True) else None
        
        if _use_two_stage:
            acn_path = model_dir / "pii_acn_binary_model.pkl"
            if acn_path.exists():
                with open(acn_path, 'rb') as f:
                    acn_data = pickle.load(f)
                _acn_model = acn_data['model']
    
    # ===== 특징 추출 (배치) =====
    # 모든 값에 대해 특징 추출
    features_list = [_extract_features(v, _text_feature_names) for v in values]
    features_df = pd.DataFrame(features_list)  # DataFrame으로 변환
    
    # 특징 순서 및 누락 특징 처리
    for name in _feature_names:
        if name not in features_df.columns:
            features_df[name] = 0.0
    features_df = features_df[_feature_names]
    
    reverse_pii_types = {v: k for k, v in _pii_types.items()}
    results = []
    
    # ===== 2단계 분류: 계좌번호 이진 분류 =====
    if _use_two_stage and _acn_model is not None:
        acn_probs = _acn_model.predict_proba(features_df)  # 모든 값에 대한 계좌번호 확률
        other_preds = _model.predict(features_df)  # 모든 값에 대한 나머지 클래스 예측
        
        for i in range(len(values)):
            acn_prob_positive = acn_probs[i][1] if len(acn_probs[i]) > 1 else acn_probs[i][0]
            if acn_prob_positive >= 0.3:  # 계좌번호로 판단
                results.append('p_acn')
            else:
                # 계좌번호가 아니면 나머지 클래스 분류 결과 사용
                pred_mapped = other_preds[i]
                pred = _reverse_mapping[pred_mapped] if _reverse_mapping else pred_mapped
                results.append(reverse_pii_types[pred])
    else:
        # ===== 1단계 분류 (기존 방식) =====
        preds = _model.predict(features_df)
        for pred in preds:
            results.append(reverse_pii_types[pred])
    
    return results
