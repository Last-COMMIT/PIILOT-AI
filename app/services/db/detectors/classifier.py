"""
개인정보 분류 모델 - 독립 실행 가능한 파일
값을 넣으면 개인정보 타입을 반환

사용법:
    from classifier_final import classify, classify_batch
    
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
import sys
import os

# 상위 디렉토리(또는 현재 디렉토리)의 bank_info import
try:
    from bank_info import BANK_CODES, MOBILE_PREFIXES, BANK_ACCT_LENGTHS, SPECIAL_ACCT_PATTERNS, COMMON_ACCT_PREFIXES
except ImportError:
    # 경로 문제 발생 시 처리
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from bank_info import BANK_CODES, MOBILE_PREFIXES, BANK_ACCT_LENGTHS, SPECIAL_ACCT_PATTERNS, COMMON_ACCT_PREFIXES

try:
    from app.core.model_manager import ModelManager
except ImportError:
    ModelManager = None

import xgboost as xgb

# XGBoost 경고 메시지 무시
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')


def _predict_with_booster(model, features_df: pd.DataFrame, feature_names: List[str], output_proba: bool = False) -> np.ndarray:
    """
    sklearn wrapper 검증 우회: Booster + DMatrix(feature_names=)로 예측.
    'data did not contain feature names' 오류 방지.
    """
    names = list(feature_names)
    try:
        if hasattr(model, 'get_booster'):
            booster_names = model.get_booster().feature_names
            if booster_names is not None and len(booster_names) == len(names):
                names = [str(n) for n in booster_names]
    except Exception:
        pass
    names = [str(n) for n in names]
    arr = features_df[names].to_numpy()
    dmat = xgb.DMatrix(arr, feature_names=names)
    preds = model.get_booster().predict(dmat, output_margin=False)
    if output_proba:
        return preds
    if preds.ndim == 2:
        return np.argmax(preds, axis=1)
    return preds


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
    value_clean = value  # 코드 가독성을 위해 별칭 사용
    
    # ===== 기본 통계 특징 =====
    # 텍스트의 기본적인 통계 정보 (길이, 숫자/문자 개수 등)
    features = {
        # 길이 및 개수 특징
        'length': len(value_clean),  # 전체 길이 (strip 후)
        'num_digits': len(re.findall(r'\d', value_clean)),  # 숫자 개수
        'num_letters': len(re.findall(r'[a-zA-Z가-힣]', value_clean)),  # 문자 개수 (영문+한글)
        'num_special': len(re.findall(r'[^a-zA-Z0-9가-힣\s]', value_clean)),  # 특수문자 개수
        'num_spaces': len(re.findall(r'\s', value_clean)),  # 공백 개수
        'num_hyphens': len(re.findall(r'-', value_clean)),  # 하이픈 개수
        'num_dots': len(re.findall(r'\.', value_clean)),  # 점 개수
        'num_at': len(re.findall(r'@', value_clean)),  # @ 기호 개수
        'num_colons': len(re.findall(r':', value_clean)),  # 콜론 개수
        
        # 비율 특징 (0~1 사이 값)
        'digit_ratio': len(re.findall(r'\d', value_clean)) / max(len(value_clean), 1),  # 숫자 비율
        'letter_ratio': len(re.findall(r'[a-zA-Z가-힣]', value_clean)) / max(len(value_clean), 1),  # 문자 비율
        'special_ratio': len(re.findall(r'[^a-zA-Z0-9가-힣\s]', value_clean)) / max(len(value_clean), 1),  # 특수문자 비율
        
        # ===== 마스킹 감지 특징 =====
        # 마스킹된 데이터는 NONE으로 분류되어야 함 (단, 일부 마스킹은 감지해야 할 수도 있음)
        'has_asterisk': 1.0 if '*' in value_clean else 0.0,  # 별표 존재 여부
        'num_asterisks': float(value_clean.count('*')),  # 별표 개수
        'has_masking_pattern': 1.0 if '*' in value_clean and value_clean.count('*') >= 2 else 0.0,  # 마스킹 패턴 (별표 2개 이상)
        'has_consecutive_asterisks': 1.0 if '**' in value_clean else 0.0,  # 연속된 별표
        
        # ===== 패턴 매칭 특징 (전처리된 값 기준) =====
        'has_rrn_pattern': 1.0 if '*' not in value_clean and re.match(r'^\d{6}-?\d{7}$', value_clean.replace('-', '')) else 0.0,  # 주민등록번호 패턴
        # 전화번호 패턴 (010-1234-5678) - 엄격한 패턴
        'has_phone_pattern': 1.0 if '*' not in value_clean and re.match(r'^0\d{1,2}-?\d{3,4}-?\d{4}$', value_clean) else 0.0,
        'has_email_pattern': 1.0 if '*' not in value_clean and bool(re.search(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', value_clean)) else 0.0,  # 이메일 패턴
        
        # 계좌번호 패턴 매칭 (기존보다 완화된 조건으로 넓게 잡음)
        'has_acn_pattern': 1.0 if '*' not in value_clean and (
            (value_clean[:3].isdigit() and value_clean.count('-') >= 1 and 10 <= len(value_clean.replace('-', '')) <= 20) or  # 하이픈 1개 이상, 길이 10~20
            (value_clean[:3].isdigit() and value_clean.count('-') == 0 and 10 <= len(value_clean) <= 16)  # 하이픈 없음, 길이 10~16
        ) else 0.0,
        
        'has_pp_pattern': 1.0 if '*' not in value_clean and re.match(r'^[A-Z]\d{8}$', value_clean) else 0.0,  # 여권번호 패턴
        'has_ip_pattern': 1.0 if '*' not in value_clean and re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', value_clean) else 0.0,  # IP주소 패턴
        'has_date_pattern': 1.0 if '*' not in value_clean and re.match(r'^\d{4}[-/]?\d{2}[-/]?\d{2}$', value_clean) else 0.0,  # 날짜 패턴
        
        # ===== 이름 특화 특징 =====
        'has_korean_name_pattern': 1.0 if bool(re.match(r'^[가-힣]{2,4}$', value_clean)) and not bool(re.search(r'[0-9]', value_clean)) else 0.0,
        'is_korean_only': 1.0 if bool(re.match(r'^[가-힣]+$', value_clean)) else 0.0,
        'name_like_length': 1.0 if 2 <= len(value_clean) <= 4 and bool(re.match(r'^[가-힣]+$', value_clean)) else 0.0,
        
        # ===== 이메일 특화 특징 =====
        'has_at_symbol': 1.0 if '@' in value_clean else 0.0,
        'has_dot_after_at': 1.0 if '@' in value_clean and len(value_clean.split('@')) > 1 and '.' in value_clean.split('@')[1] else 0.0,
        'email_like_length': 1.0 if 5 <= len(value_clean) <= 50 and '@' in value_clean else 0.0,
        
        # ===== 주소 특화 특징 =====
        'has_address_keywords': 1.0 if bool(re.search(r'(시|구|동|군|면|리|번지|로|길|대로)', value_clean)) else 0.0,
        'address_like_length': 1.0 if len(value_clean) >= 10 and bool(re.search(r'[가-힣]', value_clean)) else 0.0,
        
        # ===== 날짜 특화 특징 =====
        'has_date_keywords': 1.0 if bool(re.search(r'(년|월|일)', value_clean)) else 0.0,
        'is_korean_date': 1.0 if bool(re.search(r'\d{4}년', value_clean)) else 0.0,
        
        # ===== 길이 기반 특징 (하이픈 제외) =====
        'is_length_13_14': 1.0 if 13 <= len(value_clean.replace('-', '')) <= 14 else 0.0,
        'is_length_10_11': 1.0 if 10 <= len(value_clean.replace('-', '')) <= 11 else 0.0,
        'is_length_9': 1.0 if len(value_clean.replace('-', '')) == 9 else 0.0,
        'is_length_10_16': 1.0 if 10 <= len(value_clean.replace('-', '')) <= 16 else 0.0,
        
        # ===== 시작/끝 문자 특징 =====
        'starts_with_digit': 1.0 if value_clean and value_clean[0].isdigit() else 0.0,
        'ends_with_digit': 1.0 if value_clean and value_clean[-1].isdigit() else 0.0,
        'starts_with_letter': 1.0 if value_clean and value_clean[0].isalpha() else 0.0,
        
        # ===== 하이픈 관련 특징 =====
        'hyphen_position_ratio': value_clean.find('-') / max(len(value_clean), 1) if '-' in value_clean else 0.0,
        'num_hyphens': float(value_clean.count('-')),
        'has_multiple_hyphens': 1.0 if value_clean.count('-') >= 2 else 0.0,
        
        # ==============================================================================
        # [NEW] 계좌번호 vs 전화번호 정밀 식별 특징 (bank_info 활용)
        # ==============================================================================
    }
    
    # 전처리된 값에서 하이픈 제거
    val_digit = value_clean.replace('-', '')
    
    # 1. 은행 코드 시작 여부 & 모바일 접두어 여부
    features['starts_with_bank_code'] = 0.0
    features['starts_with_mobile_prefix'] = 0.0
    
    if len(val_digit) >= 3 and val_digit[:3].isdigit():
        prefix = val_digit[:3]
        if prefix in BANK_CODES:
            features['starts_with_bank_code'] = 1.0
        if prefix in MOBILE_PREFIXES:
            features['starts_with_mobile_prefix'] = 1.0
            
    # 2. 011과 같은 모호한 접두어 처리 (농협 vs 구형 폰번호)
    # 농협 계좌: 보통 13자리 (예: 011-01-123456)
    # 휴대폰: 10~11자리 (예: 011-234-5678)
    features['is_suspicious_01X'] = 0.0
    if features['starts_with_bank_code'] == 1.0 and features['starts_with_mobile_prefix'] == 1.0:
        # 접두어는 같지만 길이가 다르면 구분 가능
        if len(val_digit) >= 12: # 12자리 이상이면 계좌일 확률 높음
            features['is_suspicious_01X'] = -1.0 # 계좌 쪽으로 가중치
        else:
            features['is_suspicious_01X'] = 1.0 # 폰번호 쪽으로 가중치

    # 3. 은행별 구조 점수 (Structure Score)
    # 해당 은행 코드의 일반적인 길이와 일치하는지
    features['bank_structure_score'] = 0.0
    if len(val_digit) >= 3 and val_digit[:3].isdigit():
        prefix = val_digit[:3]
        if prefix in BANK_ACCT_LENGTHS:
            if len(val_digit) in BANK_ACCT_LENGTHS[prefix]:
                features['bank_structure_score'] = 1.0
            else:
                 # 코드는 맞지만 길이가 다름 -> 유령 포맷이거나 잘못된 번호
                 features['bank_structure_score'] = 0.5 
        elif prefix in BANK_CODES:
             # 길이는 모르지만 코드는 맞음
             features['bank_structure_score'] = 0.8
        
        # [NEW] 일반적인 계좌 접두어 (상품코드 등) 확인
        features['starts_with_acct_prefix'] = 0.0
        for p in COMMON_ACCT_PREFIXES:
            if val_digit.startswith(p):
                features['starts_with_acct_prefix'] = 1.0
                break
        
    # [NEW] 특수 계좌 패턴 매칭 (KB 주택, 비코드 구조 등)
    features['is_special_acct_pattern'] = 0.0
    for pattern in SPECIAL_ACCT_PATTERNS:
        if re.match(pattern, val_digit):
            features['is_special_acct_pattern'] = 1.0
            features['starts_with_bank_code'] = 1.0 # 특수 패턴이면 은행 코드로 간주 (가중치 보정)
            break

    # 4. 엄격한 모바일 패턴 (정규식)
    # -가 있거나 없거나, 010-XXXX-XXXX 형식에 정확히 부합
    features['is_strict_mobile_format'] = 0.0
    if re.match(r'^01[016789]-?\d{3,4}-?\d{4}$', value_clean):
        features['is_strict_mobile_format'] = 1.0
        
    # 5. 계좌번호인데 영문자가 없음 (강력한 힌트)
    features['no_letters_in_acn'] = 1.0 if bool(re.match(r'^[\d-]+$', value_clean)) else 0.0
    
    # 6. 하이픈 위치별 숫자 갯수 (패턴 구분용)
    # 예: 3-2-6 (계좌) vs 3-4-4 (전화)
    parts = value_clean.split('-')
    if len(parts) == 3 and all(part.isdigit() for part in parts):
        features['hyphen_structure_1'] = float(len(parts[0]))
        features['hyphen_structure_2'] = float(len(parts[1]))
        features['hyphen_structure_3'] = float(len(parts[2]))
    else:
        features['hyphen_structure_1'] = 0.0
        features['hyphen_structure_2'] = 0.0
        features['hyphen_structure_3'] = 0.0

    # ===== 연속 문자 특징 =====
    features['max_consecutive_digits'] = max([len(match) for match in re.findall(r'\d+', value_clean)] or [0])
    features['max_consecutive_letters'] = max([len(match) for match in re.findall(r'[a-zA-Z가-힣]+', value_clean)] or [0])
    
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
        model_dir: 모델 파일이 있는 디렉토리 경로 (None이면 현재 파일 기준 model 폴더)
    
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
        if model_dir is None and ModelManager is not None:
            model_path = Path(ModelManager.get_local_model_path("xgboost"))
            model_dir = model_path.parent
        elif model_dir is None:
            model_dir = Path(__file__).parent / "model"
            model_path = model_dir / "pii_xgboost_model.pkl"
        else:
            model_dir = Path(model_dir)
            model_path = model_dir / "pii_xgboost_model.pkl"
        
        # 메인 모델 로드 (다중 클래스 분류기)
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        _model = data['model']  # XGBoost 모델 객체
        _feature_names = list(data['feature_names'])  # 모델이 사용하는 특징 이름 리스트 (list로 통일)
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
    
    # Booster + DMatrix(feature_names=)로 예측 (sklearn wrapper 검증 우회)
    if _use_two_stage and _acn_model is not None:
        acn_probs = _predict_with_booster(_acn_model, features_df, _feature_names, output_proba=True)
        acn_prob = np.atleast_1d(acn_probs[0])
        acn_prob_positive = float(acn_prob[1] if len(acn_prob) > 1 else acn_prob[0])
        if acn_prob_positive >= 0.3:
            return 'p_acn'
    
    preds = _predict_with_booster(_model, features_df, _feature_names, output_proba=False)
    pred_mapped = int(preds[0])
    
    if _use_two_stage and _reverse_mapping is not None:
        pred = _reverse_mapping.get(pred_mapped, pred_mapped)
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
        model_dir: 모델 파일이 있는 디렉토리 경로 (None이면 현재 파일 기준 model 폴더)
    
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
        if model_dir is None and ModelManager is not None:
            model_path = Path(ModelManager.get_local_model_path("xgboost"))
            model_dir = model_path.parent
        elif model_dir is None:
            model_dir = Path(__file__).parent / "model"
            model_path = model_dir / "pii_xgboost_model.pkl"
        else:
            model_dir = Path(model_dir)
            model_path = model_dir / "pii_xgboost_model.pkl"
        
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        _model = data['model']
        _feature_names = list(data['feature_names'])
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
    
    # Booster + DMatrix(feature_names=)로 예측 (sklearn wrapper 검증 우회)
    if _use_two_stage and _acn_model is not None:
        acn_probs = _predict_with_booster(_acn_model, features_df, _feature_names, output_proba=True)
        other_preds = _predict_with_booster(_model, features_df, _feature_names, output_proba=False)
        
        for i in range(len(values)):
            acn_p = np.atleast_1d(acn_probs[i])
            acn_prob_positive = float(acn_p[1] if len(acn_p) > 1 else acn_p[0])
            if acn_prob_positive >= 0.3:
                results.append('p_acn')
            else:
                pred_mapped = int(other_preds[i])
                pred = _reverse_mapping.get(pred_mapped, pred_mapped) if _reverse_mapping else pred_mapped
                results.append(reverse_pii_types[pred])
    else:
        preds = _predict_with_booster(_model, features_df, _feature_names, output_proba=False)
        for pred in preds:
            results.append(reverse_pii_types[int(pred)])
    
    return results
