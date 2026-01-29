# 암호화 분류 모델

데이터베이스 컬럼의 암호화 여부를 판단하는 XGBoost 분류 모델입니다.

## 모델 파일

- `pii_xgboost_model.pkl`: 메인 다중 클래스 분류기
  - 용도: 개인정보 컬럼의 암호화 여부 분류
  - 알고리즘: XGBoost
  - 사용 위치: `app/services/db/detectors/classifier.py`

- `pii_acn_binary_model.pkl`: 계좌번호 전용 이진 분류기
  - 용도: 계좌번호 컬럼의 암호화 여부 이진 분류
  - 알고리즘: XGBoost (Binary Classifier)
  - 사용 위치: `app/services/db/detectors/classifier.py` (2단계 분류)

## 사용 방법

모델은 `classify()` 또는 `classify_batch()` 함수를 통해 자동으로 로드됩니다.
