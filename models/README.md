# Models 디렉토리

이 디렉토리는 PIILOT-AI 서비스에서 사용하는 모든 AI 모델을 저장합니다.

## 디렉토리 구조

```
models/
├── huggingface/          # HuggingFace에서 다운로드되는 모델들 (자동 생성)
│   ├── models--ParkJunSeong--PIILOT_NER_Model/
│   ├── models--intfloat--multilingual-e5-large-instruct/
│   └── models--K-intelligence--Midm-2.0-Mini-Instruct/
├── vision/              # 컴퓨터 비전 모델 (얼굴 탐지)
│   └── yolov12n-face.pt
└── encryption_classifier/  # 암호화 분류 모델
    ├── pii_xgboost_model.pkl
    └── pii_acn_binary_model.pkl
```

## 모델 목록

### HuggingFace 모델 (자동 다운로드)

FastAPI 서비스 시작 시 자동으로 `models/huggingface/`에 다운로드됩니다.

| 모델 | 용도 | 크기 (예상) |
|------|------|------------|
| `ParkJunSeong/PIILOT_NER_Model` | 개인정보 탐지 (KoELECTRA NER) | ~500MB |
| `intfloat/multilingual-e5-large-instruct` | 텍스트 임베딩 (법령 검색) | ~1.3GB |
| `K-intelligence/Midm-2.0-Mini-Instruct` | LLM (법령 질의응답) | ~2GB |
| `Whisper large-v3` | 음성 인식 (STT) | ~3GB |

**총 예상 크기: 약 7GB**

### 로컬 모델

| 모델 | 경로 | 용도 |
|------|------|------|
| YOLOv12 Face | `vision/yolov12n-face.pt` | 얼굴 탐지 |
| XGBoost 분류기 | `encryption_classifier/pii_xgboost_model.pkl` | 암호화 여부 판단 |
| XGBoost 계좌번호 분류기 | `encryption_classifier/pii_acn_binary_model.pkl` | 계좌번호 암호화 판단 |

## 모델 관리

### 자동 다운로드

FastAPI 서비스를 시작하면 모든 HuggingFace 모델이 자동으로 다운로드됩니다:

```bash
uvicorn app.main:app --reload
```

### 수동 다운로드 (선택사항)

모델을 수동으로 다운로드하려면:

```python
from app.core.model_manager import ModelManager
ModelManager.ensure_all_models()
```

### 모델 캐시 위치

- HuggingFace 모델: `models/huggingface/`
- Whisper 모델: faster-whisper 자체 캐시 사용 (환경 변수로 제어 가능)

## 주의사항

1. **Git 관리**: HuggingFace 모델은 크기가 크므로 Git에 포함하지 않습니다.
2. **디스크 공간**: 모든 모델을 다운로드하면 약 7GB의 디스크 공간이 필요합니다.
3. **네트워크**: 첫 실행 시 인터넷 연결이 필요합니다.
4. **권한**: 모델 다운로드 및 저장을 위한 쓰기 권한이 필요합니다.

## 문제 해결

### 모델 다운로드 실패

- 인터넷 연결 확인
- 디스크 공간 확인
- HuggingFace 접근 권한 확인

### 모델 로딩 실패

- 모델 파일 존재 확인: `ls models/huggingface/`
- 로그 확인: `app/core/logging.py`에서 로그 레벨 확인
