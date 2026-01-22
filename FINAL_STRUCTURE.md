# PIILOT 최종 구조

## 📁 파일 구조

```
PIILOT/
├── app/
│   ├── main.py                    # FastAPI 메인 (라우팅만)
│   ├── config.py                  # 설정 (모델 경로, API 키)
│   │
│   ├── api/                       # API 엔드포인트 (요청/응답만)
│   │   ├── __init__.py
│   │   ├── db_ai.py              # DB 관련 AI API
│   │   ├── file_ai.py            # 파일 관련 AI API
│   │   └── chat_ai.py             # AI 어시스턴트 API
│   │
│   ├── services/                  # AI 처리 서비스 (순수 AI 로직)
│   │   ├── __init__.py
│   │   │
│   │   ├── db/                   # DB AI 서비스
│   │   │   ├── __init__.py
│   │   │   ├── column_detector.py    # 개인정보 컬럼 탐지 (LLM + LangChain)
│   │   │   └── encryption_classifier.py  # 암호화 여부 판단 (분류 모델)
│   │   │
│   │   ├── file/                 # 파일 AI 서비스
│   │   │   ├── __init__.py
│   │   │   ├── document_detector.py   # 문서 개인정보 탐지 (BERT + NER)
│   │   │   ├── image_detector.py      # 이미지 얼굴 탐지 (Vision)
│   │   │   ├── audio_detector.py      # 음성 개인정보 탐지 (LLM)
│   │   │   ├── video_detector.py      # 영상 개인정보 탐지 (Vision + LLM)
│   │   │   └── masker.py              # 마스킹 처리 (공통)
│   │   │
│   │   └── chat/                 # AI 어시스턴트
│   │       ├── __init__.py
│   │       ├── assistant.py          # AI 어시스턴트 (LLM + LangChain)
│   │       └── vector_db.py          # 법령 Vector DB (읽기 전용)
│   │
│   ├── models/                    # 요청/응답 모델
│   │   ├── __init__.py
│   │   ├── request.py            # 요청 DTO
│   │   ├── response.py           # 응답 DTO
│   │   └── personal_info.py      # 개인정보 타입 상수
│   │
│   └── utils/                     # 유틸리티
│       ├── __init__.py
│       ├── logger.py              # 로깅
│       └── exceptions.py          # 커스텀 예외
│
├── models/                        # 학습된 모델 저장소
│   ├── encryption_classifier/
│   ├── bert_ner/
│   └── vision/
│
├── data/                          # 데이터 저장소
│   ├── regulations/              # 법령 데이터 (Vector DB용)
│   └── .gitkeep
│
├── tests/                         # 테스트
│   ├── __init__.py
│   ├── test_db_ai.py
│   ├── test_file_ai.py
│   └── test_chat_ai.py
│
├── scripts/                       # 스크립트
│   └── setup_vector_db.py        # Vector DB 초기화
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 🔌 API 엔드포인트

### 1. DB AI API (`/api/ai/db/`)

#### `POST /api/ai/db/detect-columns`
개인정보 컬럼 탐지
- **요청**: `ColumnDetectionRequest`
- **응답**: `ColumnDetectionResponse`
- **기능**: LLM + LangChain으로 스키마 정보에서 개인정보 컬럼 탐지

#### `POST /api/ai/db/check-encryption`
암호화 여부 확인
- **요청**: `EncryptionCheckRequest`
- **응답**: `EncryptionCheckResponse`
- **기능**: 분류 모델로 데이터 샘플의 암호화 여부 판단

### 2. File AI API (`/api/ai/file/`)

#### `POST /api/ai/file/document/detect`
문서 개인정보 탐지
- **요청**: `DocumentDetectionRequest`
- **응답**: `DocumentDetectionResponse`
- **기능**: BERT + NER로 문서에서 개인정보 탐지

#### `POST /api/ai/file/image/detect`
이미지 얼굴 탐지
- **요청**: `ImageDetectionRequest`
- **응답**: `ImageDetectionResponse`
- **기능**: Vision 모델로 이미지에서 얼굴 탐지

#### `POST /api/ai/file/audio/detect`
음성 개인정보 탐지
- **요청**: `AudioDetectionRequest`
- **응답**: `AudioDetectionResponse`
- **기능**: Whisper + LLM으로 음성에서 개인정보 탐지

#### `POST /api/ai/file/video/detect`
영상 개인정보 탐지
- **요청**: `VideoDetectionRequest`
- **응답**: `VideoDetectionResponse`
- **기능**: Vision + LLM으로 영상에서 얼굴 및 개인정보 탐지

#### `POST /api/ai/file/mask`
마스킹 처리
- **요청**: `MaskingRequest`
- **응답**: `MaskingResponse`
- **기능**: 탐지된 개인정보를 마스킹 처리

### 3. Chat AI API (`/api/ai/chat/`)

#### `POST /api/ai/chat`
자연어 질의응답
- **요청**: `ChatRequest`
- **응답**: `ChatResponse`
- **기능**: LLM + LangChain으로 자연어 질의에 응답

#### `POST /api/ai/chat/search-regulations`
법령 검색
- **요청**: `RegulationSearchRequest`
- **응답**: `RegulationSearchResponse`
- **기능**: Vector DB에서 관련 법령 검색

## 🔄 데이터 흐름

```
Spring Boot → AI 서비스 (요청)
           ← AI 서비스 (결과)
           
Spring Boot가 처리:
- 결과 저장
- 이슈 생성
- 알림 발송
- 대시보드 업데이트
```

## 📋 주요 특징

1. **순수 AI 처리**: 비즈니스 로직 없이 AI 모델 실행만 담당
2. **Stateless**: 상태 저장 없음, 요청마다 독립 처리
3. **간단한 API**: RESTful, 요청/응답만
4. **모듈화**: 기능별로 명확히 분리
   - `api/`: API 엔드포인트
   - `services/`: AI 처리 로직
   - `models/`: 요청/응답 모델

## 🛠 기술 스택

- **Framework**: FastAPI
- **AI/ML**: 
  - LangChain (챗봇, DB 탐색)
  - BERT + NER (문서 탐지)
  - Vision Models (이미지/영상)
  - LLM (음성, 챗봇)
- **Vector DB**: ChromaDB (법령 데이터)

