# PIILOT 서비스 흐름도

## 📊 전체 시스템 아키텍처

```
┌─────────────────┐
│   Spring Boot   │  (메인 백엔드)
│   Backend       │
└────────┬────────┘
         │ HTTP 요청
         │ (REST API)
         ▼
┌─────────────────┐
│   PIILOT        │  (AI 처리 전용 서비스)
│   FastAPI       │
└────────┬────────┘
         │
         ├─── DB AI 처리
         ├─── File AI 처리
         └─── Chat AI 처리
```

---

## 🔄 1. DB AI 처리 흐름

### 1.1 개인정보 컬럼 탐지

```
Spring Boot                    PIILOT Service
     │                              │
     │  POST /api/ai/db/detect-columns
     │  {schema_info: {...}}        │
     ├─────────────────────────────>│
     │                              │
     │                              ├─> API Layer (db_ai.py)
     │                              │   └─> detect_personal_info_columns()
     │                              │
     │                              ├─> Service Layer (column_detector.py)
     │                              │   └─> LLM + LangChain
     │                              │       └─> 개인정보 컬럼 분석
     │                              │
     │  {detected_columns: [...]}   │
     │<─────────────────────────────┤
     │                              │
     │  결과 저장                    │
     │  이슈 생성                    │
     │  알림 발송                    │
```

### 1.2 암호화 여부 확인

```
Spring Boot                    PIILOT Service
     │                              │
     │  POST /api/ai/db/check-encryption
     │  {data_samples: [...]}       │
     ├─────────────────────────────>│
     │                              │
     │                              ├─> API Layer (db_ai.py)
     │                              │   └─> check_encryption()
     │                              │
     │                              ├─> Service Layer (encryption_classifier.py)
     │                              │   └─> 분류 모델
     │                              │       └─> 암호화 여부 판단
     │                              │
     │  {results: [...]}            │
     │<─────────────────────────────┤
     │                              │
     │  결과 저장                    │
     │  이슈 생성 (암호화 안된 경우)  │
```

---

## 🔄 2. File AI 처리 흐름

### 2.1 문서 개인정보 탐지

```
Spring Boot                    PIILOT Service
     │                              │
     │  POST /api/ai/file/document/detect
     │  {file_content: "..."}       │
     ├─────────────────────────────>│
     │                              │
     │                              ├─> API Layer (file_ai.py)
     │                              │   └─> detect_document_personal_info()
     │                              │
     │                              ├─> Service Layer (document_detector.py)
     │                              │   └─> BERT + NER 모델
     │                              │       └─> 개인정보 탐지
     │                              │
     │  {detected_items: [...]}     │
     │<─────────────────────────────┤
     │                              │
     │  결과 저장                    │
     │  이슈 생성                    │
```

### 2.2 이미지 얼굴 탐지

```
Spring Boot                    PIILOT Service
     │                              │
     │  POST /api/ai/file/image/detect
     │  {image_data: "base64..."}   │
     ├─────────────────────────────>│
     │                              │
     │                              ├─> API Layer (file_ai.py)
     │                              │   └─> detect_image_faces()
     │                              │
     │                              ├─> Service Layer (image_detector.py)
     │                              │   └─> Vision 모델 (MediaPipe)
     │                              │       └─> 얼굴 위치 탐지
     │                              │
     │  {detected_faces: [...]}     │
     │<─────────────────────────────┤
     │                              │
     │  결과 저장                    │
     │  이슈 생성                    │
```

### 2.3 음성 개인정보 탐지

```
Spring Boot                    PIILOT Service
     │                              │
     │  POST /api/ai/file/audio/detect
     │  {audio_data: "base64..."}   │
     ├─────────────────────────────>│
     │                              │
     │                              ├─> API Layer (file_ai.py)
     │                              │   └─> detect_audio_personal_info()
     │                              │
     │                              ├─> Service Layer (audio_detector.py)
     │                              │   ├─> Whisper (음성 → 텍스트)
     │                              │   └─> LLM (텍스트 → 개인정보 탐지)
     │                              │
     │  {detected_items: [...]}     │
     │<─────────────────────────────┤
     │                              │
     │  결과 저장                    │
     │  이슈 생성                    │
```

### 2.4 영상 개인정보 탐지

```
Spring Boot                    PIILOT Service
     │                              │
     │  POST /api/ai/file/video/detect
     │  {video_data: "base64..."}   │
     ├─────────────────────────────>│
     │                              │
     │                              ├─> API Layer (file_ai.py)
     │                              │   └─> detect_video_personal_info()
     │                              │
     │                              ├─> Service Layer (video_detector.py)
     │                              │   ├─> ImageDetector (프레임별 얼굴 탐지)
     │                              │   └─> AudioDetector (오디오 개인정보 탐지)
     │                              │
     │  {faces: [...],              │
     │   personal_info_in_audio: [...]}
     │<─────────────────────────────┤
     │                              │
     │  결과 저장                    │
     │  이슈 생성                    │
```

### 2.5 마스킹 처리

```
Spring Boot                    PIILOT Service
     │                              │
     │  POST /api/ai/file/mask
     │  {file_type, file_data,      │
     │   detected_items}            │
     ├─────────────────────────────>│
     │                              │
     │                              ├─> API Layer (file_ai.py)
     │                              │   └─> apply_masking()
     │                              │
     │                              ├─> Service Layer (masker.py)
     │                              │   ├─> mask_document()   (문서)
     │                              │   ├─> mask_image()      (이미지)
     │                              │   ├─> mask_audio()      (음성)
     │                              │   └─> mask_video()      (영상)
     │                              │
     │  {masked_file: "base64..."}  │
     │<─────────────────────────────┤
     │                              │
     │  마스킹된 파일 저장            │
```

---

## 🔄 3. Chat AI 처리 흐름

### 3.1 자연어 질의응답

```
Spring Boot                    PIILOT Service
     │                              │
     │  POST /api/ai/chat
     │  {query: "...",              │
     │   context: {...}}            │
     ├─────────────────────────────>│
     │                              │
     │                              ├─> API Layer (chat_ai.py)
     │                              │   └─> chat()
     │                              │
     │                              ├─> Service Layer (assistant.py)
     │                              │   ├─> Vector DB 검색 (법령)
     │                              │   ├─> 컨텍스트 구성
     │                              │   └─> LLM + LangChain
     │                              │       └─> 응답 생성
     │                              │
     │  {answer: "...",             │
     │   sources: [...]}            │
     │<─────────────────────────────┤
     │                              │
     │  응답 표시                    │
```

### 3.2 법령 검색

```
Spring Boot                    PIILOT Service
     │                              │
     │  POST /api/ai/chat/search-regulations
     │  {query: "...", n_results: 5}│
     ├─────────────────────────────>│
     │                              │
     │                              ├─> API Layer (chat_ai.py)
     │                              │   └─> search_regulations()
     │                              │
     │                              ├─> Service Layer (vector_db.py)
     │                              │   └─> ChromaDB 검색
     │                              │       └─> 관련 법령 반환
     │                              │
     │  {results: [...]}            │
     │<─────────────────────────────┤
     │                              │
     │  법령 정보 표시                │
```

---

## 🏗️ 내부 구조 흐름

### API → Service → Model 흐름

```
┌─────────────────────────────────────────┐
│         FastAPI (app/main.py)           │
│  - 라우팅                                │
│  - CORS 설정                             │
│  - 미들웨어                              │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      API Layer (app/api/)               │
│  - db_ai.py                             │
│  - file_ai.py                           │
│  - chat_ai.py                           │
│  - 요청 검증                             │
│  - 응답 변환                             │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│    Service Layer (app/services/)        │
│  - db/                                   │
│    ├─ column_detector.py (LLM)          │
│    └─ encryption_classifier.py (ML)     │
│  - file/                                 │
│    ├─ document_detector.py (BERT+NER)   │
│    ├─ image_detector.py (Vision)        │
│    ├─ audio_detector.py (LLM)           │
│    ├─ video_detector.py (Vision+LLM)    │
│    └─ masker.py (마스킹)                 │
│  - chat/                                 │
│    ├─ assistant.py (LLM+LangChain)      │
│    └─ vector_db.py (ChromaDB)           │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      Models (app/models/)              │
│  - request.py (요청 DTO)                 │
│  - response.py (응답 DTO)               │
│  - personal_info.py (상수)              │
└─────────────────────────────────────────┘
```

---

## 📝 데이터 흐름 예시

### 예시 1: DB 스캔 전체 프로세스

```
1. Spring Boot
   └─> DB 스키마 조회
       └─> POST /api/ai/db/detect-columns
           {schema_info: {table_name: "users", columns: [...]}}

2. PIILOT
   └─> ColumnDetector.detect_personal_info_columns()
       └─> LLM 분석
           └─> 반환: [{table_name: "users", column_name: "name", ...}]

3. Spring Boot
   └─> 탐지된 컬럼에 대해 데이터 샘플 조회
       └─> POST /api/ai/db/check-encryption
           {data_samples: [{column: "users.name", sample: "홍길동"}, ...]}

4. PIILOT
   └─> EncryptionClassifier.classify()
       └─> 분류 모델 분석
           └─> 반환: [{column: "users.name", is_encrypted: false, ...}]

5. Spring Boot
   └─> 결과 저장
   └─> 이슈 생성 (암호화 안된 경우)
   └─> 알림 발송
```

### 예시 2: 파일 스캔 및 마스킹 프로세스

```
1. Spring Boot
   └─> 파일 업로드/스캔
       └─> POST /api/ai/file/document/detect
           {file_content: "홍길동의 전화번호는 010-1234-5678입니다"}

2. PIILOT
   └─> DocumentDetector.detect()
       └─> BERT + NER 분석
           └─> 반환: [{type: "name", value: "홍길동", ...}, 
                      {type: "phone", value: "010-1234-5678", ...}]

3. Spring Boot
   └─> 탐지 결과 확인
       └─> 마스킹 요청
           └─> POST /api/ai/file/mask
               {file_type: "document", 
                file_data: "원본 텍스트",
                detected_items: [...]}

4. PIILOT
   └─> Masker.mask_document()
       └─> 개인정보 마스킹 처리
           └─> 반환: {masked_file: "홍*동의 전화번호는 010-****-****입니다"}

5. Spring Boot
   └─> 마스킹된 파일 저장
```

---

## 🔑 핵심 포인트

1. **Stateless**: AI 서비스는 상태를 저장하지 않음
2. **순수 AI 처리**: 비즈니스 로직 없이 AI 모델 실행만 담당
3. **요청/응답**: Spring Boot가 모든 데이터 관리
4. **비동기 가능**: FastAPI의 async/await 활용 가능
5. **모듈화**: 각 기능이 독립적으로 동작

