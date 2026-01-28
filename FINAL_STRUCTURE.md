# PIILOT-AI 최종 폴더 구조

## 레이어 아키텍처 요약

```
요청 흐름:  Client → API → Service → (ML / CRUD) → Response
```

| 레이어 | 폴더 | 역할 | FastAPI 표준 |
|--------|------|------|:---:|
| **API** | `api/` | HTTP 요청/응답, 라우팅, 입력 검증 | O |
| **Schema** | `schemas/` | Request/Response Pydantic 모델 (DTO) | O |
| **Service** | `services/` | 비즈니스 로직 오케스트레이션 + 파일 처리 | O |
| **CRUD** | `crud/` | DB 데이터 접근, 쿼리, 커넥션 관리 | O |
| **ML** | `ml/` | ML 모델 추론 (PII 탐지, 얼굴 탐지, STT 등) | △ (프로젝트 특성) |
| **Core** | `core/` | 설정, 상수, 예외, 로깅 | O |
| **Utils** | `utils/` | 공유 유틸리티 함수 | O |

---

## 디렉토리 트리

```
app/
├── __init__.py
├── main.py
│
├── core/                                    # ① 설정·상수·예외·로깅
│   ├── __init__.py
│   ├── config.py
│   ├── constants.py
│   ├── exceptions.py
│   └── logging.py
│
├── api/                                     # ② 엔드포인트 (Controller)
│   ├── __init__.py
│   ├── deps.py
│   ├── db.py
│   ├── file.py
│   └── chat.py
│
├── schemas/                                 # ③ Request/Response DTO
│   ├── __init__.py
│   ├── db.py
│   ├── file.py
│   └── chat.py
│
├── services/                                # ④ 비즈니스 로직 (Service)
│   ├── __init__.py
│   ├── db/
│   │   ├── __init__.py
│   │   ├── column_detection_service.py
│   │   └── encryption_service.py
│   ├── file/
│   │   ├── __init__.py
│   │   ├── document_service.py
│   │   ├── image_service.py
│   │   ├── audio_service.py
│   │   ├── video_service.py
│   │   ├── masking_service.py
│   │   └── processors/                      # ④-1 파일 처리 구현체
│   │       ├── __init__.py
│   │       ├── text_extractor.py
│   │       ├── document_masker.py
│   │       ├── image_masker.py
│   │       ├── audio_masker.py
│   │       ├── video_masker.py
│   │       ├── video_processor.py
│   │       └── blur_censor.py
│   └── chat/
│       ├── __init__.py
│       ├── assistant_service.py
│       └── regulation_service.py
│
├── crud/                                    # ⑤ DB 접근 (Repository)
│   ├── __init__.py
│   ├── db_connect.py
│   ├── connection.py
│   ├── data_sampling.py
│   └── vector_db.py
│
├── ml/                                      # ⑥ ML 모델 추론
│   ├── __init__.py
│   ├── pii_detectors/
│   │   ├── __init__.py
│   │   ├── regex_detector.py
│   │   ├── dl_detector.py
│   │   └── hybrid_detector.py
│   ├── face_detector.py
│   ├── image_detector.py
│   ├── xgboost_classifier.py
│   ├── bank_info.py
│   ├── whisper_stt.py
│   └── embedding_model.py
│
└── utils/                                   # ⑦ 공유 유틸리티
    ├── __init__.py
    ├── base64_utils.py
    ├── image_utils.py
    ├── temp_file.py
    └── overlap.py
```

> **하위 호환 파일** (기존 import 경로 유지용, 새 위치로 재수출만 함):
> `app/config.py`, `app/models/`, `app/utils/logger.py`, `app/utils/exceptions.py`,
> `app/utils/image_loader.py`, `app/utils/db_connect.py`,
> `app/api/db_ai.py`, `app/api/file_ai.py`, `app/api/chat_ai.py` 등
> → 이 파일들은 기존 코드의 `import`가 깨지지 않도록 남겨둔 것이며,
>   **새로 작성하는 코드에서는 새 경로를 사용해야 합니다.**

---

## 파일별 상세 설명

### `app/main.py` — FastAPI 앱 진입점

| 항목 | 내용 |
|------|------|
| **포함하는 코드** | `FastAPI()` 인스턴스 생성, CORS 미들웨어, 글로벌 예외 핸들러 등록, 라우터 등록 (`db`, `file`, `chat`), startup/shutdown 이벤트 |
| **포함하지 않는 코드** | 비즈니스 로직, 모델 로딩, DB 접근 |

```python
# 예시
app = FastAPI(title="PIILOT")
app.add_exception_handler(Exception, global_exception_handler)
app.include_router(db.router, prefix="/api/ai/db")
app.include_router(file.router, prefix="/api/ai/file")
app.include_router(chat.router, prefix="/api/ai/chat")
```

---

### ① `app/core/` — 설정, 상수, 예외, 로깅

앱 전체에서 사용하는 **기반 설정**을 모아 두는 레이어.
다른 모든 레이어가 `core/`를 import할 수 있지만, `core/`는 다른 레이어를 import하지 않는다.

| 파일 | 역할 | 포함 코드 |
|------|------|----------|
| `config.py` | 환경 변수 / 앱 설정 | `Settings(BaseSettings)` 클래스, `settings` 싱글톤, `MODEL_PATH`, `OUTPUT_DIR` |
| `constants.py` | PII 타입 상수 (단일 소스) | `PII_CATEGORIES`, `PII_NAMES`, `CONFIDENCE_THRESHOLDS` 딕셔너리 |
| `exceptions.py` | 커스텀 예외 + 글로벌 핸들러 | `PIILOTException`, `ServerConnectionError`, `ScanError`, `ModelLoadError`, `global_exception_handler()` |
| `logging.py` | 로깅 설정 | `loguru` 기반 `logger` 인스턴스 (stdout + 파일 로테이션) |

```python
# 사용 예시
from app.core.config import settings
from app.core.constants import PII_NAMES
from app.core.logging import logger
from app.core.exceptions import PIILOTException
```

---

### ② `app/api/` — 엔드포인트 (Controller 레이어)

HTTP 요청을 받고, 스키마로 검증하고, 서비스에 위임하고, 응답을 반환하는 **얇은 컨트롤러**.

| 파일 | 역할 | 포함 코드 |
|------|------|----------|
| `deps.py` | 의존성 주입 팩토리 | `@lru_cache` 데코레이터로 서비스 싱글톤 생성 (`get_document_detector()`, `get_masker()` 등) |
| `db.py` | DB 진단 엔드포인트 | `POST /detect-columns`, `POST /check-encryption` |
| `file.py` | 파일 처리 엔드포인트 | `POST /document/detect`, `POST /image/detect`, `POST /audio/detect`, `POST /video/detect`, `POST /mask` 등 |
| `chat.py` | 챗봇 엔드포인트 | `POST /chat`, `POST /search-regulations` |

**작성 규칙:**
- `try/except` 없음 → 글로벌 핸들러가 처리
- 비즈니스 로직 없음 → 서비스에 위임
- `Depends()`로 서비스 주입

```python
# 예시: app/api/file.py
@router.post("/document/detect", response_model=DocumentDetectionResponse)
async def detect_document_personal_info(
    request: DocumentDetectionRequest,
    document_detector=Depends(get_document_detector),
):
    detected_items = document_detector.detect(request.file_content)
    return DocumentDetectionResponse(detected_items=detected_items, is_masked=False)
```

---

### ③ `app/schemas/` — Request/Response DTO

Pydantic 모델을 **도메인별로 분리**.
API의 입출력 형식만 정의하며, 비즈니스 로직은 포함하지 않는다.

| 파일 | 포함 모델 |
|------|----------|
| `db.py` | `ColumnDetectionRequest`, `ColumnDetectionResponse`, `EncryptionCheckRequest`, `EncryptionCheckResponse` 등 |
| `file.py` | `DocumentDetectionRequest`, `ImageDetectionRequest`, `AudioDetectionRequest`, `VideoDetectionRequest`, `MaskingRequest`, `MaskingResponse` 등 |
| `chat.py` | `ChatRequest`, `ChatResponse`, `RegulationSearchRequest`, `RegulationSearchResponse` 등 |

```python
# 예시: app/schemas/file.py
class DocumentDetectionRequest(BaseModel):
    file_content: str
    file_type: Optional[str] = "txt"

class DocumentDetectionResponse(BaseModel):
    detected_items: List[DetectedPersonalInfo]
    is_masked: bool = False
```

---

### ④ `app/services/` — 비즈니스 로직 (Service 레이어)

ML 모델과 CRUD를 **조합(오케스트레이션)**하여 비즈니스 로직을 수행하는 레이어.
API 레이어에서 호출되며, ML이나 CRUD를 직접 호출한다.

#### `services/db/`

| 파일 | 역할 | 포함 코드 |
|------|------|----------|
| `column_detection_service.py` | 개인정보 컬럼 탐지 | `ColumnDetector` 클래스 — LLM 기반 컬럼 분류 오케스트레이션 |
| `encryption_service.py` | 암호화 여부 확인 | `EncryptionClassifier` 클래스 — 커넥션 조회 -> 데이터 샘플링 -> XGBoost 분류 오케스트레이션 |

#### `services/file/`

| 파일 | 역할 | 포함 코드 |
|------|------|----------|
| `document_service.py` | 문서 PII 탐지 | `DocumentDetector` 클래스 — HybridPIIDetector 호출, PII 통계 수집 헬퍼 |
| `image_service.py` | 이미지 탐지+마스킹 | `ImageService` 클래스 — ImageDetector + ImageMasker 조합 |
| `audio_service.py` | 오디오 PII 탐지 | `AudioDetector` 클래스 — AudioMasker.transcribe_and_detect() 호출 |
| `video_service.py` | 영상 PII 탐지 | `VideoDetector` 클래스 — 프레임별 얼굴 감지 + 오디오 추출/탐지 오케스트레이션 |
| `masking_service.py` | 마스킹 디스패치 | `Masker` 클래스 — file_type에 따라 적절한 processor에 위임 |

#### `services/file/processors/` — 파일 처리 구현체

실제 파일 변환/마스킹 **로우레벨 처리 로직**을 담당. 서비스 레이어에서 호출된다.

| 파일 | 역할 | 포함 코드 |
|------|------|----------|
| `text_extractor.py` | 텍스트 추출 | `TextExtractor` — PDF(PyMuPDF+OCR), DOCX(python-docx), TXT, 이미지(EasyOCR) |
| `document_masker.py` | 문서 마스킹 | `DocumentMasker` — `mask_text()`, `mask_pdf()`, `mask_docx()`, `mask_txt()` |
| `image_masker.py` | 이미지 마스킹 | `ImageMasker` — 얼굴 영역 블러 처리 (OpenCV) |
| `audio_masker.py` | 오디오 마스킹 | `AudioMasker` — Whisper STT -> PII 탐지 -> 타임스탬프 매칭 -> 비프음 삽입 (pydub) |
| `video_masker.py` | 영상 마스킹 | `VideoMasker` — 얼굴 블러(VideoProcessorEnhanced) + 오디오 마스킹 + ffmpeg 합성 |
| `video_processor.py` | 영상 프레임 처리 | `VideoProcessorEnhanced` — Kalman Filter 기반 얼굴 추적 + 프레임별 블러 |
| `blur_censor.py` | 블러 적용기 | `BlurCensor` — GaussianBlur로 바운딩박스 영역 블러 |

#### `services/chat/`

| 파일 | 역할 | 포함 코드 |
|------|------|----------|
| `assistant_service.py` | AI 어시스턴트 | `AIAssistant` — VectorDB + LangChain 기반 RAG 질의응답 |
| `regulation_service.py` | 법령 검색 | `RegulationSearch` — RetrievalQA Chain + FlashrankRerank 재정렬 |

---

### ⑤ `app/crud/` — DB 접근 (Repository 레이어)

데이터베이스 **연결, 쿼리, 데이터 읽기**만 담당. 비즈니스 로직은 포함하지 않는다.

| 파일 | 역할 | 포함 코드 |
|------|------|----------|
| `db_connect.py` | DB 커넥션 풀 관리 | `create_db_engine()` (SQLAlchemy), `get_sql_database()` (LangChain), `get_connection()` (psycopg 직접 연결) |
| `connection.py` | DB 커넥션 정보 조회 | `ConnectionRepository` — 메인 DB에서 `db_server_connections` 테이블 조회 (캐싱) |
| `data_sampling.py` | 대상 DB 데이터 샘플링 | `sample_column_values()` — 특정 테이블/컬럼에서 N건 샘플링 |
| `vector_db.py` | 법령 Vector DB 검색 | `VectorDB` — pgvector 코사인 유사도 검색, EmbeddingModel 호출 |

```python
# 사용 예시 (서비스에서 CRUD 호출)
from app.crud.db_connect import create_db_engine
from app.crud.data_sampling import sample_column_values

engine = create_db_engine(user=..., password=..., host=..., database=...)
values = sample_column_values(engine, "users", "phone_number", limit=100)
```

---

### ⑥ `app/ml/` — ML 모델 추론

모델 로드, 추론(inference), 결과 반환만 담당.
**전처리/후처리 파이프라인은 서비스 레이어**가 오케스트레이션.

#### `ml/pii_detectors/` — PII 탐지 모델

| 파일 | 역할 | 포함 코드 |
|------|------|----------|
| `regex_detector.py` | 정규식 PII 탐지 | `GeneralizedRegexPIIDetector` — 전화번호, 이메일, 주소, 주민번호, IP, 여권번호 정규식 패턴 매칭 (문서+오디오 통합) |
| `dl_detector.py` | 딥러닝 PII 탐지 | `KoELECTRAPIIDetector` — KoELECTRA NER 모델 기반 토큰 분류 + 신뢰도 필터링 (문서+오디오 통합) |
| `hybrid_detector.py` | 하이브리드 탐지 | `HybridPIIDetector` — Regex + DL 결과 병합, 주소 확장, 이름 확장, 문맥 전파, 엔티티 정제 |

#### 기타 ML 모델

| 파일 | 역할 | 포함 코드 |
|------|------|----------|
| `face_detector.py` | YOLO 얼굴 감지 | `YOLOFaceDetector` — YOLOv12 모델, 이미지 전처리(CLAHE), 프레임 단위 감지 |
| `image_detector.py` | 이미지 얼굴 탐지 | `ImageDetector` — YOLO 모델로 이미지 내 얼굴 좌표 추출, 마스킹 여부 판단 |
| `xgboost_classifier.py` | 암호화 분류기 | `EncryptionDetectionClassifier` — XGBoost 기반 데이터 암호화 여부 분류 |
| `bank_info.py` | 은행 정보 탐지 | `get_card_bin_prefix()`, `is_valid_bank_account()`, `get_bank_account_patterns()` |
| `whisper_stt.py` | 음성->텍스트 변환 | `WhisperSTT` — faster-whisper large-v3 모델, 한국어 STT, 단어 타임스탬프 (싱글톤, 지연 로딩) |
| `embedding_model.py` | 텍스트 임베딩 | `EmbeddingModel` — intfloat/multilingual-e5-large-instruct, 쿼리 임베딩 생성, pgvector 형식 변환 (클래스 레벨 캐싱) |

```python
# 사용 예시 (서비스에서 ML 모델 호출)
from app.ml.pii_detectors.hybrid_detector import HybridPIIDetector

detector = HybridPIIDetector(model_path="ParkJunSeong/PIILOT_NER_Model")
entities = detector.detect_pii("홍길동의 전화번호는 010-1234-5678입니다.")
```

---

### ⑦ `app/utils/` — 공유 유틸리티

여러 레이어에서 **공통으로 사용하는 헬퍼 함수**. 비즈니스 로직이나 모델 의존성 없음.

| 파일 | 역할 | 포함 함수/클래스 |
|------|------|----------------|
| `base64_utils.py` | Base64 인코딩/디코딩 통합 | `decode_base64_data()`, `encode_to_base64()`, `is_base64()`, `decode_base64_image()`, `encode_image_to_base64()`, `decode_base64_to_temp_file()`, `get_original_image_bytes()`, `get_original_image_base64()` |
| `image_utils.py` | 이미지 로드 | `load_image()` — 파일 경로 또는 base64 문자열 -> OpenCV ndarray |
| `temp_file.py` | 임시 파일 관리 | `TempFileManager` — 컨텍스트 매니저 (`with` 문), `create()`, `add()`, `cleanup()` |
| `overlap.py` | 엔티티 겹침 검사 | `is_overlapping(entity1, entity2)` — start/end 기반 겹침 판별 |

```python
# 사용 예시
from app.utils.base64_utils import is_base64, decode_base64_to_temp_file
from app.utils.temp_file import TempFileManager

with TempFileManager() as temp:
    if is_base64(video_data):
        path = decode_base64_to_temp_file(video_data, suffix='.mp4')
        temp.add(path)
    # ... 처리 후 자동 정리
```

---

## 레이어 간 의존성 규칙

```
api/ ──> schemas/
  │  ──> services/  (Depends로 주입)
  │
services/ ──> ml/      (모델 추론 호출)
  │       ──> crud/    (DB 접근 호출)
  │       ──> utils/   (유틸리티 호출)
  │
ml/ ──> core/    (상수, 로깅)
  │ ──> utils/   (유틸리티)
  │
crud/ ──> core/  (설정, 로깅)
  │
core/ ──> (외부 라이브러리만)
utils/ ──> core/ (로깅만)
```

**금지 방향:**
- `core/` -> 다른 레이어 import 금지
- `schemas/` -> `services/`, `ml/`, `crud/` import 금지
- `ml/` -> `services/` import 금지
- `crud/` -> `services/`, `ml/` import 금지

---

## 중복 제거 매핑

| # | 중복 내용 | 기존 위치 (2곳 이상) | 통합 위치 |
|---|----------|---------------------|----------|
| 1 | Base64 비디오/이미지 디코딩 | `video_detector.py`, `masker.py`, `file_ai.py`, `image_loader.py` | `utils/base64_utils.py` |
| 2 | 글로벌 에러 핸들링 | `db_ai.py`, `file_ai.py`, `chat_ai.py` (11곳 try/except) | `core/exceptions.py` -> `main.py` 등록 |
| 3 | PII 통계 수집 | `document_detector.py` PDF/DOCX/TXT 각각 | `services/file/document_service.py._collect_pii_stats()` |
| 4 | 임시 파일 정리 | `video_detector.py`, `masker.py` (수동 try/finally) | `utils/temp_file.py` (`TempFileManager`) |
| 5 | YOLOFaceDetector 초기화 | `video_detector.py`, `masker.py` 각각 new | `api/deps.py` + `@lru_cache` 싱글톤 |
| 6 | Regex PII 탐지기 | `detectors/regex_detector.py`, `audio_masking.py` (2개 클래스) | `ml/pii_detectors/regex_detector.py` (1개) |
| 7 | KoELECTRA PII 탐지기 | `detectors/dl_detector.py`, `audio_masking.py` (2개 클래스) | `ml/pii_detectors/dl_detector.py` (1개) |
| 8 | `_is_overlapping()` | `hybrid_detector.py`, `audio_masking.py` | `utils/overlap.py` |
| 9 | PII 상수 (이름, 임계값) | `personal_info.py`, `audio_masking.py` | `core/constants.py` |

---

## API 경로 매핑 (변경 없음)

| HTTP Method | Path | 파일 | 핸들러 함수 |
|-------------|------|------|------------|
| POST | `/api/ai/db/detect-columns` | `api/db.py` | `detect_personal_info_columns()` |
| POST | `/api/ai/db/check-encryption` | `api/db.py` | `check_encryption()` |
| POST | `/api/ai/file/document/detect` | `api/file.py` | `detect_document_personal_info()` |
| POST | `/api/ai/file/image/detect` | `api/file.py` | `detect_image_faces()` |
| POST | `/api/ai/file/image/detect-and-mask` | `api/file.py` | `detect_and_mask_image()` |
| POST | `/api/ai/file/image/detect-and-mask/file` | `api/file.py` | `detect_and_mask_image_file()` |
| POST | `/api/ai/file/audio/detect` | `api/file.py` | `detect_audio_personal_info()` |
| POST | `/api/ai/file/video/detect` | `api/file.py` | `detect_video_personal_info()` |
| POST | `/api/ai/file/mask` | `api/file.py` | `apply_masking()` |
| POST | `/api/ai/chat/chat` | `api/chat.py` | `chat()` |
| POST | `/api/ai/chat/search-regulations` | `api/chat.py` | `search_regulations()` |
| GET | `/` | `main.py` | `root()` |
| GET | `/health` | `main.py` | `health_check()` |

---

## 하위 호환 파일 목록

아래 파일들은 기존 import 경로를 유지하기 위해 남겨둔 **래퍼 파일**입니다.
내부에서 새 위치의 모듈을 재수출(re-export)합니다.
새로 작성하는 코드에서는 사용하지 마세요.

| 기존 경로 (래퍼) | 새 경로 (실제 코드) |
|-----------------|-------------------|
| `app/config.py` | `app/core/config.py` + `app/core/constants.py` |
| `app/models/personal_info.py` | `app/core/constants.py` |
| `app/models/request.py` | `app/schemas/db.py` + `file.py` + `chat.py` |
| `app/models/response.py` | `app/schemas/db.py` + `file.py` + `chat.py` |
| `app/utils/logger.py` | `app/core/logging.py` |
| `app/utils/exceptions.py` | `app/core/exceptions.py` |
| `app/utils/image_loader.py` | `app/utils/image_utils.py` |
| `app/utils/db_connect.py` | `app/crud/db_connect.py` |
| `app/api/db_ai.py` | `app/api/db.py` |
| `app/api/file_ai.py` | `app/api/file.py` |
| `app/api/chat_ai.py` | `app/api/chat.py` |
| `app/services/file/detectors/*` | `app/ml/pii_detectors/*` |
| `app/services/file/extractors/*` | `app/services/file/processors/text_extractor.py` |
| `app/services/file/face_detector.py` | `app/ml/face_detector.py` |
| `app/services/file/image_detector.py` | `app/ml/image_detector.py` |
| `app/services/file/blur_censor.py` | `app/services/file/processors/blur_censor.py` |
| `app/services/file/video_processor.py` | `app/services/file/processors/video_processor.py` |
| `app/services/file/masker.py` | `app/services/file/masking_service.py` + `processors/*_masker.py` |
| `app/services/file/audio_masking.py` | `app/services/file/processors/audio_masker.py` + `app/ml/whisper_stt.py` |
| `app/services/chat/vector_db.py` | `app/crud/vector_db.py` |
| `app/services/db/detectors/*` | `app/ml/xgboost_classifier.py` + `app/ml/bank_info.py` |

---

## 기술 스택

- **Framework**: FastAPI
- **AI/ML**:
  - KoELECTRA NER (문서 PII 탐지)
  - YOLOv12 (얼굴 감지)
  - faster-whisper (음성 STT)
  - XGBoost (암호화 분류)
  - LangChain + HuggingFace LLM (법령 RAG)
  - multilingual-e5-large-instruct (임베딩)
- **Database**: PostgreSQL + pgvector
- **Document Processing**: PyMuPDF, python-docx, EasyOCR
- **Audio/Video**: pydub, OpenCV, ffmpeg
