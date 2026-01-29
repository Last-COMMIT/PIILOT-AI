# API 엔드포인트 목록

## 기본 정보
- **Base URL**: `http://localhost:8000`
- **API Prefix**: `/api/ai`

## 파일 관련 API (`/api/ai/file`)

### 문서 탐지
- **POST** `/api/ai/file/document/detect`
  - 문서에서 PII 탐지
  - 요청: `DocumentDetectionRequest`
  - 응답: `DocumentDetectionResponse`

### 이미지 탐지
- **POST** `/api/ai/file/image/detect`
  - 이미지에서 얼굴 탐지
  - 요청: `ImageDetectionRequest`
  - 응답: `ImageDetectionResponse`

- **POST** `/api/ai/file/image/detect-and-mask`
  - 이미지 탐지 및 마스킹 (한 번에 처리)
  - 요청: `ImageDetectionRequest`
  - 응답: `MaskingResponse`

- **POST** `/api/ai/file/image/detect-and-mask/file`
  - 이미지 탐지 및 마스킹 (이미지 파일로 직접 반환)
  - 요청: `ImageDetectionRequest`
  - 응답: 이미지 파일 (binary)

### 오디오 탐지
- **POST** `/api/ai/file/audio/detect`
  - 오디오에서 PII 탐지
  - 요청: `AudioDetectionRequest` (`audio_format`: "base64" 또는 "path")
  - 응답: `AudioDetectionResponse`

### 비디오 탐지
- **POST** `/api/ai/file/video/detect`
  - 비디오에서 얼굴 및 오디오 PII 탐지
  - 요청: `VideoDetectionRequest` (`video_format`: "base64" 또는 "path")
  - 응답: `VideoDetectionResponse`

### 통합 마스킹
- **POST** `/api/ai/file/mask`
  - 모든 파일 타입 마스킹 처리
  - 요청: `MaskingRequest` (`file_type`, `file_data`, `file_format`, `detected_items`)
  - 응답: `MaskingResponse`
  - 지원 타입: `document`, `image`, `audio`, `video`
  - 지원 포맷: `base64`, `path`

## DB 관련 API (`/api/ai/db`)

### 컬럼 탐지
- **POST** `/api/ai/db/detect-columns`
  - 데이터베이스 컬럼에서 개인정보 탐지
  - 요청: `ColumnDetectionRequest`
  - 응답: `ColumnDetectionResponse`

### 암호화 확인
- **POST** `/api/ai/db/check-encryption`
  - 컬럼의 암호화 여부 확인
  - 요청: `EncryptionCheckRequest`
  - 응답: `EncryptionCheckResponse`

## 채팅 관련 API (`/api/ai/chat`)

### AI 어시스턴트
- **POST** `/api/ai/chat/chat`
  - 자연어 질의응답
  - 요청: `ChatRequest`
  - 응답: `ChatResponse`

### 규정 검색
- **POST** `/api/ai/chat/search-regulations`
  - 관련 규정 검색
  - 요청: `RegulationSearchRequest`
  - 응답: `RegulationSearchResponse`

## 헬스 체크
- **GET** `/health`
  - 서비스 상태 확인
  - 응답: `{"status": "healthy"}`

## 중요 사항

⚠️ **주의**: 모든 API 경로는 `/api/ai/` prefix를 사용합니다.
- ❌ 잘못된 경로: `/api/file/document/detect`
- ✅ 올바른 경로: `/api/ai/file/document/detect`
