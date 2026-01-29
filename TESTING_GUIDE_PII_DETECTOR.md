# PII 탐지기 통합 후 테스트 가이드

## 변경 사항 요약

### 1. 통합된 PII 탐지기
- **이전**: `app/services/file/detectors/` 디렉토리에 중복된 탐지기 구현
- **이후**: `app/ml/pii_detectors/`의 통합된 탐지기 사용
  - `GeneralizedRegexPIIDetector` (정규식 기반)
  - `KoELECTRAPIIDetector` (KoELECTRA NER 기반)
  - `HybridPIIDetector` (정규식 + NER 통합)

### 2. 변경된 파일
- `app/services/file/document_detector.py`: `HybridPIIDetector` import 경로 변경
- `app/services/file/audio_masking.py`: 중복 클래스 제거, 통합 탐지기 사용
- `app/services/file/detectors/`: 디렉토리 삭제됨
- `app/services/file/audio_detector.py`: 삭제됨 (중복 코드, `audio_service.py`로 통합)
- `app/services/file/audio_service.py`: base64 디코딩 로직 추가, 표준 AudioDetector로 사용

## 테스트 항목

⚠️ **중요**: 모든 API 경로는 `/api/ai/file/` prefix를 사용합니다.
- 올바른 경로: `/api/ai/file/document/detect`
- 잘못된 경로: `/api/file/document/detect` ❌

### 1. 문서 PII 탐지 및 마스킹 (`/api/ai/file/document/detect`, `/api/ai/file/mask`)

#### 1.1 기본 탐지 기능
```bash
# 텍스트 문서 탐지 (file_content는 base64 인코딩된 파일 또는 텍스트)
curl -X POST "http://localhost:8000/api/ai/file/document/detect" \
  -H "Content-Type: application/json" \
  -d '{
    "file_content": "base64_encoded_pdf_or_text_content",
    "file_type": "txt"
  }'

# 또는 텍스트 직접 전송
curl -X POST "http://localhost:8000/api/ai/file/document/detect" \
  -H "Content-Type: application/json" \
  -d '{
    "file_content": "안녕하세요. 제 전화번호는 010-1234-5678입니다.",
    "file_type": "txt"
  }'
```

**확인 사항**:
- [✅] PII 탐지 결과가 정상적으로 반환됨
- [✅] 전화번호, 이메일, 주소, 주민등록번호 등이 정확히 탐지됨
- [✅] 탐지된 항목에 `type`, `value`, `start`, `end`, `confidence` 필드가 포함됨

#### 1.2 마스킹 기능
```bash
# 문서 마스킹은 /mask 엔드포인트 사용 (file_type: "document")
# Base64 형식
curl -X POST "http://localhost:8000/api/ai/file/mask" \
  -H "Content-Type: application/json" \
  -d '{
    "file_type": "document",
    "file_data": "base64_encoded_document_content",
    "file_format": "base64",
    "detected_items": [
      {
        "type": "p_ph",
        "value": "010-1234-5678",
        "start": 0,
        "end": 13,
        "confidence": 0.95
      }
    ]
  }'

# 파일 경로 형식 (지원됨)
curl -X POST "http://localhost:8000/api/ai/file/mask" \
  -H "Content-Type: application/json" \
  -d '{
    "file_type": "document",
    "file_data": "/path/to/document.txt",
    "file_format": "path",
    "detected_items": [...]
  }'
```

**확인 사항**:
- [ ] 마스킹된 문서가 base64 인코딩된 문자열로 반환됨
- [ ] 지정된 PII 항목이 정확히 마스킹됨
- [ ] 응답에 `masked_file` (base64)와 `file_type` 필드가 포함됨

### 2. 이미지 얼굴 탐지 및 마스킹 (`/api/ai/file/image/detect`, `/api/ai/file/mask`)

#### 2.1 Base64 형식 이미지 탐지
```bash
curl -X POST "http://localhost:8000/api/ai/file/image/detect" \
  -H "Content-Type: application/json" \
  -d '{
    "image_data": "base64_encoded_image_string",
    "image_format": "base64"
  }'
```

**확인 사항**:
- [ ] 이미지에서 얼굴 탐지가 정상 작동
- [ ] 탐지된 얼굴에 `x`, `y`, `width`, `height`, `confidence` 필드가 포함됨
- [ ] 응답에 `detected_faces` 배열이 포함됨

#### 2.2 파일 경로 형식 이미지 탐지
```bash
curl -X POST "http://localhost:8000/api/ai/file/image/detect" \
  -H "Content-Type: application/json" \
  -d '{
    "image_data": "/path/to/test.jpg",
    "image_format": "path"
  }'
```

**확인 사항**:
- [✅] 파일 경로로 이미지 파일을 정상적으로 읽음
- [✅] Base64 형식과 동일한 결과 형식 반환

#### 2.3 이미지 마스킹
```bash
# 이미지 마스킹은 /mask 엔드포인트 사용 (file_type: "image")
# Base64 형식
curl -X POST "http://localhost:8000/api/ai/file/mask" \
  -H "Content-Type: application/json" \
  -d '{
    "file_type": "image",
    "file_data": "base64_encoded_image_string",
    "file_format": "base64",
    "detected_items": [
      {
        "x": 100,
        "y": 100,
        "width": 50,
        "height": 50,
        "confidence": 0.95
      }
    ]
  }'

# 파일 경로 형식
curl -X POST "http://localhost:8000/api/ai/file/mask" \
  -H "Content-Type: application/json" \
  -d '{
    "file_type": "image",
    "file_data": "/path/to/test.jpg",
    "file_format": "path",
    "detected_items": []
  }'
```

**확인 사항**:
- [✅] 마스킹된 이미지가 base64 인코딩된 문자열로 반환됨
- [✅] 지정된 얼굴 영역이 정확히 마스킹됨
- [✅] 응답에 `masked_file` (base64)와 `file_type` 필드가 포함됨
- [✅] 파일 경로 형식도 정상 작동함
- [✅] `detected_items`가 비어있으면 자동 탐지 수행

### 3. 오디오 PII 탐지 및 마스킹 (`/api/ai/file/audio/detect`, `/api/ai/file/mask`)

#### 2.1 Base64 형식 오디오 탐지
```bash
curl -X POST "http://localhost:8000/api/ai/file/audio/detect" \
  -H "Content-Type: application/json" \
  -d '{
    "audio_data": "base64_encoded_audio_string",
    "audio_format": "base64"
  }'
```

**확인 사항**:
- [ ] STT(음성→텍스트) 변환이 정상 작동
- [ ] 텍스트에서 PII가 정확히 탐지됨
- [ ] 탐지된 항목에 타임스탬프(`start_time`, `end_time`)가 포함됨
- [ ] 응답에 `detected_items` 배열이 포함됨

#### 2.2 파일 경로 형식 오디오 탐지
```bash
curl -X POST "http://localhost:8000/api/ai/file/audio/detect" \
  -H "Content-Type: application/json" \
  -d '{
    "audio_data": "/path/to/test.mp3",
    "audio_format": "path"
  }'
```

**확인 사항**:
- [✅] 파일 경로로 오디오 파일을 정상적으로 읽음
- [✅] Base64 형식과 동일한 결과 형식 반환

#### 2.3 오디오 마스킹
```bash
# 오디오 마스킹은 /mask 엔드포인트 사용 (file_type: "audio")
# Base64 형식
curl -X POST "http://localhost:8000/api/ai/file/mask" \
  -H "Content-Type: application/json" \
  -d '{
    "file_type": "audio",
    "file_data": "base64_encoded_audio_string",
    "file_format": "base64",
    "detected_items": [
      {
        "type": "p_ph",
        "value": "010-1234-5678",
        "start_time": 1.5,
        "end_time": 2.0,
        "confidence": 0.95
      }
    ]
  }'

# 파일 경로 형식
curl -X POST "http://localhost:8000/api/ai/file/mask" \
  -H "Content-Type: application/json" \
  -d '{
    "file_type": "audio",
    "file_data": "/path/to/test.mp3",
    "file_format": "path",
    "detected_items": []
  }'
```

**확인 사항**:
- [✅] 마스킹된 오디오가 base64 인코딩된 문자열로 반환됨
- [✅] 지정된 타임스탬프 구간에 삐- 소리가 삽입됨
- [✅] 응답에 `masked_file` (base64)와 `file_type` 필드가 포함됨
- [✅] 파일 경로 형식도 정상 작동함

### 4. 비디오 PII 탐지 및 마스킹 (`/api/ai/file/video/detect`, `/api/ai/file/mask`)

#### 3.1 Base64 형식 비디오 탐지
```bash
curl -X POST "http://localhost:8000/api/ai/file/video/detect" \
  -H "Content-Type: application/json" \
  -d '{
    "video_data": "base64_encoded_video_string",
    "video_format": "base64"
  }'
```

**확인 사항**:
- [ ] 비디오에서 얼굴/오디오 탐지가 정상 작동
- [ ] 응답에 `success`, `status`, `message` 필드가 포함됨
- [ ] `status`가 "detected" 또는 "no_pii"로 반환됨

#### 3.2 파일 경로 형식 비디오 탐지
```bash
curl -X POST "http://localhost:8000/api/ai/file/video/detect" \
  -H "Content-Type: application/json" \
  -d '{
    "video_data": "/path/to/test.mp4",
    "video_format": "path"
  }'
```

**확인 사항**:
- [✅] 파일 경로로 비디오 파일을 정상적으로 읽음
- [✅] Base64 형식과 동일한 결과 형식 반환

#### 3.3 비디오 마스킹
```bash
# 비디오 마스킹은 /mask 엔드포인트 사용 (file_type: "video")
# Base64 형식
curl -X POST "http://localhost:8000/api/ai/file/mask" \
  -H "Content-Type: application/json" \
  -d '{
    "file_type": "video",
    "file_data": "base64_encoded_video_string",
    "file_format": "base64",
    "detected_items": [
      {
        "x": 100,
        "y": 100,
        "width": 50,
        "height": 50,
        "confidence": 0.95,
        "frame_number": 1
      }
    ]
  }'

# 파일 경로 형식
curl -X POST "http://localhost:8000/api/ai/file/mask" \
  -H "Content-Type: application/json" \
  -d '{
    "file_type": "video",
    "file_data": "/path/to/test.mp4",
    "file_format": "path",
    "detected_items": []
  }'
```

**확인 사항**:
- [✅] 마스킹된 비디오가 base64 인코딩된 문자열로 반환됨
- [✅] 지정된 얼굴 영역이 정확히 마스킹됨
- [✅] 파일 경로 형식도 정상 작동함

### 5. 통합 마스킹 API (`/api/ai/file/mask`)

#### 5.1 자동 탐지 후 마스킹 (이미지)
```bash
# detected_items가 비어있으면 자동 탐지 수행
# Base64 형식
curl -X POST "http://localhost:8000/api/ai/file/mask" \
  -H "Content-Type: application/json" \
  -d '{
    "file_type": "image",
    "file_data": "base64_encoded_image_string",
    "file_format": "base64",
    "detected_items": []
  }'

# 파일 경로 형식
curl -X POST "http://localhost:8000/api/ai/file/mask" \
  -H "Content-Type: application/json" \
  -d '{
    "file_type": "image",
    "file_data": "/path/to/test.jpg",
    "file_format": "path",
    "detected_items": []
  }'
```

**확인 사항**:
- [✅] `detected_items`가 비어있을 때 자동 얼굴 탐지 수행
- [✅] 탐지 후 마스킹이 정상 작동
- [✅] 로그에 "자동 탐지 완료" 메시지가 출력됨
- [✅] 얼굴이 탐지되지 않으면 원본 이미지 반환
- [✅] 파일 경로 형식도 정상 작동함

#### 5.2 자동 탐지 후 마스킹 (오디오)
```bash
# detected_items가 비어있으면 자동 탐지 수행
# Base64 형식
curl -X POST "http://localhost:8000/api/ai/file/mask" \
  -H "Content-Type: application/json" \
  -d '{
    "file_type": "audio",
    "file_data": "base64_encoded_audio_string",
    "file_format": "base64",
    "detected_items": []
  }'

# 파일 경로 형식
curl -X POST "http://localhost:8000/api/ai/file/mask" \
  -H "Content-Type: application/json" \
  -d '{
    "file_type": "audio",
    "file_data": "/path/to/test.mp3",
    "file_format": "path",
    "detected_items": []
  }'
```

**확인 사항**:
- [✅] `detected_items`가 비어있을 때 자동 탐지 수행
- [✅] 탐지 후 마스킹이 정상 작동
- [✅] 로그에 "자동 탐지 완료" 메시지가 출력됨
- [✅] 파일 경로 형식도 정상 작동함

#### 5.3 자동 탐지 후 마스킹 (비디오)
```bash
# Base64 형식
curl -X POST "http://localhost:8000/api/ai/file/mask" \
  -H "Content-Type: application/json" \
  -d '{
    "file_type": "video",
    "file_data": "base64_encoded_video_string",
    "file_format": "base64",
    "detected_items": []
  }'

# 파일 경로 형식
curl -X POST "http://localhost:8000/api/ai/file/mask" \
  -H "Content-Type: application/json" \
  -d '{
    "file_type": "video",
    "file_data": "/path/to/test.mp4",
    "file_format": "path",
    "detected_items": []
  }'
```

**확인 사항**:
- [✅] 비디오에서 얼굴과 오디오 자동 탐지 수행
- [✅] 탐지 후 마스킹이 정상 작동
- [✅] 파일 경로 형식도 정상 작동함

## 영향 범위

### 직접 영향
- ✅ 문서 PII 탐지 및 마스킹 서비스
- ✅ 이미지 얼굴 탐지 및 마스킹 서비스
- ✅ 오디오 PII 탐지 및 마스킹 서비스
- ✅ 비디오 PII 탐지 및 마스킹 서비스

### 간접 영향
- ✅ `/api/ai/file/document/detect` 엔드포인트
- ✅ `/api/ai/file/image/detect` 엔드포인트
- ✅ `/api/ai/file/audio/detect` 엔드포인트
- ✅ `/api/ai/file/video/detect` 엔드포인트
- ✅ `/api/ai/file/mask` 엔드포인트 (모든 파일 타입 통합)

### 영향 없음
- ✅ DB 암호화 분류기 (`app/services/db/detectors/classifier.py`)
- ✅ 이미지/비디오 얼굴 탐지 (`app/ml/face_detector.py`, `app/ml/image_detector.py`)
- ✅ 채팅 AI 서비스 (`app/services/chat/`)

### 추가 통합 사항
- ✅ `app/services/file/audio_detector.py` 삭제됨 (중복 코드)
- ✅ 모든 AudioDetector 사용처가 `app/services/file/audio_service.py`로 통합됨
  - `app/api/file_ai.py`
  - `app/services/file/video_detector.py`
  - `app/api/deps.py` (이미 사용 중이었음)

## 주의사항

1. **모델 로딩**: 통합된 탐지기는 `ModelManager`를 통해 중앙 관리되므로, 첫 요청 시 모델 로딩 시간이 소요될 수 있습니다.

2. **캐시 디렉토리**: HuggingFace 모델은 `models/huggingface/` 디렉토리에 캐시됩니다.

3. **로깅**: 모든 탐지 작업은 `logger`를 통해 로깅되므로, 로그를 확인하여 정상 작동 여부를 확인할 수 있습니다.

4. **에러 처리**: 통합된 탐지기는 동일한 인터페이스를 제공하므로, 기존 코드와 호환됩니다.

## 롤백 계획

만약 문제가 발생할 경우:
1. `app/services/file/detectors/` 디렉토리를 복원
2. `document_detector.py`와 `audio_masking.py`의 import 경로를 원래대로 복원
3. 중복 클래스를 다시 추가

하지만 통합된 탐지기는 기존 코드와 동일한 인터페이스를 제공하므로, 롤백이 필요할 가능성은 낮습니다.
