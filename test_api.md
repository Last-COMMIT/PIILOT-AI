# API 테스트 가이드

## 1. 서버 실행

```bash
# 가상환경 활성화
source venv/bin/activate  # Windows: venv\Scripts\activate

# 서버 실행
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

서버가 실행되면: http://localhost:8000

## 2. Postman 테스트

### 이미지 얼굴 탐지 API

**엔드포인트:**
```
POST http://localhost:8000/api/ai/file/image/detect
```

**Headers:**
```
Content-Type: application/json
```

**Request Body (JSON):**

#### 방법 1: Base64 이미지 전송
```json
{
  "image_data": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...",
  "image_format": "base64"
}
```

#### 방법 2: 파일 경로 전송 (로컬 테스트용)
```json
{
  "image_data": "test_image/input_image2.jpeg",
  "image_format": "path"
}
```

**Response 예시:**
```json
{
  "detected_faces": [
    {
      "x": 100,
      "y": 150,
      "width": 200,
      "height": 250,
      "confidence": 0.95
    }
  ]
}
```

### 이미지 탐지 및 마스킹 API (이미지 반환) ⭐

**엔드포인트:**
```
POST http://localhost:8000/api/ai/file/image/detect-and-mask
```

**Headers:**
```
Content-Type: application/json
```

**Request Body (JSON):**
```json
{
  "image_data": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...",
  "image_format": "base64"
}
```

또는
```json
{
  "image_data": "test_image/input_image2.jpeg",
  "image_format": "path"
}
```

**Response 예시:**
```json
{
  "masked_file": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...",
  "file_type": "image"
}
```

**응답 이미지 사용 방법:**
1. `masked_file` 값을 복사
2. Base64 디코딩 도구 사용 (https://base64.guru/converter/decode/image)
3. 또는 브라우저에서 직접 보기: `data:image/jpeg;base64,{masked_file}` 를 주소창에 입력

### 이미지 탐지 및 마스킹 API (이미지 파일 직접 반환) ⭐⭐

**엔드포인트:**
```
POST http://localhost:8000/api/ai/file/image/detect-and-mask/file
```

**Headers:**
```
Content-Type: application/json
```

**Request Body (JSON):**
```json
{
  "image_data": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...",
  "image_format": "base64"
}
```

**Response:**
- **Content-Type**: `image/jpeg`
- **Body**: 마스킹된 이미지 파일 (바이너리)

**Postman 사용법:**
1. 요청 전송
2. Response 탭에서 **"Send and Download"** 클릭
3. 이미지 파일이 다운로드됨
4. 또는 Response 탭에서 **"Preview"** 클릭하여 바로 확인 가능

## 3. 이미지를 Base64로 변환하는 방법

### Python 스크립트
```python
import base64

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/jpeg;base64,{encoded_string}"

# 사용 예시
base64_image = image_to_base64("test_image/input_image2.jpeg")
print(base64_image)
```

### 온라인 도구
- https://www.base64-image.de/
- https://base64.guru/converter/encode/image

## 4. Swagger UI로 테스트

서버 실행 후 브라우저에서 접속:
```
http://localhost:8000/docs
```

Swagger UI에서 직접 테스트 가능합니다.

## 5. cURL로 테스트

```bash
# 파일 경로 방식
curl -X POST "http://localhost:8000/api/ai/file/image/detect" \
  -H "Content-Type: application/json" \
  -d '{
    "image_data": "test_image/input_image2.jpeg",
    "image_format": "path"
  }'

# Base64 방식
curl -X POST "http://localhost:8000/api/ai/file/image/detect" \
  -H "Content-Type: application/json" \
  -d '{
    "image_data": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
    "image_format": "base64"
  }'
```

## 6. 주의사항

1. **모델 파일**: `models/vision/yolov12n-face.pt` 파일이 있어야 합니다.
2. **이미지 크기**: 너무 큰 이미지는 처리 시간이 오래 걸릴 수 있습니다.
3. **Base64 크기**: Base64 인코딩된 이미지는 원본보다 약 33% 큽니다.

