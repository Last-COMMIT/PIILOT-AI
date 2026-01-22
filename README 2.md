# PIILOT
## AI 기반 개인정보 보호 및 유출 관제 플랫폼 - AI 처리 전용 서비스

**AI 처리 전용 마이크로서비스**로, Spring Boot 백엔드와 통신하여 AI 관련 처리만 담당합니다.

- DB 서버의 개인정보 컬럼 탐지 및 암호화 여부 판단
- 파일 서버의 개인정보 탐지 (문서, 이미지, 음성, 영상) 및 마스킹
- 대화형 AI 어시스턴트 (법령 검색 및 자연어 질의응답)

> **참고**: 저장, 이슈 관리, 대시보드 등 비즈니스 로직은 Spring Boot에서 처리합니다.

## 주요 기능

### 1. DB AI 처리 (`/api/ai/db/`)
- **개인정보 컬럼 탐지**: LLM + LangChain으로 스키마 정보에서 개인정보 컬럼 자동 탐지
- **암호화 여부 확인**: 분류 모델로 데이터 샘플의 암호화 여부 판단

### 2. File AI 처리 (`/api/ai/file/`)
- **문서**: BERT + NER 기반 개인정보 탐지 및 마스킹
- **이미지**: Vision 기반 얼굴 탐지 및 마스킹
- **음성**: Whisper + LLM 기반 개인정보 탐지 및 마스킹
- **영상**: Vision + LLM 기반 얼굴 및 개인정보 탐지 및 마스킹

### 3. Chat AI 처리 (`/api/ai/chat/`)
- **자연어 질의응답**: LLM + LangChain으로 사용자 질의에 응답
- **법령 검색**: Vector DB에서 관련 법령 검색 (개인정보보호법, GDPR, CCPA 등)

## 설치 및 실행

### 1. 가상환경 설정
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 2. 의존성 설치
```bash
pip install -r requirements.txt
```

### 3. 환경 변수 설정
```bash
# .env 파일 생성 및 편집
OPENAI_API_KEY=your_openai_api_key_here
DATABASE_URL=postgresql://user:password@localhost:5432/piilot
# (선택) 벡터DB를 별도 DB/스키마로 쓸 경우
PGVECTOR_DATABASE_URL=postgresql://user:password@localhost:5432/piilot
```

### 4. Vector DB 초기화 (법령 데이터)
```bash
python scripts/setup_vector_db.py
```

### 5. 애플리케이션 실행
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. API 문서 확인
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 프로젝트 구조

자세한 구조는 [FINAL_STRUCTURE.md](./FINAL_STRUCTURE.md)를 참조하세요.

### 주요 디렉토리
- `app/api/`: API 엔드포인트 (요청/응답만)
- `app/services/`: AI 처리 로직 (순수 AI)
  - `db/`: DB AI 서비스
  - `file/`: 파일 AI 서비스
  - `chat/`: AI 어시스턴트
- `app/models/`: 요청/응답 모델

## 기술 스택

- **Framework**: FastAPI
- **AI/ML**: 
  - LangChain (챗봇, DB 탐색)
  - BERT + NER (문서 탐지)
  - Vision Models (이미지/영상)
  - LLM (음성, 챗봇)
- **Vector DB**: PostgreSQL + pgvector (법령 데이터)

## Spring Boot 연동

이 서비스는 Spring Boot 백엔드와 통신합니다:
- Spring Boot에서 요청 전송 → AI 처리 → 결과 반환
- 저장, 이슈 관리, 대시보드 등은 Spring Boot에서 처리