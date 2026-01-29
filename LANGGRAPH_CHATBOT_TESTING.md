# LangGraph Self-RAG 챗봇 테스트 가이드

## 사전 준비

### 1. 의존성 설치
```bash
# langgraph 패키지 설치 확인
pip install langgraph>=0.2.0

# 또는 requirements.txt 전체 설치
pip install -r requirements.txt
```

### 2. 서버 실행
```bash
# 개발 모드로 실행
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 또는 프로덕션 모드
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 3. 헬스 체크
```bash
curl http://localhost:8000/health
```

**예상 응답**:
```json
{"status": "healthy"}
```

---

## API 엔드포인트 테스트

⚠️ **중요**: LangGraph 챗봇 API 경로는 `/api/ai/chat/chat/langgraph`입니다.
- Base URL: `http://localhost:8000`
- Full Path: `http://localhost:8000/api/ai/chat/chat/langgraph`

---

## 테스트 시나리오

### 1. 기본 질문 (일반 대화)

#### 1.1 간단한 인사
```bash
curl -X POST "http://localhost:8000/api/ai/chat/chat/langgraph" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "안녕하세요",
    "conversation_id": "test-session-1"
  }'
```

**예상 응답**:
```json
{
  "answer": "안녕하세요! 무엇을 도와드릴까요?",
  "sources": [],
  "query_type": "general",
  "relevance_score": null,
  "hallucination_score": null
}
```

**확인 사항**:
- [ ] 응답이 정상적으로 반환됨
- [ ] `query_type`이 "general"로 분류됨
- [ ] `answer` 필드에 답변이 포함됨

---

### 2. 법령 검색 질문 (vector_search)

#### 2.1 개인정보보호법 관련 질문
```bash
curl -X POST "http://localhost:8000/api/ai/chat/chat/langgraph" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "개인정보보호법에서 개인정보 처리 원칙은 무엇인가요?",
    "conversation_id": "test-session-2"
  }'
```

**예상 응답**:
```json
{
  "answer": "개인정보보호법에 따르면...",
  "sources": [
    "개인정보보호법 제3조",
    "개인정보보호법 제4조"
  ],
  "query_type": "vector_search",
  "relevance_score": 0.85,
  "hallucination_score": 0.92
}
```

**확인 사항**:
- [ ] `query_type`이 "vector_search"로 분류됨
- [ ] `sources` 배열에 법령 출처가 포함됨
- [ ] `relevance_score`가 0.6 이상 (관련성 충분)
- [ ] `hallucination_score`가 0.7 이상 (근거 충분)

#### 2.2 암호화 관련 질문
```bash
curl -X POST "http://localhost:8000/api/ai/chat/chat/langgraph" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "개인정보 암호화 의무 규정은 무엇인가요?",
    "conversation_id": "test-session-3"
  }'
```

**확인 사항**:
- [ ] 법령 검색이 정상 작동
- [ ] 관련 문서가 검색됨
- [ ] 답변에 출처가 포함됨

---

### 3. 대화 맥락 테스트 (세션 유지)

#### 3.1 연속 질문
```bash
# 첫 번째 질문
curl -X POST "http://localhost:8000/api/ai/chat/chat/langgraph" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "개인정보보호법 제3조는 무엇인가요?",
    "conversation_id": "test-session-context"
  }'

# 두 번째 질문 (이전 대화 맥락 활용)
curl -X POST "http://localhost:8000/api/ai/chat/chat/langgraph" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "그 조항의 세부 내용을 알려주세요",
    "conversation_id": "test-session-context"
  }'
```

**확인 사항**:
- [ ] 같은 `conversation_id`로 연속 질문 시 대화 맥락 유지
- [ ] 두 번째 질문에서 "그 조항"이 이전 질문의 제3조를 참조함

---

### 4. Self-RAG 재시도 테스트

#### 4.1 관련성 낮은 질문 (재검색 루프)
```bash
curl -X POST "http://localhost:8000/api/ai/chat/chat/langgraph" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "우주 여행에 대한 법령은 무엇인가요?",
    "conversation_id": "test-session-retry"
  }'
```

**확인 사항**:
- [ ] 관련 문서가 없거나 관련성이 낮으면 재검색 시도
- [ ] 로그에서 `retry_count` 증가 확인
- [ ] 최종적으로 "자료 없음" 답변 또는 경고 메시지

#### 4.2 근거 부족 답변 (재생성 루프)
관련 문서는 있지만 답변이 근거 부족인 경우, 자동으로 재생성 시도됩니다.

**확인 사항**:
- [ ] 로그에서 `generation_retry_count` 증가 확인
- [ ] 재생성 시도 후 근거 점수 개선 확인

---

### 5. Python 스크립트로 테스트

#### 5.1 기본 테스트 스크립트
```python
import requests
import json

BASE_URL = "http://localhost:8000"
ENDPOINT = f"{BASE_URL}/api/ai/chat/chat/langgraph"

def test_chatbot(question: str, conversation_id: str = "test-python"):
    """LangGraph 챗봇 테스트"""
    payload = {
        "question": question,
        "conversation_id": conversation_id
    }
    
    response = requests.post(ENDPOINT, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n질문: {question}")
        print(f"답변: {result['answer']}")
        print(f"질문 유형: {result['query_type']}")
        print(f"관련성 점수: {result.get('relevance_score')}")
        print(f"근거 점수: {result.get('hallucination_score')}")
        print(f"출처: {result['sources']}")
        return result
    else:
        print(f"오류 발생: {response.status_code}")
        print(response.text)
        return None

# 테스트 실행
if __name__ == "__main__":
    # 일반 질문
    test_chatbot("안녕하세요")
    
    # 법령 검색 질문
    test_chatbot("개인정보보호법 제3조는 무엇인가요?")
    
    # 연속 질문
    session_id = "test-session-python"
    test_chatbot("개인정보보호법에 대해 알려주세요", session_id)
    test_chatbot("그 법의 주요 원칙은?", session_id)
```

#### 5.2 실행 방법
```bash
# requests 라이브러리 설치
pip install requests

# 스크립트 실행
python test_langgraph_chatbot.py
```

---

### 6. 로그 확인

#### 6.1 서버 로그 확인
서버 실행 시 콘솔에서 다음 로그들을 확인할 수 있습니다:

```
[INFO] LangGraph 챗봇 질의: 개인정보보호법 제3조는 무엇인가요?, conversation_id=test-session-1
[INFO] LangGraph 챗봇 그래프 구성 시작
[INFO] 대화 이력 로드 시작: conversation_id=test-session-1
[INFO] 질문 분류 시작: 개인정보보호법 제3조는 무엇인가요?...
[INFO] 질문 분류 완료: vector_search
[INFO] 벡터 검색 시작: 개인정보보호법 제3조는 무엇인가요?..., version=0
[INFO] 벡터 검색 완료: 20개 문서 발견
[INFO] 관련성 평가 시작: 3개 문서
[INFO] 관련성 평가 완료: score=0.85, is_relevant=True
[INFO] Rerank 시작: 20개 문서
[INFO] Rerank 완료: 5개 문서 선별
[INFO] 답변 생성 시작
[INFO] 답변 생성 완료: 245자
[INFO] 환각 검증 시작
[INFO] 환각 검증 완료: score=0.92, is_grounded=True
[INFO] 대화 이력 저장 완료: 총 2개 메시지
```

#### 6.2 주요 로그 포인트
- **질문 분류**: `query_type` 확인
- **벡터 검색**: 검색된 문서 개수 확인
- **관련성 평가**: `relevance_score` 확인
- **재시도**: `retry_count`, `search_query_version` 확인
- **환각 검증**: `hallucination_score` 확인

---

### 7. 에러 처리 테스트

#### 7.1 잘못된 요청 형식
```bash
# 필수 필드 누락
curl -X POST "http://localhost:8000/api/ai/chat/chat/langgraph" \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "test-error"
  }'
```

**예상 응답**: 422 Validation Error

#### 7.2 빈 질문
```bash
curl -X POST "http://localhost:8000/api/ai/chat/chat/langgraph" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "",
    "conversation_id": "test-error"
  }'
```

**확인 사항**:
- [ ] 적절한 에러 메시지 반환
- [ ] 서버가 계속 실행됨 (크래시 없음)

---

### 8. 성능 테스트

#### 8.1 응답 시간 측정
```python
import time
import requests

def measure_response_time(question: str):
    """응답 시간 측정"""
    start_time = time.time()
    
    response = requests.post(
        "http://localhost:8000/api/ai/chat/chat/langgraph",
        json={
            "question": question,
            "conversation_id": "test-performance"
        }
    )
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"질문: {question}")
    print(f"응답 시간: {elapsed:.2f}초")
    print(f"상태 코드: {response.status_code}")
    
    return elapsed

# 테스트
measure_response_time("안녕하세요")  # 일반 질문 (빠름)
measure_response_time("개인정보보호법 제3조는 무엇인가요?")  # 검색 질문 (느림)
```

**예상 응답 시간**:
- 일반 질문: 2-5초
- 법령 검색 질문: 5-15초 (벡터 검색 + LLM 생성)

---

## 문제 해결

### 1. 모듈 import 오류
**증상**: `ModuleNotFoundError: No module named 'langgraph'`

**해결**:
```bash
pip install langgraph>=0.2.0
```

### 2. LLM 모델 로딩 실패
**증상**: `LLM 모델 로딩 실패` 로그

**해결**:
- `ModelManager.ensure_all_models()` 실행 확인
- 모델 캐시 디렉토리 확인: `models/huggingface/`
- GPU 메모리 확인 (CUDA 사용 시)

### 3. 벡터 DB 연결 실패
**증상**: `법령 검색 오류` 로그

**해결**:
- PostgreSQL 서버 실행 확인
- `DATABASE_URL` 환경 변수 확인
- `law_data` 테이블 존재 확인

### 4. 응답이 너무 느림
**원인**:
- LLM 모델이 첫 로드 시 시간 소요
- 벡터 검색이 많은 문서를 검색

**해결**:
- 첫 요청 후 캐시 활용
- `VECTOR_SEARCH_K` 값 조정 (config.py)

---

## 체크리스트

### 기본 기능
- [ ] 서버가 정상적으로 시작됨
- [ ] 헬스 체크 엔드포인트 응답 확인
- [ ] LangGraph 챗봇 엔드포인트 응답 확인

### 질문 분류
- [ ] 일반 질문이 "general"로 분류됨
- [ ] 법령 질문이 "vector_search"로 분류됨
- [ ] DB 질문이 "db_query"로 분류됨 (구현 시)

### 벡터 검색
- [ ] 법령 검색이 정상 작동
- [ ] 검색 결과가 `sources`에 포함됨
- [ ] 관련성 평가가 정상 작동

### Self-RAG
- [ ] 관련성 낮을 때 재검색 루프 작동
- [ ] 근거 부족 시 재생성 루프 작동
- [ ] 재시도 횟수 제한 작동

### 대화 맥락
- [ ] 같은 `conversation_id`로 대화 이력 유지
- [ ] 연속 질문에서 맥락 이해

### 에러 처리
- [ ] 잘못된 요청 시 적절한 에러 반환
- [ ] 서버가 크래시하지 않음

---

## 추가 참고사항

### API 문서 확인
FastAPI 자동 생성 문서:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### 설정 조정
`app/services/chat/config.py`에서 다음 값들을 조정할 수 있습니다:
- `RELEVANCE_THRESHOLD`: 관련성 임계값 (기본: 0.6)
- `GROUNDING_THRESHOLD`: 근거 임계값 (기본: 0.7)
- `MAX_SEARCH_RETRIES`: 최대 검색 재시도 (기본: 3)
- `MAX_GENERATION_RETRIES`: 최대 생성 재시도 (기본: 2)
- `VECTOR_SEARCH_K`: 초기 검색 개수 (기본: 20)
- `RERANK_TOP_N`: Rerank 후 최종 개수 (기본: 5)
