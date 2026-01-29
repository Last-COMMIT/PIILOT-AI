# LangGraph Self-RAG 챗봇 구조 가이드

## 📋 전체 구조 개요

```
START 
  ↓
load_memory (대화이력 로드)
  ↓
classify (질문 분류)
  ↓ (조건부 분기)
  ├─→ db_query (DB 조회) → generate_answer
  ├─→ vector_search (벡터 검색) → check_relevance
  ├─→ both_query (DB + 벡터) → check_relevance
  └─→ generate_answer (일반 질문)
  
check_relevance (관련성 평가) - Self-RAG ①
  ↓ (조건부 분기)
  ├─→ rerank (관련성 높음)
  ├─→ vector_search (관련성 낮음 - 재검색 루프)
  └─→ generate_answer (재시도 초과)
  
rerank (Reranker로 재순위화)
  ↓
generate_answer (답변 생성)
  ↓
check_hallucination (환각 검증) - Self-RAG ②
  ↓ (조건부 분기)
  ├─→ save_memory (근거 충분)
  └─→ generate_answer (근거 부족 - 재생성 루프)
  
save_memory (대화이력 저장)
  ↓
END
```

---

## 🔧 필요한 노드 (Node) 및 기능

### 1. load_memory
**역할**: 대화 이력 로드  
**필요 기능**:
- 이전 대화 메시지 불러오기 (session_id 기준)
- 메시지 개수 제한 (최근 N개만, 토큰 제한 고려)
- retry_count, search_query_version 등 카운터 초기화

---

### 2. classify
**역할**: 질문 유형 분류  
**필요 기능**:
- LLM 호출하여 질문 유형 판단
  - `db_query`: DB 조회 필요
  - `vector_search`: 법령/내규 검색 필요
  - `both`: 둘 다 필요
  - `general`: 일반 대화
- 이전 대화 맥락 포함하여 판단 (대명사 해석 등)

---

### 3. db_query
**역할**: PostgreSQL DB 조회  
**필요 기능**:
- Text-to-SQL: 자연어 질문 → SQL 쿼리 생성
- SQL 실행 및 결과 반환
- 이전 대화 맥락 고려 (예: "그 회사" → 이전 언급된 회사명)
- 에러 핸들링

---

### 4. vector_search
**역할**: pgvector로 법령/내규 검색  
**필요 기능**:
- 검색 쿼리 최적화 (대화 맥락 포함)
- **Self-RAG**: 재시도 시 쿼리 개선
  - `search_query_version` 확인하여 다른 표현 사용
  - 동의어, 상위/하위 개념, 관련 키워드 추가
- pgvector 유사도 검색 (k=20 정도, 넉넉하게)
- 검색 결과 + similarity score 저장

---

### 5. both_query
**역할**: DB와 Vector 검색 동시 실행  
**필요 기능**:
- `db_query` 실행
- `vector_search` 실행
- 두 결과를 state에 모두 저장

---

### 6. check_relevance ⭐ Self-RAG ①
**역할**: 검색 결과의 관련성 평가  
**필요 기능**:
- LLM으로 검색된 문서와 질문의 관련성 평가
- 관련성 점수 계산 (0.0 ~ 1.0)
- 결과 저장:
  - `is_relevant`: bool
  - `relevance_score`: float
- 상위 3개 문서만 평가 (비용 절감)

**판단 기준**:
- 문서가 질문에 답할 수 있는 정보를 포함하는가?
- 주제가 일치하는가?

---

### 7. rerank
**역할**: Reranker로 검색 결과 재순위화  
**필요 기능**:
- Cohere Rerank (또는 다른 Reranker) 호출
- 대화 맥락 포함하여 재순위화
- Top-N (5개 정도) 선별
- 재순위화된 문서 + relevance score 저장

---

### 8. generate_answer
**역할**: 최종 답변 생성  
**필요 기능**:
- LLM 호출하여 답변 생성
- 입력:
  - 대화 이력 (최근 5개)
  - DB 조회 결과 (있으면)
  - Reranked 문서들 (있으면)
  - 사용자 질문
- **Self-RAG 프롬프트 강화**:
  - "제공된 자료만 사용"
  - "근거 없으면 추측 금지"
  - "출처 명시"
- 답변을 state에 저장

---

### 9. check_hallucination ⭐ Self-RAG ②
**역할**: 생성된 답변의 환각(hallucination) 검증  
**필요 기능**:
- LLM으로 답변이 검색 결과를 근거로 하는지 평가
- 근거 점수 계산 (0.0 ~ 1.0)
- 결과 저장:
  - `is_grounded`: bool
  - `hallucination_score`: float
  - `unsupported_claims`: list (근거 없는 주장들)

**판단 기준**:
- 답변의 각 주장이 제공된 문서에 근거하는가?
- 추측이나 외부 지식을 사용했는가?

**특수 케이스**:
- DB 조회만 한 경우: 환각 체크 스킵 (is_grounded=True)
- 검색 결과 없음: "자료 없음" 답변이면 OK

---

### 10. save_memory
**역할**: 대화 이력 저장  
**필요 기능**:
- 현재 질문 + 답변을 messages에 추가
- DB에 영구 저장 (선택사항)
- Session 관리

---

## 🔀 라우팅 함수 (Conditional Edges)

### route_after_classify
```python
query_type 기준:
- "db_query" → db_query 노드
- "vector_search" → vector_search 노드  
- "both" → both_query 노드
- "general" → generate_answer 노드 (검색 스킵)
```

### route_after_db
```python
항상 generate_answer로 (관련성 체크 불필요)
```

### route_after_vector
```python
항상 check_relevance로 (Self-RAG 평가)
```

### route_after_relevance ⭐ 중요
```python
if is_relevant and relevance_score >= THRESHOLD:
    → rerank
elif retry_count < MAX_SEARCH_RETRIES:
    retry_count += 1
    search_query_version += 1
    → vector_search (재검색 루프!)
else:
    → generate_answer (재시도 초과, "자료 없음" 답변)
```

### route_after_hallucination ⭐ 중요
```python
if is_grounded and hallucination_score >= THRESHOLD:
    → save_memory
elif generation_retry_count < MAX_GENERATION_RETRIES:
    generation_retry_count += 1
    → generate_answer (재생성 루프!)
else:
    final_answer 앞에 "⚠️ [답변 품질 경고]" 추가
    → save_memory (경고와 함께 저장)
```

---

## 📊 State 정의

```python
class ChatbotState(TypedDict):
    # 기본
    messages: List[Message]  # 대화 이력
    user_question: str
    conversation_id: str
    
    # 분류
    query_type: str  # "db_query" | "vector_search" | "both" | "general"
    
    # DB
    db_result: str
    
    # Vector
    vector_docs: List[dict]  # [{content, metadata, similarity_score}]
    
    # Rerank
    reranked_docs: List[dict]  # [{content, metadata, score}]
    
    # 답변
    final_answer: str
    
    # Self-RAG
    relevance_score: float
    is_relevant: bool
    hallucination_score: float
    is_grounded: bool
    
    # 재시도
    retry_count: int  # 검색 재시도 횟수
    search_query_version: int  # 검색 쿼리 버전
    generation_retry_count: int  # 답변 재생성 횟수
```

---

## ⚙️ 설정값

```python
# 임계값
RELEVANCE_THRESHOLD = 0.6  # 관련성 최소 점수
GROUNDING_THRESHOLD = 0.7  # 근거 최소 점수

# 재시도 제한
MAX_SEARCH_RETRIES = 3  # 최대 검색 재시도
MAX_GENERATION_RETRIES = 2  # 최대 생성 재시도

# Vector 검색
VECTOR_SEARCH_K = 20  # 초기 검색 개수 (많이 가져오기)
RERANK_TOP_N = 5  # Rerank 후 최종 개수
```

---

## 🔄 그래프 구성 코드 골격

```python
from langgraph.graph import StateGraph, END

workflow = StateGraph(ChatbotState)

# 노드 추가
workflow.add_node("load_memory", load_memory_func)
workflow.add_node("classify", classify_func)
workflow.add_node("db_query", db_query_func)
workflow.add_node("vector_search", vector_search_func)
workflow.add_node("both_query", both_query_func)
workflow.add_node("check_relevance", check_relevance_func)
workflow.add_node("rerank", rerank_func)
workflow.add_node("generate_answer", generate_answer_func)
workflow.add_node("check_hallucination", check_hallucination_func)
workflow.add_node("save_memory", save_memory_func)

# 시작점
workflow.set_entry_point("load_memory")

# 고정 엣지
workflow.add_edge("load_memory", "classify")
workflow.add_edge("both_query", "check_relevance")
workflow.add_edge("rerank", "generate_answer")
workflow.add_edge("generate_answer", "check_hallucination")
workflow.add_edge("save_memory", END)

# 조건부 엣지
workflow.add_conditional_edges(
    "classify",
    route_after_classify,
    {
        "db_query": "db_query",
        "vector_search": "vector_search",
        "both_query": "both_query",
        "generate_answer": "generate_answer"
    }
)

workflow.add_conditional_edges(
    "db_query",
    route_after_db,
    {"generate_answer": "generate_answer"}
)

workflow.add_conditional_edges(
    "vector_search",
    route_after_vector,
    {"check_relevance": "check_relevance"}
)

workflow.add_conditional_edges(
    "check_relevance",
    route_after_relevance,
    {
        "rerank": "rerank",
        "vector_search": "vector_search",  # 재검색 루프
        "generate_answer": "generate_answer"
    }
)

workflow.add_conditional_edges(
    "check_hallucination",
    route_after_hallucination,
    {
        "save_memory": "save_memory",
        "generate_answer": "generate_answer"  # 재생성 루프
    }
)

# 컴파일
app = workflow.compile(checkpointer=MemorySaver())
```

---

## 📦 필요한 라이브러리

```bash
pip install langgraph
pip install langchain
pip install langchain-openai
pip install langchain-postgres  # pgvector
pip install langchain-cohere  # Cohere Rerank
pip install psycopg2-binary  # PostgreSQL
```

---

## 🎯 핵심 Self-RAG 포인트

1. **관련성 평가**: Vector 검색 후 실제로 관련 있는 문서인지 검증
   - 낮으면 쿼리 개선 후 재검색 (최대 3회)
   
2. **환각 검증**: 답변 생성 후 근거가 충분한지 검증
   - 부족하면 프롬프트 조정 후 재생성 (최대 2회)

3. **재시도 루프**: 
   - `check_relevance` ⇄ `vector_search`
   - `check_hallucination` ⇄ `generate_answer`

4. **실패 처리**:
   - 재검색 초과 → "관련 자료를 찾을 수 없습니다"
   - 재생성 초과 → "⚠️ [답변 품질 경고]" 표시

---

## 💡 구현 팁

1. **LLM 호출 최적화**
   - 관련성/환각 평가는 저렴한 모델 (gpt-4o-mini)
   - 답변 생성은 고성능 모델 (gpt-4o)

2. **로깅**
   - 재시도 발생 시 로그 출력 (디버깅용)
   - 관련성/환각 점수 추적

3. **비용 절감**
   - 관련성 평가는 상위 3개 문서만
   - 재시도 횟수 제한

4. **메모리 관리**
   - MemorySaver로 세션별 대화 유지
   - 또는 DB에 직접 저장

5. **에러 처리**
   - DB 연결 실패, Vector 검색 실패 등 예외 처리
   - Fallback 답변 준비
