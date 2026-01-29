"""
LangGraph 챗봇 설정값
"""
LLM_TYPE = "openai"  # "huggingface" 또는 "openai"
OPENAI_MODEL = "gpt-4o-mini"  # "gpt-4o-mini", "gpt-3.5-turbo" 등

# 임계값
RELEVANCE_THRESHOLD = 0.6  # 관련성 최소 점수
GROUNDING_THRESHOLD = 0.7  # 근거 최소 점수

# 재시도 제한
MAX_SEARCH_RETRIES = 3  # 최대 검색 재시도
MAX_GENERATION_RETRIES = 2  # 최대 생성 재시도

# Vector 검색
VECTOR_SEARCH_K = 20  # 초기 검색 개수 (많이 가져오기)
RERANK_TOP_N = 5  # Rerank 후 최종 개수

# 메모리 관리
MAX_MESSAGES = 10  # 최대 대화 이력 개수
MAX_TOKENS = 2000  # 최대 토큰 수 (대화 이력 제한용)
