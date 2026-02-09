"""
LangGraph 챗봇 State 정의
"""
from typing import TypedDict, List, Dict, Optional, Annotated
from operator import add
from app.core.logging import logger


def reduce_messages(left: List[Dict], right: List[Dict]) -> List[Dict]:
    """
    messages 리스트 병합 reducer
    LangGraph에서 여러 노드가 messages를 업데이트할 때 사용
    
    옵션 A: load_memory에서 설정한 제한된 메시지 리스트를 우선시
    - right가 제한된 길이(MAX_MESSAGES 이하)이면 우선시 (load_memory에서 설정)
    - left가 비정상적으로 길면(누적 버그) right를 우선시
    """
    from app.services.chat.config import MAX_MESSAGES
    
    if not left:
        return right if right else []
    if not right:
        return left
    
    # 옵션 A: load_memory에서 설정한 제한된 메시지 우선시
    # right가 MAX_MESSAGES 이하이면 load_memory에서 제한한 것으로 간주하여 우선시
    if len(right) <= MAX_MESSAGES:
        return right
    
    # left가 비정상적으로 길면(누적 버그) right를 우선시
    if len(left) > MAX_MESSAGES * 10:
        logger.warning(f"⚠️ reduce_messages: left가 비정상적으로 길어서 right 우선시: {len(left)}개 -> {len(right)}개")
        return right if right else []
    
    # 일반적인 경우: 두 리스트를 합치되, 중복 제거는 하지 않음 (순서 유지)
    # 단, 결과가 MAX_MESSAGES를 초과하지 않도록 제한
    merged = left + right
    if len(merged) > MAX_MESSAGES * 2:
        # 비정상적으로 길어지면 최근 것만 유지
        logger.warning(f"⚠️ reduce_messages: 병합 결과가 너무 길어서 제한: {len(merged)}개 -> {MAX_MESSAGES * 2}개")
        return merged[-(MAX_MESSAGES * 2):]
    
    return merged


def reduce_conversation_summary(left: Optional[str], right: Optional[str]) -> Optional[str]:
    """
    conversation_summary 병합 reducer
    LangGraph에서 여러 노드가 conversation_summary를 업데이트할 때 사용
    - 둘 다 None이면 None 반환
    - 하나만 있으면 그 값 반환
    - 둘 다 있으면 오른쪽(최신) 값 우선 (덮어쓰기)
    """
    if not left and not right:
        return None
    if not left:
        return right
    if not right:
        return left
    # 둘 다 있으면 오른쪽(최신) 값 우선
    return right


def reduce_user_question(left: str, right: str) -> str:
    """
    user_question 병합 reducer
    LangGraph에서 initial_state와 체크포인터 state가 병합될 때 사용
    - 둘 다 있으면 오른쪽(최신) 값 우선
    - 하나만 있으면 그 값 반환
    """
    if not left:
        return right if right else ""
    if not right:
        return left
    # 둘 다 있으면 오른쪽(최신) 값 우선
    return right


def reduce_conversation_id(left: str, right: str) -> str:
    """
    conversation_id 병합 reducer
    LangGraph에서 initial_state와 체크포인터 state가 병합될 때 사용
    - 둘 다 있으면 오른쪽(최신) 값 우선
    - 하나만 있으면 그 값 반환
    """
    if not left:
        return right if right else "default"
    if not right:
        return left
    # 둘 다 있으면 오른쪽(최신) 값 우선
    return right


# 범용 reducer 함수들
def reduce_str(left: str, right: str) -> str:
    """문자열 필드 reducer: 최신 값 우선"""
    return right if right else (left if left else "")


def reduce_optional_str(left: Optional[str], right: Optional[str]) -> Optional[str]:
    """Optional[str] 필드 reducer: 최신 값 우선"""
    return right if right is not None else left


def reduce_optional_float(left: Optional[float], right: Optional[float]) -> Optional[float]:
    """Optional[float] 필드 reducer: 최신 값 우선"""
    return right if right is not None else left


def reduce_optional_bool(left: Optional[bool], right: Optional[bool]) -> Optional[bool]:
    """Optional[bool] 필드 reducer: 최신 값 우선"""
    return right if right is not None else left


def reduce_int(left: int, right: int) -> int:
    """int 필드 reducer: 최신 값 우선 (단, 카운터는 증가하는 값만 허용)"""
    # 카운터의 경우: 더 큰 값을 반환 (감소 방지)
    return max(left, right)


def reduce_counter_int(left: int, right: int) -> int:
    """
    카운터용 int 필드 reducer: 초기화(0) 우선, 그 외에는 증가하는 값만 허용
    - 0이 오면 항상 0 반환 (초기화 의미)
    - 그 외에는 더 큰 값 반환 (감소 방지)
    """
    # 초기화(0)가 오면 항상 0 반환
    if right == 0:
        return 0
    if left == 0:
        return right
    # 둘 다 0이 아니면 더 큰 값 반환
    return max(left, right)


def reduce_list(left: List, right: List) -> List:
    """리스트 필드 reducer: 병합 (일반적인 경우)"""
    if not left:
        return right if right else []
    if not right:
        return left
    return left + right


def reduce_replace_list(left: List, right: List) -> List:
    """
    리스트 필드 reducer: 완전 교체 (문서 리스트용)
    문서 검색 결과는 항상 최신 값으로 완전히 교체되어야 함
    
    주의: 빈 리스트 []도 유효한 교체값임 (초기화 용도)
    - right가 None이 아니면 항상 right로 교체 ([]도 포함)
    - right가 None일 때만 left 유지
    """
    if right is not None:
        return right
    return left if left else []


def reduce_optional_list(left: Optional[List], right: Optional[List]) -> Optional[List]:
    """Optional[List] 필드 reducer: 병합"""
    if left is None and right is None:
        return None
    if left is None:
        return right if right else []
    if right is None:
        return left if left else []
    return left + right


def reduce_dict(left: Dict, right: Dict) -> Dict:
    """Dict 필드 reducer: 병합 (오른쪽 우선)"""
    if not left:
        return right if right else {}
    if not right:
        return left
    result = left.copy()
    result.update(right)
    return result


def reduce_optional_dict(left: Optional[Dict], right: Optional[Dict]) -> Optional[Dict]:
    """Optional[Dict] 필드 reducer: 병합 (오른쪽 우선)"""
    if left is None and right is None:
        return None
    if left is None:
        return right if right else {}
    if right is None:
        return left if left else {}
    result = left.copy()
    result.update(right)
    return result


class ChatbotState(TypedDict):
    """LangGraph 챗봇 State - 모든 필드에 reducer 추가하여 State 충돌 방지"""
    # 기본
    messages: Annotated[List[Dict], reduce_messages]  # 대화 이력 [{"role": "user"|"assistant", "content": str}]
    conversation_summary: Annotated[Optional[str], reduce_conversation_summary]  # 오래된 대화 요약(누적)
    user_question: Annotated[str, reduce_user_question]  # 현재 사용자 질문
    conversation_id: Annotated[str, reduce_conversation_id]  # 대화 세션 ID
    
    # 분류
    query_type: Annotated[str, reduce_str]  # "db_query" | "vector_search" | "both" | "general"
    
    # DB
    db_result: Annotated[Optional[str], reduce_optional_str]  # DB 조회 결과
    
    # Vector
    vector_docs: Annotated[List[Dict], reduce_replace_list]  # [{content, metadata, similarity_score}]
    
    # Rerank
    reranked_docs: Annotated[List[Dict], reduce_replace_list]  # [{content, metadata, score}]
    
    # 답변
    final_answer: Annotated[str, reduce_str]  # 최종 생성된 답변
    
    # Self-RAG 평가
    relevance_score: Annotated[Optional[float], reduce_optional_float]  # 관련성 점수 (0.0 ~ 1.0)
    is_relevant: Annotated[Optional[bool], reduce_optional_bool]  # 관련성 여부
    hallucination_score: Annotated[Optional[float], reduce_optional_float]  # 근거 점수 (0.0 ~ 1.0)
    is_grounded: Annotated[Optional[bool], reduce_optional_bool]  # 근거 충분 여부
    
    # 고도화: Multi-Aspect Evaluation
    accuracy_score: Annotated[Optional[float], reduce_optional_float]  # 정확성 점수 (0.0 ~ 1.0)
    completeness_score: Annotated[Optional[float], reduce_optional_float]  # 완전성 점수 (0.0 ~ 1.0)
    confidence_score: Annotated[Optional[float], reduce_optional_float]  # 신뢰도 점수 (0.0 ~ 1.0)
    
    # 고도화: Structured Memory
    entities: Annotated[Optional[Dict[str, List[str]]], reduce_optional_dict]  # 엔티티 메모리 {type: [values]}
    facts: Annotated[Optional[List[str]], reduce_optional_list]  # 사실 메모리
    
    # 고도화: Output Validation
    validation_issues: Annotated[Optional[List[str]], reduce_optional_list]  # 검증 이슈 리스트
    
    # 재시도 카운터 (초기화 우선)
    retry_count: Annotated[int, reduce_counter_int]  # 검색 재시도 횟수
    search_query_version: Annotated[int, reduce_counter_int]  # 검색 쿼리 버전 (재시도 시 개선)
    generation_retry_count: Annotated[int, reduce_counter_int]  # 답변 재생성 횟수
