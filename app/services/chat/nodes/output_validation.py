"""
Output Validation 노드 - 답변 품질 검증 강화
"""
from app.services.chat.state import ChatbotState
from app.core.logging import logger
import re


def validate_output(state: ChatbotState) -> ChatbotState:
    """
    답변 품질 검증 강화
    
    Args:
        state: 현재 상태
    
    Returns:
        업데이트된 상태 (검증 결과 포함)
    """
    try:
        final_answer = state.get("final_answer", "")
        user_question = state.get("user_question", "")
        
        if not final_answer:
            logger.warning("답변이 비어있어 검증 스킵")
            return state
        
        logger.info("답변 품질 검증 시작")
        
        validation_issues = []
        
        # 1. 길이 검증
        if len(final_answer.strip()) < 10:
            validation_issues.append("답변이 너무 짧습니다")
        elif len(final_answer.strip()) > 5000:
            validation_issues.append("답변이 너무 깁니다")
        
        # 2. 불완전한 답변 패턴 감지
        incomplete_patterns = [
            r'\.\.\.$',  # "..."로 끝남
            r'^\.\.\.',  # "..."로 시작
            r'죄송.*?답변.*?없',  # "죄송합니다. 답변을 찾을 수 없습니다" (자료 없음은 제외)
            r'정보.*?없.*?제공',
        ]
        
        for pattern in incomplete_patterns:
            if re.search(pattern, final_answer, re.IGNORECASE):
                # 단, "자료 없음"은 정상적인 답변이므로 제외
                if "자료 없음" not in final_answer and "찾을 수 없" not in final_answer:
                    validation_issues.append("불완전한 답변 패턴 감지")
                    break
        
        # 3. 에러 메시지 패턴 감지
        error_patterns = [
            r'error',
            r'오류',
            r'exception',
            r'실패',
            r'failed',
        ]
        
        for pattern in error_patterns:
            if re.search(pattern, final_answer, re.IGNORECASE):
                # 단, "오류 없음" 같은 긍정적 표현은 제외
                if "오류 없" not in final_answer and "error.*?없" not in final_answer.lower():
                    validation_issues.append("에러 메시지 패턴 감지")
                    break
        
        # 4. 중복 내용 검증
        sentences = final_answer.split('.')
        unique_sentences = set(s.strip().lower() for s in sentences if s.strip())
        if len(sentences) > len(unique_sentences) * 1.5:
            validation_issues.append("중복 내용이 많습니다")
        
        # 5. 질문과의 관련성 간단 검증 (키워드 기반)
        if user_question:
            question_keywords = set(re.findall(r'\w+', user_question.lower()))
            answer_keywords = set(re.findall(r'\w+', final_answer.lower()))
            
            # 공통 키워드가 너무 적으면 관련성 낮음
            common_keywords = question_keywords & answer_keywords
            if len(question_keywords) > 0 and len(common_keywords) / len(question_keywords) < 0.2:
                validation_issues.append("질문과 답변의 관련성이 낮습니다")
        
        # 검증 결과 저장
        if validation_issues:
            logger.warning(f"답변 검증 이슈 발견: {validation_issues}")
            state["validation_issues"] = validation_issues
        else:
            logger.info("답변 검증 통과")
            state["validation_issues"] = []
        
        return state
        
    except Exception as e:
        logger.error(f"답변 검증 실패: {str(e)}", exc_info=True)
        state["validation_issues"] = []
        return state
