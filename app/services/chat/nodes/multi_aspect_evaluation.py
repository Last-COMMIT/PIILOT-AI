"""
Multi-Aspect Evaluation 노드 - 고도화된 평가
정확성, 완전성, 신뢰도를 종합 평가
"""
from app.services.chat.state import ChatbotState
from app.services.chat.utils.llm_client import get_evaluation_llm
from app.core.logging import logger
from langchain_core.prompts import PromptTemplate


MULTI_ASPECT_PROMPT = """당신은 답변 품질 평가 전문가입니다. 다음 답변을 다각도로 평가하세요.

사용자 질문: {question}

생성된 답변:
{answer}

참고 자료:
{context}

DB 조회 결과:
{db_result}

평가 항목:
1. 정확성 (Accuracy): 답변이 사실에 부합하는가? 오류가 없는가?
2. 완전성 (Completeness): 질문에 대한 답변이 충분히 완전한가? 중요한 정보가 누락되지 않았는가?
3. 신뢰도 (Confidence): 답변의 확실성은 어느 정도인가? 추측이 아닌 근거가 있는가?

각 항목을 0.0 ~ 1.0 사이의 점수로 평가하세요.

응답 형식 (반드시 이 형식으로만):
정확성 점수: 0.85
완전성 점수: 0.90
신뢰도 점수: 0.88"""


def multi_aspect_evaluation(state: ChatbotState) -> ChatbotState:
    """
    Multi-Aspect Evaluation: 정확성, 완전성, 신뢰도 평가
    
    Args:
        state: 현재 상태
    
    Returns:
        업데이트된 상태 (accuracy_score, completeness_score, confidence_score 설정)
    """
    try:
        user_question = state.get("user_question", "")
        final_answer = state.get("final_answer", "")
        db_result = state.get("db_result")
        reranked_docs = state.get("reranked_docs", [])
        vector_docs = state.get("vector_docs", [])
        
        logger.info("Multi-Aspect Evaluation 시작")
        
        # 참고 자료 구성
        context_docs = reranked_docs if reranked_docs else vector_docs
        context_text = "\n\n---\n\n".join([
            f"문서 {i+1}:\n{doc.get('content', '')[:500]}"
            for i, doc in enumerate(context_docs[:5])
        ]) if context_docs else "참고 문서 없음"
        
        db_result_text = db_result if db_result else "DB 조회 결과 없음"
        
        # 프롬프트 생성
        prompt = PromptTemplate(
            template=MULTI_ASPECT_PROMPT,
            input_variables=["question", "answer", "context", "db_result"]
        )
        
        # LLM 호출
        llm = get_evaluation_llm()
        formatted_prompt = prompt.format(
            question=user_question,
            answer=final_answer,
            context=context_text,
            db_result=db_result_text
        )
        response = llm.invoke(formatted_prompt)
        
        # 응답 파싱
        response_text = response.content.strip()
        logger.debug(f"Multi-Aspect Evaluation LLM 응답: {response_text[:200]}...")
        
        # 점수 추출
        accuracy_score = 0.5
        completeness_score = 0.5
        confidence_score = 0.5
        
        import re
        for line in response_text.split("\n"):
            line_lower = line.lower()
            if "정확성" in line or "accuracy" in line_lower:
                match = re.search(r'(\d+\.?\d*)', line)
                if match:
                    accuracy_score = max(0.0, min(1.0, float(match.group(1))))
            elif "완전성" in line or "completeness" in line_lower:
                match = re.search(r'(\d+\.?\d*)', line)
                if match:
                    completeness_score = max(0.0, min(1.0, float(match.group(1))))
            elif "신뢰도" in line or "confidence" in line_lower:
                match = re.search(r'(\d+\.?\d*)', line)
                if match:
                    confidence_score = max(0.0, min(1.0, float(match.group(1))))
        
        # 종합 신뢰도 계산 (가중 평균)
        overall_confidence = (
            accuracy_score * 0.4 +
            completeness_score * 0.3 +
            confidence_score * 0.3
        )
        
        state["accuracy_score"] = accuracy_score
        state["completeness_score"] = completeness_score
        state["confidence_score"] = overall_confidence
        
        logger.info(f"Multi-Aspect Evaluation 완료: accuracy={accuracy_score:.2f}, completeness={completeness_score:.2f}, confidence={overall_confidence:.2f}")
        return state
        
    except Exception as e:
        logger.error(f"Multi-Aspect Evaluation 실패: {str(e)}", exc_info=True)
        # 기본값 설정
        state["accuracy_score"] = 0.5
        state["completeness_score"] = 0.5
        state["confidence_score"] = 0.5
        return state
