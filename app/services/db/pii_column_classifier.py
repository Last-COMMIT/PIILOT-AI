"""
PII 컬럼 분류 서비스 (RAG 기반)
- 컬럼명을 받아 VectorDB 검색 후 LLM으로 PII 유형 판정
"""
import json
import re
import time
from typing import List, Dict, Optional
import torch

from langchain_huggingface import ChatHuggingFace

from app.services.db.pii_vector_db import PIIVectorDB
from app.core.logging import logger


class PIIColumnClassifier:
    """PII 컬럼 분류기 - RAG 기반"""

    # 허용된 PII 유형 목록
    ALLOWED_PII_TYPES = {"NM", "RRN", "ADD", "IP", "PH", "ACN", "PP", "EM"}

    PROMPT_TEMPLATE = """당신은 데이터베이스 컬럼명을 보고 개인정보(PII) 여부를 판단하는 전문가입니다.

## 허용된 개인정보 유형 (이 목록에 있는 것만 사용)
- NM: 이름
- RRN: 주민등록번호
- ADD: 주소
- IP: IP주소
- PH: 전화번호
- ACN: 계좌번호
- PP: 여권번호
- EM: 이메일

## 기업 컬럼 작명 규칙 (참고용)
{context}

## 분류 규칙
1. 입력된 컬럼명을 기업 작명 규칙과 비교하여 개인정보 여부 판단
2. piiType은 반드시 위 허용된 목록(NM, RRN, ADD, IP, PH, ACN, PP, EM) 중에서만 선택
3. columnName은 입력값을 그대로 반환 (절대 변경하지 말 것)
4. 개인정보가 아닌 컬럼(id, seq, created_at, amount 등)은 결과에서 제외
5. 컬럼명이 개인정보인지 확실하지 않으면 제외

## 예시

입력:
- users.user_nm
- users.email   
- users.phone
- users.reg_dt

출력:
```json
[
  {{"tableName": "users", "columnName": "user_nm", "piiType": "NM"}},
  {{"tableName": "users", "columnName": "email", "piiType": "EM"}},
  {{"tableName": "users", "columnName": "phone", "piiType": "PH"}}
]
```

## 분류할 컬럼
{columns}

## 출력
JSON 배열만 출력하세요. 설명 없이 JSON만 반환하세요.
```json
"""

    def __init__(self):
        init_start_time = time.time()
        logger.info("PIIColumnClassifier 초기화 시작")

        # VectorDB 초기화
        logger.info("PIIVectorDB 초기화 중...")
        self.vector_db = PIIVectorDB()
        logger.info("✓ PIIVectorDB 초기화 완료")

        # LLM 초기화 (Midm-2.0-Mini-Instruct)
        model_name = "K-intelligence/Midm-2.0-Mini-Instruct"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        llm_start_time = time.time()

        try:
            self.llm = ChatHuggingFace.from_model_id(
                model_id=model_name,
                task="text-generation",
                device_map=device,
                model_kwargs={
                    "torch_dtype": torch.bfloat16,
                    "trust_remote_code": True,
                },
                tokenizer_kwargs={
                    "trust_remote_code": True,
                },
                temperature=0.001,
                max_new_tokens=1024,
            )
            llm_load_time = time.time() - llm_start_time
            logger.info(f"✓ LLM 모델 로딩 완료 (소요 시간: {llm_load_time:.2f}초)")
        except Exception as e:
            logger.error(f"LLM 모델 로딩 실패: {str(e)}", exc_info=True)
            raise

        total_init_time = time.time() - init_start_time
        logger.info(f"✓ PIIColumnClassifier 초기화 완료 (총 소요 시간: {total_init_time:.2f}초)")

    def _build_context(self, search_results: Dict[str, List[Dict]]) -> str:
        """
        VectorDB 검색 결과에서 중복 제거 후 context 문자열 생성

        Args:
            search_results: {컬럼명: [{abbr, chunk_text, similarity}, ...]}

        Returns:
            중복 제거된 chunk_text 목록 문자열
        """
        # 중복 제거를 위해 abbr 기준으로 unique한 chunk_text 수집
        seen_abbrs = set()
        unique_chunks = []

        for column_name, results in search_results.items():
            for result in results:
                abbr = result["abbr"]
                if abbr not in seen_abbrs:
                    seen_abbrs.add(abbr)
                    unique_chunks.append(result["chunk_text"])

        # 줄바꿈으로 구분된 문자열 생성
        context_lines = [f"- {chunk}" for chunk in unique_chunks]
        return "\n".join(context_lines)

    def _build_columns_input(self, tables: List[Dict]) -> str:
        """
        테이블/컬럼 정보를 LLM 입력 형식으로 변환

        Args:
            tables: [{"tableName": "users", "columns": ["id", "name", ...]}]

        Returns:
            "- users.id\n- users.name\n..." 형식 문자열
        """
        lines = []
        for table in tables:
            table_name = table["tableName"]
            for column in table["columns"]:
                lines.append(f"- {table_name}.{column}")
        return "\n".join(lines)

    def _parse_llm_response(self, response: str) -> List[Dict]:
        """
        LLM 응답에서 JSON 배열 파싱

        Args:
            response: LLM 응답 문자열

        Returns:
            [{"tableName": ..., "columnName": ..., "piiType": ...}, ...]
        """
        try:
            # JSON 블록 추출 (```json ... ``` 또는 [ ... ])
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
            if json_match:
                json_str = json_match.group(1)
            else:
                # ```json 없이 바로 JSON 배열인 경우
                json_match = re.search(r'\[[\s\S]*\]', response)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    logger.warning(f"JSON 파싱 실패: {response[:200]}")
                    return []

            result = json.loads(json_str)

            # 결과 검증
            if not isinstance(result, list):
                logger.warning(f"JSON이 배열이 아님: {type(result)}")
                return []

            # 필수 필드 검증
            validated_results = []
            for item in result:
                if all(key in item for key in ["tableName", "columnName", "piiType"]):
                    validated_results.append({
                        "tableName": item["tableName"],
                        "columnName": item["columnName"],
                        "piiType": item["piiType"]
                    })

            return validated_results

        except json.JSONDecodeError as e:
            logger.error(f"JSON 파싱 오류: {e}, 응답: {response[:200]}")
            return []
        except Exception as e:
            logger.error(f"응답 파싱 오류: {e}", exc_info=True)
            return []

    def classify(self, tables: List[Dict], top_k: int = 3) -> List[Dict]:
        """
        PII 컬럼 분류 메인 메서드

        Args:
            tables: [{"tableName": "users", "columns": ["id", "user_brdt", ...]}]
            top_k: 각 컬럼당 검색할 유사 표준단어 수 (기본값: 3)

        Returns:
            [{"tableName": "users", "columnName": "user_brdt", "piiType": "BRDT"}, ...]
        """
        try:
            total_start_time = time.time()
            logger.info(f"[PII 컬럼 분류 시작] {len(tables)}개 테이블, top_k={top_k}")

            # 1. 모든 컬럼명 추출
            all_columns = []
            for table in tables:
                all_columns.extend(table["columns"])

            if not all_columns:
                logger.warning("분류할 컬럼이 없습니다.")
                return []

            logger.info(f"총 {len(all_columns)}개 컬럼 분류 예정")

            # 2. VectorDB 배치 검색
            logger.info("VectorDB 검색 중...")
            search_start_time = time.time()
            search_results = self.vector_db.batch_search(all_columns, top_k=top_k)
            search_time = time.time() - search_start_time
            logger.info(f"✓ VectorDB 검색 완료 (소요 시간: {search_time:.2f}초)")

            # 3. Context 구성 (중복 제거된 chunk_text)
            logger.info("Context 구성 중...")
            context = self._build_context(search_results)
            logger.info(f"✓ Context 구성 완료 (유니크 표준단어 수: {context.count(chr(10)) + 1}개)")

            # 4. 컬럼 입력 구성
            columns_input = self._build_columns_input(tables)

            # 5. LLM 호출
            logger.info("LLM 호출 중... (시간이 걸릴 수 있습니다)")
            llm_start_time = time.time()

            prompt = self.PROMPT_TEMPLATE.format(context=context, columns=columns_input)
            response = self.llm.invoke(prompt)

            # 응답에서 텍스트 추출
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)

            llm_time = time.time() - llm_start_time
            logger.info(f"✓ LLM 호출 완료 (소요 시간: {llm_time:.2f}초)")

            # 6. 응답 파싱
            logger.info("LLM 응답 파싱 중...")
            pii_columns = self._parse_llm_response(response_text)
            logger.info(f"✓ 파싱 완료: {len(pii_columns)}개 PII 컬럼 탐지")

            total_time = time.time() - total_start_time
            logger.info(f"[PII 컬럼 분류 완료] 총 소요 시간: {total_time:.2f}초")

            return pii_columns

        except Exception as e:
            logger.error(f"PII 컬럼 분류 오류: {str(e)}", exc_info=True)
            return []
