"""
E5 임베딩 모델 래퍼 (vector_db.py에서 분리)
"""
from typing import Optional
from transformers import AutoTokenizer, AutoModel
from torch import Tensor
import torch
import torch.nn.functional as F
from app.core.logging import logger


class EmbeddingModel:
    """intfloat/multilingual-e5-large-instruct 임베딩 모델"""

    _tokenizer: Optional[AutoTokenizer] = None
    _model: Optional[AutoModel] = None

    def __init__(self, model_name: str = 'intfloat/multilingual-e5-large-instruct'):
        self.model_name = model_name

    @staticmethod
    def _average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """평균 풀링으로 임베딩 생성"""
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    @staticmethod
    def _get_detailed_instruct(task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery: {query}'

    def _load_model(self):
        """임베딩 모델 로드 (한 번만 로드)"""
        if EmbeddingModel._tokenizer is None or EmbeddingModel._model is None:
            import time
            model_start_time = time.time()
            logger.info(f"임베딩 모델 로드 시작: {self.model_name}")
            try:
                EmbeddingModel._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                EmbeddingModel._model = AutoModel.from_pretrained(self.model_name)
                EmbeddingModel._model.eval()
                model_load_time = time.time() - model_start_time
                logger.info(f"임베딩 모델 로드 완료 (소요 시간: {model_load_time:.2f}초)")
            except Exception as e:
                logger.error(f"임베딩 모델 로드 실패: {str(e)}", exc_info=True)
                raise
        return EmbeddingModel._tokenizer, EmbeddingModel._model

    def generate_query_embedding(self, question: str) -> Tensor:
        """질문을 임베딩 벡터로 변환"""
        tokenizer, model = self._load_model()
        task = '법률 질문에 대해 관련된 법령 조항을 검색합니다'
        formatted_text = self._get_detailed_instruct(task, question)

        batch_dict = tokenizer(
            [formatted_text],
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors='pt',
        )

        with torch.no_grad():
            outputs = model(**batch_dict)
            embedding = self._average_pool(
                outputs.last_hidden_state,
                batch_dict['attention_mask'],
            )

        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding[0]

    @staticmethod
    def vector_to_str(vector: Tensor) -> str:
        """벡터를 PostgreSQL pgvector 형식 문자열로 변환"""
        return '[' + ','.join(map(str, vector.tolist())) + ']'
