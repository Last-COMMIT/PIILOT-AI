"""
임베딩 유틸 (E5 instruct)
- 문서/청크 임베딩 생성에 사용
- VectorDB.search()에서 쿼리 임베딩에도 재사용 가능
"""

import json
from typing import List, Dict, Any
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

# 프로젝트 경로 (로컬)
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
json_path = PROJECT_ROOT / "test_pdf" / "chunks_output.json"


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """평균 풀링으로 임베딩 생성"""
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def get_detailed_instruct(task_description: str, query: str) -> str:
    """임베딩용 instruction 포맷"""
    return f'Instruct: {task_description}\nQuery: {query}'

# JSON 파일
def load_chunks_from_json(json_path: str) -> List[Dict[str, Any]]:
    """JSON 파일에서 청크 데이터 로드"""
    with open(json_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    return chunks

def generate_embeddings_batch(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    task: str = '법률 질문에 대해 관련된 법령 조항을 검색합니다',
    batch_size: int = 32
) -> Tensor:
    """배치 단위로 임베딩 생성"""
    all_embeddings = []
    formatted_texts = [get_detailed_instruct(task, text) for text in texts]
    
    for i in range(0, len(formatted_texts), batch_size):
        batch_dict = tokenizer(
            formatted_texts[i:i + batch_size],
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            outputs = model(**batch_dict)
            embeddings = average_pool(
                outputs.last_hidden_state,
                batch_dict['attention_mask']
            )
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings)

    
    return torch.cat(all_embeddings, dim=0)

