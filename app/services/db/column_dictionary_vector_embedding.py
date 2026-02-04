import re
import sys
from pathlib import Path
from app.core.logging import logger
from app.core.config import settings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from typing import List, Dict
from torch import Tensor
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

# 프로젝트 루트를 sys.path에 추가 (스크립트 직접 실행 시 모듈 import를 위해)
# config.py의 get_project_root()와 동일한 로직 사용
def _find_project_root() -> Path:
    """프로젝트 루트 경로 반환 (requirements.txt 기준) - config.py와 동일한 로직"""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / 'requirements.txt').exists():
            return current
        current = current.parent
    raise RuntimeError("프로젝트 루트(requirements.txt)를 찾을 수 없습니다.")

_project_root = _find_project_root()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# DB 연결 (기존 방식 사용)
from app.crud.db_connect import get_connection
from urllib.parse import urlparse


"""
PDF 표에서 메타데이터 자동 추출

메타데이터 구조:
{
  'abbr': 'TEL',
  'korean': '전화',
  'english': 'Telephone',
  'description': '전화번호 (유선)'
}
"""

# 텍스트 추출
def load_pdf_lines(file_path: Path) -> List[str]:
    """
    PDF에서 줄 단위로 직접 추출
    
    Args:
        file_path: PDF 파일 경로
        
    Returns:
        List[str]: 줄 단위 리스트
    """
    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")
    
    loader = PyMuPDFLoader(str(file_path))
    docs = loader.load()
    
    # 각 페이지의 줄들을 바로 리스트로 추가
    all_lines = []
    for doc in docs:
        lines = [line.strip() for line in doc.page_content.split('\n') if line.strip()]
        all_lines.extend(lines)
    
    return all_lines

class RowBasedMetadataExtractor:
    """행 기반 메타데이터 추출 클래스"""
    
    REQUIRED_HEADERS = ['영문약어', '한글명', '영문풀네임', '설명', 'PI여부']
    
    def __init__(self, lines: List[str]):
        """
        Args:
            lines: PDF에서 추출한 줄 단위 리스트
        """
        self.lines = lines
    
    def is_abbreviation(self, text: str) -> bool:
        """영문약어인지 판단 (대문자 1-10자)"""
        return bool(re.match(r'^[A-Z]{1,10}$', text))
    
    def find_header_blocks(self) -> List[int]:
        """
        헤더 블록 찾기 (연속된 줄들에 필수 헤더가 모두 포함)
        
        Returns:
            List[int]: 헤더 블록 시작 인덱스
        """
        header_starts = []
        num_headers = len(self.REQUIRED_HEADERS)
        
        for i in range(len(self.lines) - num_headers + 1):
            # 연속된 num_headers개 줄 가져오기
            consecutive_lines = [self.lines[i+j].strip() for j in range(num_headers)]
            
            # 필수 헤더가 모두 포함되어 있는지 확인
            headers_found = []
            for header in self.REQUIRED_HEADERS:
                for line in consecutive_lines:
                    if header in line:
                        headers_found.append(header)
                        break
            
            # 모든 필수 헤더가 발견되면
            if len(headers_found) == len(self.REQUIRED_HEADERS):
                header_starts.append(i)
        return header_starts
    
    def extract_table_rows(self, header_start: int, next_header_start: int = None) -> List[Dict]:
        """
        표를 행 단위로 추출
        
        세로로 나열된 데이터를 헤더 개수만큼 묶어서 하나의 행으로 변환
        
        Args:
            header_start: 헤더 블록 시작 인덱스
            next_header_start: 다음 헤더 시작 (없으면 None)
            
        Returns:
            List[Dict]: 메타데이터 리스트
        """
        metadata_list = []
        
        # 헤더 개수 (범용성)
        num_columns = len(self.REQUIRED_HEADERS)
        
        # 데이터 시작: 헤더 다음부터
        data_start = header_start + num_columns
        
        # 데이터 끝
        if next_header_start:
            data_end = next_header_start
        else:
            data_end = len(self.lines)
        
        # 데이터 영역 추출
        data_lines = []
        for i in range(data_start, data_end):
            line = self.lines[i]
            
            # 새로운 섹션 시작하면 중단
            if re.match(r'^\d+\.\d+', line):
                break
            
            data_lines.append(line)
        
        # num_columns개씩 묶어서 하나의 행으로 변환
        i = 0
        while i + num_columns - 1 < len(data_lines):
            # 영문약어로 시작하는지 확인
            if self.is_abbreviation(data_lines[i]):
                abbr = data_lines[i]
                
                # 다음 필드들 수집
                korean = data_lines[i + 1]
                english = data_lines[i + 2]
                description = data_lines[i + 3]
                # PI여부는 사용하지 않음 (data_lines[i + 4])
                
                # 다음 줄이 영문약어가 아니어야 유효
                if not self.is_abbreviation(korean):
                    metadata = {
                        'abbr': abbr,
                        'korean': korean,
                        'english': english,
                        'description': description,
                    }
                    
                    metadata_list.append(metadata)
                    
                    # num_columns개 건너뛰기
                    i += num_columns
                    continue
            
            i += 1
        return metadata_list
    
    def extract_all(self) -> List[Dict]:
        """전체 메타데이터 추출"""
        
        # 1. 헤더 블록 찾기
        header_starts = self.find_header_blocks()
        
        if not header_starts:
            logger.warning("헤더 블록을 찾을 수 없습니다.")
            return []
        
        # 2. 각 표에서 데이터 추출
        all_metadata = []
        
        for idx, header_start in enumerate(header_starts, 1):
            next_header = header_starts[idx] if idx < len(header_starts) else None
            table_metadata = self.extract_table_rows(header_start, next_header)
            
            all_metadata.extend(table_metadata)
        
        return all_metadata


# 청크(Document) 생성
def create_documents(metadata_list: List[Dict]) -> List[Document]:
    """
    메타데이터를 Document로 변환 (풍부한 청크 + 메타데이터)
    메타데이터와 청크에 동일한 정보 저장
    
    Args:
        metadata_list: 메타데이터 리스트
        
    Returns:
        List[Document]: 벡터DB용 Document 리스트
    """
    documents = []
    for metadata in metadata_list:
        # 풍부한 청크 생성 (임베딩용)
        page_content = (
            f"{metadata['korean']}({metadata['english']})은 "
            f"{metadata['description']}을 의미한다. "
            f"데이터베이스 표준 약어는 {metadata['abbr']}이다."
        )
    
        # Document 생성 (동일한 정보를 메타데이터에도 저장)
        doc = Document(
            page_content=page_content,
            metadata={
                'abbr': metadata['abbr'],
                'korean': metadata['korean'],
                'english': metadata['english'],
                'description': metadata['description'],
            }
        )

        documents.append(doc)
    return documents


# 임베딩(batch)
"""
intfloat/multilingual-e5-*-instruct 계열 기준:
- Instruct 포맷 적용
- Attention mask 기반 mean pooling
- L2 normalize (cosine 검색 전제)
- 배치 임베딩
"""

def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """평균 풀링으로 임베딩 생성"""
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def get_detailed_instruct(task_description: str, query: str) -> str:
    """임베딩용 instruction 포맷"""
    return f'Instruct: {task_description}\nQuery: {query}'

def generate_embeddings_batch(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    task: str = '데이터베이스 컬럼명을 분석하여 개인정보 포함 여부와 유형을 식별합니다',
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


# PGVector 데이터 삽입
def insert_to_database(documents: List[Document], embeddings: Tensor, batch_size: int = 100):
    """
    데이터베이스에 표준단어 문서와 임베딩 삽입
    
    Args:
        documents: LangChain Document 리스트
        embeddings: 임베딩 텐서 [num_docs, embedding_dim]
        batch_size: 배치 크기
    """
    # config.py의 DATABASE_URL 파싱하여 연결
    parsed_url = urlparse(settings.DATABASE_URL.replace('postgresql+psycopg://', 'postgresql://'))
    conn = get_connection(
        user=parsed_url.username,
        password=parsed_url.password,
        host=parsed_url.hostname,
        port=parsed_url.port or 5432,
        database=parsed_url.path.lstrip('/')
    )
    
    insert_query = """
        INSERT INTO pii_standard_words (
            chunk_text,
            embedding,
            abbr,
            korean,
            english,
            description
        ) VALUES (
            %s, %s, %s, %s, %s, %s
        )
    """

    try:
        with conn.cursor() as cursor:
            inserted_count = 0
            
            for batch_start in range(0, len(documents), batch_size):
                batch_data = []
                
                for doc, embedding in zip(
                    documents[batch_start:batch_start + batch_size],
                    embeddings[batch_start:batch_start + batch_size]
                ):
                    batch_data.append((
                        doc.page_content,           # chunk_text
                        embedding.tolist(),         # embedding
                        doc.metadata.get('abbr'),   # abbr
                        doc.metadata.get('korean'), # korean
                        doc.metadata.get('english'), # english
                        doc.metadata.get('description') # description
                    ))
                cursor.executemany(insert_query, batch_data)
                conn.commit()
                inserted_count += len(batch_data)
    finally:
        conn.close()

def column_dictionary_to_vector(file_path: str):
    """메인 실행 함수 (로컬 파일 또는 URL 지원)"""
    import os
    from app.utils.temp_file import download_file_from_url, is_url

    temp_file = None
    local_path = file_path

    # URL인 경우 다운로드
    if is_url(file_path):
        logger.info(f"URL에서 PDF 다운로드: {file_path}")
        temp_file = download_file_from_url(file_path)
        local_path = temp_file
    else:
        local_path = Path(file_path)

    try:
        # 1. PDF 줄 단위 추출
        lines = load_pdf_lines(local_path)
        logger.info(f"텍스트 추출 완료 (총 {len(lines):,} 줄)")

        # 2. 메타데이터 추출
        extractor = RowBasedMetadataExtractor(lines)
        metadata_list = extractor.extract_all()

        if not metadata_list:
            logger.warning("추출된 메타데이터가 없습니다.")
            return []
        logger.info(f"{len(metadata_list)}개 metadata 생성 완료")

        # 3. Document 생성
        documents = create_documents(metadata_list)
        logger.info(f"{len(documents)}개 Document 생성 완료")

        # 4. 임베딩 모델 로드 (ModelManager의 캐시 디렉토리 사용)
        from app.core.model_manager import ModelManager
        ModelManager.setup_cache_dir()  # 캐시 디렉토리 설정
        cache_dir = ModelManager.get_cache_dir()

        model_name = ModelManager.HUGGINGFACE_MODELS["embedding"]["name"]
        logger.info(f"임베딩 모델 로드 시작: {model_name} (캐시 디렉토리: {cache_dir})")
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        model.eval()
        logger.info("임베딩 모델 로드 완료")

        # 5. 배치 임베딩
        texts = [doc.page_content for doc in documents]
        embeddings = generate_embeddings_batch(
            texts=texts,
            tokenizer=tokenizer,
            model=model,
            task='데이터베이스 컬럼명을 분석하여 개인정보 포함 여부와 유형을 식별합니다',
            batch_size=32
        )
        logger.info(f"배치 임베딩 완료! (shape: {embeddings.shape})")

        # 6. PGVector 저장
        insert_to_database(
            documents=documents,
            embeddings=embeddings
        )
        logger.info("VectorDB 저장 완료!")

        return documents

    finally:
        # 임시 파일 정리
        if temp_file and temp_file.exists():
            try:
                os.unlink(temp_file)
                logger.debug(f"임시 파일 삭제: {temp_file}")
            except Exception as e:
                logger.warning(f"임시 파일 삭제 실패: {temp_file} - {e}")
