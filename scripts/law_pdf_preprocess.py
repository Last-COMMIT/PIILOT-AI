"""
법령/내규 파일 전처리
"""
import fitz
import re
import os
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

# 프로젝트 경로 (로컬)
from pathlib import Path

def extract_pdf_blocks(file_path):
    doc = fitz.open(file_path)
    blocks = []

    for page_index, page in enumerate(doc):
        page_no = page_index + 1
        page_blocks = page.get_text("blocks")

        for block in page_blocks:
            x0, y0, x1, y1, text, block_no, block_type = block

            # 텍스트 블록만 사용 (이미지 제외)
            if block_type == 0:
                cleaned = text.strip()
                if cleaned:
                    blocks.append({
                        "page": page_no,
                        "text": cleaned
                    })

    return blocks


# 최소 정규화/보호 규칙
def _normalize_for_repeat(text: str) -> str:
    """
    반복 텍스트 비교 안정화를 위한 '최소' 정규화.
    (의미 기반 처리 금지: 공백/대소문자/페이지번호 정도만)
    """
    t = (text or "").strip()
    t = re.sub(r"\s+", " ", t)
    # 흔한 페이지 번호/장식 패턴 제거(짧은 문구에만 효과적으로 작동)
    t = re.sub(r"^\s*[-–—]?\s*\d+\s*[-–—]?\s*$", "", t)  # - 3 - / 3
    t = re.sub(r"^(page\s*)?\d+\s*(of\s*\d+)?$", "", t, flags=re.I)  # Page 3 / Page 3 of 10
    return t.lower().strip()

def _is_structural_token(text: str) -> bool:
    """
    법령/내규 본문 구조를 대표하는 토큰은 반복 제거 단계에서 보호.
    """
    t = (text or "").strip()

    # 제6조 / 제 6 조 / 제6조의2
    if re.match(r"^제\s*\d+\s*조(\s*의\s*\d+)?\b", t):
        return True
    # 제1항
    if re.match(r"^제\s*\d+\s*항\b", t):
        return True
    # ①②③...
    if re.match(r"^[①②③④⑤⑥⑦⑧⑨⑩]\b", t):
        return True
    # 1. / (1) / 1)
    if re.match(r"^(\(?\d+\)?[.)])\s+\S+", t):
        return True
    # 부칙/별표/별지
    if re.match(r"^(부칙|별표|별지)\b", t):
        return True

    return False


# 추출: page + bbox 포함
def extract_pdf_blocks(file_path: str) -> List[Dict[str, Any]]:
    """
    PyMuPDF blocks 추출(텍스트 블록만)
    반환 원소:
      { "page": int(1-based), "text": str, "bbox": (x0,y0,x1,y1), "page_h": float }
    """
    doc = fitz.open(file_path)
    blocks: List[Dict[str, Any]] = []

    for page_index, page in enumerate(doc):
        page_no = page_index + 1
        h = float(page.rect.height)

        for block in page.get_text("blocks"):
            x0, y0, x1, y1, text, _, block_type = block
            if block_type != 0:
                continue

            cleaned = (text or "").strip()
            if not cleaned:
                continue

            blocks.append({
                "page": page_no,
                "text": cleaned,
                "bbox": (float(x0), float(y0), float(x1), float(y1)),
                "page_h": h,
            })

    return blocks


# 헤더/푸터 제거
def remove_header_footer_by_position(
    blocks: List[Dict[str, Any]],
    top_ratio: float = 0.12,      # 상단 12%를 헤더 영역으로 간주
    bottom_ratio: float = 0.88,   # 하단 12%를 푸터 영역으로 간주
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    상/하단 영역에 들어가는 블록을 제거.
    반환: (kept, removed)
    """
    kept, removed = [], []

    for b in blocks:
        _, y0, _, y1 = b["bbox"]
        h = b["page_h"]

        in_header = (y1 <= top_ratio * h)
        in_footer = (y0 >= bottom_ratio * h)

        if in_header or in_footer:
            removed.append(b)
        else:
            kept.append(b)

    return kept, removed

def remove_repeated_text(
    blocks: List[Dict[str, Any]],
    min_page_ratio: float = 0.35, 
    max_len: int = 100,            
) -> Tuple[List[Dict[str, Any]], Dict[str, int], set]:
    """
    페이지별 유니크 텍스트(정규화)를 집계해,
    여러 페이지에 반복 등장하는 짧은 문구를 제거.
    반환: (kept, page_presence_counts, repeated_set)
    """
    pages = sorted({b["page"] for b in blocks})
    total_pages = len(pages)
    if total_pages == 0:
        return blocks, {}, set()

    per_page_norms = defaultdict(set)

    for b in blocks:
        raw = b["text"]
        if _is_structural_token(raw):
            continue

        norm = _normalize_for_repeat(raw)
        if not norm:
            continue
        if len(norm) > max_len:
            continue

        per_page_norms[b["page"]].add(norm)

    presence = Counter()
    for p in pages:
        for norm in per_page_norms.get(p, set()):
            presence[norm] += 1

    threshold = max(2, int(total_pages * min_page_ratio))
    repeated_set = {t for t, c in presence.items() if c >= threshold}

    kept: List[Dict[str, Any]] = []
    for b in blocks:
        raw = b["text"]
        if _is_structural_token(raw):
            kept.append(b)
            continue

        norm = _normalize_for_repeat(raw)
        if norm in repeated_set:
            continue

        kept.append(b)

    return kept, dict(presence), repeated_set


# 최종: 추출 -> 1차(위치) -> 2차(반복) -> (page,text)
def extract_and_remove_noise(
    file_path: str,
    top_ratio: float = 0.08,
    bottom_ratio: float = 0.92,
    min_page_ratio: float = 0.35,
    max_repeat_len: int = 100,
) -> Dict[str, Any]:
    raw = extract_pdf_blocks(file_path)

    after_pos, removed_pos = remove_header_footer_by_position(
        raw, top_ratio=top_ratio, bottom_ratio=bottom_ratio
    )

    after_rep, presence, repeated_set = remove_repeated_text(
        after_pos, min_page_ratio=min_page_ratio, max_len=max_repeat_len
    )

    cleaned_blocks = [{"page": b["page"], "text": b["text"]} for b in after_rep]

    debug = {
        "total_pages": len({b["page"] for b in raw}),
        "raw_blocks": len(raw),
        "removed_by_position": len(removed_pos),
        "kept_after_position": len(after_pos),
        "repeated_text_count": len(repeated_set),
        "kept_after_repeat": len(after_rep),
        "repeated_threshold_pages": max(2, int(len({b["page"] for b in raw}) * min_page_ratio))
                                  if len({b["page"] for b in raw}) else 0,
    }

    return {"cleaned_blocks": cleaned_blocks, "debug": debug}


# block들을 같은 페이지 안에서 이어붙임 (개행 유지)
def combine_blocks_by_page(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pages = defaultdict(list)
    for b in blocks:
        pages[b["page"]].append(b["text"])

    out = []
    for page in sorted(pages.keys()):
        text = "\n".join(t for t in pages[page] if t and t.strip())
        out.append({"page": page, "text": text})
    return out

# 문서 상수 규칙
_STRUCT_START = re.compile(
    r"^(제\s*\d+\s*조(\s*의\s*\d+)?\b|제\s*\d+\s*항\b|[①②③④⑤⑥⑦⑧⑨⑩]\b|\(?\d+\)?[.)])"
)
_SENT_END = re.compile(r"(다\.|함\.|됨\.|임\.|니다\.|습니까\?|\.|!|\?|」|’|\)|]|\})\s*$")
_ALNUM_KO_END = re.compile(r"[A-Za-z0-9가-힣]$")
_ALNUM_KO_START = re.compile(r"^[A-Za-z0-9가-힣]")

def merge_soft_linebreaks_in_page(text: str) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines()]

    merged: List[str] = []
    buf = ""

    for line in lines:
        if not line:
            if buf:
                merged.append(buf)
                buf = ""
            merged.append("")  # 빈 줄 유지
            continue

        if not buf:
            buf = line
            continue

        # 구조 토큰 시작이면 병합 금지
        if _STRUCT_START.match(line):
            merged.append(buf)
            buf = line
            continue

        # 문장 종결이면 병합 금지
        if _SENT_END.search(buf):
            merged.append(buf)
            buf = line
            continue

        # 기본은 '공백 1칸' 넣어서 병합(붙어버림 방지)
        need_space = bool(_ALNUM_KO_END.search(buf) and _ALNUM_KO_START.search(line))
        buf = buf + (" " if need_space else "") + line

    if buf:
        merged.append(buf)

    # 연속 빈 줄 2개 이상 -> 1개
    out_lines: List[str] = []
    prev_empty = False
    for ln in merged:
        is_empty = (ln == "")
        if is_empty and prev_empty:
            continue
        out_lines.append(ln)
        prev_empty = is_empty

    return "\n".join(out_lines).strip()


def normalize_pages_linebreaks(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for p in pages:
        fixed = merge_soft_linebreaks_in_page(p["text"])
        if fixed:
            out.append({"page": p["page"], "text": fixed})
    return out

def extract_and_fix_pages(file_path: str) -> List[Dict[str, Any]]:
    """PDF 파일에서 fixed_pages 생성"""
    result = extract_and_remove_noise(file_path)
    cleaned_blocks = result["cleaned_blocks"]
    combined = combine_blocks_by_page(cleaned_blocks)
    fixed_pages = normalize_pages_linebreaks(combined)
    return fixed_pages

# 제n조 / 제n조의m
RE_ARTICLE = re.compile(r"^(제\s*\d+\s*조(\s*의\s*\d+)?)(.*)$")
# 제n항 (문서에 실제로 "제1항"처럼 쓰인 경우)
RE_PARAGRAPH_WORD = re.compile(r"^(제\s*\d+\s*항)(.*)$")
# ①②③ 같은 항 표기
RE_PARAGRAPH_CIRCLE = re.compile(r"^([①②③④⑤⑥⑦⑧⑨⑩])(.*)$")
# 호(號): 숫자로 시작하는 항목 (예: "1.", "2.", "1의2.")
RE_ITEM_NUMBER = re.compile(r"^(\d+(?:의\d+)?)\.\s*(.*)$")
# 목(目): 한글 자모로 시작하는 항목 (예: "가.", "나.", "다.")
RE_ITEM_HANGUL = re.compile(r"^([가-힣])\.\s*(.*)$")

def _clean_token(s: str) -> str:
    return re.sub(r"\s+", "", s or "").strip()

def _circle_to_paragraph(circle: str) -> str:
    mapping = {"①":"제1항","②":"제2항","③":"제3항","④":"제4항","⑤":"제5항",
               "⑥":"제6항","⑦":"제7항","⑧":"제8항","⑨":"제9항","⑩":"제10항"}
    return mapping.get(circle, circle)

def _number_to_item(num_str: str) -> str:
    """숫자를 호(號) 형식으로 변환 (예: "1" -> "제1호", "1의2" -> "제1의2호")"""
    return f"제{num_str}호"

def _hangul_to_item(hangul: str) -> str:
    """한글 자모를 목(目) 형식으로 변환 (예: "가" -> "가목")"""
    return f"{hangul}목"

def extract_metadata_from_pdf(file_path: str, fixed_pages: List[Dict[str, Any]]) -> Dict[str, Optional[str]]:
    """
    PDF 파일명과 내용에서 메타데이터 자동 추출
    
    Args:
        file_path: PDF 파일 경로
        fixed_pages: 페이지별 텍스트 리스트
    
    Returns:
        dict: document_title, law_name, effective_date를 포함한 메타데이터
    """
    metadata = {}


# 1. document_title: 파일명에서 추출
    filename = os.path.basename(file_path)
    document_title = filename.replace('.pdf', '').replace('.PDF', '').strip()
    metadata['document_title'] = document_title
    
    # 2. law_name: PDF 첫 페이지에서 추출 시도
    law_name = None
    if fixed_pages:
        first_page_text = fixed_pages[0]['text']
        # 법령명 패턴 찾기 (첫 15줄에서)
        lines = first_page_text.split('\n')[:15]
        for line in lines:
            line = line.strip()
            # "○○법", "○○규정", "○○규칙", "○○지침", "○○지시", "○○규정서" 등 패턴 매칭
            match = re.search(r'^([가-힣\s]+(?:법|규정|규칙|지침|지시|규정서|규정서))', line)
            if match:
                law_name = match.group(1).strip()
                # 너무 짧거나 헤더/푸터 같은 것 제외
                if len(law_name) >= 3 and len(law_name) <= 50:
                    break
    
    # law_name 추출 실패 시 document_title 사용
    metadata['law_name'] = law_name or document_title
    
    # 3. effective_date: PDF 첫 페이지에서 날짜 추출 시도
    effective_date = None
    if fixed_pages:
        first_page_text = fixed_pages[0]['text']
        # 날짜 패턴 찾기 (시행일, 개정일, 공포일 등)
        date_patterns = [
            r'시행.*?(\d{4})[.\s](\d{1,2})[.\s](\d{1,2})',
            r'개정.*?(\d{4})[.\s](\d{1,2})[.\s](\d{1,2})',
            r'공포.*?(\d{4})[.\s](\d{1,2})[.\s](\d{1,2})',
            r'(\d{4})[.\s](\d{1,2})[.\s](\d{1,2})',  # 일반 날짜 패턴
        ]
        
        dates_found = []
        for pattern in date_patterns:
            matches = re.finditer(pattern, first_page_text)
            for match in matches:
                year, month, day = match.groups()
                # 유효한 날짜 범위 체크
                if 1900 <= int(year) <= 2100 and 1 <= int(month) <= 12 and 1 <= int(day) <= 31:
                    date_str = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                    dates_found.append(date_str)
        
        # 가장 최근 날짜 선택 (시행일/개정일이 보통 여러 개 있음)
        if dates_found:
            effective_date = sorted(dates_found, reverse=True)[0]
    
    metadata['effective_date'] = effective_date
    
    return metadata


def build_chunks_with_metadata(
    fixed_pages: List[Dict[str, Any]],
    file_path: Optional[str] = None,
    document_title: Optional[str] = None,
    law_name: Optional[str] = None,
    effective_date: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    입력: 
        fixed_pages: [{"page": n, "text": "...(page 전체 텍스트)..."}]
        file_path: PDF 파일 경로 (메타데이터 자동 추출용, 선택적)
        document_title: 문서 제목 (None이면 자동 추출)
        law_name: 법령명/규정명 (None이면 자동 추출)
        effective_date: 시행일/개정일 (None이면 자동 추출 시도)
    출력: [{"chunk_text": "...", "metadata": {...}}, ...]
    metadata 스키마는 사용자 정의 형태에 맞춘다.
    """
    # 메타데이터 자동 추출
    if file_path:
        auto_metadata = extract_metadata_from_pdf(file_path, fixed_pages)
        document_title = document_title or auto_metadata['document_title']
        law_name = law_name or auto_metadata['law_name']
        effective_date = effective_date or auto_metadata['effective_date']
    else:
        # file_path 없으면 기본값 사용
        document_title = document_title or "문서"
        law_name = law_name or document_title
        effective_date = effective_date  # None 가능
    
    # ===== 청크 생성 =====
    chunks: List[Dict[str, Any]] = []

    current_article: Optional[str] = None
    current_paragraph: Optional[str] = None
    current_item: Optional[str] = None
    current_article: Optional[str] = None
    current_paragraph: Optional[str] = None
    current_item: Optional[str] = None

    # 현재 chunk 누적 버퍼
    buf_lines: List[str] = []
    buf_start_page: Optional[int] = None
    buf_article: Optional[str] = None
    buf_paragraph: Optional[str] = None
    buf_item: Optional[str] = None  # 현재 호 (단일 값)
    buf_subitem: Optional[str] = None  # 현재 목 (단일 값)


    def flush():
        nonlocal buf_lines, buf_start_page, buf_article, buf_paragraph, buf_item, buf_subitem
        text = "\n".join([ln for ln in buf_lines if ln.strip()]).strip()
        if not text:
            buf_lines, buf_start_page, buf_article, buf_paragraph, buf_item, buf_subitem = [], None, None, None, None, None
            return

        chunks.append({
            "chunk_text": text,
            "metadata": {
                "document_title": document_title,
                "law_name": law_name,
                "article": buf_article,
                "page": buf_start_page,
                "effective_date": effective_date,
            }
        })

        buf_lines, buf_start_page, buf_article, buf_paragraph, buf_item, buf_subitem = [], None, None, None, None, None

    for page_obj in fixed_pages:
        page_no = page_obj["page"]
        lines = [ln.strip() for ln in (page_obj["text"] or "").splitlines() if ln.strip()]

        for line in lines:
            # 1) 조 감지
            m_art = RE_ARTICLE.match(line)
            if m_art:
                # 새 조 시작 => 이전 버퍼 flush
                flush()

                art = _clean_token(m_art.group(1))  # "제 6 조" -> "제6조"
                rest = (m_art.group(3) or "").strip()

                current_article = art
                current_paragraph = None  # 조 바뀌면 항 컨텍스트 리셋
                current_item = None  # 조 바뀌면 호 컨텍스트 리셋
                current_subitem = None  # 조 바뀌면 목 컨텍스트 리셋
                current_item_subitems = []  # 목 리스트 초기화
                
                # 같은 줄에 동그라미 항이 있는지 확인 (예: "제3조 ... ① ...")
                m_c_inline = RE_PARAGRAPH_CIRCLE.search(rest)
                if m_c_inline:
                    # 동그라미 항이 있으면 항으로 처리
                    par = _circle_to_paragraph(m_c_inline.group(1))
                    current_paragraph = par
                    buf_paragraph = current_paragraph
                else:
                    buf_paragraph = None  # 조 제목 라인은 항 없음

                # 새 버퍼 시작
                buf_start_page = page_no
                buf_article = current_article
                buf_item = None
                buf_subitem = None
                buf_lines = [line] if not rest else [f"{art} {rest}".strip()]
                continue

            # 2) 항 감지(제n항) - 항은 조에 포함 (별도 청크 아님)
            m_p = RE_PARAGRAPH_WORD.match(line)
            if m_p:
                # flush() 제거 - 항은 같은 조에 포함

                par = _clean_token(m_p.group(1))  # "제 1 항" -> "제1항"
                rest = (m_p.group(2) or "").strip()
                current_paragraph = par
                current_item = None  # 항 바뀌면 호 컨텍스트 리셋
                current_subitem = None  # 항 바뀌면 목 컨텍스트 리셋
                current_item_subitems = []  # 목 리스트 초기화
    
                buf_start_page = page_no if buf_start_page is None else min(buf_start_page, page_no)
                buf_article = current_article
                buf_paragraph = current_paragraph
                buf_item = None
                buf_subitem = None
                # 조에 포함되므로 기존 버퍼에 추가 (새 버퍼 시작하지 않음)
                if buf_lines:
                    buf_lines.append(line if not rest else f"{par} {rest}".strip())
                else:
                    buf_lines = [line if not rest else f"{par} {rest}".strip()]
                continue

            # 3) 항 감지(①②③…) - 항은 조에 포함 (별도 청크 아님)
            m_c = RE_PARAGRAPH_CIRCLE.match(line)
            if m_c:
                # flush() 제거 - 항은 같은 조에 포함

                par = _circle_to_paragraph(m_c.group(1))
                rest = (m_c.group(2) or "").strip()

                current_paragraph = par
                current_item = None  # 항 바뀌면 호 컨텍스트 리셋
                current_subitem = None  # 항 바뀌면 목 컨텍스트 리셋
                current_item_subitems = []  # 목 리스트 초기화
                
                buf_start_page = page_no if buf_start_page is None else min(buf_start_page, page_no)
                buf_article = current_article
                buf_paragraph = current_paragraph
                buf_item = None
                buf_subitems = []
                # 조에 포함되므로 기존 버퍼에 추가 (새 버퍼 시작하지 않음)
                if buf_lines:
                    buf_lines.append(line if not rest else f"{m_c.group(1)} {rest}".strip())
                else:
                    buf_lines = [line if not rest else f"{m_c.group(1)} {rest}".strip()]
                continue

            # 4) 호(號) 감지(1., 2., 1의2. 등) - 호는 조에 포함
            m_item_num = RE_ITEM_NUMBER.match(line)
            if m_item_num:
                # flush() 제거 - 호는 같은 조에 포함

                item_num = _number_to_item(m_item_num.group(1))  # "1" -> "제1호"
                rest = (m_item_num.group(2) or "").strip()
                current_item = item_num
                current_subitem = None  # 호 바뀌면 목 컨텍스트 리셋
                current_item_subitems = []  # 새 호 시작 시 목 리스트 초기화

                # 호는 조에 포함되므로 기존 버퍼에 추가 (새 버퍼 시작하지 않음)
                buf_start_page = page_no if buf_start_page is None else min(buf_start_page, page_no)
                buf_article = current_article
                buf_paragraph = current_paragraph
                buf_item = item_num
                buf_subitem = None  # 호 시작 시 목 없음
                if buf_lines:
                    buf_lines.append(line if not rest else f"{m_item_num.group(1)}. {rest}".strip())
                else:
                    buf_lines = [line if not rest else f"{m_item_num.group(1)}. {rest}".strip()]
                continue

            # 5) 목(目) 감지(가., 나., 다. 등) - 목은 조에 포함 (별도 청크 아님)
            m_item_hangul = RE_ITEM_HANGUL.match(line)
            if m_item_hangul:
                # flush() 제거 - 목은 같은 호 내에서 계속 누적

                item_hangul = _hangul_to_item(m_item_hangul.group(1))  # "가" -> "가목"
                rest = (m_item_hangul.group(2) or "").strip()

                current_subitem = item_hangul

                # 목은 현재 호에 추가 (새 청크 시작하지 않음)
                buf_subitem = item_hangul  # 현재 목 업데이트
                # 목 라인을 버퍼에 추가 (새 버퍼 시작하지 않음)
                if buf_lines:  # 이미 버퍼에 내용이 있으면
                    buf_lines.append(line if not rest else f"{m_item_hangul.group(1)}. {rest}".strip())
                else:  # 버퍼가 비어있으면 (호가 없이 목만 있는 경우)
                    buf_start_page = page_no if buf_start_page is None else min(buf_start_page, page_no)
                    buf_article = current_article
                    buf_paragraph = current_paragraph
                    buf_item = current_item
                    buf_lines = [line if not rest else f"{m_item_hangul.group(1)}. {rest}".strip()]
                continue

            # 6) 일반 라인: 현재 버퍼에 누적
            # (조가 아직 안 잡힌 구간은 스킵하거나 별도 처리 가능)
            if buf_start_page is None:
                # 조/항 시작 전의 머리말(목차/서문 등)일 가능성이 큼 -> 필요하면 따로 저장
                continue

            # 일반 라인 처리 시 현재 메타데이터 유지 (이미 설정된 경우)
            if buf_article is None:
                buf_article = current_article
            if buf_paragraph is None:
                buf_paragraph = current_paragraph
            # buf_items, buf_subitems는 이미 배열로 관리되므로 별도 처리 불필요

            buf_lines.append(line)

    flush()
    return chunks