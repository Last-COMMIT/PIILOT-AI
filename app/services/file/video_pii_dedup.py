"""
영상 PII 프레임 단위 → 트랙(고유 등장) 단위 집계
- 같은 프레임 내: 머지하지 않음
- 프레임 간: IoU/위치 유사 + 프레임 차이로 같은 대상이면 하나의 트랙으로 묶음
"""
from typing import List, Dict, Any

# 기본값 (필요 시 config로 이전)
DEFAULT_MAX_FRAME_GAP = 30
DEFAULT_IOU_THRESHOLD = 0.25


def _bbox_to_xyxy(item: Dict) -> tuple:
    """(x, y, width, height) -> (x1, y1, x2, y2)"""
    x = item.get("x", 0)
    y = item.get("y", 0)
    w = item.get("width", 0)
    h = item.get("height", 0)
    return (x, y, x + w, y + h)


def _iou(box1: tuple, box2: tuple) -> float:
    """두 bbox (x1,y1,x2,y2) 의 IoU"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    inter = (xi2 - xi1) * (yi2 - yi1)
    a1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    a2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def dedup_video_faces(
    faces: List[Dict],
    max_frame_gap: int = DEFAULT_MAX_FRAME_GAP,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> int:
    """
    프레임별 얼굴 리스트를 트랙으로 묶어 고유 얼굴 수 반환.
    같은 프레임 내 얼굴은 절대 머지하지 않음.
    """
    if not faces:
        return 0
    # 프레임 오름차순 정렬
    sorted_faces = sorted(faces, key=lambda f: (f.get("frame_number", 0), f.get("x", 0)))
    tracks: List[List[Dict]] = []  # 각 트랙 = 해당 얼굴이 나온 프레임들의 bbox 목록

    for face in sorted_faces:
        fn = face.get("frame_number", 0)
        box = _bbox_to_xyxy(face)
        assigned = False
        for track in tracks:
            # 같은 프레임이면 이 트랙에 넣지 않음 (다른 사람)
            last = track[-1]
            if last.get("frame_number") == fn:
                continue
            last_fn = last.get("frame_number", 0)
            if abs(fn - last_fn) > max_frame_gap:
                continue
            last_box = _bbox_to_xyxy(last)
            if _iou(box, last_box) >= iou_threshold:
                track.append(face)
                assigned = True
                break
        if not assigned:
            tracks.append([face])

    return len(tracks)


def dedup_video_text_regions(
    text_pii_regions: List[Dict],
    max_frame_gap: int = DEFAULT_MAX_FRAME_GAP,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> Dict[str, int]:
    """
    키프레임별 텍스트 PII 영역을 label별로 트랙 머지해 고유 등장 수 반환.
    같은 프레임 내 영역은 머지하지 않음.
    Returns:
        { "NM": n, "PH": m, ... } (label -> PII 코드, 코드는 호출측에서 매핑)
    """
    if not text_pii_regions:
        return {}
    # label별로 그룹
    by_label: Dict[str, List[Dict]] = {}
    for r in text_pii_regions:
        label = r.get("label", "p_nm")
        by_label.setdefault(label, []).append(r)

    counts: Dict[str, int] = {}
    for label, regions in by_label.items():
        sorted_regions = sorted(regions, key=lambda r: (r.get("frame_number", 0), r.get("x", 0)))
        tracks: List[List[Dict]] = []
        for reg in sorted_regions:
            fn = reg.get("frame_number", 0)
            box = _bbox_to_xyxy(reg)
            assigned = False
            for track in tracks:
                if track[-1].get("frame_number") == fn:
                    continue
                last_fn = track[-1].get("frame_number", 0)
                if abs(fn - last_fn) > max_frame_gap:
                    continue
                last_box = _bbox_to_xyxy(track[-1])
                if _iou(box, last_box) >= iou_threshold:
                    track.append(reg)
                    assigned = True
                    break
            if not assigned:
                tracks.append([reg])
        counts[label] = len(tracks)
    return counts
