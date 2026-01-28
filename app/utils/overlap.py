"""
엔티티 겹침 검사 공유 유틸리티
"""
from typing import Dict


def is_overlapping(entity1: Dict, entity2: Dict) -> bool:
    """
    두 엔티티가 텍스트 위치상 겹치는지 확인

    Args:
        entity1: {'start': int, 'end': int, ...}
        entity2: {'start': int, 'end': int, ...}

    Returns:
        겹치면 True
    """
    return not (entity1['end'] <= entity2['start'] or entity2['end'] <= entity1['start'])
