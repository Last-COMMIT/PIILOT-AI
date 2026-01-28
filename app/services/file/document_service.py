"""
문서 PII 탐지 서비스
(기존 document_detector.py 리팩토링)
"""
from typing import List, Dict
from app.ml.pii_detectors.hybrid_detector import HybridPIIDetector
from app.core.constants import PII_NAMES
from app.core.logging import logger


class DocumentDetector:
    """문서 PII 탐지 (Hybrid: KoELECTRA NER + 정규식)"""

    def __init__(self, model_path: str, confidence_thresholds: Dict[str, float] = None):
        logger.info(f"DocumentDetector 초기화: {model_path}")
        self.hybrid_detector = HybridPIIDetector(model_path, confidence_thresholds)
        self.context_names = set()

    def detect(self, text: str) -> List[Dict]:
        """
        텍스트에서 PII 탐지

        Returns:
            [{"type": str, "value": str, "start": int, "end": int, "confidence": float}, ...]
        """
        logger.info(f"문서 PII 탐지 시작 (길이: {len(text)})")

        entities = self.hybrid_detector.detect_pii(text, self.context_names)

        # 이름 컨텍스트 업데이트
        for entity in entities:
            if entity.get('label') == 'p_nm':
                self.context_names.add(entity['text'])

        # API 응답 형식으로 변환 + 통계 수집
        result = []
        stats = self._collect_pii_stats(entities)

        for entity in entities:
            result.append({
                "type": PII_NAMES.get(entity['label'], entity['label']),
                "value": entity['text'],
                "start": entity.get('start'),
                "end": entity.get('end'),
                "confidence": entity.get('confidence', 0.0),
            })

        logger.info(f"문서 PII 탐지 완료: {len(result)}개 항목 발견")
        for pii_type, count in stats.items():
            logger.info(f"  - {pii_type}: {count}개")

        return result

    def _collect_pii_stats(self, entities: List[Dict]) -> Dict[str, int]:
        """PII 통계 수집 (PDF/DOCX/TXT 공통)"""
        stats = {}
        for entity in entities:
            label = entity.get('label', 'unknown')
            pii_name = PII_NAMES.get(label, label)
            stats[pii_name] = stats.get(pii_name, 0) + 1
        return stats
