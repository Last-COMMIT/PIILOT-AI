import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForTokenClassification
from typing import List, Dict, Optional
import os
from ....models.personal_info import CONFIDENCE_THRESHOLDS, PII_NAMES

class KoELECTRAPIIDetector:
    """학습된 KoELECTRA NER 모델로 PII 탐지"""
    def __init__(self, model_path: str, confidence_thresholds: Dict[str, float] = None):
        print(f"KoELECTRA PII 탐지 모델 로드: {model_path}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForTokenClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            self.id2label = self.model.config.id2label
            self.label2id = self.model.config.label2id
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            raise e

        self.confidence_thresholds = confidence_thresholds or CONFIDENCE_THRESHOLDS
        print(f"KoELECTRA 모델 로드 완료 (디바이스: {self.device})")

    def detect_pii(self, text: str, return_confidence: bool = True) -> List[Dict]:
        if not text.strip(): return []
        
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, return_offsets_mapping=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = F.softmax(outputs.logits, dim=-1)[0]
            predictions = torch.argmax(probabilities, dim=-1)

        predictions = predictions.cpu().numpy()
        offsets = inputs["offset_mapping"][0].cpu().numpy()

        entities = []
        current_entity = None

        for idx, (pred, offset) in enumerate(zip(predictions, offsets)):
            if offset[0] == 0 and offset[1] == 0: continue
            pred_label = self.id2label[pred]
            confidence = probabilities[idx][pred].item()

            if pred_label == 'O':
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
                continue

            if pred_label.startswith('B-'):
                if current_entity: entities.append(current_entity)
                current_entity = {
                    'label': pred_label[2:],
                    'start': offset[0],
                    'end': offset[1],
                    'confidences': [confidence],
                    'text_fragments': [text[offset[0]:offset[1]]]
                }
                
            elif pred_label.startswith('I-'):
                if current_entity and (current_entity['label'] == pred_label[2:]):
                    current_entity['end'] = offset[1]
                    current_entity['confidences'].append(confidence)
                else:
                    if current_entity: entities.append(current_entity)
                    current_entity = {
                        'label': pred_label[2:],
                        'start': offset[0],
                        'end': offset[1],
                        'confidences': [confidence],
                        'text_fragments': [text[offset[0]:offset[1]]]
                    }

        if current_entity: entities.append(current_entity)

        final_results = []
        for ent in entities:
            full_text = text[ent['start']:ent['end']]
            avg_conf = sum(ent['confidences']) / len(ent['confidences'])
            final_results.append({
                'text': full_text,
                'label': ent['label'],
                'start': ent['start'],
                'end': ent['end'],
                'confidence': avg_conf
            })

        return self.apply_confidence_filter(final_results)

    def apply_confidence_filter(self, entities: List[Dict]) -> List[Dict]:
        filtered = []
        for entity in entities:
            label = entity['label']
            confidence = entity['confidence']
            threshold = self.confidence_thresholds.get(label, 0.75)
            if confidence >= threshold:
                filtered.append(entity)
            else:
                pass
        return filtered

    def update_confidence_threshold(self, entity_type: str, new_threshold: float):
        if entity_type in self.confidence_thresholds:
            self.confidence_thresholds[entity_type] = new_threshold