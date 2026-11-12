import json
import logging
import os
import uuid
from typing import Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelRegistry:
    def __init__(self, registry_dir: str = 'e:\\CardioMind\\ml\\models'):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.registry_dir / 'registry.json'
        if not self.registry_path.exists():
            with open(self.registry_path, 'w') as f:
                json.dump([], f)
                
    def _load_registry(self) -> List[Dict]:
        with open(self.registry_path, 'r') as f:
            return json.load(f)
            
    def _save_registry(self, registry: List[Dict]):
        with open(self.registry_path, 'w') as f:
            json.dump(registry, f, indent=4)
            
    def register(self, model, metrics: Dict[str, float], config: Dict[str, Any], features: List[str], dataset: str) -> str:
        model_id = str(uuid.uuid4())
        model_path = self.registry_dir / f"{model_id}.model"
        
        # Determine how to save based on class (simple for now, assuming sklearn-like joblib or native save)
        if hasattr(model, 'save'):
            model.save(str(model_path))
        else:
            import joblib
            joblib.dump(model, str(model_path))
            
        metadata = {
            'model_id': model_id,
            'model_name': getattr(model, 'metadata', {}).get('model_name', model.__class__.__name__),
            'metrics': metrics,
            'features': features,
            'dataset': dataset,
            'path': str(model_path)
        }
        
        registry = self._load_registry()
        registry.append(metadata)
        self._save_registry(registry)
        
        logger.info(f"Registered model {model_id}")
        return model_id
        
    def load(self, model_id: str) -> Dict[str, Any]:
        registry = self._load_registry()
        for m in registry:
            if m['model_id'] == model_id:
                return m
        raise ValueError(f"Model {model_id} not found in registry.")
        
    def list_models(self) -> List[Dict]:
        return self._load_registry()
        
    def get_best_model(self, metric: str = 'roc_auc') -> Dict[str, Any]:
        registry = self._load_registry()
        if not registry:
            return None
        return max(registry, key=lambda x: x.get('metrics', {}).get(metric, 0))
