import logging
import json
from .metrics import MetricsCalculator

logger = logging.getLogger(__name__)

class ExternalValidator:
    def __init__(self):
        self.metrics_calc = MetricsCalculator()
        
    def validate(self, trained_model, external_dataset_path: str, adapter) -> dict:
        try:
            X, y, features, info = adapter.load(external_dataset_path)
            
            # Very basic check for feature compatibility
            model_features = getattr(trained_model, 'metadata', {}).get('features', [])
            if model_features and set(model_features) != set(features):
                logger.warning(f"Feature mismatch between model and dataset {external_dataset_path}. Validation may fail or be inaccurate.")
                
            # Assume data is appropriately preprocessed (in a real scenario, we'd run the preprocessing pipeline's transform)
            
            preds = trained_model.predict_proba(X.values)
            metrics = self.metrics_calc.calculate(y.values, preds)
            
            return {
                'dataset_name': info['name'],
                'metrics': metrics
            }
        except Exception as e:
            logger.error(f"External validation failed for {external_dataset_path}: {e}")
            return None
