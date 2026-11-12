import logging
import numpy as np

logger = logging.getLogger(__name__)

class SHAPExplainer:
    """
    Computes feature attributions using SHAP.
    IMPORTANT: These are feature attributions, NOT causal effects.
    """
    def __init__(self):
        self.explainer = None
        self.is_tree = False
        
    def fit(self, model, X_background):
        try:
            import shap
            # Very basic model type detection
            model_name = getattr(model, '__class__', None).__name__
            if 'Boost' in model_name or 'Forest' in model_name or 'LGBM' in model_name:
                self.is_tree = True
                # Access underlying model if it's our wrapper
                base_model = getattr(model, 'model', model)
                self.explainer = shap.TreeExplainer(base_model)
            else:
                base_model = getattr(model, 'model', model)
                self.explainer = shap.LinearExplainer(base_model, X_background)
        except Exception as e:
            logger.error(f"Failed to fit SHAP explainer: {e}")
            self.explainer = None
            
    def explain(self, X_instance):
        if self.explainer is None:
            return np.zeros(X_instance.shape[1])
            
        shap_values = self.explainer.shap_values(X_instance)
        # Handle list return for some models (e.g. classification returning [neg, pos])
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
            
        return shap_values

    def get_top_features(self, shap_values, feature_names, n=5):
        if shap_values.ndim > 1:
            shap_values = shap_values[0] # Take first instance if batch
            
        abs_shap = np.abs(shap_values)
        top_indices = np.argsort(abs_shap)[::-1][:n]
        
        results = []
        for idx in top_indices:
            val = shap_values[idx]
            results.append({
                'feature': feature_names[idx],
                'contribution': float(val),
                'direction': 'increases risk' if val > 0 else 'decreases risk',
                'magnitude': float(abs_shap[idx]),
                'feature_value': None # To be filled by caller
            })
        return results
