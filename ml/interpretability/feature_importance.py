import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

logger = logging.getLogger(__name__)

class FeatureImportanceAnalyzer:
    def get_global_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        base_model = getattr(model, 'model', model)
        
        importances = None
        if hasattr(base_model, 'feature_importances_'):
            importances = base_model.feature_importances_
        elif hasattr(base_model, 'coef_'):
            importances = np.abs(base_model.coef_[0])
            
        if importances is None:
            logger.warning("Could not extract feature importances from model")
            return {name: 0.0 for name in feature_names}
            
        # Normalize
        importances = importances / np.sum(importances)
        
        return {name: float(imp) for name, imp in zip(feature_names, importances)}

    def plot_importance(self, importance_dict: Dict[str, float], save_path: str):
        sorted_items = sorted(importance_dict.items(), key=lambda x: x[1])
        features = [x[0] for x in sorted_items][-15:] # Top 15
        scores = [x[1] for x in sorted_items][-15:]
        
        plt.figure(figsize=(10, 8))
        plt.barh(features, scores, color='skyblue')
        plt.xlabel('Relative Importance')
        plt.title('Global Feature Importance')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
