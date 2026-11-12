import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class InterpretabilityEngine:
    """
    Post-hoc SHAP explanation for the trained neural cardiovascular predictor.
    Explains model decisions without impacting the training objective.
    """
    def __init__(self, model, background_data: np.ndarray, feature_names: List[str]):
        self.model = model
        self.background_data = background_data
        self.feature_names = feature_names
        self.explainer = None
        self._setup_explainer()

    def _setup_explainer(self):
        try:
            import shap
            # Take representative background sample of 50 records for computational efficiency
            bg_sample = self.background_data[:50]
            self.explainer = shap.KernelExplainer(self.model.predict_proba, bg_sample)
            logger.info("Initialized SHAP KernelExplainer for neural tabular model.")
        except Exception as e:
            logger.warning(f"SHAP initialization fallback: {e}")
            self.explainer = None

    def explain_instance(self, instance: np.ndarray) -> Dict[str, Any]:
        """Generates patient-level explanation with top risk factors."""
        if instance.ndim == 1:
            instance = instance.reshape(1, -1)

        pred_prob = float(self.model.predict_proba(instance)[0])

        if self.explainer is not None:
            try:
                shap_values = self.explainer.shap_values(instance, nsamples=100)
                if isinstance(shap_values, list):
                    vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
                else:
                    vals = shap_values[0] if shap_values.ndim > 1 else shap_values
            except Exception as e:
                logger.warning(f"SHAP compute error, using perturbation fallback: {e}")
                vals = self._perturbation_importance(instance)
        else:
            vals = self._perturbation_importance(instance)

        # Rank features
        ranked_indices = np.argsort(np.abs(vals))[::-1]
        top_factors = []
        for idx in ranked_indices[:5]:
            contrib = float(vals[idx])
            top_factors.append({
                'feature': self.feature_names[idx],
                'contribution': round(contrib, 4),
                'direction': 'Increasing Risk (↑)' if contrib > 0 else 'Decreasing Risk (↓)',
                'magnitude': round(abs(contrib), 4)
            })

        return {
            'predicted_risk_probability': round(pred_prob, 4),
            'top_risk_factors': top_factors,
            'all_shap_values': {self.feature_names[i]: float(vals[i]) for i in range(len(self.feature_names))}
        }

    def _perturbation_importance(self, instance: np.ndarray) -> np.ndarray:
        base_pred = self.model.predict_proba(instance)[0]
        importances = []
        for j in range(instance.shape[1]):
            perturbed = instance.copy()
            perturbed[0, j] += 0.5 # delta perturbation
            p_perturbed = self.model.predict_proba(perturbed)[0]
            importances.append(p_perturbed - base_pred)
        return np.array(importances)

    def compute_global_importance(self, test_samples: np.ndarray) -> pd.DataFrame:
        """Computes mean absolute SHAP values across test dataset."""
        logger.info(f"Computing global SHAP feature importances on {min(len(test_samples), 100)} samples...")
        sample = test_samples[:min(len(test_samples), 100)]
        
        if self.explainer is not None:
            try:
                shap_vals = self.explainer.shap_values(sample, nsamples=60)
                if isinstance(shap_vals, list):
                    vals = np.abs(shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]).mean(axis=0)
                else:
                    vals = np.abs(shap_vals).mean(axis=0)
            except Exception:
                vals = np.array([np.abs(self._perturbation_importance(sample[i:i+1])).mean() for i in range(len(sample))])
        else:
            all_pert = [self._perturbation_importance(sample[i:i+1]) for i in range(len(sample))]
            vals = np.abs(np.array(all_pert)).mean(axis=0)

        df_imp = pd.DataFrame({
            'feature': self.feature_names,
            'importance': vals
        }).sort_values(by='importance', ascending=False).reset_index(drop=True)

        return df_imp
