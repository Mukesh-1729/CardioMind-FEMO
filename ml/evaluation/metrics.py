import logging
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, brier_score_loss

logger = logging.getLogger(__name__)

class MetricsCalculator:
    """Calculates standard ML evaluation metrics."""
    
    def calculate(self, y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5) -> dict:
        """Calculates classification metrics given probabilities and true labels."""
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_true, y_pred_proba)),
            'pr_auc': float(average_precision_score(y_true, y_pred_proba)),
            'brier_score': float(brier_score_loss(y_true, y_pred_proba)),
            'worst_client_performance': 0.0, # Placeholder for federated
            'client_variance': 0.0,          # Placeholder for federated
            'training_time': 0.0,
            'communication_cost': 0.0
        }
        
        return metrics

    def calculate_ece(self, y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10) -> float:
        """Calculates Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
            prop_in_bin = in_bin.mean()
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_pred_proba[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
        return float(ece)
