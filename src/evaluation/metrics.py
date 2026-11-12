import logging
from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss
)

logger = logging.getLogger(__name__)

def calculate_ece(y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            acc = y_true[in_bin].mean()
            conf = y_pred_proba[in_bin].mean()
            ece += np.abs(conf - acc) * prop_in_bin
            
    return float(ece)

class MetricsSuite:
    """
    Standard comprehensive metrics evaluator for centralized and federated models.
    """
    def compute_all_metrics(
        self, y_true: np.ndarray, y_pred_proba: np.ndarray, training_time: float = 0.0,
        comm_cost_kb: float = 0.0, n_rounds: int = 0, client_variance: float = 0.0
    ) -> Dict[str, float]:
        
        preds = np.clip(y_pred_proba, 0.0, 1.0)
        pred_labels = (preds >= 0.5).astype(int)

        try:
            auc = float(roc_auc_score(y_true, preds))
        except Exception:
            auc = 0.5

        try:
            pr_auc = float(average_precision_score(y_true, preds))
        except Exception:
            pr_auc = 0.5

        acc = float(accuracy_score(y_true, pred_labels))
        prec = float(precision_score(y_true, pred_labels, zero_division=0))
        rec = float(recall_score(y_true, pred_labels, zero_division=0))
        f1 = float(f1_score(y_true, pred_labels, zero_division=0))
        brier = float(brier_score_loss(y_true, preds))
        ece = float(calculate_ece(y_true, preds))

        return {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'roc_auc': auc,
            'pr_auc': pr_auc,
            'brier_score': brier,
            'ece': ece,
            'training_time_sec': round(training_time, 2),
            'communication_cost_kb': round(comm_cost_kb, 2),
            'n_rounds': n_rounds,
            'client_variance': round(client_variance, 4)
        }
