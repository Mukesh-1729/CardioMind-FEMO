import logging
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.calibration import IsotonicRegression, calibration_curve
from sklearn.metrics import brier_score_loss

logger = logging.getLogger(__name__)

class CalibrationEngine:
    """
    Post-hoc probability calibration and analysis:
    - Fits Isotonic / Platt calibrator on validation predictions
    - Evaluates calibration quality: Brier Score, ECE, MCE
    """
    def __init__(self, method: str = 'isotonic'):
        self.method = method
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.is_fitted = False

    def fit(self, val_pred_probas: np.ndarray, y_val: np.ndarray):
        """Fits calibrator strictly on validation predictions."""
        self.calibrator.fit(val_pred_probas, y_val)
        self.is_fitted = True
        logger.info("Fitted post-hoc probability calibrator on validation data.")

    def calibrate(self, test_pred_probas: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return test_pred_probas
        calibrated = self.calibrator.transform(test_pred_probas)
        return np.clip(calibrated, 0.0, 1.0)

    def evaluate_calibration(
        self, y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10
    ) -> Dict[str, Any]:
        prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins)
        brier = float(brier_score_loss(y_true, y_pred_proba))

        # ECE & MCE
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        bin_diffs = []
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
            prop_in_bin = in_bin.mean()
            if prop_in_bin > 0:
                acc = y_true[in_bin].mean()
                conf = y_pred_proba[in_bin].mean()
                diff = np.abs(conf - acc)
                ece += diff * prop_in_bin
                bin_diffs.append(diff)

        mce = float(max(bin_diffs)) if bin_diffs else 0.0

        return {
            'brier_score': brier,
            'ece': float(ece),
            'mce': mce,
            'prob_true': prob_true.tolist(),
            'prob_pred': prob_pred.tolist()
        }
