import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve, IsotonicRegression
from sklearn.metrics import brier_score_loss
from .metrics import MetricsCalculator

logger = logging.getLogger(__name__)

class CalibrationAnalyzer:
    def __init__(self):
        self.metrics_calc = MetricsCalculator()

    def analyze(self, y_true, y_pred_proba, n_bins=10):
        prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins)
        ece = self.metrics_calc.calculate_ece(y_true, y_pred_proba, n_bins)
        brier = brier_score_loss(y_true, y_pred_proba)
        mce = np.max(np.abs(prob_true - prob_pred))
        
        return {
            'prob_true': prob_true.tolist(),
            'prob_pred': prob_pred.tolist(),
            'ece': ece,
            'mce': mce,
            'brier': brier
        }

    def calibrate(self, model, X_val, y_val, method='isotonic'):
        val_preds = model.predict_proba(X_val)
        if method == 'isotonic':
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(val_preds, y_val)
            
            # Monkey patch predict_proba to use calibrator
            original_predict_proba = model.predict_proba
            model.predict_proba = lambda X: calibrator.transform(original_predict_proba(X))
        return model

    def plot_calibration_curve(self, results, save_path):
        plt.figure(figsize=(8, 8))
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.plot(results['prob_pred'], results['prob_true'], "s-", label=f"Model (ECE={results['ece']:.3f})")
        plt.ylabel("Fraction of positives")
        plt.xlabel("Mean predicted value")
        plt.legend(loc="lower right")
        plt.savefig(save_path)
        plt.close()
