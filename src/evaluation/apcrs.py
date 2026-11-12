import logging
from typing import Dict, Any, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

class APCRSEngine:
    """
    Adaptive Personalized Cardiovascular Risk Score (APCRS):
    APCRS_t = 100 * clip(alpha * p_t + beta * R_t + gamma * delta_R_t, 0, 1)
    where alpha + beta + gamma = 1
    """
    def __init__(self, alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.2):
        total = alpha + beta + gamma
        self.alpha = alpha / total
        self.beta = beta / total
        self.gamma = gamma / total

    def compute_apcrs(
        self, pred_prob: float, clinical_risk: float = None, prev_risk: float = None
    ) -> Dict[str, Any]:
        p_t = float(np.clip(pred_prob, 0.0, 1.0))
        R_t = float(np.clip(clinical_risk if clinical_risk is not None else p_t, 0.0, 1.0))

        if prev_risk is not None:
            delta_R = float(np.clip(R_t - prev_risk, -1.0, 1.0))
        else:
            delta_R = 0.0

        raw_score = self.alpha * p_t + self.beta * R_t + self.gamma * delta_R
        apcrs_score = round(float(np.clip(raw_score, 0.0, 1.0)) * 100.0, 2)

        # Risk stratification
        if apcrs_score >= 60.0:
            risk_tier = 'High'
        elif apcrs_score >= 30.0:
            risk_tier = 'Moderate'
        else:
            risk_tier = 'Low'

        # Trend analysis
        if delta_R > 0.10:
            trend = 'Rapidly Increasing'
        elif delta_R > 0.02:
            trend = 'Moderately Increasing'
        elif delta_R < -0.02:
            trend = 'Improving'
        else:
            trend = 'Stable'

        return {
            'apcrs_score': apcrs_score,
            'risk_tier': risk_tier,
            'prediction_component': round(self.alpha * p_t * 100.0, 2),
            'clinical_component': round(self.beta * R_t * 100.0, 2),
            'trend_component': round(self.gamma * delta_R * 100.0, 2),
            'delta_risk': round(delta_R, 4),
            'trend': trend
        }
