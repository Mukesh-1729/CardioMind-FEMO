import logging
import numpy as np
from typing import Dict, Any

logger = logging.getLogger(__name__)

class APCRSCalculator:
    """
    APCRS_t = 100 × clip(α·p_t + β·R_t + γ·ΔR_t, 0, 1)
    DISCLAIMER: This is a research prototype score, not clinically validated.
    """
    def __init__(self, config: Dict[str, Any] = None):
        if config is None:
            self.config = {'apcrs': {'alpha': 0.5, 'beta': 0.3, 'gamma': 0.2}}
        else:
            self.config = config

    def compute_risk_factor_index(self, patient_features: Dict[str, float]) -> float:
        """R_t: normalized 0-1 composite of risk factors"""
        # Very simplified composite risk indexing
        sbp = patient_features.get('systolic_bp', 120)
        chol = patient_features.get('cholesterol', 150)
        bmi = patient_features.get('bmi', 25)
        
        # Normalize roughly
        n_sbp = min(max((sbp - 90) / (180 - 90), 0), 1)
        n_chol = min(max((chol - 100) / (300 - 100), 0), 1)
        n_bmi = min(max((bmi - 18) / (40 - 18), 0), 1)
        
        return (n_sbp + n_chol + n_bmi) / 3.0

    def compute_delta_risk(self, current_R: float, previous_R: float) -> float:
        """ΔR_t: Change in risk index. Higher is worse."""
        if previous_R is None:
            return 0.5 # Neutral baseline if no history
            
        # Map change to 0-1 where 0.5 is no change, 1 is terrible worsening, 0 is great improvement
        delta = current_R - previous_R
        # Assume max change is +/- 0.5
        mapped_delta = (delta + 0.5) 
        return float(np.clip(mapped_delta, 0.0, 1.0))

    def compute_apcrs(self, p_t: float, R_t: float, delta_R: float, config: Dict[str, Any] = None) -> float:
        cfg = config or self.config
        alpha = cfg['apcrs']['alpha']
        beta = cfg['apcrs']['beta']
        gamma = cfg['apcrs']['gamma']
        
        raw_score = alpha * p_t + beta * R_t + gamma * delta_R
        return 100.0 * float(np.clip(raw_score, 0.0, 1.0))

    def classify_risk(self, apcrs_score: float, thresholds: Dict[str, float]) -> str:
        score_norm = apcrs_score / 100.0
        if score_norm <= thresholds.get('low_max', 0.3):
            return 'Low'
        elif score_norm <= thresholds.get('moderate_max', 0.6):
            return 'Moderate'
        elif score_norm <= thresholds.get('high_max', 1.0):
            return 'High'
        else:
            return 'Very High'

    def compute_trend(self, history: list) -> str:
        if not history or len(history) < 2:
            return 'Stable'
            
        diff = history[-1] - history[-2]
        if diff > 5.0:
            return 'Worsening'
        elif diff < -5.0:
            return 'Improving'
        return 'Stable'
