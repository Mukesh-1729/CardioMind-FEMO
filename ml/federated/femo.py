import logging
import numpy as np
from typing import List, Dict
import json

logger = logging.getLogger(__name__)

class FEMO:
    """
    L_FEMO = L_pred + λc·L_cal + λp·L_prox + λr·L_risk
    """
    
    def compute_pred_loss(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Binary cross entropy"""
        eps = 1e-15
        p = np.clip(y_pred_proba, eps, 1 - eps)
        return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
        
    def compute_cal_loss(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Brier-style: (1/nk) Σ (pi - yi)²"""
        return np.mean((y_pred_proba - y_true) ** 2)
        
    def compute_prox_loss(self, params: np.ndarray, global_params: np.ndarray) -> float:
        """(1/2)||θk - θt||²"""
        return 0.5 * np.linalg.norm(params - global_params) ** 2
        
    def compute_risk_loss(self, current_risks: np.ndarray, prev_risks: np.ndarray, delta_features: np.ndarray) -> float:
        """(1/N) Σ |ri,t - ri,t-1 - g(Δxi,t)|"""
        if prev_risks is None or delta_features is None:
            return 0.0
        g_delta = np.sum(delta_features, axis=1) * 0.01 # Simplified g()
        return np.mean(np.abs(current_risks - prev_risks - g_delta))
        
    def compute_total_loss(self, y_true, y_pred_proba, params, global_params, prev_risks, delta_features, config):
        l_pred = self.compute_pred_loss(y_true, y_pred_proba)
        l_cal = self.compute_cal_loss(y_true, y_pred_proba)
        l_prox = self.compute_prox_loss(params, global_params)
        l_risk = self.compute_risk_loss(current_risks=y_pred_proba, prev_risks=prev_risks, delta_features=delta_features)
        
        femo_cfg = config['federated']['femo']
        l_total = l_pred + femo_cfg['lambda_cal'] * l_cal + femo_cfg['lambda_prox'] * l_prox + femo_cfg['lambda_risk'] * l_risk
        return l_total, l_pred, l_cal, l_prox, l_risk
        
    def log_loss_components(self, round_num, client_id, components):
        logger.info(f"Round {round_num} Client {client_id} Losses: {json.dumps(components)}")
        
    def aggregate(self, client_updates: List[Dict], weights_config: Dict = None) -> np.ndarray:
        from .aggregation import AdaptiveAggregation
        return AdaptiveAggregation().aggregate(client_updates, weights_config)
