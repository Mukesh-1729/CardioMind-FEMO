import logging
import json
import numpy as np
from typing import List, Dict

logger = logging.getLogger(__name__)

class AdaptiveAggregation:
    def calculate_reliability(self, client_metrics: Dict) -> float:
        """qk from validation AUC/Brier"""
        auc = client_metrics.get('roc_auc', 0.5)
        brier = client_metrics.get('brier_score', 0.25)
        # Higher AUC and lower Brier means higher reliability
        return max(0.1, auc + (0.25 - brier))
        
    def calculate_stability(self, prev_params: np.ndarray, curr_params: np.ndarray) -> float:
        """sk from cosine similarity"""
        if prev_params is None or curr_params is None:
            return 1.0
        norm_p = np.linalg.norm(prev_params)
        norm_c = np.linalg.norm(curr_params)
        if norm_p == 0 or norm_c == 0:
            return 1.0
        cos_sim = np.dot(prev_params, curr_params) / (norm_p * norm_c)
        return float(np.clip(cos_sim, 0.1, 1.0))
        
    def calculate_weights(self, n_samples: int, reliability: float, stability: float, 
                          a: float = 1.0, b: float = 0.5, c: float = 0.5) -> float:
        """uk = nk^a × qk^b × sk^c"""
        return (n_samples ** a) * (reliability ** b) * (stability ** c)
        
    def aggregate(self, client_updates: List[Dict], weights_config: Dict = None) -> np.ndarray:
        if not client_updates:
            return None
            
        a = weights_config.get('alpha_agg', 1.0) if weights_config else 1.0
        b = weights_config.get('beta_agg', 0.5) if weights_config else 0.5
        c = weights_config.get('gamma_agg', 0.5) if weights_config else 0.5
        
        uks = []
        for update in client_updates:
            n = update['metadata']['n_samples']
            metrics = update['metadata']['local_metrics']
            rel = self.calculate_reliability(metrics)
            
            # Simple stability estimation for now
            stab = 1.0 
            
            uk = self.calculate_weights(n, rel, stab, a, b, c)
            uks.append(uk)
            
        sum_uk = sum(uks)
        wks = [uk / sum_uk for uk in uks]
        
        global_params = np.zeros_like(client_updates[0]['params'])
        for i, update in enumerate(client_updates):
            global_params += wks[i] * update['params']
            
        return global_params
        
    def log_weights(self, round_num: int, weights_dict: Dict):
        logger.info(f"Round {round_num} Aggregation Weights: {json.dumps(weights_dict)}")
