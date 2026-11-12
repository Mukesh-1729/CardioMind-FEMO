import logging
import numpy as np
from typing import List, Dict

logger = logging.getLogger(__name__)

class FedProx:
    def compute_loss(self, pred_loss, params, global_params, mu):
        """Local objective: L_FedProx = L_pred + (μ/2) × ||θk - θt||²"""
        prox_term = (mu / 2) * np.linalg.norm(params - global_params) ** 2
        return pred_loss + prox_term
        
    def aggregate(self, client_updates: List[Dict]) -> np.ndarray:
        """Same as FedAvg aggregation"""
        total_samples = sum([u['metadata']['n_samples'] for u in client_updates])
        global_params = np.zeros_like(client_updates[0]['params'])
        
        for update in client_updates:
            weight = update['metadata']['n_samples'] / total_samples
            global_params += weight * update['params']
            
        return global_params
