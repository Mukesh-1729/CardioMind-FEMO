import logging
import numpy as np
from typing import List, Dict

logger = logging.getLogger(__name__)

class FedAvg:
    def aggregate(self, client_updates: List[Dict]) -> np.ndarray:
        total_samples = sum([u['metadata']['n_samples'] for u in client_updates])
        
        if total_samples == 0:
            return None
            
        global_params = np.zeros_like(client_updates[0]['params'])
        
        for update in client_updates:
            weight = update['metadata']['n_samples'] / total_samples
            logger.info(f"Client {update['client_id']} weight: {weight:.4f}")
            global_params += weight * update['params']
            
        return global_params
