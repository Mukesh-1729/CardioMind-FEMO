import logging
from typing import Dict, Any, List
import numpy as np

logger = logging.getLogger(__name__)

class FedAvgAggregator:
    """
    Standard Federated Averaging Aggregator:
    w_k = n_k / sum_j(n_j)
    """
    def aggregate(self, client_updates: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        if not client_updates:
            raise ValueError("No client updates provided.")

        total_samples = sum([u['metadata']['sample_count'] for u in client_updates])
        weights = [u['metadata']['sample_count'] / float(total_samples) for u in client_updates]

        first_params = client_updates[0]['parameters']
        aggregated_params = {}

        for key in first_params.keys():
            accumulated = np.zeros_like(first_params[key], dtype=np.float32)
            for i, u in enumerate(client_updates):
                accumulated += weights[i] * u['parameters'][key]
            aggregated_params[key] = accumulated

        return aggregated_params
