import os
import logging
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class NonIIDPartitioner:
    """
    Partitions the training set across 5 simulated hospital clients with
    deliberate, algorithmically generated Non-IID distributions:
    - Client 1: Predominantly lower-risk population
    - Client 2: Predominantly higher BP / hypertensive skew
    - Client 3: Older age cohort skew
    - Client 4: High cholesterol / metabolic risk skew
    - Client 5: Heterogeneous mixed demographic
    """
    def __init__(self, n_clients: int = 5):
        self.n_clients = n_clients

    def partition_training_data(
        self, X_train: np.ndarray, y_train: np.ndarray, feature_names: List[str], seed: int = 42
    ) -> List[Dict[str, Any]]:
        np.random.seed(seed)
        N = len(y_train)
        indices = np.arange(N)

        # Map key feature indices for non-iid skewing if present in feature names
        name_to_idx = {name.lower(): i for i, name in enumerate(feature_names)}
        
        sbp_idx = name_to_idx.get('ap_hi') or name_to_idx.get('systolic_bp') or 0
        age_idx = name_to_idx.get('age') or 0
        chol_idx = name_to_idx.get('cholesterol') or name_to_idx.get('cholesterol_3') or 0

        # Compute sampling weights for each hospital
        # Base uniform weight
        w1 = np.exp(-1.2 * y_train)  # Hospital 1: lower-risk
        w2 = np.exp(1.5 * (X_train[:, sbp_idx] > 0.5).astype(float) + 0.8 * y_train)  # Hospital 2: high BP
        w3 = np.exp(1.5 * (X_train[:, age_idx] > 0.5).astype(float))  # Hospital 3: older age
        w4 = np.exp(1.5 * (X_train[:, chol_idx] > 0.5).astype(float) + 0.5 * y_train)  # Hospital 4: metabolic
        w5 = np.ones(N)  # Hospital 5: mixed

        # Normalize probability distributions
        weights_matrix = np.column_stack([w1, w2, w3, w4, w5])
        # Add small Dirichlet noise for realistic variance across seeds
        dirichlet_noise = np.random.dirichlet([0.6] * 5, size=N)
        weights_matrix = weights_matrix * dirichlet_noise
        client_assignments = np.argmax(weights_matrix, axis=1)

        clients = []
        stats_rows = []

        hospital_names = [
            "Hospital 1 (Metro Prevention)",
            "Hospital 2 (CardioVascular Specialty)",
            "Hospital 3 (Geriatric Care Center)",
            "Hospital 4 (Metabolic & Lipid Clinic)",
            "Hospital 5 (Community General)"
        ]

        for i in range(self.n_clients):
            c_mask = (client_assignments == i)
            # Ensure each client gets at least some samples
            if c_mask.sum() < 100:
                c_idx = np.array_split(indices, self.n_clients)[i]
            else:
                c_idx = indices[c_mask]

            c_X = X_train[c_idx]
            c_y = y_train[c_idx]
            pos_rate = float(c_y.mean()) if len(c_y) > 0 else 0.0

            client_info = {
                'client_id': f"Hospital_{i+1}",
                'hospital_name': hospital_names[i],
                'X': c_X,
                'y': c_y,
                'sample_count': len(c_y),
                'positive_rate': pos_rate,
                'indices': c_idx
            }
            clients.append(client_info)

            stats_rows.append({
                'client_id': f"Hospital_{i+1}",
                'hospital_name': hospital_names[i],
                'sample_count': len(c_y),
                'positive_rate': round(pos_rate, 4),
                'pos_samples': int(c_y.sum()),
                'neg_samples': int(len(c_y) - c_y.sum())
            })

            logger.info(f"Generated {hospital_names[i]}: {len(c_y)} samples (positive rate: {pos_rate:.1%})")

        self.client_stats_df = pd.DataFrame(stats_rows)
        return clients

    def save_distribution_report(self, output_path: str = 'results/client_distribution.csv'):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if hasattr(self, 'client_stats_df'):
            self.client_stats_df.to_csv(output_path, index=False)
            logger.info(f"Saved Non-IID client distribution report to {output_path}")
