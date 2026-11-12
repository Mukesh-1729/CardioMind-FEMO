import logging
import time
import copy
from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss

from ..models.neural_tabular import NeuralTabularModel
from .femo import FEMOObjective
from .fedprox import FedProxObjective

logger = logging.getLogger(__name__)

class FederatedClient:
    """
    Simulated Hospital Client executing local federated neural training
    with support for CardioMind-FEMO multi-objective optimization,
    FedAvg, and FedProx.
    """
    def __init__(self, client_id: str, hospital_name: str, X: np.ndarray, y: np.ndarray, config: Dict[str, Any]):
        self.client_id = client_id
        self.hospital_name = hospital_name
        self.config = config
        self.sample_count = len(y)

        # Split 15% of local partition for local validation quality measurement (qk)
        if len(y) > 30:
            from sklearn.model_selection import train_test_split
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                X, y, test_size=0.15, stratify=y, random_state=42
            )
        else:
            self.X_train, self.X_val = X, X
            self.y_train, self.y_val = y, y

        input_dim = X.shape[1]
        hidden_1 = config.get('model', {}).get('hidden_dim_1', 64)
        hidden_2 = config.get('model', {}).get('hidden_dim_2', 32)
        dropout = config.get('model', {}).get('dropout_rate', 0.2)

        self.model = NeuralTabularModel(input_dim=input_dim, hidden_dim_1=hidden_1, hidden_dim_2=hidden_2, dropout_rate=dropout)
        self.global_params = None
        self.prev_global_params = None
        self.prev_update = None
        self.last_metrics = {}

        self.femo_obj = FEMOObjective(config)
        self.fedprox_obj = FedProxObjective(mu=config.get('femo', {}).get('lambda_prox', 0.01))

    def receive_global_model(self, global_params: Dict[str, np.ndarray]):
        """Sets local model parameters from global broadcast."""
        self.prev_global_params = copy.deepcopy(self.global_params) if self.global_params is not None else None
        self.global_params = copy.deepcopy(global_params)
        self.model.set_parameter_dict(global_params)

    def _get_flat_params(self, param_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Flattens a parameter dictionary into a 1D numpy array."""
        return np.concatenate([v.flatten() for v in param_dict.values()])

    def train_local(self, strategy: str = 'femo', round_num: int = 1) -> Dict[str, Any]:
        """Executes local epochs of training on private partition."""
        local_epochs = self.config.get('local_epochs', 5)
        batch_size = self.config.get('batch_size', 128)
        lr = self.config.get('learning_rate', 0.001)

        dataset = TensorDataset(
            torch.tensor(self.X_train, dtype=torch.float32),
            torch.tensor(self.y_train, dtype=torch.float32)
        )
        loader = DataLoader(dataset, batch_size=min(batch_size, len(self.X_train)), shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)

        # Frozen snapshot of baseline model for temporal risk consistency
        prior_model = None
        if strategy in ['femo', 'femo_pcr']:
            prior_model = copy.deepcopy(self.model)
            prior_model.eval()

        self.model.train()

        for epoch in range(local_epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                pred_proba = self.model(batch_x)

                prev_risks = None
                if prior_model is not None:
                    with torch.no_grad():
                        prev_risks = prior_model(batch_x).detach()

                if strategy == 'fedavg':
                    loss = self.femo_obj.compute_pred_loss(pred_proba, batch_y)
                elif strategy == 'fedprox':
                    loss = self.fedprox_obj.compute_loss(pred_proba, batch_y, self.model, self.global_params)
                else:
                    # FEMO and ablation variants (femo_base, femo_p, femo_pc, femo_pcr, femo)
                    loss, _ = self.femo_obj.compute_total_loss(
                        model=self.model,
                        y_pred_proba=pred_proba,
                        y_true=batch_y,
                        global_params=self.global_params,
                        round_num=round_num,
                        prev_risks=prev_risks,
                        strategy=strategy
                    )

                loss.backward()
                optimizer.step()

        self.last_metrics = self.evaluate_local()
        return self.last_metrics

    def evaluate_local(self) -> Dict[str, float]:
        """Calculates local validation metrics (qk estimation)."""
        self.model.eval()
        with torch.no_grad():
            tensor_X = torch.tensor(self.X_val, dtype=torch.float32)
            preds = self.model(tensor_X).cpu().numpy().flatten()

        y_val = self.y_val
        pred_labels = (preds >= 0.5).astype(int)

        try:
            auc = float(roc_auc_score(y_val, preds))
        except Exception:
            auc = 0.5

        acc = float(accuracy_score(y_val, pred_labels))
        brier = float(brier_score_loss(y_val, preds))

        return {
            'roc_auc': auc,
            'accuracy': acc,
            'brier_score': brier
        }

    def compute_update_stability(self) -> float:
        """
        Calculates parameter update cosine stability sk = cos_sim(Δθ_k^{(t)}, Δθ_k^{(t-1)})
        where Δθ_k = θ_k - θ_global.
        """
        if self.global_params is None:
            return 1.0

        curr_flat = self._get_flat_params(self.model.get_parameter_dict())
        glob_flat = self._get_flat_params(self.global_params)
        curr_delta = curr_flat - glob_flat

        if self.prev_update is None:
            self.prev_update = curr_delta
            return 1.0

        norm_curr = np.linalg.norm(curr_delta)
        norm_prev = np.linalg.norm(self.prev_update)
        if norm_curr < 1e-8 or norm_prev < 1e-8:
            self.prev_update = curr_delta
            return 1.0

        cos_sim = float(np.dot(curr_delta, self.prev_update) / (norm_curr * norm_prev))
        self.prev_update = curr_delta
        return float(np.clip(cos_sim, 0.0, 1.0))

    def get_update(self) -> Dict[str, Any]:
        """Returns local model weights and privacy-preserving metadata."""
        return {
            'client_id': self.client_id,
            'parameters': self.model.get_parameter_dict(),
            'metadata': {
                'sample_count': self.sample_count,
                'validation_metrics': self.last_metrics or self.evaluate_local(),
                'stability': self.compute_update_stability(),
                'timestamp': time.time()
            }
        }
