import logging
from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
from .fedavg import FedAvgAggregator

logger = logging.getLogger(__name__)

class FedProxObjective:
    """
    FedProx Local Objective:
    L_FedProx = L_pred + (mu / 2) * ||theta_k - theta_t||^2
    """
    def __init__(self, mu: float = 0.01):
        self.mu = mu

    def compute_loss(
        self, y_pred_proba: torch.Tensor, y_true: torch.Tensor,
        model: nn.Module, global_params: Dict[str, np.ndarray]
    ) -> torch.Tensor:
        eps = 1e-7
        p = torch.clamp(y_pred_proba.view(-1), eps, 1.0 - eps)
        y = y_true.view(-1).float()
        l_pred = - torch.mean(y * torch.log(p) + (1.0 - y) * torch.log(1.0 - p))

        prox_term = torch.tensor(0.0, dtype=torch.float32, device=next(model.parameters()).device)
        if global_params is not None:
            for name, param in model.named_parameters():
                if name in global_params:
                    glob_t = torch.tensor(global_params[name], dtype=torch.float32, device=param.device)
                    prox_term += 0.5 * self.mu * torch.sum((param - glob_t) ** 2)

        return l_pred + prox_term


class FedProxAggregator(FedAvgAggregator):
    """FedProx uses sample-weighted FedAvg aggregation."""
    pass
