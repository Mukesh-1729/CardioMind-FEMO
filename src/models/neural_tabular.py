import os
import copy
import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class NeuralTabularModel(nn.Module):
    """
    Federatable Neural Tabular Architecture for Cardiovascular Risk Assessment:
    Input Features
          ↓
    Dense (e.g. 64)
          ↓
    BatchNorm1d
          ↓
    ReLU
          ↓
    Dropout (0.2)
          ↓
    Dense (e.g. 32)
          ↓
    ReLU
          ↓
    Dense (1)
          ↓
    Sigmoid
          ↓
    CVD Probability
    """
    def __init__(self, input_dim: int, hidden_dim_1: int = 64, hidden_dim_2: int = 32, dropout_rate: float = 0.2):
        super(NeuralTabularModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2

        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.bn1 = nn.BatchNorm1d(hidden_dim_1)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(hidden_dim_2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If batch size is 1, batchnorm needs eval mode or bypass
        if x.size(0) == 1 and self.training:
            self.eval()
            out = self.sigmoid(self.forward_logits(x))
            self.train()
            return out
        return self.sigmoid(self.forward_logits(x))

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc1(x)
        if x.size(0) > 1:
            out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.relu2(out)
        
        logits = self.fc3(out)
        return logits

    def get_parameter_dict(self) -> Dict[str, np.ndarray]:
        """Exports state dict as serializable NumPy dictionary for federated transport."""
        return {k: v.cpu().detach().clone().numpy() for k, v in self.state_dict().items()}

    def set_parameter_dict(self, param_dict: Dict[str, np.ndarray]):
        """Loads parameters from NumPy dictionary."""
        state_dict = {}
        for k, v in param_dict.items():
            state_dict[k] = torch.tensor(v, dtype=torch.float32)
        self.load_state_dict(state_dict)

    def get_flat_parameters(self) -> np.ndarray:
        """Returns 1D flattened vector of trainable parameters for cosine similarity."""
        params = []
        for p in self.parameters():
            if p.requires_grad:
                params.append(p.cpu().detach().view(-1).numpy())
        return np.concatenate(params) if params else np.array([])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """NumPy prediction wrapper."""
        self.eval()
        with torch.no_grad():
            tensor_X = torch.tensor(X, dtype=torch.float32)
            probas = self.forward(tensor_X).cpu().numpy().flatten()
        return probas
