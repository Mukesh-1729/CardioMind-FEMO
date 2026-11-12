import logging
import time
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss

logger = logging.getLogger(__name__)

def sigmoid(z):
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))

class FederatedClient:
    """Simulated hospital client executing local federated training."""
    
    def __init__(self, client_id: str, data_path: str, config: Dict[str, Any]):
        self.client_id = client_id
        self.data_path = data_path
        self.config = config
        self.local_params = None  # Shape (D+1,): weights w and intercept b
        self.prev_params = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.n_samples = 0
        self.last_metrics = {}
        
    def load_local_data(self):
        """Loads client local data and splits 15% for local validation."""
        data = np.load(self.data_path)
        X = data['X']
        y = data['y']
        self.n_samples = len(y)
        
        # Local train/val split for validation reliability estimation
        if len(y) > 20:
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                X, y, test_size=0.15, stratify=y, random_state=42
            )
        else:
            self.X_train, self.X_val = X, X
            self.y_train, self.y_val = y, y
            
        logger.info(f"Client {self.client_id} loaded {self.n_samples} samples ({len(self.X_train)} train, {len(self.X_val)} val).")
        
    def receive_global_model(self, global_params: np.ndarray):
        """Updates local model parameters from server broadcast."""
        self.prev_params = np.copy(self.local_params) if self.local_params is not None else None
        self.local_params = np.copy(global_params)
        
    def train_local(self, strategy: str = 'fedavg', config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Performs local SGD training on private partition."""
        if config is None:
            config = self.config
            
        fed_cfg = config.get('federated', {})
        local_epochs = fed_cfg.get('local_epochs', 3)
        mu = fed_cfg.get('mu', 0.01)
        lr = fed_cfg.get('lr', 0.05)
        
        femo_cfg = fed_cfg.get('femo', {
            'lambda_cal': 0.1,
            'lambda_prox': 0.01,
            'lambda_risk': 0.05
        })
        
        X = self.X_train
        y = self.y_train
        N = len(y)
        D = X.shape[1]
        
        # Initialize if not yet set
        if self.local_params is None or len(self.local_params) != D + 1:
            self.local_params = np.zeros(D + 1)
            
        w = self.local_params[:D].copy()
        b = self.local_params[D]
        
        global_w = self.local_params[:D].copy()
        global_b = self.local_params[D]
        
        # Local mini-batch / full-batch gradient descent
        batch_size = min(256, N)
        n_batches = int(np.ceil(N / batch_size))
        
        for epoch in range(local_epochs):
            indices = np.random.permutation(N)
            for i in range(n_batches):
                b_idx = indices[i * batch_size : (i + 1) * batch_size]
                X_b, y_b = X[b_idx], y[b_idx]
                n_b = len(y_b)
                
                # Forward pass
                z = np.dot(X_b, w) + b
                p = sigmoid(z)
                
                # 1. Prediction loss gradient (BCE)
                # dL_pred/dz = p - y
                err = p - y_b
                grad_w = np.dot(X_b.T, err) / n_b
                grad_b = np.sum(err) / n_b
                
                # 2. Strategy specific regularizations
                if strategy == 'fedprox':
                    # Proximal term: (mu / 2) * ||w - global_w||^2
                    grad_w += mu * (w - global_w)
                    grad_b += mu * (b - global_b)
                    
                elif strategy == 'femo':
                    # Calibration loss gradient: Brier (p - y)^2
                    # dL_cal/dz = 2 * (p - y) * p * (1 - p)
                    d_cal = 2.0 * err * p * (1.0 - p)
                    grad_cal_w = np.dot(X_b.T, d_cal) / n_b
                    grad_cal_b = np.sum(d_cal) / n_b
                    
                    # Proximal drift gradient
                    grad_prox_w = (w - global_w)
                    grad_prox_b = (b - global_b)
                    
                    # Personalized risk stability gradient
                    grad_risk_w = np.sign(w - (self.prev_params[:D] if self.prev_params is not None else global_w))
                    
                    lambda_cal = femo_cfg.get('lambda_cal', 0.1)
                    lambda_prox = femo_cfg.get('lambda_prox', 0.01)
                    lambda_risk = femo_cfg.get('lambda_risk', 0.05)
                    
                    grad_w += lambda_cal * grad_cal_w + lambda_prox * grad_prox_w + lambda_risk * grad_risk_w
                    grad_b += lambda_cal * grad_cal_b + lambda_prox * grad_prox_b
                
                # Update parameters
                w -= lr * grad_w
                b -= lr * grad_b
                
        self.local_params = np.append(w, b)
        self.last_metrics = self.calculate_validation_metrics()
        return self.last_metrics
        
    def calculate_validation_metrics(self) -> Dict[str, float]:
        """Calculates local validation metrics (AUC, Accuracy, Brier)."""
        if self.X_val is None or self.y_val is None or self.local_params is None:
            return {'roc_auc': 0.5, 'accuracy': 0.5, 'brier_score': 0.25}
            
        D = self.X_val.shape[1]
        w = self.local_params[:D]
        b = self.local_params[D]
        
        z = np.dot(self.X_val, w) + b
        p = sigmoid(z)
        y_pred = (p >= 0.5).astype(int)
        
        try:
            auc = float(roc_auc_score(self.y_val, p))
        except Exception:
            auc = 0.5
            
        acc = float(accuracy_score(self.y_val, y_pred))
        brier = float(brier_score_loss(self.y_val, p))
        
        return {
            'roc_auc': auc,
            'accuracy': acc,
            'brier_score': brier
        }
        
    def calculate_update_stability(self) -> float:
        """Calculates parameter update cosine stability sk."""
        if self.prev_params is None or self.local_params is None:
            return 1.0
        norm_p = np.linalg.norm(self.prev_params)
        norm_c = np.linalg.norm(self.local_params)
        if norm_p < 1e-8 or norm_c < 1e-8:
            return 1.0
        cos_sim = float(np.dot(self.prev_params, self.local_params) / (norm_p * norm_c))
        return float(np.clip(cos_sim, 0.1, 1.0))
        
    def get_update(self) -> Dict[str, Any]:
        """Returns model parameters and metadata (No raw patient data)."""
        return {
            'client_id': self.client_id,
            'params': np.copy(self.local_params),
            'metadata': {
                'n_samples': self.n_samples,
                'local_metrics': self.last_metrics or self.calculate_validation_metrics(),
                'stability': self.calculate_update_stability(),
                'timestamp': time.time()
            }
        }
