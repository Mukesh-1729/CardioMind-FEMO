import logging
import json
from pathlib import Path
import numpy as np
from typing import Dict, Any, List
from .fedavg import FedAvg
from .fedprox import FedProx
from .femo import FEMO
from ..evaluation.metrics import MetricsCalculator

logger = logging.getLogger(__name__)

def sigmoid(z):
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))

class FederatedServer:
    def __init__(self, config: Dict[str, Any], n_clients: int):
        self.config = config
        self.n_clients = n_clients
        self.global_params = None
        self.metrics_calc = MetricsCalculator()
        self.round_logs = []
        
    def initialize_global_model(self, dim: int):
        """Initializes global parameter vector (D weights + 1 intercept)."""
        self.global_params = np.zeros(dim + 1)
        
    def broadcast_global_model(self) -> np.ndarray:
        return np.copy(self.global_params)
        
    def collect_updates(self, client_updates: List[Dict]) -> List[Dict]:
        """Validates and filters incoming client model updates."""
        valid_updates = []
        for update in client_updates:
            if 'params' in update and 'client_id' in update and update['params'] is not None:
                valid_updates.append(update)
            else:
                logger.warning(f"Invalid update from {update.get('client_id', 'unknown')}")
        return valid_updates
        
    def aggregate(self, updates: List[Dict], strategy: str = 'fedavg') -> np.ndarray:
        if not updates:
            return self.global_params
            
        if strategy == 'fedavg':
            aggregator = FedAvg()
            self.global_params = aggregator.aggregate(updates)
        elif strategy == 'fedprox':
            aggregator = FedProx()
            self.global_params = aggregator.aggregate(updates)
        elif strategy == 'femo':
            aggregator = FEMO()
            femo_cfg = self.config.get('federated', {}).get('femo', {})
            self.global_params = aggregator.aggregate(updates, femo_cfg)
        else:
            aggregator = FedAvg()
            self.global_params = aggregator.aggregate(updates)
            
        return self.global_params
        
    def evaluate_global_model(self, test_data: Dict[str, np.ndarray], clients: List = None) -> Dict[str, float]:
        X = test_data['X']
        y = test_data['y']
        D = X.shape[1]
        
        w = self.global_params[:D]
        b = self.global_params[D]
        
        z = np.dot(X, w) + b
        preds = sigmoid(z)
        
        metrics = self.metrics_calc.calculate(y, preds)
        metrics['ece'] = self.metrics_calc.calculate_ece(y, preds)
        
        # Calculate per-client performance on validation sets
        if clients:
            client_aucs = []
            for client in clients:
                if client.X_val is not None and len(client.y_val) > 0:
                    z_c = np.dot(client.X_val, w) + b
                    p_c = sigmoid(z_c)
                    try:
                        from sklearn.metrics import roc_auc_score
                        auc_c = float(roc_auc_score(client.y_val, p_c))
                    except Exception:
                        auc_c = 0.5
                    client_aucs.append(auc_c)
                    
            if client_aucs:
                metrics['worst_client_performance'] = float(np.min(client_aucs))
                metrics['client_variance'] = float(np.std(client_aucs))
                metrics['client_aucs'] = client_aucs
                
        return metrics
        
    def log_round(self, round_num: int, metrics: Dict[str, Any], client_metadata: List[Dict]):
        log_entry = {
            'round': round_num,
            'metrics': metrics,
            'clients': client_metadata
        }
        self.round_logs.append(log_entry)
        
    def run_simulation(self, n_rounds: int, strategy: str, clients: List, test_data: Dict[str, np.ndarray]):
        D = test_data['X'].shape[1]
        self.initialize_global_model(dim=D)
        self.round_logs = []
        
        # Load local data for each client if not loaded
        for client in clients:
            if client.X_train is None:
                client.load_local_data()
                
        for r in range(n_rounds):
            global_params = self.broadcast_global_model()
            
            client_updates = []
            for client in clients:
                client.receive_global_model(global_params)
                client.train_local(strategy=strategy, config=self.config)
                client_updates.append(client.get_update())
                
            valid_updates = self.collect_updates(client_updates)
            self.aggregate(valid_updates, strategy=strategy)
            
            metrics = self.evaluate_global_model(test_data, clients=clients)
            client_metadata = [u['metadata'] for u in client_updates]
            self.log_round(r + 1, metrics, client_metadata)
            
            logger.info(f"[{strategy.upper()} Round {r+1:02d}/{n_rounds:02d}] Global AUC: {metrics['roc_auc']:.4f} | Acc: {metrics['accuracy']:.4f} | Brier: {metrics['brier_score']:.4f} | ECE: {metrics['ece']:.4f} | Worst Client: {metrics.get('worst_client_performance', 0):.4f}")
            
        return self.round_logs
