import logging
import copy
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss

from ..models.neural_tabular import NeuralTabularModel
from .fedavg import FedAvgAggregator
from .fedprox import FedProxAggregator
from .femo import AdaptiveAggregator

logger = logging.getLogger(__name__)

def calculate_ece(y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_pred_proba[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return float(ece)

class FederatedServer:
    """
    Federation Server orchestrating global neural model broadcast,
    adaptive multi-objective aggregation, per-round validation tracking,
    and locked test set evaluation on the validation-selected best checkpoint.
    """
    def __init__(self, input_dim: int, config: Dict[str, Any]):
        self.config = config
        hidden_1 = config.get('model', {}).get('hidden_dim_1', 64)
        hidden_2 = config.get('model', {}).get('hidden_dim_2', 32)
        dropout = config.get('model', {}).get('dropout_rate', 0.2)

        self.global_model = NeuralTabularModel(input_dim=input_dim, hidden_dim_1=hidden_1, hidden_dim_2=hidden_2, dropout_rate=dropout)
        
        # Aggregators
        self.fedavg_agg = FedAvgAggregator()
        self.fedprox_agg = FedProxAggregator()
        femo_cfg = config.get('femo', {})
        self.adaptive_agg = AdaptiveAggregator(
            eta_quality=femo_cfg.get('eta_quality', 0.5),
            mu_stability=femo_cfg.get('mu_stability', 0.5),
            clip_min=femo_cfg.get('weight_clip_min', 0.05),
            clip_max=femo_cfg.get('weight_clip_max', 0.50)
        )

        self.val_weights = femo_cfg.get('val_weights', {'auc': 1.0, 'brier': 0.5, 'ece': 0.5})
        self.round_history = []
        self.best_checkpoint_params = None
        self.best_round = 1
        self.best_val_score = -float('inf')

    def get_global_parameters(self) -> Dict[str, np.ndarray]:
        return self.global_model.get_parameter_dict()

    def aggregate_updates(
        self, client_updates: List[Dict[str, Any]], strategy: str
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        if strategy in ['fedavg', 'femo_base', 'femo_p', 'femo_pc', 'femo_pcr']:
            new_params = self.fedavg_agg.aggregate(client_updates)
            weights_dict = {u['client_id']: round(1.0 / len(client_updates), 4) for u in client_updates}
        elif strategy == 'fedprox':
            new_params = self.fedprox_agg.aggregate(client_updates)
            weights_dict = {u['client_id']: round(1.0 / len(client_updates), 4) for u in client_updates}
        else:  # Flagship CardioMind-FEMO with Adaptive Aggregation
            new_params, weights_dict = self.adaptive_agg.aggregate(client_updates)

        self.global_model.set_parameter_dict(new_params)
        return new_params, weights_dict

    def evaluate_global_model(
        self, X_eval: np.ndarray, y_eval: np.ndarray, clients: List = None
    ) -> Dict[str, float]:
        preds = self.global_model.predict_proba(X_eval)
        pred_labels = (preds >= 0.5).astype(int)

        try:
            auc = float(roc_auc_score(y_eval, preds))
        except Exception:
            auc = 0.5

        acc = float(accuracy_score(y_eval, pred_labels))
        prec = float(precision_score(y_eval, pred_labels, zero_division=0))
        rec = float(recall_score(y_eval, pred_labels, zero_division=0))
        f1 = float(f1_score(y_eval, pred_labels, zero_division=0))
        brier = float(brier_score_loss(y_eval, preds))
        ece = float(calculate_ece(y_eval, preds))

        metrics = {
            'roc_auc': auc,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'brier_score': brier,
            'ece': ece
        }

        # Measure client variance and worst-client performance on their local validations
        if clients:
            client_aucs = []
            for c in clients:
                c_preds = self.global_model.predict_proba(c.X_val)
                try:
                    c_auc = float(roc_auc_score(c.y_val, c_preds))
                except Exception:
                    c_auc = 0.5
                client_aucs.append(c_auc)

            metrics['worst_client_performance'] = float(np.min(client_aucs))
            metrics['client_variance'] = float(np.std(client_aucs))
            metrics['client_aucs'] = client_aucs

        return metrics

    def compute_val_selection_score(self, val_metrics: Dict[str, float]) -> float:
        """
        Computes composite validation score:
        S_val = w_auc * AUC - w_brier * Brier - w_ece * ECE
        """
        w_auc = self.val_weights.get('auc', 1.0)
        w_brier = self.val_weights.get('brier', 0.5)
        w_ece = self.val_weights.get('ece', 0.5)
        return float(w_auc * val_metrics['roc_auc'] - w_brier * val_metrics['brier_score'] - w_ece * val_metrics['ece'])

    def run_simulation(
        self, clients: List, X_val: np.ndarray, y_val: np.ndarray,
        X_test: np.ndarray, y_test: np.ndarray, strategy: str = 'femo'
    ) -> List[Dict[str, Any]]:
        """
        Runs federated rounds, evaluates validation set for checkpoint selection,
        tracks convergence metrics, and loads best validation checkpoint for final evaluation.
        """
        n_rounds = self.config.get('rounds', 30)
        self.round_history = []
        self.best_val_score = -float('inf')
        self.best_checkpoint_params = None
        self.best_round = 1

        # Model size in KB for communication cost calculation
        param_bytes = sum(p.nbytes for p in self.get_global_parameters().values())
        param_kb = param_bytes / 1024.0
        cum_comm_cost = 0.0

        for r in range(1, n_rounds + 1):
            global_params = self.get_global_parameters()
            
            client_updates = []
            for client in clients:
                client.receive_global_model(global_params)
                client.train_local(strategy=strategy, round_num=r)
                client_updates.append(client.get_update())

            # Server aggregation
            new_params, weights_dict = self.aggregate_updates(client_updates, strategy=strategy)

            # 1. Server-side Validation Evaluation (For Checkpoint Selection)
            val_metrics = self.evaluate_global_model(X_val, y_val, clients=clients)
            val_score = self.compute_val_selection_score(val_metrics)

            if val_score > self.best_val_score:
                self.best_val_score = val_score
                self.best_round = r
                self.best_checkpoint_params = copy.deepcopy(new_params)

            # 2. Test Set Evaluation (Trajectory Logging Only, Never used for Selection)
            test_metrics = self.evaluate_global_model(X_test, y_test, clients=clients)

            # Communication accounting
            round_comm_cost = 2.0 * param_kb * len(clients)
            cum_comm_cost += round_comm_cost
            test_metrics['communication_cost_kb'] = cum_comm_cost
            test_metrics['round'] = r
            test_metrics['strategy'] = strategy
            test_metrics['weights'] = weights_dict
            test_metrics['val_roc_auc'] = val_metrics['roc_auc']
            test_metrics['val_brier'] = val_metrics['brier_score']
            test_metrics['val_ece'] = val_metrics['ece']
            test_metrics['val_score'] = val_score

            self.round_history.append(test_metrics)

            if r == 1 or r % 5 == 0 or r == n_rounds:
                logger.info(
                    f"[{strategy.upper()} Round {r:02d}/{n_rounds:02d}] "
                    f"Val AUC: {val_metrics['roc_auc']:.4f} | "
                    f"Test AUC: {test_metrics['roc_auc']:.4f} | "
                    f"Acc: {test_metrics['accuracy']:.4f} | "
                    f"F1: {test_metrics['f1']:.4f} | "
                    f"Brier: {test_metrics['brier_score']:.4f} | "
                    f"ECE: {test_metrics['ece']:.4f} | "
                    f"Worst Client: {test_metrics.get('worst_client_performance', 0):.4f}"
                )

        # Restore optimal model checkpoint selected exclusively by validation score
        if self.best_checkpoint_params is not None:
            logger.info(f"[{strategy.upper()}] Selecting best checkpoint from Round {self.best_round} (Val Score: {self.best_val_score:.4f})")
            self.global_model.set_parameter_dict(self.best_checkpoint_params)

        return self.round_history
