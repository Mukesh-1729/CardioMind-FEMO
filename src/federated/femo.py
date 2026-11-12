import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class FEMOObjective:
    """
    CardioMind-FEMO Multi-Objective Loss with Huber Risk Regularization,
    Running Exponential Moving Average (EMA) Loss Normalization, and
    Progressive Round Scheduling:
    
    L_FEMO = \tilde{L}_pred + λc(t) * \tilde{L}_cal + λp(t) * \tilde{L}_prox + λr(t) * \tilde{L}_risk
    where \tilde{L}_j = L_j / (EMA(L_j) + ε)
    """
    def __init__(self, config: Dict[str, Any] = None):
        femo_cfg = config.get('femo', {}) if config else {}
        self.lambda_cal_max = femo_cfg.get('lambda_cal_max', 0.05)
        self.lambda_prox_base = femo_cfg.get('lambda_prox', 0.01)
        self.lambda_risk_max = femo_cfg.get('lambda_risk_max', 0.01)
        self.huber_delta = femo_cfg.get('huber_delta', 0.1)
        self.ema_decay = femo_cfg.get('ema_decay', 0.90)
        self.warmup_rounds = femo_cfg.get('warmup_rounds', 3)
        self.total_rounds = config.get('rounds', 30) if config else 30

        # Running EMA loss magnitudes initialized to reasonable tabular scales
        self.ema_losses = {
            'pred': 0.50,
            'cal': 0.18,
            'prox': 0.01,
            'risk': 0.02
        }

    def get_scheduled_lambdas(self, round_num: int) -> Tuple[float, float, float]:
        """
        Calculates progressive loss coefficients λc(t), λp(t), λr(t).
        Allows primary prediction representation to stabilize before scaling auxiliary terms.
        """
        # Smooth progressive scaling from round 1 to total_rounds
        progress = float(round_num) / float(max(1, self.total_rounds))
        
        # Proximal drift regularizer warm-up
        prox_scale = min(1.0, float(round_num) / float(max(1, self.warmup_rounds)))
        lambda_prox = self.lambda_prox_base * prox_scale

        # Calibration and Risk-consistency progressive ramp-up
        lambda_cal = self.lambda_cal_max * progress
        lambda_risk = self.lambda_risk_max * progress

        return lambda_cal, lambda_prox, lambda_risk

    def compute_pred_loss(self, y_pred_proba: torch.Tensor, y_true: torch.Tensor, pos_weight: float = 1.0) -> torch.Tensor:
        """Weighted Binary Cross-Entropy Loss."""
        eps = 1e-7
        p = torch.clamp(y_pred_proba.view(-1), eps, 1.0 - eps)
        y = y_true.view(-1).float()
        loss = - (pos_weight * y * torch.log(p) + (1.0 - y) * torch.log(1.0 - p))
        return torch.mean(loss)

    def compute_cal_loss(self, y_pred_proba: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Brier-style Calibration Loss: (1/N) * sum((p_i - y_i)^2)."""
        p = y_pred_proba.view(-1)
        y = y_true.view(-1).float()
        return torch.mean((p - y) ** 2)

    def compute_prox_loss(self, model: nn.Module, global_params: Dict[str, np.ndarray]) -> torch.Tensor:
        """Proximal drift constraint: 0.5 * ||theta_k - theta_t||^2."""
        prox_loss = torch.tensor(0.0, dtype=torch.float32, device=next(model.parameters()).device)
        for name, param in model.named_parameters():
            if name in global_params:
                glob_t = torch.tensor(global_params[name], dtype=torch.float32, device=param.device)
                prox_loss += 0.5 * torch.sum((param - glob_t) ** 2)
        return prox_loss

    def compute_risk_loss(
        self, current_risks: torch.Tensor, prev_risks: torch.Tensor, delta_features: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Bounded Huber-based Risk Consistency Loss:
        L_risk = (1/N) * sum( rho_delta(r_i,t - r_i,t-1 - g(delta x)) )
        where rho_delta is a Huber loss preventing gradient explosion from outlier fluctuations.
        """
        if prev_risks is None:
            return torch.tensor(0.0, dtype=torch.float32, device=current_risks.device)
        
        diff = current_risks.view(-1) - prev_risks.view(-1)
        target_zeros = torch.zeros_like(diff)
        return F.huber_loss(diff, target_zeros, delta=self.huber_delta, reduction='mean')

    def _update_ema(self, key: str, value: float):
        """Updates exponential moving average of running loss magnitude."""
        if value > 0:
            current_ema = self.ema_losses.get(key, value)
            self.ema_losses[key] = self.ema_decay * current_ema + (1.0 - self.ema_decay) * value

    def compute_total_loss(
        self, model: nn.Module, y_pred_proba: torch.Tensor, y_true: torch.Tensor,
        global_params: Dict[str, np.ndarray], round_num: int, prev_risks: torch.Tensor = None,
        strategy: str = 'femo'
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes normalized multi-objective loss with progressive scheduling.
        Supports ablation variants:
        - 'femo_base': L_pred only
        - 'femo_p': L_pred + L_prox
        - 'femo_pc': L_pred + L_prox + L_cal
        - 'femo_pcr': L_pred + L_prox + L_cal + L_risk
        - 'femo': Full CardioMind-FEMO with scheduled normalized objectives
        """
        lambda_cal, lambda_prox, lambda_risk = self.get_scheduled_lambdas(round_num)

        # Raw component losses
        l_pred = self.compute_pred_loss(y_pred_proba, y_true)
        l_cal = self.compute_cal_loss(y_pred_proba, y_true)
        l_prox = self.compute_prox_loss(model, global_params) if global_params is not None else torch.tensor(0.0, device=y_pred_proba.device)
        l_risk = self.compute_risk_loss(y_pred_proba, prev_risks) if prev_risks is not None else torch.tensor(0.0, device=y_pred_proba.device)

        # Update running EMA scales (detached float)
        self._update_ema('pred', float(l_pred.item()))
        self._update_ema('cal', float(l_cal.item()))
        if global_params is not None:
            self._update_ema('prox', float(l_prox.item()))
        if prev_risks is not None:
            self._update_ema('risk', float(l_risk.item()))

        # Normalized loss components
        eps = 1e-6
        norm_pred = l_pred / (self.ema_losses['pred'] + eps)
        norm_cal = l_cal / (self.ema_losses['cal'] + eps)
        norm_prox = l_prox / (self.ema_losses['prox'] + eps)
        norm_risk = l_risk / (self.ema_losses['risk'] + eps)

        if strategy == 'femo_base' or strategy == 'fedavg':
            total_loss = l_pred
        elif strategy == 'femo_p' or strategy == 'fedprox':
            total_loss = l_pred + self.lambda_prox_base * norm_prox
        elif strategy == 'femo_pc':
            total_loss = l_pred + self.lambda_prox_base * norm_prox + self.lambda_cal_max * norm_cal
        elif strategy == 'femo_pcr':
            total_loss = l_pred + self.lambda_prox_base * norm_prox + self.lambda_cal_max * norm_cal + self.lambda_risk_max * norm_risk
        else:
            # Full Flagship CardioMind-FEMO with normalized objectives and dynamic scheduling
            total_loss = norm_pred + lambda_cal * norm_cal + lambda_prox * norm_prox + lambda_risk * norm_risk

        loss_breakdown = {
            'total_loss': float(total_loss.item()),
            'l_pred_raw': float(l_pred.item()),
            'l_cal_raw': float(l_cal.item()),
            'l_prox_raw': float(l_prox.item()),
            'l_risk_raw': float(l_risk.item()),
            'lambda_cal': lambda_cal,
            'lambda_prox': lambda_prox,
            'lambda_risk': lambda_risk
        }
        return total_loss, loss_breakdown


class AdaptiveAggregator:
    """
    CardioMind-FEMO Reliability and Stability Aware Adaptive Aggregator:
    w_k = [ n_k * (1 + eta * q_k') * (1 + mu * s_k') ] / sum_j [ n_j * (1 + eta * q_j') * (1 + mu * s_j') ]
    where:
    - n_k = client sample count
    - q_k' = bounded validation quality in [0.0, 1.0] (AUC + (0.25 - Brier))
    - s_k' = bounded parameter update cosine stability in [0.0, 1.0]
    - w_k is clipped to [w_min, w_max] and renormalized.
    """
    def __init__(
        self, eta_quality: float = 0.5, mu_stability: float = 0.5,
        clip_min: float = 0.05, clip_max: float = 0.50
    ):
        self.eta_quality = eta_quality
        self.mu_stability = mu_stability
        self.clip_min = clip_min
        self.clip_max = clip_max

    def aggregate(self, client_updates: List[Dict[str, Any]]) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        if not client_updates:
            raise ValueError("No client updates to aggregate.")

        raw_weights = []
        for u in client_updates:
            n_k = u['metadata']['sample_count']
            metrics = u['metadata']['validation_metrics']
            
            # Validation Quality: AUC + (0.25 - Brier), bounded to [0.0, 1.0]
            auc = metrics.get('roc_auc', 0.5)
            brier = metrics.get('brier_score', 0.25)
            q_k = float(np.clip(auc + (0.25 - brier), 0.0, 1.0))

            # Update Stability: cosine similarity of updates, bounded to [0.0, 1.0]
            s_k = u['metadata'].get('stability', 1.0)
            s_k = float(np.clip(s_k, 0.0, 1.0))

            score = n_k * (1.0 + self.eta_quality * q_k) * (1.0 + self.mu_stability * s_k)
            raw_weights.append(score)

        # Normalize and clip extreme weights to prevent client domination
        total_score = sum(raw_weights)
        norm_weights = [w / total_score for w in raw_weights]
        clipped_weights = [float(np.clip(w, self.clip_min, self.clip_max)) for w in norm_weights]
        final_sum = sum(clipped_weights)
        final_weights = [w / final_sum for w in clipped_weights]

        weights_dict = {
            u['client_id']: round(final_weights[i], 4) for i, u in enumerate(client_updates)
        }

        # Aggregate parameters across all model tensors
        first_params = client_updates[0]['parameters']
        aggregated_params = {}

        for key in first_params.keys():
            accumulated = np.zeros_like(first_params[key], dtype=np.float32)
            for i, u in enumerate(client_updates):
                accumulated += final_weights[i] * u['parameters'][key]
            aggregated_params[key] = accumulated

        return aggregated_params, weights_dict
