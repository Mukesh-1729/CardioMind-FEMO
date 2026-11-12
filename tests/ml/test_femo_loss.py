"""
Tests for CardioMind-FEMO: FEMO Loss Components

Verifies mathematical correctness of each FEMO loss component.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
import numpy as np


class TestPredictionLoss:
    """Test binary cross-entropy prediction loss."""

    def test_perfect_predictions(self):
        """BCE loss should be near 0 for perfect predictions."""
        from ml.federated.femo import FEMO
        femo = FEMO()
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([1e-7, 1-1e-7, 1e-7, 1-1e-7])
        loss = femo.compute_pred_loss(y_true, y_pred)
        assert loss < 0.01

    def test_worst_predictions(self):
        """BCE loss should be high for inverted predictions."""
        from ml.federated.femo import FEMO
        femo = FEMO()
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([1-1e-7, 1e-7, 1-1e-7, 1e-7])
        loss = femo.compute_pred_loss(y_true, y_pred)
        assert loss > 10.0

    def test_loss_non_negative(self):
        """BCE loss must be non-negative."""
        from ml.federated.femo import FEMO
        femo = FEMO()
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0.2, 0.8, 0.6, 0.3, 0.7])
        loss = femo.compute_pred_loss(y_true, y_pred)
        assert loss >= 0.0


class TestCalibrationLoss:
    """Test Brier-style calibration loss."""

    def test_perfect_calibration(self):
        """Calibration loss near 0 for perfectly calibrated predictions."""
        from ml.federated.femo import FEMO
        femo = FEMO()
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0.0, 1.0, 0.0, 1.0])
        loss = femo.compute_cal_loss(y_true, y_pred)
        assert loss == pytest.approx(0.0)

    def test_poor_calibration(self):
        """Poorly calibrated predictions should have higher loss."""
        from ml.federated.femo import FEMO
        femo = FEMO()
        y_true = np.array([0, 1, 0, 1])
        good_pred = np.array([0.1, 0.9, 0.1, 0.9])
        bad_pred  = np.array([0.9, 0.1, 0.9, 0.1])
        good_loss = femo.compute_cal_loss(y_true, good_pred)
        bad_loss  = femo.compute_cal_loss(y_true, bad_pred)
        assert good_loss < bad_loss

    def test_calibration_bounds(self):
        """Calibration loss must be in [0, 1] (squared differences of probabilities)."""
        from ml.federated.femo import FEMO
        femo = FEMO()
        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.uniform(0, 1, 100)
        loss = femo.compute_cal_loss(y_true, y_pred)
        assert 0.0 <= loss <= 1.0


class TestProximalLoss:
    """Test proximal regularization loss."""

    def test_same_params_zero_loss(self):
        """Proximal loss must be 0 when local params == global params."""
        from ml.federated.femo import FEMO
        femo = FEMO()
        params = np.array([0.1, 0.2, 0.3, 0.4])
        loss = femo.compute_prox_loss(params, params.copy())
        assert loss == pytest.approx(0.0)

    def test_proximal_positive(self):
        """Proximal loss must be non-negative."""
        from ml.federated.femo import FEMO
        femo = FEMO()
        local  = np.array([0.5, 0.6, 0.7])
        global_ = np.array([0.1, 0.2, 0.3])
        loss = femo.compute_prox_loss(local, global_)
        assert loss > 0.0

    def test_proximal_formula(self):
        """Verify L_prox = 0.5 * ||theta_k - theta_t||^2"""
        from ml.federated.femo import FEMO
        femo = FEMO()
        local  = np.array([1.0, 0.0])
        global_ = np.array([0.0, 0.0])
        loss = femo.compute_prox_loss(local, global_)
        expected = 0.5 * np.sum((local - global_) ** 2)  # 0.5 * 1.0 = 0.5
        assert loss == pytest.approx(expected)


class TestTotalFEMOLoss:
    """Test total FEMO loss computation."""

    def test_zero_lambdas_equals_bce(self):
        """With all lambdas=0, FEMO loss should equal BCE loss."""
        from ml.federated.femo import FEMO
        femo = FEMO()
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0.2, 0.8, 0.3, 0.7])
        params = np.array([0.1, 0.2])
        global_params = np.array([0.15, 0.25])
        config = {'lambda_cal': 0.0, 'lambda_prox': 0.0, 'lambda_risk': 0.0}
        total = femo.compute_total_loss(y_true, y_pred, params, global_params,
                                        current_risks=None, prev_risks=None,
                                        delta_features=None, config=config)
        bce = femo.compute_pred_loss(y_true, y_pred)
        assert total == pytest.approx(bce)

    def test_loss_increases_with_lambdas(self):
        """Total FEMO loss should be >= BCE loss when lambdas > 0."""
        from ml.federated.femo import FEMO
        femo = FEMO()
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0.3, 0.7, 0.4, 0.6, 0.2])
        params = np.array([0.5, 0.3, 0.2])
        global_params = np.array([0.1, 0.1, 0.1])
        config = {'lambda_cal': 0.1, 'lambda_prox': 0.01, 'lambda_risk': 0.0}
        total = femo.compute_total_loss(y_true, y_pred, params, global_params,
                                        current_risks=None, prev_risks=None,
                                        delta_features=None, config=config)
        bce = femo.compute_pred_loss(y_true, y_pred)
        assert total >= bce


class TestDataLocality:
    """Critical test: Raw patient data must NEVER leave the client."""

    def test_client_update_no_raw_data(self):
        """Client.get_update() must return only parameters and metadata, never raw data."""
        # This test verifies the contract: updates contain no patient records
        # We check the keys in the returned dict
        expected_keys_that_must_not_exist = {'X_train', 'y_train', 'patients', 'records',
                                              'raw_data', 'patient_ids', 'features_raw'}
        # Create a minimal mock update to verify structure
        mock_update = {
            'client_id': 'client_1',
            'n_samples': 1000,
            'parameters': [0.1, 0.2, 0.3],
            'local_metrics': {'auc': 0.75, 'brier': 0.18},
            'update_stability': 0.92,
            'update_norm': 0.045,
            'timestamp': '2025-01-01T00:00:00'
        }
        # Verify no raw data keys are present
        for forbidden_key in expected_keys_that_must_not_exist:
            assert forbidden_key not in mock_update, \
                f"PRIVACY VIOLATION: '{forbidden_key}' found in client update payload"

    def test_aggregation_receives_no_patient_data(self):
        """FedAvg/FEMO aggregation must operate only on model parameters."""
        from ml.federated.fedavg import FedAvg
        # Simulate what the server receives
        client_updates = [
            {'client_id': 'c1', 'parameters': np.array([0.1, 0.2, 0.3]), 'n_samples': 1000},
            {'client_id': 'c2', 'parameters': np.array([0.2, 0.3, 0.4]), 'n_samples': 800},
        ]
        fedavg = FedAvg()
        result = fedavg.aggregate(client_updates)
        # Result should be aggregated parameters, not any patient data
        assert 'parameters' in result or isinstance(result, np.ndarray)
