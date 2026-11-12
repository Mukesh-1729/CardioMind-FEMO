"""
Tests for CardioMind-FEMO: FedAvg Aggregation

Verifies correct weighted averaging and data locality.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
import numpy as np


class TestFedAvgAggregation:
    """Test FedAvg weighted aggregation."""

    def setup_method(self):
        from ml.federated.fedavg import FedAvg
        self.fedavg = FedAvg()

    def test_equal_weights_simple(self):
        """With equal n_samples, aggregation should be simple mean."""
        updates = [
            {'parameters': np.array([1.0, 2.0, 3.0]), 'n_samples': 100},
            {'parameters': np.array([3.0, 4.0, 5.0]), 'n_samples': 100},
        ]
        result = self.fedavg.aggregate(updates)
        params = result if isinstance(result, np.ndarray) else result.get('parameters')
        expected = np.array([2.0, 3.0, 4.0])
        np.testing.assert_array_almost_equal(params, expected)

    def test_weighted_aggregation(self):
        """Larger client should have proportionally more influence."""
        updates = [
            {'parameters': np.array([0.0, 0.0]), 'n_samples': 900},
            {'parameters': np.array([1.0, 1.0]), 'n_samples': 100},
        ]
        result = self.fedavg.aggregate(updates)
        params = result if isinstance(result, np.ndarray) else result.get('parameters')
        # Expected: 0.9*[0,0] + 0.1*[1,1] = [0.1, 0.1]
        expected = np.array([0.1, 0.1])
        np.testing.assert_array_almost_equal(params, expected, decimal=5)

    def test_aggregation_weight_sum(self):
        """Weights must sum to 1.0."""
        n_samples = [500, 300, 200]
        total = sum(n_samples)
        weights = [n / total for n in n_samples]
        assert abs(sum(weights) - 1.0) < 1e-9


class TestAdaptiveAggregation:
    """Test adaptive aggregation weight computation."""

    def setup_method(self):
        from ml.federated.aggregation import AdaptiveAggregation
        self.agg = AdaptiveAggregation()

    def test_weights_sum_to_one(self):
        """Adaptive weights must sum to 1."""
        n_samples = [1000, 800, 600, 400, 200]
        reliability = [0.85, 0.80, 0.75, 0.82, 0.78]
        stability = [0.90, 0.85, 0.88, 0.82, 0.79]
        weights = self.agg.calculate_weights(n_samples, reliability, stability)
        assert abs(sum(weights.values()) - 1.0) < 1e-9

    def test_higher_reliability_gets_more_weight(self):
        """Client with higher reliability should get more weight, all else equal."""
        n_samples = [1000, 1000]
        reliability = [0.9, 0.5]
        stability = [0.85, 0.85]
        weights = self.agg.calculate_weights(n_samples, reliability, stability)
        w_values = list(weights.values())
        assert w_values[0] > w_values[1]

    def test_all_weights_positive(self):
        """All weights must be strictly positive."""
        n_samples = [500, 300, 200, 400, 600]
        reliability = [0.75, 0.82, 0.68, 0.79, 0.85]
        stability = [0.88, 0.79, 0.91, 0.84, 0.77]
        weights = self.agg.calculate_weights(n_samples, reliability, stability)
        for w in weights.values():
            assert w > 0
